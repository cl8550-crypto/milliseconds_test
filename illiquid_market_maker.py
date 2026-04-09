"""
illiquid_market_maker.py — Week 3 "Illiquid" strategy

Two-sided market maker for small-cap / micro-cap stocks.

Core idea
---------
Post a resting limit BUY 1 tick inside the current best bid and a resting
limit SELL 1 tick inside the current best ask simultaneously.  Both fills
earn the +$0.002/share limit-order rebate.  The round-trip profit is:

    (ask_fill – bid_fill) + $0.004/share  (positive as long as spread > 0)

Sharpe maximisation
-------------------
Consistent, spread-captured income has very low return variance — the ideal
profile for Sharpe.  Inventory skewing (nudging quotes toward the flat side
when we're carrying a position) ensures we rarely sit long / short for long,
which would add variance.

Quote refresh
-------------
Cancel and repost when EITHER:
  • quotes are older than quote_max_age seconds, OR
  • mid-price has moved ≥ price_move_pct since the quotes were placed.
Stale quotes in a moving market are the biggest risk for a market maker.

Inventory limits
----------------
Hard cap at ±max_position_lots × 100 shares.  If the cap is reached on one
side we stop posting quotes on that side until the position normalises.
"""

import shift
import numpy as np
from time import sleep, time
from utils import (
    log, get_position, get_best_prices,
    submit_limit_buy, submit_limit_sell,
    cancel_orders_for, close_position,
)
from risk_manager import RiskManager


class IlliquidMarketMaker:

    TICK = 0.01  # minimum price step on SHIFT

    def __init__(
        self,
        trader: shift.Trader,
        risk_manager: RiskManager,
        ticker: str,
        lot_size: int = 1,             # 1 lot = 100 shares (keep small)
        max_position_lots: int = 2,    # hard cap: ±200 shares
        min_spread: float = 0.06,      # absolute floor — never quote below this
        vol_window: int = 20,          # ticks used to estimate per-tick volatility
        vol_multiplier: float = 2.0,   # spread must be ≥ vol_multiplier × per-tick-vol
        quote_offset: float = 0.01,    # improve best price by 1 tick
        skew_ticks: int = 1,           # extra aggression on the unwind side
        quote_max_age: float = 3.0,    # seconds before a forced repost
        price_move_pct: float = 0.003, # 0.3% mid move triggers repost
        check_freq: float = 1.0,       # main-loop sleep (seconds)
        warmup_ticks: int = 15,        # price samples before first quote
    ):
        self.trader           = trader
        self.risk_manager     = risk_manager
        self.ticker           = ticker
        self.lot_size         = lot_size
        self.max_pos_shares   = max_position_lots * 100
        self.min_spread       = min_spread          # absolute floor
        self.vol_window       = vol_window
        self.vol_multiplier   = vol_multiplier
        self.quote_offset     = quote_offset
        self.skew_ticks       = skew_ticks
        self.quote_max_age    = quote_max_age
        self.price_move_pct   = price_move_pct
        self.check_freq       = check_freq
        self.warmup_ticks     = warmup_ticks

        # State
        self.price_history:   list[float] = []
        self.spread_samples:  list[float] = []
        self.bid_posted:      float | None = None  # price of our resting buy
        self.ask_posted:      float | None = None  # price of our resting sell
        self.quote_posted_at: float        = 0.0
        self.trade_count:     int          = 0     # total orders placed
        self.round_trips:     int          = 0     # full buy+sell cycles
        self._prev_position:  int          = 0
        self.running:         bool         = False

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #
    def log(self, msg: str):
        log("MM", self.ticker, msg)

    def has_pending(self) -> bool:
        return any(
            o.symbol == self.ticker
            for o in self.trader.get_waiting_list()
        )

    def cancel_quotes(self):
        cancel_orders_for(self.trader, self.ticker)
        self.bid_posted = None
        self.ask_posted = None

    # ------------------------------------------------------------------ #
    #  Dynamic spread floor                                                #
    # ------------------------------------------------------------------ #
    def compute_dynamic_spread_floor(self, current_price: float) -> float:
        """
        Raise the minimum required spread when the stock is moving fast.

        Rationale: high volatility → high adverse-selection risk.  A wide
        spread is no protection if price gaps $0.50 while our quote sits in
        the book.  We require the spread to be at least vol_multiplier times
        the per-tick price volatility so we have genuine edge after expected
        adverse moves.

        Formula:
            per_tick_vol  = std(log-returns over vol_window ticks) × price
            dynamic_floor = max(min_spread, vol_multiplier × per_tick_vol)

        Examples (vol_multiplier = 2.0, price = $10):
            calm  vol 0.1%/tick → per_tick_vol=$0.010 → floor = $0.06 (base)
            normal vol 0.5%/tick → per_tick_vol=$0.050 → floor = $0.10
            spiking 2%/tick  → per_tick_vol=$0.200 → floor = $0.40
        """
        if len(self.price_history) < self.vol_window + 1 or current_price <= 0:
            return self.min_spread

        prices     = np.array(self.price_history[-(self.vol_window + 1):])
        log_rets   = np.diff(np.log(prices + 1e-10))   # guard against log(0)
        vol_per_tick = float(np.std(log_rets)) * current_price

        dynamic_floor = max(self.min_spread, self.vol_multiplier * vol_per_tick)
        return round(dynamic_floor, 4)

    # ------------------------------------------------------------------ #
    #  Stale-quote check                                                   #
    # ------------------------------------------------------------------ #
    def quotes_stale(self, bid: float, ask: float) -> bool:
        # Age check
        if time() - self.quote_posted_at > self.quote_max_age:
            return True
        # Price-movement check (only when we have posted prices to compare)
        if self.bid_posted and self.ask_posted:
            mid_then = (self.bid_posted + self.ask_posted) / 2
            mid_now  = (bid + ask) / 2
            if mid_then > 0:
                move = abs(mid_now - mid_then) / mid_then
                if move > self.price_move_pct:
                    return True
        return False

    # ------------------------------------------------------------------ #
    #  Quote posting                                                       #
    # ------------------------------------------------------------------ #
    def post_quotes(self, bid: float, ask: float, position: int) -> bool:
        """
        Post two-sided limit orders, skewed toward the flat side.

        Returns True if at least one order was submitted.
        """
        mid    = (bid + ask) / 2
        spread = ask - bid

        # Dynamic floor: rises automatically during high-volatility periods
        eff_min = self.compute_dynamic_spread_floor(mid)
        if spread < eff_min:
            raised = eff_min > self.min_spread
            self.log(
                f"Spread ${spread:.4f} < "
                f"{'dynamic' if raised else 'base'} floor ${eff_min:.4f} — skip"
                + (f" (vol raised floor from ${self.min_spread:.2f})" if raised else "")
            )
            return False

        # ---- Compute quote prices ---- #
        # Start 1 tick inside each best price (we lead the queue)
        my_bid = round(bid + self.quote_offset, 2)
        my_ask = round(ask - self.quote_offset, 2)

        # Never let our quotes cross
        if my_ask <= my_bid:
            my_bid, my_ask = round(bid, 2), round(ask, 2)

        # ---- Inventory skewing ---- #
        # If long → more aggressive ask (lower price) to sell faster
        # If short → more aggressive bid (higher price) to cover faster
        skew = round(self.skew_ticks * self.TICK, 2)
        if position > 0:
            my_ask = round(my_ask - skew, 2)
        elif position < 0:
            my_bid = round(my_bid + skew, 2)

        # Re-check crossing after skew
        if my_ask <= my_bid:
            my_bid, my_ask = round(bid, 2), round(ask, 2)

        bid_ok = ask_ok = False

        # ---- BUY side: only if below max long ---- #
        if position < self.max_pos_shares:
            bp_needed = my_bid * self.lot_size * 100
            if self.risk_manager.has_buying_power(bp_needed):
                if submit_limit_buy(
                    self.trader, self.ticker, self.lot_size, my_bid
                ):
                    self.bid_posted = my_bid
                    bid_ok = True
                    self.trade_count += 1

        # ---- SELL side: only if above max short ---- #
        if position > -self.max_pos_shares:
            if submit_limit_sell(
                self.trader, self.ticker, self.lot_size, my_ask
            ):
                self.ask_posted = my_ask
                ask_ok = True
                self.trade_count += 1

        if bid_ok or ask_ok:
            self.quote_posted_at = time()
            self.log(
                f"QUOTE | "
                f"bid={my_bid:.2f}[{'Y' if bid_ok else 'N'}] "
                f"ask={my_ask:.2f}[{'Y' if ask_ok else 'N'}] | "
                f"mkt={bid:.2f}/{ask:.2f} spread=${spread:.4f} | "
                f"pos={position} | orders={self.trade_count}"
            )
        return bid_ok or ask_ok

    # ------------------------------------------------------------------ #
    #  Main loop                                                           #
    # ------------------------------------------------------------------ #
    def run(self, end_time):
        self.running = True
        self.log(
            f"Start | lot={self.lot_size} | "
            f"max_pos=±{self.max_pos_shares} shares | "
            f"spread_floor=${self.min_spread:.2f} (base) + "
            f"{self.vol_multiplier}×vol (dynamic) | "
            f"vol_window={self.vol_window} ticks | "
            f"offset=±${self.quote_offset:.2f} | "
            f"skew={self.skew_ticks} tick(s) | "
            f"refresh={self.quote_max_age}s / {self.price_move_pct*100:.2f}%"
        )

        self.trader.sub_order_book(self.ticker)

        # ---- Warmup: collect price / spread samples ---- #
        ticks = 0
        self.log(f"Warmup — collecting {self.warmup_ticks} price samples...")
        while ticks < self.warmup_ticks:
            if self.risk_manager.all_halted:
                self.trader.unsub_order_book(self.ticker)
                return
            prices = get_best_prices(self.trader, self.ticker)
            if prices:
                bid, ask = prices
                self.price_history.append((bid + ask) / 2)
                self.spread_samples.append(ask - bid)
                ticks += 1
            sleep(self.check_freq)

        avg_sp = np.mean(self.spread_samples) if self.spread_samples else 0.0
        self.log(
            f"Warmup done | avg_spread=${avg_sp:.4f} | "
            f"{'WILL QUOTE' if avg_sp >= self.min_spread else 'SPREAD TOO TIGHT — idle'}"
        )

        # ---- Main trading loop ---- #
        while self.running and self.trader.get_last_trade_time() < end_time:
            if self.risk_manager.all_halted:
                self.log("Global halt — stopping")
                break

            try:
                prices = get_best_prices(self.trader, self.ticker)
                if prices is None:
                    sleep(self.check_freq)
                    continue

                bid, ask = prices
                mid      = (bid + ask) / 2
                self.price_history.append(mid)
                if len(self.price_history) > 300:
                    self.price_history = self.price_history[-300:]

                position = get_position(self.trader, self.ticker)

                # Detect completed round trips (position returned to 0)
                if self._prev_position != 0 and position == 0:
                    self.round_trips += 1
                    self.log(f"Round trip #{self.round_trips} complete")
                self._prev_position = position

                self.log(
                    f"bid={bid:.2f} ask={ask:.2f} "
                    f"spread=${ask - bid:.4f} | pos={position}"
                )

                if self.has_pending():
                    # Quotes are sitting in the book — only disturb if stale
                    if self.quotes_stale(bid, ask):
                        self.cancel_quotes()
                        sleep(0.2)
                        self.post_quotes(bid, ask, position)
                    # else: leave resting orders untouched
                else:
                    # No pending orders — either just filled or never posted
                    self.bid_posted = None
                    self.ask_posted = None
                    self.post_quotes(bid, ask, position)

                sleep(self.check_freq)

            except KeyboardInterrupt:
                break
            except Exception as e:
                self.log(f"Error: {e}")
                sleep(self.check_freq)

        # ---- End-of-trading cleanup ---- #
        self.log("Trading window closed — cancelling quotes and closing position")
        self.cancel_quotes()
        sleep(1)
        close_position(self.trader, self.ticker)
        self.trader.unsub_order_book(self.ticker)

        try:
            final_pl = self.trader.get_portfolio_item(
                self.ticker
            ).get_realized_pl()
        except Exception:
            final_pl = 0.0

        self.log(
            f"DONE | P&L=${final_pl:.2f} | "
            f"orders_placed={self.trade_count} | "
            f"round_trips={self.round_trips}"
        )
        self.running = False
