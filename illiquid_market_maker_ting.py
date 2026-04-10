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
        phase_manager=None,
        volume_min_spread: float | None = None,
        volume_vol_multiplier: float | None = None,
        volume_quote_max_age: float | None = None,
        volume_price_move_pct: float | None = None,
        volume_quote_offset: float | None = None,
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
        self.phase_manager    = phase_manager

        # Volume-mode overrides used until trade-count target is reached.
        self.volume_min_spread     = volume_min_spread if volume_min_spread is not None else min_spread
        self.volume_vol_multiplier = volume_vol_multiplier if volume_vol_multiplier is not None else vol_multiplier
        self.volume_quote_max_age  = volume_quote_max_age if volume_quote_max_age is not None else quote_max_age
        self.volume_price_move_pct = volume_price_move_pct if volume_price_move_pct is not None else price_move_pct
        self.volume_quote_offset   = volume_quote_offset if volume_quote_offset is not None else quote_offset

        # State
        self.price_history:   list[float] = []
        self.spread_samples:  list[float] = []
        self.bid_posted:      float | None = None  # price of our resting buy
        self.ask_posted:      float | None = None  # price of our resting sell
        self.quote_posted_at: float        = 0.0
        self.trade_count:     int          = 0     # total orders placed by this strategy
        self.round_trips:     int          = 0     # full buy+sell cycles
        self._prev_position:  int          = 0
        self.running:         bool         = False
        self._last_phase:     str | None   = None

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

    def current_phase(self) -> str:
        if self.phase_manager is None:
            return "sharpe"
        return self.phase_manager.current_phase()

    def get_mode_params(self) -> dict[str, float]:
        phase = self.current_phase()
        if phase == "volume":
            return {
                "phase": phase,
                "min_spread": self.volume_min_spread,
                "vol_multiplier": self.volume_vol_multiplier,
                "quote_max_age": self.volume_quote_max_age,
                "price_move_pct": self.volume_price_move_pct,
                "quote_offset": self.volume_quote_offset,
            }
        return {
            "phase": phase,
            "min_spread": self.min_spread,
            "vol_multiplier": self.vol_multiplier,
            "quote_max_age": self.quote_max_age,
            "price_move_pct": self.price_move_pct,
            "quote_offset": self.quote_offset,
        }

    def log_phase_change_if_needed(self, params: dict[str, float]):
        phase = str(params["phase"])
        if phase == self._last_phase:
            return
        self._last_phase = phase
        if phase == "volume":
            self.log(
                "PHASE → VOLUME | "
                f"min_spread=${params['min_spread']:.4f} | "
                f"vol_mult={params['vol_multiplier']:.2f} | "
                f"quote_age={params['quote_max_age']:.2f}s"
            )
        else:
            self.log(
                "PHASE → SHARPE | "
                f"min_spread=${params['min_spread']:.4f} | "
                f"vol_mult={params['vol_multiplier']:.2f} | "
                f"quote_age={params['quote_max_age']:.2f}s"
            )

    def record_order_submission(self, count: int = 1):
        self.trade_count += count
        if self.phase_manager is not None:
            self.phase_manager.record_orders(count)

    # ------------------------------------------------------------------ #
    #  Dynamic spread floor                                                #
    # ------------------------------------------------------------------ #
    def compute_dynamic_spread_floor(
        self,
        current_price: float,
        min_spread_floor: float,
        vol_multiplier: float,
    ) -> float:
        """
        Raise the minimum required spread when the stock is moving fast.

        Rationale: high volatility → high adverse-selection risk.  A wide
        spread is no protection if price gaps $0.50 while our quote sits in
        the book.  We require the spread to be at least vol_multiplier times
        the per-tick price volatility so we have genuine edge after expected
        adverse moves.
        """
        if len(self.price_history) < self.vol_window + 1 or current_price <= 0:
            return min_spread_floor

        prices       = np.array(self.price_history[-(self.vol_window + 1):])
        log_rets     = np.diff(np.log(prices + 1e-10))
        vol_per_tick = float(np.std(log_rets)) * current_price

        dynamic_floor = max(min_spread_floor, vol_multiplier * vol_per_tick)
        return round(dynamic_floor, 4)

    # ------------------------------------------------------------------ #
    #  Stale-quote check                                                   #
    # ------------------------------------------------------------------ #
    def quotes_stale(self, bid: float, ask: float, params: dict[str, float]) -> bool:
        if time() - self.quote_posted_at > float(params["quote_max_age"]):
            return True
        if self.bid_posted and self.ask_posted:
            mid_then = (self.bid_posted + self.ask_posted) / 2
            mid_now  = (bid + ask) / 2
            if mid_then > 0:
                move = abs(mid_now - mid_then) / mid_then
                if move > float(params["price_move_pct"]):
                    return True
        return False

    # ------------------------------------------------------------------ #
    #  Quote posting                                                       #
    # ------------------------------------------------------------------ #
    def post_quotes(self, bid: float, ask: float, position: int, params: dict[str, float]) -> bool:
        """
        Post two-sided limit orders, skewed toward the flat side.

        Returns True if at least one order was submitted.
        """
        mid    = (bid + ask) / 2
        spread = ask - bid

        eff_min = self.compute_dynamic_spread_floor(
            current_price=mid,
            min_spread_floor=float(params["min_spread"]),
            vol_multiplier=float(params["vol_multiplier"]),
        )
        if spread < eff_min:
            raised = eff_min > float(params["min_spread"])
            self.log(
                f"Spread ${spread:.4f} < "
                f"{'dynamic' if raised else 'base'} floor ${eff_min:.4f} — skip"
                + (f" (vol raised floor from ${float(params['min_spread']):.2f})" if raised else "")
            )
            return False

        my_bid = round(bid + float(params["quote_offset"]), 2)
        my_ask = round(ask - float(params["quote_offset"]), 2)

        if my_ask <= my_bid:
            my_bid, my_ask = round(bid, 2), round(ask, 2)

        skew = round(self.skew_ticks * self.TICK, 2)
        if position > 0:
            my_ask = round(my_ask - skew, 2)
        elif position < 0:
            my_bid = round(my_bid + skew, 2)

        if my_ask <= my_bid:
            my_bid, my_ask = round(bid, 2), round(ask, 2)

        bid_ok = ask_ok = False

        if position < self.max_pos_shares:
            bp_needed = my_bid * self.lot_size * 100
            if self.risk_manager.has_buying_power(bp_needed):
                if submit_limit_buy(self.trader, self.ticker, self.lot_size, my_bid):
                    self.bid_posted = my_bid
                    bid_ok = True
                    self.record_order_submission(1)

        if position > -self.max_pos_shares:
            if submit_limit_sell(self.trader, self.ticker, self.lot_size, my_ask):
                self.ask_posted = my_ask
                ask_ok = True
                self.record_order_submission(1)

        if bid_ok or ask_ok:
            self.quote_posted_at = time()
            phase = str(params["phase"]).upper()
            total_orders = self.phase_manager.total_orders() if self.phase_manager else self.trade_count
            self.log(
                f"QUOTE[{phase}] | "
                f"bid={my_bid:.2f}[{'Y' if bid_ok else 'N'}] "
                f"ask={my_ask:.2f}[{'Y' if ask_ok else 'N'}] | "
                f"mkt={bid:.2f}/{ask:.2f} spread=${spread:.4f} | "
                f"pos={position} | local_orders={self.trade_count} | "
                f"team_orders={total_orders}"
            )
        return bid_ok or ask_ok

    # ------------------------------------------------------------------ #
    #  Main loop                                                           #
    # ------------------------------------------------------------------ #
    def run(self, end_time):
        self.running = True
        self.log("Strategy started")
        self.log(
            f"Config | min_spread=${self.min_spread:.4f} | "
            f"vol_mult={self.vol_multiplier:.2f} | "
            f"quote_age={self.quote_max_age:.2f}s | "
            f"move_pct={self.price_move_pct*100:.2f}% | "
            f"lot={self.lot_size}"
        )

        self.trader.sub_order_book(self.ticker)

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

                params = self.get_mode_params()
                self.log_phase_change_if_needed(params)
                position = get_position(self.trader, self.ticker)

                if self._prev_position != 0 and position == 0:
                    self.round_trips += 1
                    self.log(f"Round trip #{self.round_trips} complete")
                self._prev_position = position

                phase = str(params["phase"]).upper()
                self.log(
                    f"{phase} | bid={bid:.2f} ask={ask:.2f} "
                    f"spread=${ask - bid:.4f} | pos={position}"
                )

                if self.has_pending():
                    if self.quotes_stale(bid, ask, params):
                        self.cancel_quotes()
                        sleep(0.2)
                        self.post_quotes(bid, ask, position, params)
                else:
                    self.bid_posted = None
                    self.ask_posted = None
                    self.post_quotes(bid, ask, position, params)

                sleep(self.check_freq)

            except KeyboardInterrupt:
                break
            except Exception as e:
                self.log(f"Error: {e}")
                sleep(self.check_freq)

        self.log("Trading window closed — cancelling quotes and closing position")
        self.cancel_quotes()
        sleep(1)
        close_position(self.trader, self.ticker)
        self.trader.unsub_order_book(self.ticker)

        try:
            final_pl = self.trader.get_portfolio_item(self.ticker).get_realized_pl()
        except Exception:
            final_pl = 0.0

        total_orders = self.phase_manager.total_orders() if self.phase_manager else self.trade_count
        self.log(
            f"DONE | P&L=${final_pl:.2f} | "
            f"local_orders={self.trade_count} | "
            f"team_orders={total_orders} | "
            f"round_trips={self.round_trips}"
        )
        self.running = False
