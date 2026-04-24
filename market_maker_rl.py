"""
market_maker_rl.py — Market Making for "The Learning Curve" (Week 5)

Week 5 setup:
  CS1 and CS2 each driven by 6 RL agents (not hundreds of ZI agents).
  RL agents learn purposefully — they create persistent directional
  moves and adapt over time.  Mean reversion breaks down; market making
  captures the +$0.002/share rebate regardless of direction.

Core idea (Avellaneda-Stoikov 2008):
  r  = mid - q * gamma * spread * (T-t) + OFI * ofi_scale * spread
  k  = gamma / (exp(gamma * 0.70 * spread / 2) - 1)
  d  = gamma * sigma2 * (T-t) + (2/gamma) * ln(1 + gamma/k)
  bid = r - d/2,  ask = r + d/2

Why OFI matters here:
  With only 6 RL agents per market the depth imbalance is a strong
  signal — if 70% of resting volume is on the bid, the RL agents are
  leaning long and price will drift up.  We shift r upward so our bid
  gets more aggressive and our ask eases off, riding their flow.

Tuning vs. Week 1/2 A-S:
  gamma:        0.5 → 0.8   stronger inventory skew (RL trends harder)
  max_inventory: 3  → 2     tighter cap (prevent one-sided accumulation)
  quote_refresh: 3.0 → 2.0  faster (RL agents react quickly)
  ofi_scale:    0.0 → 0.30  OFI signal now active
  ofi_depth:          3     read top-3 price levels

Rate budget (2 tickers, 2.0 s refresh):
  cancel 2 + post 2 = 4 msgs / 2.0 s = 2.0 msg/s per ticker
  Total for 2 tickers: 4.0 msg/s  (limit = 5 msg/s, burst = 10)
"""

import shift
import numpy as np
from time import sleep, time
from datetime import datetime
from utils import (
    log, get_position,
    submit_limit_buy, submit_limit_sell,
    submit_market_buy, submit_market_sell,
    cancel_orders_for,
)
from risk_manager import RiskManager


class MarketMakerRL:

    def __init__(
        self,
        trader:         shift.Trader,
        risk_manager:   RiskManager,
        ticker:         str,
        session_start:  datetime,
        session_end:    datetime,
        gamma:          float = 0.8,
        max_inventory:  int   = 2,
        vol_window:     int   = 30,
        quote_refresh:  float = 2.0,
        min_spread_pct: float = 0.0020,
        ofi_scale:      float = 0.30,
        ofi_depth:      int   = 3,
        check_freq:     float = 1.0,
    ):
        self.trader         = trader
        self.risk_manager   = risk_manager
        self.ticker         = ticker
        self.session_start  = session_start
        self.session_end    = session_end
        self.gamma          = gamma
        self.max_inventory  = max_inventory
        self.vol_window     = vol_window
        self.quote_refresh  = quote_refresh
        self.min_spread_pct = min_spread_pct
        self.ofi_scale      = ofi_scale
        self.ofi_depth      = ofi_depth
        self.check_freq     = check_freq

        self.price_history   = []
        self.trade_count     = 0
        self.running         = False
        self.last_quote_time = 0.0
        self.last_bid        = 0.0
        self.last_ask        = 0.0

    def log(self, msg: str):
        log("MMRL", self.ticker, msg)

    # ------------------------------------------------------------------ #
    #  Time remaining [0.0 → 1.0]                                         #
    # ------------------------------------------------------------------ #
    def time_remaining(self) -> float:
        now     = self.trader.get_last_trade_time()
        total   = (self.session_end - self.session_start).total_seconds()
        elapsed = (now - self.session_start).total_seconds()
        if total <= 0:
            return 0.0
        return max(0.0, min(1.0, 1.0 - elapsed / total))

    # ------------------------------------------------------------------ #
    #  Rolling volatility                                                  #
    # ------------------------------------------------------------------ #
    def update_price(self, mid: float):
        self.price_history.append(mid)
        if len(self.price_history) > self.vol_window * 3:
            self.price_history = self.price_history[-self.vol_window * 3:]

    def estimate_sigma2(self) -> float:
        if len(self.price_history) < self.vol_window:
            return 1e-6
        prices  = np.array(self.price_history[-self.vol_window:])
        returns = np.diff(np.log(np.maximum(prices, 1e-8)))
        return max(float(np.var(returns)), 1e-8)

    # ------------------------------------------------------------------ #
    #  Global order book (avoids recursive-spiral from our own quotes)     #
    # ------------------------------------------------------------------ #
    def get_global_book(self):
        """Returns (bid_price, ask_price, bid_book, ask_book) or None."""
        try:
            bid_book = self.trader.get_order_book(
                self.ticker, shift.OrderBookType.GLOBAL_BID, self.ofi_depth
            )
            ask_book = self.trader.get_order_book(
                self.ticker, shift.OrderBookType.GLOBAL_ASK, self.ofi_depth
            )
            if not bid_book or not ask_book:
                return None
            bid_px = float(bid_book[0].price)
            ask_px = float(ask_book[0].price)
            if bid_px <= 0 or ask_px <= 0 or ask_px <= bid_px:
                return None
            return bid_px, ask_px, bid_book, ask_book
        except Exception:
            return None

    # ------------------------------------------------------------------ #
    #  OFI: order-flow imbalance in [-1, +1]                              #
    # ------------------------------------------------------------------ #
    def compute_ofi(self, bid_book, ask_book) -> float:
        try:
            total_bid = sum(row.size for row in bid_book[:self.ofi_depth])
            total_ask = sum(row.size for row in ask_book[:self.ofi_depth])
            total = total_bid + total_ask
            if total <= 0:
                return 0.0
            return (total_bid - total_ask) / total
        except Exception:
            return 0.0

    # ------------------------------------------------------------------ #
    #  A-S quote computation with OFI bias                                #
    # ------------------------------------------------------------------ #
    def compute_quotes(
        self,
        mid:       float,
        bid_mkt:   float,
        ask_mkt:   float,
        inventory: float,
        ofi:       float,
    ) -> tuple:
        """Returns (bid_quote, ask_quote, r, spread, kappa)."""
        sigma2        = self.estimate_sigma2()
        t_rem         = self.time_remaining()
        actual_spread = max(ask_mkt - bid_mkt, 0.01)

        # Beat competing market makers by targeting 70% of their spread
        kappa = self.gamma / (
            np.exp(self.gamma * actual_spread * 0.70 / 2) - 1
        )
        kappa = max(kappa, 0.01)

        # Reservation price: skew away from inventory, lean with RL flow
        r = (
            mid
            - inventory * self.gamma * actual_spread * t_rem
            + ofi * self.ofi_scale * actual_spread
        )

        term1  = self.gamma * sigma2 * t_rem
        term2  = (2.0 / self.gamma) * np.log(1.0 + self.gamma / kappa)
        spread = max(term1 + term2, mid * self.min_spread_pct)

        bid_q = round(r - spread / 2, 2)
        ask_q = round(r + spread / 2, 2)
        return bid_q, ask_q, r, spread, kappa

    # ------------------------------------------------------------------ #
    #  Requote trigger                                                     #
    # ------------------------------------------------------------------ #
    def price_moved(self, bid: float, ask: float) -> bool:
        return (
            abs(bid - self.last_bid) > 0.02 or
            abs(ask - self.last_ask) > 0.02
        )

    # ------------------------------------------------------------------ #
    #  Post one bid + one ask (respects inventory limits)                  #
    # ------------------------------------------------------------------ #
    def post_quotes(
        self,
        bid_q:     float,
        ask_q:     float,
        r:         float,
        spread:    float,
        kappa:     float,
        inventory: float,
        ofi:       float,
    ):
        posted_bid = False
        posted_ask = False

        if inventory < self.max_inventory:
            if self.risk_manager.can_trade(self.ticker, True, bid_q):
                submit_limit_buy(self.trader, self.ticker, 1, bid_q)
                self.trade_count += 1
                posted_bid = True

        if inventory > -self.max_inventory:
            if self.risk_manager.can_trade(self.ticker, False, ask_q):
                submit_limit_sell(self.trader, self.ticker, 1, ask_q)
                self.trade_count += 1
                posted_ask = True

        if posted_bid or posted_ask:
            sides = (
                "BID+ASK" if posted_bid and posted_ask
                else "ASK only" if posted_ask
                else "BID only"
            )
            self.log(
                f"QUOTED {sides} | "
                f"bid={bid_q:.2f} ask={ask_q:.2f} | "
                f"r={r:.2f} spread={spread:.4f} κ={kappa:.2f} | "
                f"OFI={ofi:+.2f} inv={int(inventory)} | "
                f"trades={self.trade_count}"
            )
            self.last_bid        = bid_q
            self.last_ask        = ask_q
            self.last_quote_time = time()

    # ------------------------------------------------------------------ #
    #  Flatten inventory (market orders)                                   #
    # ------------------------------------------------------------------ #
    def close_inventory(self, reason: str):
        cancel_orders_for(self.trader, self.ticker)
        sleep(0.3)
        position = get_position(self.trader, self.ticker)
        if position > 0:
            lots = int(position / 100)
            if lots > 0:
                submit_market_sell(self.trader, self.ticker, lots)
                self.trade_count += 1
                self.log(f"CLOSE ({reason}) MARKET_SELL {lots}L")
        elif position < 0:
            lots = int(-position / 100)
            if lots > 0:
                submit_market_buy(self.trader, self.ticker, lots)
                self.trade_count += 1
                self.log(f"CLOSE ({reason}) MARKET_BUY {lots}L")

    # ------------------------------------------------------------------ #
    #  Main loop                                                           #
    # ------------------------------------------------------------------ #
    def run(self, end_time):
        self.running = True
        self.log("Strategy started — Week 5 RL Market Maker")
        self.log(
            f"γ={self.gamma} | max_inv=±{self.max_inventory} | "
            f"refresh={self.quote_refresh}s | "
            f"OFI_scale={self.ofi_scale} ofi_depth={self.ofi_depth} | "
            f"min_spread={self.min_spread_pct*100:.2f}%"
        )

        self.trader.sub_order_book(self.ticker)
        sleep(0.5)

        # Warmup: collect sigma2 history using global book
        self.log(f"Warming up — need {self.vol_window} samples...")
        while len(self.price_history) < self.vol_window:
            if self.risk_manager.all_halted:
                return
            result = self.get_global_book()
            if result:
                bid_mkt, ask_mkt, _, _ = result
                self.update_price((bid_mkt + ask_mkt) / 2)
            sleep(self.check_freq)
        self.log("Warmup complete — quoting started")

        while (
            self.running and
            self.trader.get_last_trade_time() < end_time
        ):
            if self.risk_manager.all_halted:
                self.log("Global halt — stopping")
                break

            try:
                result = self.get_global_book()
                if result is None:
                    sleep(self.check_freq)
                    continue

                bid_mkt, ask_mkt, bid_book, ask_book = result
                mid = (bid_mkt + ask_mkt) / 2
                self.update_price(mid)

                ofi       = self.compute_ofi(bid_book, ask_book)
                position  = get_position(self.trader, self.ticker)
                inventory = position / 100

                bid_q, ask_q, r, spread, kappa = self.compute_quotes(
                    mid, bid_mkt, ask_mkt, inventory, ofi
                )

                self.log(
                    f"mid={mid:.2f} | "
                    f"mkt=[{bid_mkt:.2f}/{ask_mkt:.2f}] | "
                    f"our=[{bid_q:.2f}/{ask_q:.2f}] | "
                    f"OFI={ofi:+.2f} inv={int(inventory)} "
                    f"T-t={self.time_remaining():.2f}"
                )

                time_expired  = time() - self.last_quote_time >= self.quote_refresh
                price_shifted = self.price_moved(bid_mkt, ask_mkt)

                if time_expired or price_shifted:
                    cancel_orders_for(self.trader, self.ticker)
                    sleep(0.2)

                    # Re-read position after cancels settle
                    position  = get_position(self.trader, self.ticker)
                    inventory = position / 100

                    bid_q, ask_q, r, spread, kappa = self.compute_quotes(
                        mid, bid_mkt, ask_mkt, inventory, ofi
                    )
                    self.post_quotes(
                        bid_q, ask_q, r, spread, kappa, inventory, ofi
                    )

                sleep(self.check_freq)

            except KeyboardInterrupt:
                break
            except Exception as e:
                self.log(f"Error: {e}")
                sleep(self.check_freq)

        self.log("End time — closing inventory")
        self.close_inventory("end_of_session")
        self.trader.unsub_order_book(self.ticker)

        try:
            final_pl = self.trader.get_portfolio_item(self.ticker).get_realized_pl()
            self.log(f"DONE | P&L=${final_pl:.2f} | trades={self.trade_count}")
        except Exception:
            self.log(f"DONE | trades={self.trade_count}")

        self.running = False
