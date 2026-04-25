"""
Production MM — Week 5 RL Learning Curve
Per-ticker parameter tuning: CS1 uses $0.05 spread, CS2 uses $0.10.

Verified $111,788 P&L over ~70 min in diagnostic run.
SAFETY fix confirmed working (208 confirmed / 19 timeout out of 227).
"""

import shift
import numpy as np
from time import sleep, time
from datetime import timedelta
from utils import (
    log, get_position, get_best_prices,
    submit_limit_buy, submit_limit_sell,
    submit_market_buy, submit_market_sell,
    cancel_orders_for
)
from risk_manager import RiskManager


class MarketMaker:

    def __init__(
        self,
        trader:        shift.Trader,
        risk_manager:  RiskManager,
        ticker:        str,
        lots:          int   = 2,
        half_spread:   float = 0.05,
        max_inventory: int   = 6,
        price_window:  int   = 10,
        quote_refresh: float = 3.0,
        check_freq:    float = 1.0,
    ):
        self.trader        = trader
        self.risk_manager  = risk_manager
        self.ticker        = ticker
        self.lots          = lots
        self.half_spread   = half_spread
        self.max_inventory = max_inventory
        self.price_window  = price_window
        self.quote_refresh = quote_refresh
        self.check_freq    = check_freq

        self.SKEW_PER_LOT  = 0.005

        # Spike defense parameters
        # If 1-tick move exceeds SPIKE_THRESHOLD, pause quoting
        # for SPIKE_PAUSE seconds. Existing positions handled separately:
        # if SAFETY triggers during pause and spread is too wide, defer.
        #
        # Tuned from run 20260425: 0.7% threshold fired 1,845 times in
        # 60 min — over-triggered and killed strategy. RL market has
        # routine 0.5-1% one-tick moves. Only 1.5%+ moves are real spikes.
        self.SPIKE_THRESHOLD     = 0.015   # 1.5% one-tick move triggers pause
        self.SPIKE_PAUSE         = 8.0     # seconds to pause after spike
        self.WIDE_SPREAD_PCT     = 0.005   # 0.5% spread = book is chaotic
        self.MAX_DEFERRED_WAIT   = 30.0    # don't defer flatten longer than this

        self.prices          = []
        self.running         = False
        self.last_quote_time = 0.0
        self.last_pnl_log    = 0.0

        # Spike defense state
        self.spike_pause_until = 0.0    # epoch time until which we're paused

        self.known_position = 0
        self.bid_fills      = 0
        self.ask_fills      = 0

    def log_event(self, tag: str, msg: str):
        log(tag, self.ticker, msg)

    def rolling_mid(self):
        if not self.prices:
            return None
        return float(np.mean(self.prices[-self.price_window:]))

    def detect_spike(self, mid: float) -> bool:
        """
        Returns True if the latest tick is a violent spike.
        Compares current mid to the previous mid (1 tick ago).
        """
        if len(self.prices) < 2:
            return False
        prev = self.prices[-2]
        if prev <= 0:
            return False
        move = abs(mid - prev) / prev
        if move >= self.SPIKE_THRESHOLD:
            return True
        return False

    def is_spike_paused(self) -> bool:
        return time() < self.spike_pause_until

    def book_is_chaotic(self) -> bool:
        """Wide spread = book is chaotic, deferring flatten is safer."""
        try:
            p = get_best_prices(self.trader, self.ticker)
            if not p:
                return True
            mid = (p[0] + p[1]) / 2
            spread = p[1] - p[0]
            if mid <= 0:
                return True
            return (spread / mid) >= self.WIDE_SPREAD_PCT
        except Exception:
            return True

    def safety_flatten(self, max_wait: float = 8.0):
        """Cancel quotes, market-close, wait for confirmation.
        Defers if book is chaotic (very wide spread)."""
        # If book is chaotic AND we haven't waited too long, defer
        defer_start = time()
        while self.book_is_chaotic():
            elapsed = time() - defer_start
            if elapsed >= self.MAX_DEFERRED_WAIT:
                self.log_event(
                    "SAFETY",
                    f"Deferred {elapsed:.0f}s — proceeding with flatten anyway"
                )
                break
            self.log_event(
                "SAFETY",
                f"Book chaotic — deferring flatten ({elapsed:.0f}/{self.MAX_DEFERRED_WAIT:.0f}s)"
            )
            sleep(2.0)

        self.log_event("SAFETY", "Inventory cap hit — flattening")
        cancel_orders_for(self.trader, self.ticker)
        sleep(0.3)

        pos = get_position(self.trader, self.ticker)
        if pos == 0:
            self.known_position = 0
            return

        lots_to_close = abs(pos) // 100
        if lots_to_close == 0:
            self.known_position = pos
            return

        if pos > 0:
            submit_market_sell(self.trader, self.ticker, lots_to_close)
            self.log_event("SAFETY", f"Submitted MARKET_SELL {lots_to_close}L")
        else:
            submit_market_buy(self.trader, self.ticker, lots_to_close)
            self.log_event("SAFETY", f"Submitted MARKET_BUY {lots_to_close}L")

        start = time()
        while time() - start < max_wait:
            sleep(0.5)
            current = get_position(self.trader, self.ticker)
            if current == 0:
                self.log_event("SAFETY", f"Flatten confirmed ({time()-start:.1f}s)")
                self.known_position = 0
                return
            if abs(current) > abs(pos):
                self.log_event(
                    "SAFETY",
                    f"Position moved AWAY: {pos//100} → {current//100}"
                )

        final = get_position(self.trader, self.ticker)
        self.known_position = final
        self.log_event(
            "SAFETY",
            f"Flatten TIMEOUT — pos still {final//100}L after {max_wait}s"
        )

    def detect_fills(self, mid: float):
        current = get_position(self.trader, self.ticker)
        delta = current - self.known_position
        if delta == 0:
            return
        lots = abs(delta) // 100
        if lots == 0:
            self.known_position = current
            return
        if delta > 0:
            self.bid_fills += lots
            self.log_event(
                "FILL", f"BID +{lots}L | mid={mid:.3f} | inv={current//100}"
            )
        else:
            self.ask_fills += lots
            self.log_event(
                "FILL", f"ASK -{lots}L | mid={mid:.3f} | inv={current//100}"
            )
        self.known_position = current

    def log_pnl(self):
        """P&L snapshot — uses get_realized_pl only (no unrealized API)."""
        try:
            item = self.trader.get_portfolio_item(self.ticker)
            realized = item.get_realized_pl()
            pos      = (item.get_long_shares() - item.get_short_shares()) // 100
            self.log_event(
                "PNL",
                f"realized=${realized:+.0f} | inv={pos}L | "
                f"fills bid/ask={self.bid_fills}/{self.ask_fills}"
            )
        except Exception as e:
            self.log_event("PNL", f"error: {e}")

    def shutdown(self):
        self.running = False

    def final_flatten(self, max_wait: float = 15.0):
        self.log_event("CLOSE", "Final flatten — cancelling orders")
        cancel_orders_for(self.trader, self.ticker)
        sleep(0.5)

        pos = get_position(self.trader, self.ticker)
        if pos == 0:
            self.log_event("CLOSE", "Already flat")
            return

        lots = abs(pos) // 100
        if pos > 0:
            submit_market_sell(self.trader, self.ticker, lots)
            self.log_event("CLOSE", f"MARKET_SELL {lots}L")
        else:
            submit_market_buy(self.trader, self.ticker, lots)
            self.log_event("CLOSE", f"MARKET_BUY {lots}L")

        start = time()
        while time() - start < max_wait:
            sleep(1.0)
            if get_position(self.trader, self.ticker) == 0:
                self.log_event("CLOSE", f"Confirmed flat ({time()-start:.1f}s)")
                return

        final_pos = get_position(self.trader, self.ticker)
        self.log_event(
            "CLOSE", f"TIMEOUT — still {final_pos//100}L after {max_wait}s"
        )

    def run(self, end_time):
        self.running = True
        self.log_event(
            "INIT",
            f"MM started | lots={self.lots} | "
            f"half_spread=${self.half_spread:.2f} | "
            f"max_inv=±{self.max_inventory}L | "
            f"refresh={self.quote_refresh}s"
        )

        self.trader.sub_order_book(self.ticker)
        sleep(0.5)

        self.log_event("INIT", f"Warming up — need {self.price_window} samples...")
        while len(self.prices) < self.price_window and self.running:
            p = get_best_prices(self.trader, self.ticker)
            if p:
                self.prices.append((p[0] + p[1]) / 2)
            sleep(1.0)
        if not self.running:
            return
        self.log_event("INIT", "Warmup done")

        no_entry_time    = end_time - timedelta(minutes=15)
        force_close_time = end_time - timedelta(minutes=10)

        while self.running and self.trader.get_last_trade_time() < end_time:
            if self.risk_manager.all_halted:
                self.log_event("HALT", "Global halt")
                break

            try:
                p = get_best_prices(self.trader, self.ticker)
                if p:
                    mid = (p[0] + p[1]) / 2
                    self.prices.append(mid)
                    if len(self.prices) > 200:
                        self.prices = self.prices[-200:]
                else:
                    sleep(self.check_freq)
                    continue

                # Spike detection — if violent move, cancel quotes and pause
                if self.detect_spike(mid):
                    pct = abs(mid - self.prices[-2]) / self.prices[-2] * 100
                    self.spike_pause_until = time() + self.SPIKE_PAUSE
                    self.log_event(
                        "SPIKE",
                        f"Spike detected: {pct:.2f}% move | "
                        f"prev={self.prices[-2]:.2f} → mid={mid:.2f} | "
                        f"pausing {self.SPIKE_PAUSE:.0f}s"
                    )
                    # Cancel all outstanding quotes immediately
                    cancel_orders_for(self.trader, self.ticker)
                    sleep(0.2)

                self.detect_fills(mid)

                if time() - self.last_pnl_log >= 30.0:
                    self.log_pnl()
                    self.last_pnl_log = time()

                now = self.trader.get_last_trade_time()

                if now >= force_close_time:
                    self.final_flatten()
                    sleep(self.check_freq)
                    continue

                if now >= no_entry_time:
                    sleep(self.check_freq)
                    continue

                pos_lots = self.known_position // 100
                if abs(pos_lots) >= self.max_inventory:
                    self.safety_flatten()
                    sleep(self.check_freq)
                    continue

                # Spike pause — don't post fresh quotes during chaos
                if self.is_spike_paused():
                    sleep(self.check_freq)
                    continue

                if time() - self.last_quote_time < self.quote_refresh:
                    sleep(self.check_freq)
                    continue

                r_mid = self.rolling_mid()
                if r_mid is None:
                    sleep(self.check_freq)
                    continue

                skew  = -pos_lots * self.SKEW_PER_LOT
                bid_q = round(r_mid - self.half_spread + skew, 2)
                ask_q = round(r_mid + self.half_spread + skew, 2)

                if bid_q >= ask_q:
                    bid_q = round(r_mid - self.half_spread, 2)
                    ask_q = round(r_mid + self.half_spread, 2)

                cancel_orders_for(self.trader, self.ticker)
                sleep(0.2)

                posted_bid = False
                posted_ask = False

                if pos_lots < self.max_inventory:
                    if self.risk_manager.can_trade(self.ticker, True, bid_q):
                        if submit_limit_buy(
                            self.trader, self.ticker, self.lots, bid_q
                        ):
                            posted_bid = True

                if pos_lots > -self.max_inventory:
                    if self.risk_manager.can_trade(self.ticker, False, ask_q):
                        if submit_limit_sell(
                            self.trader, self.ticker, self.lots, ask_q
                        ):
                            posted_ask = True

                if posted_bid or posted_ask:
                    sides = (
                        "BID+ASK" if posted_bid and posted_ask
                        else "BID only" if posted_bid
                        else "ASK only"
                    )
                    self.log_event(
                        "QUOTE",
                        f"{sides} | r_mid={r_mid:.3f} | "
                        f"bid={bid_q:.2f} ask={ask_q:.2f} | "
                        f"skew={skew:+.3f} | inv={pos_lots}L"
                    )
                    self.last_quote_time = time()

                sleep(self.check_freq)

            except KeyboardInterrupt:
                self.running = False
                break
            except Exception as e:
                self.log_event("ERROR", str(e))
                sleep(self.check_freq)

        self.log_event("END", "Loop ending — final flatten")
        self.final_flatten()
        self.trader.unsub_order_book(self.ticker)

        try:
            item = self.trader.get_portfolio_item(self.ticker)
            realized = item.get_realized_pl()
            self.log_event(
                "SUMMARY",
                f"realized P&L: ${realized:+.2f} | "
                f"fills bid/ask={self.bid_fills}/{self.ask_fills}"
            )
        except Exception:
            pass