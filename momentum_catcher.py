import shift
import numpy as np
from time import sleep, time
from datetime import datetime
from utils import (
    log, get_position,
    submit_limit_buy, submit_limit_sell,
    submit_market_buy, submit_market_sell,
    cancel_orders_for, close_position,
    get_best_prices
)
from risk_manager import RiskManager


# ------------------------------------------------------------------ #
#  Per-ticker calibration                                              #
# ------------------------------------------------------------------ #
TICKER_CONFIG = {
    "NVDA": {"spike_threshold": 0.020, "momentum_hold": 25, "reversal_hold": 45},
    "AAPL": {"spike_threshold": 0.015, "momentum_hold": 30, "reversal_hold": 45},
    "AMZN": {"spike_threshold": 0.018, "momentum_hold": 25, "reversal_hold": 40},
    "CRM":  {"spike_threshold": 0.020, "momentum_hold": 25, "reversal_hold": 40},
    "NKE":  {"spike_threshold": 0.015, "momentum_hold": 20, "reversal_hold": 40},
    "BA":   {"spike_threshold": 0.015, "momentum_hold": 30, "reversal_hold": 50},
    "CAT":  {"spike_threshold": 0.013, "momentum_hold": 25, "reversal_hold": 45},
    "HD":   {"spike_threshold": 0.012, "momentum_hold": 25, "reversal_hold": 40},
    "GS":   {"spike_threshold": 0.015, "momentum_hold": 30, "reversal_hold": 50},
    "JPM":  {"spike_threshold": 0.012, "momentum_hold": 30, "reversal_hold": 50},
    "AXP":  {"spike_threshold": 0.012, "momentum_hold": 30, "reversal_hold": 45},
    "V":    {"spike_threshold": 0.010, "momentum_hold": 30, "reversal_hold": 45},
}

DEFAULT_CONFIG = {
    "spike_threshold": 0.015,
    "momentum_hold":   30,
    "reversal_hold":   45,
}


class MomentumCatcher:
    def __init__(
        self,
        trader: shift.Trader,
        risk_manager: RiskManager,
        ticker: str,
        order_size: int = 3,
        momentum_window: int = 10,
        reversal_threshold: float = 0.005,
        origin_exit_pct: float = 0.003,
        stop_loss_pct: float = 0.025,
        check_freq: float = 1.0,
        cooldown: float = 20.0,
        max_history: int = 300,
    ):
        self.trader             = trader
        self.risk_manager       = risk_manager
        self.ticker             = ticker
        self.order_size         = order_size
        self.momentum_window    = momentum_window
        self.reversal_threshold = reversal_threshold
        self.origin_exit_pct    = origin_exit_pct
        self.stop_loss_pct      = stop_loss_pct
        self.check_freq         = check_freq
        self.cooldown           = cooldown
        self.max_history        = max_history

        cfg = TICKER_CONFIG.get(ticker, DEFAULT_CONFIG)
        self.spike_threshold  = cfg["spike_threshold"]
        self.momentum_hold    = cfg["momentum_hold"]
        self.reversal_hold    = cfg["reversal_hold"]

        self.price_history    = []
        self.trade_count      = 0
        self.last_trade_time  = 0
        self.running          = False

        self.phase             = None
        self.phase_entry_time  = None
        self.phase_entry_price = None
        self.spike_origin      = None
        self.spike_magnitude   = 0.0

        self.spikes_detected   = 0
        self.momentum_trades   = 0
        self.reversal_trades   = 0
        self.stop_loss_hits    = 0

    # ------------------------------------------------------------------ #
    #  Logging                                                             #
    # ------------------------------------------------------------------ #
    def log(self, message: str):
        log("MOM", self.ticker, message)

    # ------------------------------------------------------------------ #
    #  Cooldown / pending                                                  #
    # ------------------------------------------------------------------ #
    def is_cooling_down(self) -> bool:
        return (time() - self.last_trade_time) < self.cooldown

    def has_pending_orders(self) -> bool:
        return any(
            o.symbol == self.ticker
            for o in self.trader.get_waiting_list()
        )

    # ------------------------------------------------------------------ #
    #  Price history                                                       #
    # ------------------------------------------------------------------ #
    def update_history(self, mid: float):
        self.price_history.append((time(), mid))
        if len(self.price_history) > self.max_history:
            self.price_history = self.price_history[-self.max_history:]

    def current_price(self) -> float | None:
        return self.price_history[-1][1] if self.price_history else None

    def price_n_seconds_ago(self, n: int) -> float | None:
        cutoff = time() - n
        window = [(ts, p) for ts, p in self.price_history if ts >= cutoff]
        return window[0][1] if window else None

    # ------------------------------------------------------------------ #
    #  Spike detection                                                     #
    # ------------------------------------------------------------------ #
    def detect_spike(self) -> tuple[str, float] | tuple[None, None]:
        current = self.current_price()
        past    = self.price_n_seconds_ago(self.momentum_window)
        if not current or not past or past <= 0:
            return None, None
        pct = (current - past) / past
        if pct > self.spike_threshold:
            return "SPIKE_UP", abs(pct)
        elif pct < -self.spike_threshold:
            return "SPIKE_DOWN", abs(pct)
        return None, None

    # ------------------------------------------------------------------ #
    #  Reversal detection                                                  #
    # ------------------------------------------------------------------ #
    def detect_early_reversal(self) -> bool:
        current = self.current_price()
        if not current or self.phase_entry_price is None:
            return False
        if self.phase == "momentum_long":
            return (self.phase_entry_price - current) / self.phase_entry_price > self.reversal_threshold
        elif self.phase == "momentum_short":
            return (current - self.phase_entry_price) / self.phase_entry_price > self.reversal_threshold
        return False

    def reversal_complete(self) -> bool:
        current = self.current_price()
        if not current or self.spike_origin is None:
            return False
        return abs(current - self.spike_origin) / self.spike_origin < self.origin_exit_pct

    # ------------------------------------------------------------------ #
    #  Hold time checks                                                    #
    # ------------------------------------------------------------------ #
    def momentum_hold_expired(self) -> bool:
        if self.phase_entry_time is None:
            return False
        return (time() - self.phase_entry_time) > self.momentum_hold

    def reversal_hold_expired(self) -> bool:
        if self.phase_entry_time is None:
            return False
        return (time() - self.phase_entry_time) > self.reversal_hold

    # ------------------------------------------------------------------ #
    #  Stop loss                                                           #
    # ------------------------------------------------------------------ #
    def stop_loss_hit(self) -> bool:
        current = self.current_price()
        if not current or self.phase_entry_price is None:
            return False

        if self.phase in ["momentum_long", "reversal_long"]:
            loss = (self.phase_entry_price - current) / self.phase_entry_price
        elif self.phase in ["momentum_short", "reversal_short"]:
            loss = (current - self.phase_entry_price) / self.phase_entry_price
        else:
            return False

        if loss > self.stop_loss_pct:
            self.log(
                f"STOP LOSS | entry={self.phase_entry_price:.2f} | "
                f"current={current:.2f} | loss={loss*100:.2f}%"
            )
            self.stop_loss_hits += 1
            return True
        return False

    # ------------------------------------------------------------------ #
    #  Phase reset                                                         #
    # ------------------------------------------------------------------ #
    def reset_phase(self):
        self.phase            = None
        self.phase_entry_time  = None
        self.phase_entry_price = None
        self.spike_origin      = None
        self.spike_magnitude   = 0.0

    # ------------------------------------------------------------------ #
    #  Phase transitions                                                   #
    # ------------------------------------------------------------------ #
    def enter_momentum_long(self, magnitude: float):
        prices = get_best_prices(self.trader, self.ticker)
        if prices is None:
            return
        bid, ask = prices

        lots = self.risk_manager.get_safe_lot_size(ask, self.order_size)
        if lots <= 0 or not self.risk_manager.can_trade(self.ticker, True, ask):
            return

        result = submit_limit_buy(self.trader, self.ticker, lots, round(ask, 2))
        if result:
            self.phase            = "momentum_long"
            self.phase_entry_time  = time()
            self.phase_entry_price = ask
            self.last_trade_time   = time()
            self.trade_count      += 1
            self.momentum_trades  += 1
            self.log(
                f"🔴 MOMENTUM LONG | spike={magnitude*100:.2f}% | "
                f"BUY @ {ask:.2f} | hold_max={self.momentum_hold}s | "
                f"trades={self.trade_count}"
            )

    def enter_momentum_short(self, magnitude: float):
        prices = get_best_prices(self.trader, self.ticker)
        if prices is None:
            return
        bid, ask = prices

        lots = self.risk_manager.get_safe_lot_size(bid, self.order_size)
        if lots <= 0 or not self.risk_manager.can_trade(self.ticker, False, bid):
            return

        result = submit_limit_sell(self.trader, self.ticker, lots, round(bid, 2))
        if result:
            self.phase            = "momentum_short"
            self.phase_entry_time  = time()
            self.phase_entry_price = bid
            self.last_trade_time   = time()
            self.trade_count      += 1
            self.momentum_trades  += 1
            self.log(
                f"🔵 MOMENTUM SHORT | spike={magnitude*100:.2f}% | "
                f"SELL @ {bid:.2f} | hold_max={self.momentum_hold}s | "
                f"trades={self.trade_count}"
            )

    def enter_reversal_short(self, reason: str):
        self.force_close_position()
        sleep(1)

        prices = get_best_prices(self.trader, self.ticker)
        if prices is None:
            return
        bid, ask = prices

        lots = self.risk_manager.get_safe_lot_size(bid, self.order_size)
        if lots <= 0 or not self.risk_manager.can_trade(self.ticker, False, bid):
            return

        result = submit_limit_sell(self.trader, self.ticker, lots, round(bid, 2))
        if result:
            self.phase            = "reversal_short"
            self.phase_entry_time  = time()
            self.phase_entry_price = bid
            self.last_trade_time   = time()
            self.trade_count      += 1
            self.reversal_trades  += 1
            self.log(
                f"↘ REVERSAL SHORT | reason={reason} | "
                f"SELL @ {bid:.2f} | origin={self.spike_origin:.2f} | "
                f"hold_max={self.reversal_hold}s | trades={self.trade_count}"
            )

    def enter_reversal_long(self, reason: str):
        self.force_close_position()
        sleep(1)

        prices = get_best_prices(self.trader, self.ticker)
        if prices is None:
            return
        bid, ask = prices

        lots = self.risk_manager.get_safe_lot_size(ask, self.order_size)
        if lots <= 0 or not self.risk_manager.can_trade(self.ticker, True, ask):
            return

        result = submit_limit_buy(self.trader, self.ticker, lots, round(ask, 2))
        if result:
            self.phase            = "reversal_long"
            self.phase_entry_time  = time()
            self.phase_entry_price = ask
            self.last_trade_time   = time()
            self.trade_count      += 1
            self.reversal_trades  += 1
            self.log(
                f"↗ REVERSAL LONG | reason={reason} | "
                f"BUY @ {ask:.2f} | origin={self.spike_origin:.2f} | "
                f"hold_max={self.reversal_hold}s | trades={self.trade_count}"
            )

    # ------------------------------------------------------------------ #
    #  Position closing                                                    #
    # ------------------------------------------------------------------ #
    def force_close_position(self):
        cancel_orders_for(self.trader, self.ticker)
        sleep(0.5)

        item         = self.trader.get_portfolio_item(self.ticker)
        long_shares  = item.get_long_shares()
        short_shares = item.get_short_shares()

        if long_shares > 0:
            lots = int(long_shares / 100)
            submit_market_sell(self.trader, self.ticker, lots)
            self.trade_count += 1
            self.log(f"FORCE CLOSE LONG | {long_shares} shares")

        if short_shares > 0:
            lots = int(short_shares / 100)
            submit_market_buy(self.trader, self.ticker, lots)
            self.trade_count += 1
            self.log(f"FORCE CLOSE SHORT | {short_shares} shares")

        self.reset_phase()
        self.last_trade_time = time()

    def close_all(self):
        cancel_orders_for(self.trader, self.ticker)
        sleep(2)
        close_position(self.trader, self.ticker)

    # ------------------------------------------------------------------ #
    #  Main loop                                                           #
    # ------------------------------------------------------------------ #
    def run(self, end_time):
        self.running = True
        self.log("Strategy started")
        self.log(
            f"Config | spike={self.spike_threshold*100:.1f}% | "
            f"mom_hold={self.momentum_hold}s | rev_hold={self.reversal_hold}s | "
            f"stop={self.stop_loss_pct*100:.1f}%"
        )

        self.trader.sub_order_book(self.ticker)

        warmup_needed = self.momentum_window * 2
        self.log(f"Warming up — need {warmup_needed} samples...")

        while len(self.price_history) < warmup_needed:
            if self.risk_manager.all_halted:
                self.log("Halt during warmup — exiting")
                self.trader.unsub_order_book(self.ticker)
                return

            prices = get_best_prices(self.trader, self.ticker)
            if prices:
                bid, ask = prices
                self.update_history((bid + ask) / 2)
            sleep(self.check_freq)

        self.log("Warmup complete — monitoring for spikes")

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
                position = get_position(self.trader, self.ticker)
                self.update_history(mid)

                self.log(
                    f"mid={mid:.2f} | phase={self.phase} | pos={position}"
                )

                # ---- FLAT ---- #
                if self.phase is None:
                    if self.is_cooling_down() or self.has_pending_orders():
                        sleep(self.check_freq)
                        continue

                    spike, magnitude = self.detect_spike()

                    if spike == "SPIKE_UP":
                        self.spike_origin    = self.price_n_seconds_ago(self.momentum_window)
                        self.spike_magnitude = magnitude
                        self.spikes_detected += 1
                        self.log(
                            f"🔴 SPIKE UP | {magnitude*100:.2f}% | "
                            f"origin={self.spike_origin:.2f} | "
                            f"spike#{self.spikes_detected}"
                        )
                        self.enter_momentum_long(magnitude)

                    elif spike == "SPIKE_DOWN":
                        self.spike_origin    = self.price_n_seconds_ago(self.momentum_window)
                        self.spike_magnitude = magnitude
                        self.spikes_detected += 1
                        self.log(
                            f"🔵 SPIKE DOWN | {magnitude*100:.2f}% | "
                            f"origin={self.spike_origin:.2f} | "
                            f"spike#{self.spikes_detected}"
                        )
                        self.enter_momentum_short(magnitude)

                # ---- MOMENTUM LONG ---- #
                elif self.phase == "momentum_long":
                    if self.stop_loss_hit():
                        self.force_close_position()
                    elif self.detect_early_reversal():
                        self.log(f"Early reversal — flipping to reversal short")
                        self.enter_reversal_short("early_reversal")
                    elif self.momentum_hold_expired():
                        self.log(f"Hold expired — flipping to reversal short")
                        self.enter_reversal_short("hold_expired")

                # ---- MOMENTUM SHORT ---- #
                elif self.phase == "momentum_short":
                    if self.stop_loss_hit():
                        self.force_close_position()
                    elif self.detect_early_reversal():
                        self.log(f"Early reversal — flipping to reversal long")
                        self.enter_reversal_long("early_reversal")
                    elif self.momentum_hold_expired():
                        self.log(f"Hold expired — flipping to reversal long")
                        self.enter_reversal_long("hold_expired")

                # ---- REVERSAL SHORT ---- #
                elif self.phase == "reversal_short":
                    if self.stop_loss_hit():
                        self.force_close_position()
                    elif self.reversal_complete():
                        self.log(f"Reversal complete — closing")
                        self.force_close_position()
                    elif self.reversal_hold_expired():
                        self.log(f"Reversal hold expired — closing")
                        self.force_close_position()

                # ---- REVERSAL LONG ---- #
                elif self.phase == "reversal_long":
                    if self.stop_loss_hit():
                        self.force_close_position()
                    elif self.reversal_complete():
                        self.log(f"Reversal complete — closing")
                        self.force_close_position()
                    elif self.reversal_hold_expired():
                        self.log(f"Reversal hold expired — closing")
                        self.force_close_position()

                sleep(self.check_freq)

            except KeyboardInterrupt:
                self.log("Interrupted")
                break
            except Exception as e:
                self.log(f"Error: {e}")
                sleep(self.check_freq)
                continue

        self.log("End time reached — closing all positions")
        self.close_all()
        self.trader.unsub_order_book(self.ticker)

        final_pl = self.trader.get_portfolio_item(self.ticker).get_realized_pl()
        self.log("="*40)
        self.log(f"DONE | P&L: ${final_pl:.2f} | trades={self.trade_count} | "
                 f"spikes={self.spikes_detected} | mom={self.momentum_trades} | "
                 f"rev={self.reversal_trades} | stops={self.stop_loss_hits}")
        self.log("="*40)
        self.running = False