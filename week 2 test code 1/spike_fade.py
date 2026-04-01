import shift
import numpy as np
from time import sleep, time
from utils import (
    log, get_mid_price, get_position,
    cancel_orders_for, submit_limit_buy, submit_limit_sell,
    submit_market_buy, submit_market_sell, close_position
)
from risk_manager import RiskManager


class SpikeFadeStrategy:
    """
    Spike & Dip Fade Strategy — Week 2 Theme: "Spikes and Dips"

    Maintains a rolling median baseline price. When price deviates more than
    spike_threshold (default 1.5%) from baseline, enters a contra-directional
    limit order to fade the move and capture the recovery.

    Exit conditions (whichever comes first):
      - Take profit:  price recovers within recovery_threshold of baseline
      - Stop loss:    move extends to stop_loss_mult * spike_threshold
      - Time stop:    position held longer than time_stop_minutes

    Baseline uses median (not mean) of recent prices so it is robust to
    the spike itself corrupting the reference level. Baseline is also frozen
    during active spike periods so it cannot drift with the move.
    """

    def __init__(
        self,
        trader: shift.Trader,
        risk_manager: RiskManager,
        ticker: str,
        order_size: int = 3,
        baseline_window: int = 60,       # ticks for rolling baseline (~30s at 0.5s freq)
        spike_threshold: float = 0.015,  # 1.5% deviation triggers entry
        recovery_threshold: float = 0.005,  # 0.5% from baseline = take profit
        stop_loss_mult: float = 2.0,     # stop if move reaches 2x spike_threshold (3%)
        time_stop_minutes: float = 3.0,  # abandon position after N minutes
        check_freq: float = 0.5,         # seconds between price checks
        cooldown: float = 5.0,           # seconds before re-entering after a trade
    ):
        self.trader             = trader
        self.risk_manager       = risk_manager
        self.ticker             = ticker
        self.order_size         = order_size
        self.baseline_window    = baseline_window
        self.spike_threshold    = spike_threshold
        self.recovery_threshold = recovery_threshold
        self.stop_loss_mult     = stop_loss_mult
        self.time_stop_seconds  = time_stop_minutes * 60
        self.check_freq         = check_freq
        self.cooldown           = cooldown

        self.price_history      = []
        self.trade_count        = 0
        self.last_entry_time    = 0
        self.entry_time         = None
        self.running            = False

    # ------------------------------------------------------------------ #
    #  Logging                                                             #
    # ------------------------------------------------------------------ #
    def log(self, message: str):
        log("FADE", self.ticker, message)

    # ------------------------------------------------------------------ #
    #  State helpers                                                       #
    # ------------------------------------------------------------------ #
    def is_cooling_down(self) -> bool:
        return (time() - self.last_entry_time) < self.cooldown

    def has_pending_orders(self) -> bool:
        return any(
            o.symbol == self.ticker
            for o in self.trader.get_waiting_list()
        )

    def get_position(self) -> int:
        return get_position(self.trader, self.ticker)

    # ------------------------------------------------------------------ #
    #  Baseline — median of last N ticks, frozen during active spikes      #
    # ------------------------------------------------------------------ #
    def update_baseline(self, mid: float):
        self.price_history.append(mid)
        if len(self.price_history) > self.baseline_window * 2:
            self.price_history = self.price_history[-self.baseline_window * 2:]

    def get_baseline(self) -> float | None:
        if len(self.price_history) < max(10, self.baseline_window // 2):
            return None
        window = self.price_history[-self.baseline_window:]
        return float(np.median(window))

    def get_deviation(self, mid: float, baseline: float) -> float:
        """Signed % deviation from baseline. Positive = above, negative = below."""
        return (mid - baseline) / baseline

    # ------------------------------------------------------------------ #
    #  Order helpers                                                       #
    # ------------------------------------------------------------------ #
    def cancel_pending(self):
        cancel_orders_for(self.trader, self.ticker)
        sleep(0.3)

    def enter_long(self, bid: float) -> bool:
        """Buy on dip — fade downward spike."""
        price = round(bid + 0.01, 2)
        if not self.risk_manager.can_trade(self.ticker, True, price):
            self.log(f"Risk check failed — skipping LONG @ {price:.2f}")
            return False
        result = submit_limit_buy(self.trader, self.ticker, self.order_size, price)
        if result:
            self.trade_count    += 1
            self.last_entry_time = time()
            self.entry_time      = time()
            self.log(
                f"ENTER LONG (dip fade) @ {price:.2f} | "
                f"size={self.order_size} | trades={self.trade_count}"
            )
            return True
        return False

    def enter_short(self, ask: float) -> bool:
        """Sell on spike — fade upward spike."""
        price = round(ask - 0.01, 2)
        if not self.risk_manager.can_trade(self.ticker, False, price):
            self.log(f"Risk check failed — skipping SHORT @ {price:.2f}")
            return False
        result = submit_limit_sell(self.trader, self.ticker, self.order_size, price)
        if result:
            self.trade_count    += 1
            self.last_entry_time = time()
            self.entry_time      = time()
            self.log(
                f"ENTER SHORT (spike fade) @ {price:.2f} | "
                f"size={self.order_size} | trades={self.trade_count}"
            )
            return True
        return False

    def exit_long_limit(self, ask: float):
        """Close long with limit sell (take profit, earns rebate)."""
        price = round(ask - 0.01, 2)
        if not self.risk_manager.can_close(self.ticker, price):
            self._force_close_long()
            return
        result = submit_limit_sell(self.trader, self.ticker, self.order_size, price)
        if result:
            self.trade_count += 1
            self.entry_time   = None
            self.log(f"EXIT LONG limit @ {price:.2f} | trades={self.trade_count}")

    def exit_short_limit(self, bid: float):
        """Close short with limit buy (take profit, earns rebate)."""
        price = round(bid + 0.01, 2)
        if not self.risk_manager.can_close(self.ticker, price):
            self._force_close_short()
            return
        result = submit_limit_buy(self.trader, self.ticker, self.order_size, price)
        if result:
            self.trade_count += 1
            self.entry_time   = None
            self.log(f"EXIT SHORT limit @ {price:.2f} | trades={self.trade_count}")

    def _force_close_long(self):
        item = self.trader.get_portfolio_item(self.ticker)
        lots = int(item.get_long_shares() / 100)
        if lots > 0:
            submit_market_sell(self.trader, self.ticker, lots)
            self.trade_count += 1
            self.entry_time   = None
            self.log(f"EXIT LONG market (forced) {lots} lots")

    def _force_close_short(self):
        item = self.trader.get_portfolio_item(self.ticker)
        lots = int(item.get_short_shares() / 100)
        if lots > 0:
            submit_market_buy(self.trader, self.ticker, lots)
            self.trade_count += 1
            self.entry_time   = None
            self.log(f"EXIT SHORT market (forced) {lots} lots")

    def close_all(self):
        self.cancel_pending()
        sleep(1)
        close_position(self.trader, self.ticker)

    # ------------------------------------------------------------------ #
    #  Main loop                                                           #
    # ------------------------------------------------------------------ #
    def run(self, end_time):
        self.running = True
        self.log(
            f"Strategy started | "
            f"spike_threshold={self.spike_threshold*100:.1f}% | "
            f"stop={self.spike_threshold*self.stop_loss_mult*100:.1f}% | "
            f"time_stop={self.time_stop_seconds/60:.1f}min"
        )
        self.trader.sub_order_book(self.ticker)

        # Warmup — collect enough ticks to form a reliable baseline
        warmup_needed = max(10, self.baseline_window // 2)
        self.log(f"Warming up — collecting {warmup_needed} price ticks...")
        while len(self.price_history) < warmup_needed:
            if self.risk_manager.all_halted:
                self.log("Halt during warmup — exiting")
                self.trader.unsub_order_book(self.ticker)
                return
            mid = get_mid_price(self.trader, self.ticker)
            if mid and mid > 0:
                self.update_baseline(mid)
            sleep(self.check_freq)
        self.log("Warmup complete — baseline ready")

        stop_threshold = self.spike_threshold * self.stop_loss_mult

        while self.running and self.trader.get_last_trade_time() < end_time:

            if self.risk_manager.all_halted:
                self.log("Global halt — stopping")
                break

            try:
                mid = get_mid_price(self.trader, self.ticker)
                if not mid or mid <= 0:
                    sleep(self.check_freq)
                    continue

                baseline = self.get_baseline()
                if not baseline or baseline <= 0:
                    self.update_baseline(mid)
                    sleep(self.check_freq)
                    continue

                deviation = self.get_deviation(mid, baseline)
                position  = self.get_position()

                best = self.trader.get_best_price(self.ticker)
                bid  = best.get_bid_price()
                ask  = best.get_ask_price()
                if bid <= 0 or ask <= 0:
                    sleep(self.check_freq)
                    continue

                # Only update baseline when price is calm (not spiking)
                # This prevents the spike from drifting the reference level
                if abs(deviation) < self.spike_threshold * 0.5:
                    self.update_baseline(mid)

                self.log(
                    f"mid={mid:.2f} | baseline={baseline:.2f} | "
                    f"dev={deviation*100:+.2f}% | pos={position}"
                )

                # ---- MANAGE OPEN LONG (bought the dip) ---- #
                if position > 0:
                    time_held = time() - self.entry_time if self.entry_time else 0

                    if deviation > -self.recovery_threshold:
                        # Price recovered close to baseline — take profit
                        self.log(
                            f"RECOVERY (dev={deviation*100:+.2f}%) — "
                            f"taking profit on LONG"
                        )
                        self.cancel_pending()
                        self.exit_long_limit(ask)

                    elif deviation < -stop_threshold:
                        # Dip deepened past stop — cut loss
                        self.log(
                            f"STOP LOSS (dev={deviation*100:+.2f}%) — closing LONG"
                        )
                        self.cancel_pending()
                        self._force_close_long()

                    elif time_held > self.time_stop_seconds:
                        # Held too long without recovery — time stop
                        self.log(
                            f"TIME STOP ({time_held:.0f}s) — closing LONG"
                        )
                        self.cancel_pending()
                        self._force_close_long()

                # ---- MANAGE OPEN SHORT (sold the spike) ---- #
                elif position < 0:
                    time_held = time() - self.entry_time if self.entry_time else 0

                    if deviation < self.recovery_threshold:
                        # Spike reversed back to baseline — take profit
                        self.log(
                            f"RECOVERY (dev={deviation*100:+.2f}%) — "
                            f"taking profit on SHORT"
                        )
                        self.cancel_pending()
                        self.exit_short_limit(bid)

                    elif deviation > stop_threshold:
                        # Spike extended past stop — cut loss
                        self.log(
                            f"STOP LOSS (dev={deviation*100:+.2f}%) — closing SHORT"
                        )
                        self.cancel_pending()
                        self._force_close_short()

                    elif time_held > self.time_stop_seconds:
                        self.log(
                            f"TIME STOP ({time_held:.0f}s) — closing SHORT"
                        )
                        self.cancel_pending()
                        self._force_close_short()

                # ---- LOOK FOR NEW ENTRY (flat) ---- #
                elif position == 0 and not self.has_pending_orders():

                    if self.is_cooling_down():
                        sleep(self.check_freq)
                        continue

                    if deviation > self.spike_threshold:
                        # Price spiked UP — fade with short
                        self.log(
                            f"SPIKE UP detected (dev={deviation*100:+.2f}%) — "
                            f"entering SHORT"
                        )
                        self.enter_short(ask)

                    elif deviation < -self.spike_threshold:
                        # Price dipped DOWN — fade with long
                        self.log(
                            f"DIP DOWN detected (dev={deviation*100:+.2f}%) — "
                            f"entering LONG"
                        )
                        self.enter_long(bid)

                sleep(self.check_freq)

            except KeyboardInterrupt:
                self.log("Interrupted")
                break
            except Exception as e:
                self.log(f"Error: {e}")
                sleep(self.check_freq)
                continue

        # Cleanup
        self.log("End time reached — closing all positions")
        self.close_all()
        self.trader.unsub_order_book(self.ticker)

        final_pl = self.trader.get_portfolio_item(self.ticker).get_realized_pl()
        self.log(
            f"DONE | Final P&L: ${final_pl:.2f} | "
            f"Total trades: {self.trade_count}"
        )
        self.running = False
