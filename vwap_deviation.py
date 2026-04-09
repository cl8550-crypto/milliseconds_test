import shift
import numpy as np
from time import sleep, time
from utils import (
    log, get_position, get_best_prices,
    submit_limit_buy, submit_limit_sell,
    submit_market_buy, submit_market_sell,
    cancel_orders_for, close_position
)
from risk_manager import RiskManager


class VWAPDeviationStrategy:
    def __init__(
        self,
        trader: shift.Trader,
        risk_manager: RiskManager,
        ticker: str,
        order_size: int = None,
        vwap_window: int = 60,
        zscore_window: int = 20,
        zscore_threshold: float = 1.5,
        vwap_threshold: float = 0.005,
        exit_zscore: float = 0.3,
        exit_vwap: float = 0.001,
        check_freq: float = 1.0,
        cooldown: float = 30.0,
        max_history: int = 200,
        pending_timeout: float = 30.0,
        trend_window: int = 10,
        trend_threshold: float = 0.003,
    ):
        self.trader           = trader
        self.risk_manager     = risk_manager
        self.ticker           = ticker
        self.order_size       = order_size
        self.vwap_window      = vwap_window
        self.zscore_window    = zscore_window
        self.zscore_threshold = zscore_threshold
        self.vwap_threshold   = vwap_threshold
        self.exit_zscore      = exit_zscore
        self.exit_vwap        = exit_vwap
        self.check_freq       = check_freq
        self.cooldown         = cooldown
        self.max_history      = max_history
        self.pending_timeout  = pending_timeout
        self.trend_window     = trend_window
        self.trend_threshold  = trend_threshold

        self.price_history     = []
        self.trade_count       = 0
        self.last_trade_time   = 0
        self.running           = False

        self.owns_position     = False
        self.flat_ticks        = 0
        self.pending_since     = 0.0

        self.confirmed_signals = 0
        self.filtered_signals  = 0

    # ------------------------------------------------------------------ #
    #  Logging                                                             #
    # ------------------------------------------------------------------ #
    def log(self, message: str):
        log("VWAP", self.ticker, message)

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

    def check_pending_with_timeout(self) -> bool:
        if not self.has_pending_orders():
            self.pending_since = 0.0
            return False
        if self.pending_since == 0.0:
            self.pending_since = time()
            return True
        elapsed = time() - self.pending_since
        if elapsed > self.pending_timeout:
            self.log(
                f"Pending timeout ({elapsed:.0f}s) — "
                f"cancelling stale order"
            )
            cancel_orders_for(self.trader, self.ticker)
            self.pending_since = 0.0
            # FIX 2: release ownership if entry never filled
            position = get_position(self.trader, self.ticker)
            if position == 0:
                self.owns_position = False
                self.log("Entry never filled — releasing ownership")
            return False
        return True

    # ------------------------------------------------------------------ #
    #  Dynamic order size                                                  #
    # ------------------------------------------------------------------ #
    def get_lot_size(self, price: float) -> int:
        if self.order_size is not None:
            return self.risk_manager.get_safe_lot_size(
                price, self.order_size
            )
        dynamic = self.risk_manager.get_dynamic_order_size(price)
        return self.risk_manager.get_safe_lot_size(price, dynamic)

    # ------------------------------------------------------------------ #
    #  Price history                                                       #
    # ------------------------------------------------------------------ #
    def update_history(self, mid: float):
        self.price_history.append((time(), mid))
        if len(self.price_history) > self.max_history:
            self.price_history = self.price_history[-self.max_history:]

    def get_prices_only(self) -> list[float]:
        return [p for _, p in self.price_history]

    # ------------------------------------------------------------------ #
    #  VWAP                                                                #
    # ------------------------------------------------------------------ #
    def compute_vwap(self) -> float | None:
        if not self.price_history:
            return None
        cutoff = time() - self.vwap_window
        window = [p for ts, p in self.price_history if ts >= cutoff]
        if len(window) < 5:
            return None
        return float(np.mean(window))

    def compute_vwap_deviation(self, price: float) -> float | None:
        vwap = self.compute_vwap()
        if vwap is None or vwap <= 0:
            return None
        return (price - vwap) / vwap

    # ------------------------------------------------------------------ #
    #  Z-score                                                             #
    # ------------------------------------------------------------------ #
    def compute_zscore(self) -> float | None:
        prices = self.get_prices_only()
        if len(prices) < self.zscore_window:
            return None
        series = np.array(prices[-self.zscore_window:])
        mean   = np.mean(series)
        std    = np.std(series)
        if std == 0:
            return None
        return float((prices[-1] - mean) / std)

    # ------------------------------------------------------------------ #
    #  FIX 3: Trend filter                                                 #
    # ------------------------------------------------------------------ #
    def is_trending(self) -> bool:
        prices = self.get_prices_only()
        if len(prices) < self.trend_window:
            return False
        recent     = prices[-self.trend_window:]
        total_move = (recent[-1] - recent[0]) / recent[0]
        trending   = abs(total_move) > self.trend_threshold
        if trending:
            direction = "DOWN" if total_move < 0 else "UP"
            self.log(
                f"Trend detected ({direction} "
                f"{total_move*100:.3f}% over {self.trend_window} ticks) "
                f"— skipping entry"
            )
        return trending

    # ------------------------------------------------------------------ #
    #  Ownership tracking                                                  #
    # ------------------------------------------------------------------ #
    def update_ownership(self, position: int):
        if position == 0:
            self.flat_ticks += 1
            if self.flat_ticks >= 5 and not self.is_cooling_down():
                if self.owns_position:
                    self.log(
                        "Position confirmed flat — releasing ownership"
                    )
                self.owns_position = False
        else:
            self.flat_ticks = 0

    def claim_inherited_position(self, position: int):
        """Called ONCE at startup only — never in main loop."""
        if position != 0 and not self.owns_position:
            self.owns_position = True
            self.flat_ticks    = 0
            self.log(
                f"Inherited position: {position} shares — "
                f"claiming ownership"
            )

    # ------------------------------------------------------------------ #
    #  Combined signal                                                     #
    # ------------------------------------------------------------------ #
    def get_signal(
        self, price: float
    ) -> tuple[str, float, float] | tuple[None, None, None]:
        zscore   = self.compute_zscore()
        vwap_dev = self.compute_vwap_deviation(price)

        if zscore is None or vwap_dev is None:
            return None, None, None

        if (zscore < -self.zscore_threshold and
                vwap_dev < -self.vwap_threshold):
            return "BUY", zscore, vwap_dev

        if (zscore > self.zscore_threshold and
                vwap_dev > self.vwap_threshold):
            return "SELL", zscore, vwap_dev

        if abs(zscore) > self.zscore_threshold:
            self.filtered_signals += 1
            self.log(
                f"FILTERED | z={zscore:.2f} | "
                f"vwap_dev={vwap_dev*100:.3f}% "
                f"(need {self.vwap_threshold*100:.2f}%)"
            )

        return None, None, None

    def should_exit(self, price: float) -> bool:
        zscore   = self.compute_zscore()
        vwap_dev = self.compute_vwap_deviation(price)

        if zscore is None or vwap_dev is None:
            return False

        if abs(zscore) < self.exit_zscore:
            self.log(f"EXIT z-score | z={zscore:.2f}")
            return True

        if abs(vwap_dev) < self.exit_vwap:
            self.log(f"EXIT VWAP | dev={vwap_dev*100:.3f}%")
            return True

        return False

    # ------------------------------------------------------------------ #
    #  Order execution                                                     #
    # ------------------------------------------------------------------ #
    def submit_buy(self, price: float, lots: int) -> bool:
        if not self.risk_manager.can_trade(self.ticker, True, price):
            self.log(f"Risk failed — skipping BUY @ {price:.2f}")
            return False
        result = submit_limit_buy(
            self.trader, self.ticker, lots, round(price, 2)
        )
        if result:
            self.trade_count      += 1
            self.confirmed_signals += 1
            self.last_trade_time   = time()
            self.owns_position     = True
            self.flat_ticks        = 0
            self.pending_since     = time()
            self.log(
                f"✅ LIMIT_BUY @ {price:.2f} | "
                f"lots={lots} | cost≈${price*lots*100:,.0f} | "
                f"trades={self.trade_count}"
            )
            return True
        return False

    def submit_sell(self, price: float, lots: int) -> bool:
        if not self.risk_manager.can_trade(self.ticker, False, price):
            self.log(f"Risk failed — skipping SELL @ {price:.2f}")
            return False
        result = submit_limit_sell(
            self.trader, self.ticker, lots, round(price, 2)
        )
        if result:
            self.trade_count      += 1
            self.confirmed_signals += 1
            self.last_trade_time   = time()
            self.owns_position     = True
            self.flat_ticks        = 0
            self.pending_since     = time()
            self.log(
                f"✅ LIMIT_SELL @ {price:.2f} | "
                f"lots={lots} | cost≈${price*lots*100:,.0f} | "
                f"trades={self.trade_count}"
            )
            return True
        return False

    def submit_close_buy(self) -> bool:
        """
        Close short with MARKET order.
        FIX 1: Do NOT set owns_position=False if lots==0.
        """
        item = self.trader.get_portfolio_item(self.ticker)
        lots = int(item.get_short_shares() / 100)
        if lots <= 0:
            return False  # don't touch owns_position here

        result = submit_market_buy(self.trader, self.ticker, lots)
        if result:
            self.trade_count    += 1
            self.last_trade_time = time()
            self.owns_position   = False
            self.log(
                f"MARKET_CLOSE_BUY | lots={lots} | "
                f"trades={self.trade_count}"
            )
            return True
        return False

    def submit_close_sell(self) -> bool:
        """
        Close long with MARKET order.
        FIX 1: Do NOT set owns_position=False if lots==0.
        """
        item = self.trader.get_portfolio_item(self.ticker)
        lots = int(item.get_long_shares() / 100)
        if lots <= 0:
            return False  # don't touch owns_position here

        result = submit_market_sell(self.trader, self.ticker, lots)
        if result:
            self.trade_count    += 1
            self.last_trade_time = time()
            self.owns_position   = False
            self.log(
                f"MARKET_CLOSE_SELL | lots={lots} | "
                f"trades={self.trade_count}"
            )
            return True
        return False

    def close_all(self):
        cancel_orders_for(self.trader, self.ticker)
        sleep(2)
        close_position(self.trader, self.ticker)
        self.owns_position = False

    # ------------------------------------------------------------------ #
    #  Main loop                                                           #
    # ------------------------------------------------------------------ #
    def run(self, end_time):
        self.running = True
        self.log("Strategy started")
        self.log(
            f"Config | vwap_window={self.vwap_window}s | "
            f"z_thresh={self.zscore_threshold} | "
            f"vwap_thresh={self.vwap_threshold*100:.2f}% | "
            f"exit_vwap={self.exit_vwap*100:.2f}% | "
            f"cooldown={self.cooldown}s | "
            f"trend_window={self.trend_window} | "
            f"trend_threshold={self.trend_threshold*100:.2f}%"
        )

        self.trader.sub_order_book(self.ticker)
        self.trader.request_sample_prices(
            [self.ticker],
            sampling_frequency=1.0,
            sampling_window=self.vwap_window
        )

        # Claim inherited positions ONCE at startup
        existing = get_position(self.trader, self.ticker)
        self.claim_inherited_position(existing)

        warmup_needed = max(self.vwap_window, self.zscore_window)
        self.log(f"Warming up — need {warmup_needed} samples...")

        while len(self.price_history) < warmup_needed:
            if self.risk_manager.all_halted:
                self.log("Halt during warmup — exiting")
                self.trader.unsub_order_book(self.ticker)
                return
            prices = get_best_prices(self.trader, self.ticker)
            if prices:
                self.update_history((prices[0] + prices[1]) / 2)
            sleep(self.check_freq)

        self.log("Warmup complete — trading started")

        while (
            self.running and
            self.trader.get_last_trade_time() < end_time
        ):
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
                self.update_history(mid)

                vwap     = self.compute_vwap()
                zscore   = self.compute_zscore()
                vwap_dev = self.compute_vwap_deviation(mid)
                position = get_position(self.trader, self.ticker)

                # Update ownership (never claim here — startup only)
                self.update_ownership(position)

                self.log(
                    f"mid={mid:.2f} | "
                    f"vwap={f'{vwap:.2f}' if vwap else 'N/A'} | "
                    f"dev={f'{vwap_dev*100:.3f}%' if vwap_dev is not None else 'N/A'} | "
                    f"z={f'{zscore:.2f}' if zscore is not None else 'N/A'} | "
                    f"pos={position} | owns={self.owns_position}"
                )

                # ---- Exit if we own a position ---- #
                if self.owns_position and position != 0:
                    if self.should_exit(mid):
                        cancel_orders_for(self.trader, self.ticker)
                        sleep(0.5)
                        if position > 0:
                            self.submit_close_sell()
                        elif position < 0:
                            self.submit_close_buy()
                        sleep(self.check_freq)
                        continue

                # ---- Pending with timeout ---- #
                if self.check_pending_with_timeout():
                    sleep(self.check_freq)
                    continue

                # ---- Cooldown ---- #
                if self.is_cooling_down():
                    sleep(self.check_freq)
                    continue

                # ---- Entry ---- #
                if position == 0 and not self.owns_position:
                    signal, z, dev = self.get_signal(mid)

                    if signal is None:
                        sleep(self.check_freq)
                        continue

                    # FIX 3: skip entry if price is trending
                    if self.is_trending():
                        sleep(self.check_freq)
                        continue

                    lots = self.get_lot_size(mid)
                    if lots <= 0:
                        sleep(self.check_freq)
                        continue

                    if signal == "BUY":
                        self.log(
                            f"✅ BUY | z={z:.2f} | "
                            f"dev={dev*100:.3f}% | lots={lots}"
                        )
                        self.submit_buy(round(bid + 0.01, 2), lots)

                    elif signal == "SELL":
                        self.log(
                            f"✅ SELL | z={z:.2f} | "
                            f"dev={dev*100:.3f}% | lots={lots}"
                        )
                        self.submit_sell(round(ask - 0.01, 2), lots)

                sleep(self.check_freq)

            except KeyboardInterrupt:
                self.log("Interrupted")
                break
            except Exception as e:
                self.log(f"Error: {e}")
                sleep(self.check_freq)
                continue

        self.log("End time — closing all positions")
        self.trader.cancel_sample_prices_request([self.ticker])
        self.close_all()
        self.trader.unsub_order_book(self.ticker)

        total_sig = self.confirmed_signals + self.filtered_signals
        filter_rt = (
            f"{self.filtered_signals/total_sig*100:.1f}%"
            if total_sig > 0 else "N/A"
        )
        final_pl = self.trader.get_portfolio_item(
            self.ticker
        ).get_realized_pl()
        self.log(
            f"DONE | P&L: ${final_pl:.2f} | "
            f"trades={self.trade_count} | "
            f"confirmed={self.confirmed_signals} | "
            f"filtered={self.filtered_signals} | "
            f"filter_rate={filter_rt}"
        )
        self.running = False