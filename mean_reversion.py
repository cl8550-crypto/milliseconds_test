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


class MeanReversionStrategy:
    def __init__(
        self,
        trader: shift.Trader,
        risk_manager: RiskManager,
        ticker: str,
        order_size: int = None,
        window: int = 30,
        entry_threshold: float = 2.0,
        exit_threshold: float = 0.3,
        check_freq: float = 1.0,
        cooldown: float = 45.0,
        max_history: int = 100,
        pending_timeout: float = 30.0,
        trend_window: int = 10,
        trend_threshold: float = 0.003,
    ):
        self.trader           = trader
        self.risk_manager     = risk_manager
        self.ticker           = ticker
        self.order_size       = order_size
        self.window           = window
        self.entry_threshold  = entry_threshold
        self.exit_threshold   = exit_threshold
        self.check_freq       = check_freq
        self.cooldown         = cooldown
        self.max_history      = max_history
        self.pending_timeout  = pending_timeout
        self.trend_window     = trend_window
        self.trend_threshold  = trend_threshold

        self.price_history    = []
        self.trade_count      = 0
        self.last_trade_time  = 0
        self.running          = False

        self.owns_position    = False
        self.flat_ticks       = 0
        self.pending_since    = 0.0

    # ------------------------------------------------------------------ #
    #  Logging                                                             #
    # ------------------------------------------------------------------ #
    def log(self, message: str):
        log("MR", self.ticker, message)

    # ------------------------------------------------------------------ #
    #  Cooldown                                                            #
    # ------------------------------------------------------------------ #
    def is_cooling_down(self) -> bool:
        return (time() - self.last_trade_time) < self.cooldown

    # ------------------------------------------------------------------ #
    #  Pending order check with timeout                                    #
    # ------------------------------------------------------------------ #
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
            # FIX 2: if position is still flat, entry never filled
            # release ownership so strategy can re-evaluate
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
    #  Price history / z-score                                             #
    # ------------------------------------------------------------------ #
    def update_history(self, mid: float):
        self.price_history.append(mid)
        if len(self.price_history) > self.max_history:
            self.price_history = self.price_history[-self.max_history:]

    def compute_zscore(self) -> float | None:
        if len(self.price_history) < self.window:
            return None
        series = np.array(self.price_history[-self.window:])
        mean   = np.mean(series)
        std    = np.std(series)
        if std == 0:
            return None
        return float((self.price_history[-1] - mean) / std)

    # ------------------------------------------------------------------ #
    #  FIX 3: Trend filter                                                 #
    #  Skips entry if price has moved >trend_threshold in one direction    #
    #  over the last trend_window ticks.                                   #
    #  Prevents MR from buying into sustained downtrends or shorting       #
    #  into sustained uptrends.                                            #
    # ------------------------------------------------------------------ #
    def is_trending(self) -> bool:
        if len(self.price_history) < self.trend_window:
            return False
        recent     = self.price_history[-self.trend_window:]
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
        """
        Release ownership after 5 flat ticks AND not in cooldown.
        threshold=5 tolerates SHIFT fill delay.
        Cooldown guard prevents premature release just after fill.
        """
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
    #  Order helpers                                                       #
    # ------------------------------------------------------------------ #
    def submit_buy(self, price: float, lots: int) -> bool:
        if not self.risk_manager.can_trade(self.ticker, True, price):
            self.log(f"Risk failed — skipping BUY @ {price:.2f}")
            return False
        result = submit_limit_buy(
            self.trader, self.ticker, lots, round(price, 2)
        )
        if result:
            self.trade_count    += 1
            self.last_trade_time = time()
            self.owns_position   = True
            self.flat_ticks      = 0
            self.pending_since   = time()
            self.log(
                f"LIMIT_BUY @ {price:.2f} | "
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
            self.trade_count    += 1
            self.last_trade_time = time()
            self.owns_position   = True
            self.flat_ticks      = 0
            self.pending_since   = time()
            self.log(
                f"LIMIT_SELL @ {price:.2f} | "
                f"lots={lots} | cost≈${price*lots*100:,.0f} | "
                f"trades={self.trade_count}"
            )
            return True
        return False

    def submit_close_buy(self) -> bool:
        """
        Close short with MARKET order.
        FIX 1: Do NOT set owns_position=False if lots==0.
        A zero read from SHIFT is often a timing delay, not a flat position.
        Only release ownership when an actual order is submitted.
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
            f"Config | window={self.window} | "
            f"entry={self.entry_threshold} | "
            f"exit={self.exit_threshold} | "
            f"cooldown={self.cooldown}s | "
            f"pending_timeout={self.pending_timeout}s | "
            f"trend_window={self.trend_window} | "
            f"trend_threshold={self.trend_threshold*100:.2f}% | "
            f"size={'dynamic' if self.order_size is None else self.order_size}"
        )

        self.trader.sub_order_book(self.ticker)

        # Claim inherited positions ONCE at startup
        existing = get_position(self.trader, self.ticker)
        self.claim_inherited_position(existing)

        # Warmup
        self.log(f"Warming up — need {self.window} samples...")
        while len(self.price_history) < self.window:
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

                zscore   = self.compute_zscore()
                position = get_position(self.trader, self.ticker)

                if zscore is None:
                    sleep(self.check_freq)
                    continue

                # Update ownership state (never claim here — startup only)
                self.update_ownership(position)

                self.log(
                    f"mid={mid:.2f} | z={zscore:.2f} | "
                    f"pos={position} | owns={self.owns_position} | "
                    f"flat_ticks={self.flat_ticks}"
                )

                # ---- Exit if we own a position ---- #
                if self.owns_position and position != 0:
                    if abs(zscore) < self.exit_threshold:
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
                    self.log("Pending orders — waiting for fill")
                    sleep(self.check_freq)
                    continue

                # ---- Cooldown ---- #
                if self.is_cooling_down():
                    remaining = self.cooldown - (
                        time() - self.last_trade_time
                    )
                    self.log(
                        f"Cooling down — {remaining:.0f}s remaining"
                    )
                    sleep(self.check_freq)
                    continue

                # ---- Entry ---- #
                if position == 0 and not self.owns_position:
                    lots = self.get_lot_size(mid)

                    if lots <= 0:
                        self.log(
                            f"Lot size 0 at ${mid:.2f} — skipping"
                        )
                        sleep(self.check_freq)
                        continue

                    # FIX 3: skip entry if price is trending
                    if self.is_trending():
                        sleep(self.check_freq)
                        continue

                    if zscore < -self.entry_threshold:
                        self.submit_buy(round(bid + 0.01, 2), lots)
                    elif zscore > self.entry_threshold:
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
        self.close_all()
        self.trader.unsub_order_book(self.ticker)

        final_pl = self.trader.get_portfolio_item(
            self.ticker
        ).get_realized_pl()
        self.log(
            f"DONE | P&L: ${final_pl:.2f} | "
            f"trades={self.trade_count}"
        )
        self.running = False