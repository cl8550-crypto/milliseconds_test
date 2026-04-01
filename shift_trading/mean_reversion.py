import shift
import numpy as np
from time import sleep, time
from datetime import datetime
from utils import (
    log, get_mid_price, get_zscore, get_position,
    cancel_orders_for, submit_limit_buy, submit_limit_sell,
    submit_market_buy, submit_market_sell, close_position
)
from risk_manager import RiskManager


class MeanReversionStrategy:
    def __init__(
        self,
        trader: shift.Trader,
        risk_manager: RiskManager,
        ticker: str,
        order_size: int = 5,
        window: int = 20,
        entry_threshold: float = 1.5,
        exit_threshold: float = 0.5,
        check_freq: float = 1.0,
        cooldown: float = 10.0,
        max_history: int = 100,
    ):
        self.trader         = trader
        self.risk_manager   = risk_manager
        self.ticker         = ticker
        self.order_size     = order_size
        self.window         = window
        self.entry_threshold  = entry_threshold
        self.exit_threshold   = exit_threshold
        self.check_freq     = check_freq
        self.cooldown       = cooldown
        self.max_history    = max_history

        self.price_history   = []
        self.trade_count     = 0
        self.last_trade_time = 0
        self.running         = False

    # ------------------------------------------------------------------ #
    #  Logging                                                             #
    # ------------------------------------------------------------------ #
    def log(self, message: str):
        log("MR", self.ticker, message)

    # ------------------------------------------------------------------ #
    #  Cooldown check                                                      #
    # ------------------------------------------------------------------ #
    def is_cooling_down(self) -> bool:
        return (time() - self.last_trade_time) < self.cooldown

    # ------------------------------------------------------------------ #
    #  Pending orders check                                                #
    # ------------------------------------------------------------------ #
    def has_pending_orders(self) -> bool:
        return any(
            o.symbol == self.ticker
            for o in self.trader.get_waiting_list()
        )

    # ------------------------------------------------------------------ #
    #  Price history                                                       #
    # ------------------------------------------------------------------ #
    def update_price_history(self, mid: float):
        self.price_history.append(mid)
        if len(self.price_history) > self.max_history:
            self.price_history = self.price_history[-self.max_history:]

    # ------------------------------------------------------------------ #
    #  Order helpers                                                       #
    # ------------------------------------------------------------------ #
    def cancel_pending_orders(self):
        cancel_orders_for(self.trader, self.ticker)
        sleep(0.5)

    def get_position(self) -> int:
        return get_position(self.trader, self.ticker)

    def submit_buy(self, price: float):
        """Open/increase position — full risk check"""
        if not self.risk_manager.can_trade(self.ticker, True, price):
            self.log(f"Risk check failed — skipping BUY @ {price:.2f}")
            return
        result = submit_limit_buy(
            self.trader, self.ticker, self.order_size, price
        )
        if result:
            self.trade_count    += 1
            self.last_trade_time = time()
            self.log(
                f"LIMIT_BUY @ {price:.2f} | "
                f"size={self.order_size} | "
                f"trades={self.trade_count}"
            )

    def submit_sell(self, price: float):
        """Open/increase position — full risk check"""
        if not self.risk_manager.can_trade(self.ticker, False, price):
            self.log(f"Risk check failed — skipping SELL @ {price:.2f}")
            return
        result = submit_limit_sell(
            self.trader, self.ticker, self.order_size, price
        )
        if result:
            self.trade_count    += 1
            self.last_trade_time = time()
            self.log(
                f"LIMIT_SELL @ {price:.2f} | "
                f"size={self.order_size} | "
                f"trades={self.trade_count}"
            )

    def submit_close_buy(self, price: float):
        """Close short position — lightweight risk check only"""
        if not self.risk_manager.can_close(self.ticker, price):
            # Force market close if limit close fails
            self.log(f"can_close failed — forcing MARKET_BUY")
            item = self.trader.get_portfolio_item(self.ticker)
            lots = int(item.get_short_shares() / 100)
            if lots > 0:
                submit_market_buy(self.trader, self.ticker, lots)
            return
        result = submit_limit_buy(
            self.trader, self.ticker, self.order_size, price
        )
        if result:
            self.trade_count    += 1
            self.last_trade_time = time()
            self.log(f"CLOSE_BUY @ {price:.2f} | trades={self.trade_count}")

    def submit_close_sell(self, price: float):
        """Close long position — lightweight risk check only"""
        if not self.risk_manager.can_close(self.ticker, price):
            # Force market close if limit close fails
            self.log(f"can_close failed — forcing MARKET_SELL")
            item = self.trader.get_portfolio_item(self.ticker)
            lots = int(item.get_long_shares() / 100)
            if lots > 0:
                submit_market_sell(self.trader, self.ticker, lots)
            return
        result = submit_limit_sell(
            self.trader, self.ticker, self.order_size, price
        )
        if result:
            self.trade_count    += 1
            self.last_trade_time = time()
            self.log(f"CLOSE_SELL @ {price:.2f} | trades={self.trade_count}")

    # ------------------------------------------------------------------ #
    #  Position closing                                                    #
    # ------------------------------------------------------------------ #
    def close_all(self):
        self.cancel_pending_orders()
        sleep(1)
        close_position(self.trader, self.ticker)

    # ------------------------------------------------------------------ #
    #  Main strategy loop                                                  #
    # ------------------------------------------------------------------ #
    def run(self, end_time):
        self.running = True
        self.log("Strategy started")

        self.trader.sub_order_book(self.ticker)

        # Warmup
        self.log(f"Warming up — collecting {self.window} prices...")
        while len(self.price_history) < self.window:
            # Stop warmup early if global halt triggered
            if self.risk_manager.all_halted:
                self.log("Halt detected during warmup — exiting")
                self.trader.unsub_order_book(self.ticker)
                return
            mid = get_mid_price(self.trader, self.ticker)
            if mid and mid > 0:
                self.update_price_history(mid)
            sleep(self.check_freq)

        self.log("Warmup complete — trading started")

        while self.running and self.trader.get_last_trade_time() < end_time:

            # ---- Check global halt first ---- #
            if self.risk_manager.all_halted:
                self.log("Global halt detected — stopping strategy loop")
                break

            try:
                mid = get_mid_price(self.trader, self.ticker)

                if not mid or mid <= 0:
                    self.log("Invalid mid price — skipping")
                    sleep(self.check_freq)
                    continue

                self.update_price_history(mid)
                zscore = get_zscore(self.price_history, self.window)

                if zscore is None:
                    sleep(self.check_freq)
                    continue

                position = self.get_position()
                self.log(
                    f"mid={mid:.2f} | z={zscore:.2f} | "
                    f"position={position}"
                )

                # Skip if cooling down
                if self.is_cooling_down():
                    remaining = self.cooldown - (time() - self.last_trade_time)
                    self.log(f"Cooling down — {remaining:.0f}s remaining")
                    sleep(self.check_freq)
                    continue

                # Skip if pending orders exist
                if self.has_pending_orders():
                    self.log("Pending orders exist — waiting for fill")
                    sleep(self.check_freq)
                    continue

                # Get best bid/ask
                best = self.trader.get_best_price(self.ticker)
                bid  = best.get_bid_price()
                ask  = best.get_ask_price()

                if bid <= 0 or ask <= 0:
                    self.log(f"Invalid bid/ask — skipping")
                    sleep(self.check_freq)
                    continue

                # ---- Entry signals ---- #
                if zscore < -self.entry_threshold and position <= 0:
                    # Price LOW → expect bounce → BUY
                    self.submit_buy(round(bid + 0.01, 2))

                elif zscore > self.entry_threshold and position >= 0:
                    # Price HIGH → expect pullback → SELL
                    self.submit_sell(round(ask - 0.01, 2))

                # ---- Exit signals — use can_close() ---- #
                elif abs(zscore) < self.exit_threshold and position != 0:
                    if position > 0:
                        # Close long
                        self.submit_close_sell(round(ask - 0.01, 2))
                    elif position < 0:
                        # Close short
                        self.submit_close_buy(round(bid + 0.01, 2))

                sleep(self.check_freq)

            except KeyboardInterrupt:
                self.log("Interrupted")
                break
            except Exception as e:
                self.log(f"Error: {e}")
                sleep(self.check_freq)
                continue

        # Cleanup
        self.log("Closing all positions...")
        self.close_all()
        self.trader.unsub_order_book(self.ticker)

        final_pl = self.trader.get_portfolio_item(self.ticker).get_realized_pl()
        self.log(
            f"DONE | Final P&L: ${final_pl:.2f} | "
            f"Total trades: {self.trade_count}"
        )
        self.running = False