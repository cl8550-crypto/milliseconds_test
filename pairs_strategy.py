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


class PairsStrategy:
    def __init__(
        self,
        trader: shift.Trader,
        risk_manager: RiskManager,
        ticker1: str,
        ticker2: str,
        order_size: int = 2,
        window: int = 30,
        entry_threshold: float = 2.0,
        exit_threshold: float = 0.5,
        check_freq: float = 1.0,
        cooldown: float = 15.0,
        max_history: int = 100,
        leg_stop_loss: float = -500.0,   # ← NEW: close spread if any leg loses this much
    ):
        self.trader          = trader
        self.risk_manager    = risk_manager
        self.ticker1         = ticker1
        self.ticker2         = ticker2
        self.order_size      = order_size
        self.window          = window
        self.entry_threshold = entry_threshold
        self.exit_threshold  = exit_threshold
        self.check_freq      = check_freq
        self.cooldown        = cooldown
        self.max_history     = max_history
        self.leg_stop_loss   = leg_stop_loss

        self.spread_history  = []
        self.trade_count     = 0
        self.last_trade_time = 0
        self.running         = False

        self.pair_position   = None

    # ------------------------------------------------------------------ #
    #  Logging                                                             #
    # ------------------------------------------------------------------ #
    def log(self, message: str):
        pair = f"{self.ticker1}/{self.ticker2}"
        log("PAIRS", pair, message)

    # ------------------------------------------------------------------ #
    #  Cooldown check                                                      #
    # ------------------------------------------------------------------ #
    def is_cooling_down(self) -> bool:
        return (time() - self.last_trade_time) < self.cooldown

    # ------------------------------------------------------------------ #
    #  Pending orders check                                                #
    # ------------------------------------------------------------------ #
    def has_pending_orders(self) -> bool:
        waiting = self.trader.get_waiting_list()
        return any(
            o.symbol in [self.ticker1, self.ticker2]
            for o in waiting
        )

    # ------------------------------------------------------------------ #
    #  Spread helpers                                                      #
    # ------------------------------------------------------------------ #
    def get_spread(self) -> float | None:
        mid1 = get_mid_price(self.trader, self.ticker1)
        mid2 = get_mid_price(self.trader, self.ticker2)
        if mid1 and mid2 and mid1 > 0 and mid2 > 0:
            return round(mid1 - mid2, 4)
        return None

    def get_spread_zscore(self) -> float | None:
        return get_zscore(self.spread_history, self.window)

    def update_spread_history(self, spread: float):
        self.spread_history.append(spread)
        if len(self.spread_history) > self.max_history:
            self.spread_history = self.spread_history[-self.max_history:]

    # ------------------------------------------------------------------ #
    #  Price validation                                                    #
    # ------------------------------------------------------------------ #
    def get_best_prices(self):
        try:
            best1 = self.trader.get_best_price(self.ticker1)
            best2 = self.trader.get_best_price(self.ticker2)

            bid1 = best1.get_bid_price()
            ask1 = best1.get_ask_price()
            bid2 = best2.get_bid_price()
            ask2 = best2.get_ask_price()

            if bid1 <= 0 or ask1 <= 0 or bid2 <= 0 or ask2 <= 0:
                return None
            return bid1, ask1, bid2, ask2
        except Exception as e:
            self.log(f"Error getting best prices: {e}")
            return None

    # ------------------------------------------------------------------ #
    #  Leg stop-loss check — NEW                                           #
    # ------------------------------------------------------------------ #
    def check_leg_stop_loss(self) -> bool:
        """
        Returns True if stop-loss is breached on either leg.
        Checks realized + unrealized P&L for each ticker.
        If either leg loses more than leg_stop_loss → close entire spread.
        """
        if self.pair_position is None:
            return False

        try:
            pl1         = self.trader.get_portfolio_item(self.ticker1).get_realized_pl()
            unrealized1 = self.trader.get_unrealized_pl(self.ticker1)
            total_pl1   = pl1 + unrealized1

            pl2         = self.trader.get_portfolio_item(self.ticker2).get_realized_pl()
            unrealized2 = self.trader.get_unrealized_pl(self.ticker2)
            total_pl2   = pl2 + unrealized2

            if total_pl1 < self.leg_stop_loss:
                self.log(
                    f"LEG STOP-LOSS HIT: {self.ticker1} P&L=${total_pl1:.2f} "
                    f"< ${self.leg_stop_loss:.2f} — closing spread"
                )
                return True

            if total_pl2 < self.leg_stop_loss:
                self.log(
                    f"LEG STOP-LOSS HIT: {self.ticker2} P&L=${total_pl2:.2f} "
                    f"< ${self.leg_stop_loss:.2f} — closing spread"
                )
                return True

        except Exception as e:
            self.log(f"Error checking leg stop-loss: {e}")

        return False

    # ------------------------------------------------------------------ #
    #  Order helpers                                                       #
    # ------------------------------------------------------------------ #
    def cancel_pair_orders(self):
        cancel_orders_for(self.trader, self.ticker1)
        cancel_orders_for(self.trader, self.ticker2)
        sleep(0.5)

    # ------------------------------------------------------------------ #
    #  Opening positions — uses can_trade()                                #
    # ------------------------------------------------------------------ #
    def open_long_spread(self):
        """Long ticker1 + short ticker2"""
        prices = self.get_best_prices()
        if prices is None:
            self.log("Invalid prices — skipping long spread")
            return

        bid1, ask1, bid2, ask2 = prices

        lot1 = self.risk_manager.get_safe_lot_size(ask1, self.order_size)
        lot2 = self.risk_manager.get_safe_lot_size(ask2, self.order_size)
        lots = min(lot1, lot2)

        if lots <= 0:
            self.log("Lot size 0 — skipping long spread")
            return

        if not self.risk_manager.can_trade(self.ticker1, True, ask1):
            return
        if not self.risk_manager.can_trade(self.ticker2, False, bid2):
            return

        submit_limit_buy(
            self.trader, self.ticker1, lots,
            round(bid1 + 0.01, 2)
        )
        self.trade_count += 1

        submit_limit_sell(
            self.trader, self.ticker2, lots,
            round(ask2 - 0.01, 2)
        )
        self.trade_count += 1

        self.pair_position   = "long_spread"
        self.last_trade_time = time()
        self.log(
            f"OPEN LONG SPREAD | "
            f"BUY {self.ticker1} @ {bid1+0.01:.2f} | "
            f"SELL {self.ticker2} @ {ask2-0.01:.2f} | "
            f"lots={lots} | trades={self.trade_count}"
        )

    def open_short_spread(self):
        """Short ticker1 + long ticker2"""
        prices = self.get_best_prices()
        if prices is None:
            self.log("Invalid prices — skipping short spread")
            return

        bid1, ask1, bid2, ask2 = prices

        lot1 = self.risk_manager.get_safe_lot_size(bid1, self.order_size)
        lot2 = self.risk_manager.get_safe_lot_size(bid2, self.order_size)
        lots = min(lot1, lot2)

        if lots <= 0:
            self.log("Lot size 0 — skipping short spread")
            return

        if not self.risk_manager.can_trade(self.ticker1, False, bid1):
            return
        if not self.risk_manager.can_trade(self.ticker2, True, ask2):
            return

        submit_limit_sell(
            self.trader, self.ticker1, lots,
            round(ask1 - 0.01, 2)
        )
        self.trade_count += 1

        submit_limit_buy(
            self.trader, self.ticker2, lots,
            round(bid2 + 0.01, 2)
        )
        self.trade_count += 1

        self.pair_position   = "short_spread"
        self.last_trade_time = time()
        self.log(
            f"OPEN SHORT SPREAD | "
            f"SELL {self.ticker1} @ {ask1-0.01:.2f} | "
            f"BUY {self.ticker2} @ {bid2+0.01:.2f} | "
            f"lots={lots} | trades={self.trade_count}"
        )

    # ------------------------------------------------------------------ #
    #  Closing positions — uses can_close()                                #
    # ------------------------------------------------------------------ #
    def close_leg(self, ticker: str, is_long: bool):
        try:
            best  = self.trader.get_best_price(ticker)
            bid   = best.get_bid_price()
            ask   = best.get_ask_price()
            item  = self.trader.get_portfolio_item(ticker)
            long  = item.get_long_shares()
            short = item.get_short_shares()

            if is_long and long > 0:
                lots = int(long / 100)
                if self.risk_manager.can_close(ticker, bid):
                    submit_limit_sell(
                        self.trader, ticker, lots,
                        round(ask - 0.01, 2)
                    )
                    self.trade_count += 1
                    self.log(f"CLOSE_SELL {ticker} @ {ask-0.01:.2f} | lots={lots}")
                else:
                    submit_market_sell(self.trader, ticker, lots)
                    self.log(f"FORCE MARKET_SELL {ticker} | lots={lots}")

            elif not is_long and short > 0:
                lots = int(short / 100)
                if self.risk_manager.can_close(ticker, ask):
                    submit_limit_buy(
                        self.trader, ticker, lots,
                        round(bid + 0.01, 2)
                    )
                    self.trade_count += 1
                    self.log(f"CLOSE_BUY {ticker} @ {bid+0.01:.2f} | lots={lots}")
                else:
                    submit_market_buy(self.trader, ticker, lots)
                    self.log(f"FORCE MARKET_BUY {ticker} | lots={lots}")

        except Exception as e:
            self.log(f"Error closing leg {ticker}: {e}")
            close_position(self.trader, ticker)

    def close_spread(self):
        """Close both legs — always allowed regardless of halt state"""
        self.cancel_pair_orders()
        sleep(3)

        if self.pair_position == "long_spread":
            self.close_leg(self.ticker1, is_long=True)
            sleep(1)
            self.close_leg(self.ticker2, is_long=False)

        elif self.pair_position == "short_spread":
            self.close_leg(self.ticker1, is_long=False)
            sleep(1)
            self.close_leg(self.ticker2, is_long=True)

        else:
            self.cancel_pair_orders()

        sleep(1)
        self.pair_position   = None
        self.last_trade_time = time()
        self.log("SPREAD CLOSED")

    # ------------------------------------------------------------------ #
    #  Main strategy loop                                                  #
    # ------------------------------------------------------------------ #
    def run(self, end_time):
        self.running = True
        self.log("Strategy started")

        self.trader.sub_order_book(self.ticker1)
        self.trader.sub_order_book(self.ticker2)

        # Warmup
        self.log(f"Warming up — collecting {self.window} spread samples...")
        while len(self.spread_history) < self.window:
            if self.risk_manager.all_halted:
                self.log("Halt detected during warmup — exiting")
                self.trader.unsub_order_book(self.ticker1)
                self.trader.unsub_order_book(self.ticker2)
                return
            spread = self.get_spread()
            if spread is not None:
                self.update_spread_history(spread)
            sleep(self.check_freq)

        self.log("Warmup complete — trading started")

        while self.running and self.trader.get_last_trade_time() < end_time:

            # ---- Check global halt ---- #
            if self.risk_manager.all_halted:
                self.log("Global halt detected — stopping strategy loop")
                break

            try:
                spread = self.get_spread()
                if spread is None:
                    sleep(self.check_freq)
                    continue

                self.update_spread_history(spread)
                zscore = self.get_spread_zscore()

                if zscore is None:
                    sleep(self.check_freq)
                    continue

                self.log(
                    f"spread={spread:.4f} | z={zscore:.2f} | "
                    f"position={self.pair_position}"
                )

                # ---- Leg stop-loss check — highest priority ---- #
                if self.check_leg_stop_loss():
                    self.close_spread()
                    sleep(self.check_freq)
                    continue

                # ---- Skip if cooling down ---- #
                if self.is_cooling_down():
                    remaining = self.cooldown - (time() - self.last_trade_time)
                    self.log(f"Cooling down — {remaining:.0f}s remaining")
                    sleep(self.check_freq)
                    continue

                # ---- Skip if pending orders ---- #
                if self.has_pending_orders():
                    self.log("Pending orders exist — waiting")
                    sleep(self.check_freq)
                    continue

                # ---- Entry logic ---- #
                if self.pair_position is None:
                    if zscore < -self.entry_threshold:
                        self.open_long_spread()
                    elif zscore > self.entry_threshold:
                        self.open_short_spread()

                # ---- Exit logic ---- #
                elif abs(zscore) < self.exit_threshold:
                    self.log(f"Spread reverted (z={zscore:.2f}) → closing")
                    self.close_spread()

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
        self.close_spread()

        self.trader.unsub_order_book(self.ticker1)
        self.trader.unsub_order_book(self.ticker2)

        pl1 = self.trader.get_portfolio_item(self.ticker1).get_realized_pl()
        pl2 = self.trader.get_portfolio_item(self.ticker2).get_realized_pl()
        self.log(
            f"Final P&L | "
            f"{self.ticker1}: ${pl1:.2f} | "
            f"{self.ticker2}: ${pl2:.2f} | "
            f"Combined: ${pl1+pl2:.2f} | "
            f"Total trades: {self.trade_count}"
        )
        self.running = False