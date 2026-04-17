"""
Mean Reversion Strategy — Week 4 ZI Day

Why mean reversion works on ZI prices:
  ZI agents submit random limit orders clustered around the current
  mid-price. When a burst of market sells pushes price down 0.15%,
  the cluster of resting limit buys absorbs the move and price snaps
  back. We buy into the dip and sell into the spike, capturing the
  reversion.

  Confirmed from run_20260415: all 10 momentum trades exited
  hold_complete with price moving against entry = price reverted.
  Flipping the signal makes those same reversions profitable.

Why market orders only:
  Limit orders in ZI local book create a feedback spiral — our quote
  becomes the best bid/ask, get_best_prices() returns it, we requote
  inside our own quote each tick, quotes drift far from market.
  Market orders always fill at current price. Zero cancellations needed.
  Rate limit impact: ~0.13 msg/sec — negligible.

P&L math per trade (5 lots = 500 shares):
  Entry move: 0.15% × $100 = $0.15/share
  Expected reversion: ~50% = $0.075/share profit
  Spread cost entry + exit: ~2 × $0.025 = $0.05/share
  Net: $0.075 - $0.05 = $0.025/share × 500 = $12.50 per trade
  At 2 trades/hour/ticker × 3 tickers × 6 hours = 36 trades → ~$450
  Compounded with correlation boost on CS1/CS2/CS3 moving together.
"""

import shift
from time import sleep, time
from utils import (
    log, get_position, get_best_prices,
    submit_market_buy, submit_market_sell
)
from risk_manager import RiskManager


class MeanReversionZI:

    def __init__(
        self,
        trader:             shift.Trader,
        risk_manager:       RiskManager,
        ticker:             str,
        lookback:           int   = 5,      # ticks to measure move over
        entry_threshold:    float = 0.0015, # 0.15% move triggers fade
        hold_ticks:         int   = 8,      # ticks to hold before forced exit
        stop_pct:           float = 0.003,  # 0.30% stop loss
        take_profit_pct:    float = 0.001,  # 0.10% take profit (lock in early)
        lots:               int   = 5,
        cooldown:           float = 15.0,   # seconds between trades
        check_freq:         float = 1.0,
    ):
        self.trader             = trader
        self.risk_manager       = risk_manager
        self.ticker             = ticker
        self.lookback           = lookback
        self.entry_threshold    = entry_threshold
        self.hold_ticks         = hold_ticks
        self.stop_pct           = stop_pct
        self.take_profit_pct    = take_profit_pct
        self.lots               = lots
        self.cooldown           = cooldown
        self.check_freq         = check_freq

        self.prices          = []
        self.trade_count     = 0
        self.wins            = 0
        self.losses          = 0
        self.running         = False
        self.last_trade_time = 0.0

        # Position state
        self.direction    = None    # "LONG" or "SHORT"
        self.entry_price  = None
        self.hold_count   = 0

    def log(self, msg):
        log("MR", self.ticker, msg)

    def is_cooling(self):
        return (time() - self.last_trade_time) < self.cooldown

    def get_signal(self):
        """
        Mean reversion signal: fade large moves.
        Price spiked up → SHORT (it will come back down).
        Price dropped    → LONG  (it will bounce back up).
        """
        if len(self.prices) < self.lookback + 1:
            return None
        past = self.prices[-(self.lookback + 1)]
        now  = self.prices[-1]
        if past <= 0:
            return None
        move = (now - past) / past
        if move >  self.entry_threshold:
            return "SHORT"   # spiked up → fade
        if move < -self.entry_threshold:
            return "LONG"    # dropped → buy dip
        return None

    def run(self, end_time):
        self.running = True
        self.log("Strategy started")
        self.log(
            f"lookback={self.lookback}t | "
            f"entry={self.entry_threshold*100:.2f}% | "
            f"hold={self.hold_ticks}t | "
            f"stop={self.stop_pct*100:.2f}% | "
            f"tp={self.take_profit_pct*100:.2f}% | "
            f"lots={self.lots} | cooldown={self.cooldown}s"
        )

        # Wind-down times
        # Stop new entries 15 mins before end (preserve profitable positions)
        # Force-close all positions 10 mins before end (lock in gains)
        from datetime import timedelta
        no_entry_time  = end_time - timedelta(minutes=15)
        force_close_time = end_time - timedelta(minutes=10)

        self.trader.sub_order_book(self.ticker)
        sleep(0.5)

        # Warmup: collect enough price history
        self.log(f"Warming up — need {self.lookback + 1} samples...")
        while len(self.prices) < self.lookback + 1:
            if self.risk_manager.all_halted:
                return
            p = get_best_prices(self.trader, self.ticker)
            if p:
                self.prices.append((p[0] + p[1]) / 2)
            sleep(self.check_freq)
        self.log("Warmup done — trading active")

        while (
            self.running and
            self.trader.get_last_trade_time() < end_time
        ):
            if self.risk_manager.all_halted:
                break

            try:
                p = get_best_prices(self.trader, self.ticker)
                if p is None:
                    sleep(self.check_freq)
                    continue

                mid = (p[0] + p[1]) / 2
                self.prices.append(mid)
                if len(self.prices) > 200:
                    self.prices = self.prices[-200:]

                now = self.trader.get_last_trade_time()

                # ---- Force-close 10 mins before end ---- #
                if now >= force_close_time and self.direction is not None:
                    pos = get_position(self.trader, self.ticker)
                    if pos != 0:
                        self.log(
                            f"WIND-DOWN: force-closing {self.direction} "
                            f"position with {(end_time - now).seconds}s remaining"
                        )
                        if pos > 0:
                            submit_market_sell(
                                self.trader, self.ticker, int(pos / 100)
                            )
                            self.trade_count += 1
                        elif pos < 0:
                            submit_market_buy(
                                self.trader, self.ticker, int(-pos / 100)
                            )
                            self.trade_count += 1
                    self.direction       = None
                    self.entry_price     = None
                    self.hold_count      = 0
                    self.last_trade_time = time()
                    sleep(self.check_freq)
                    continue

                # ---- Manage open position ---- #
                if self.direction is not None:
                    self.hold_count += 1
                    pnl_pct = (
                        (mid - self.entry_price) / self.entry_price
                        if self.direction == "LONG"
                        else (self.entry_price - mid) / self.entry_price
                    )

                    exit_reason = None
                    if pnl_pct >= self.take_profit_pct:
                        exit_reason = f"take_profit pnl={pnl_pct*100:.3f}%"
                    elif pnl_pct <= -self.stop_pct:
                        exit_reason = f"stop_loss pnl={pnl_pct*100:.3f}%"
                    elif self.hold_count >= self.hold_ticks:
                        exit_reason = f"hold_complete pnl={pnl_pct*100:.3f}%"

                    if exit_reason:
                        pos  = get_position(self.trader, self.ticker)
                        ok   = False
                        if pos > 0:
                            ok = submit_market_sell(
                                self.trader, self.ticker, int(pos / 100)
                            )
                            self.trade_count += 1
                        elif pos < 0:
                            ok = submit_market_buy(
                                self.trader, self.ticker, int(-pos / 100)
                            )
                            self.trade_count += 1

                        if pnl_pct > 0:
                            self.wins += 1
                        else:
                            self.losses += 1

                        self.log(
                            f"EXIT {self.direction} | {exit_reason} | "
                            f"W/L={self.wins}/{self.losses} | "
                            f"trades={self.trade_count}"
                        )
                        self.direction       = None
                        self.entry_price     = None
                        self.hold_count      = 0
                        self.last_trade_time = time()

                    sleep(self.check_freq)
                    continue

                # ---- Look for signal ---- #
                if now >= no_entry_time:
                    sleep(self.check_freq)
                    continue   # no new entries in last 15 mins

                if self.is_cooling():
                    sleep(self.check_freq)
                    continue

                signal = self.get_signal()
                if signal is None:
                    sleep(self.check_freq)
                    continue

                if not self.risk_manager.can_trade(
                    self.ticker, signal == "LONG", mid
                ):
                    sleep(self.check_freq)
                    continue

                # Capture the triggering move BEFORE entering
                past_price = self.prices[-(self.lookback + 1)]
                move_pct   = (mid - past_price) / past_price * 100

                # Enter with market order
                if signal == "LONG":
                    ok = submit_market_buy(
                        self.trader, self.ticker, self.lots
                    )
                else:
                    ok = submit_market_sell(
                        self.trader, self.ticker, self.lots
                    )

                if ok:
                    self.trade_count    += 1
                    self.direction       = signal
                    self.entry_price     = mid
                    self.hold_count      = 0
                    self.last_trade_time = time()
                    self.log(
                        f"ENTER {signal} | mid={mid:.3f} | "
                        f"move={move_pct:+.3f}% over {self.lookback}t | "
                        f"trades={self.trade_count}"
                    )

                sleep(self.check_freq)

            except KeyboardInterrupt:
                break
            except Exception as e:
                self.log(f"Error: {e}")
                sleep(self.check_freq)

        # Close out
        self.log("End time — closing position")
        pos = get_position(self.trader, self.ticker)
        if pos > 0:
            submit_market_sell(self.trader, self.ticker, int(pos / 100))
            self.trade_count += 1
        elif pos < 0:
            submit_market_buy(self.trader, self.ticker, int(-pos / 100))
            self.trade_count += 1

        self.trader.unsub_order_book(self.ticker)

        pl = self.trader.get_portfolio_item(self.ticker).get_realized_pl()
        total = self.wins + self.losses
        wr = f"{self.wins/total*100:.0f}%" if total > 0 else "N/A"
        self.log(
            f"DONE | P&L=${pl:.2f} | "
            f"trades={self.trade_count} | "
            f"W/L={self.wins}/{self.losses} ({wr})"
        )
        self.running = False
