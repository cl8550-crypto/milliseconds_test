import shift
import numpy as np
from time import sleep, time
from datetime import datetime
from utils import (
    log, get_position, get_best_prices,
    submit_limit_buy, submit_limit_sell,
    submit_market_buy, submit_market_sell,
    cancel_orders_for
)
from risk_manager import RiskManager


class AvellanedaStoikov:
    """
    Avellaneda-Stoikov optimal market making (2008).
    Used by 3rd-place team (Money Never Sleeps) in HFTC-25.

    Core formulas:
        r = mid - q * gamma * spread * (T-t)      [reservation price]
        k = gamma / (exp(gamma * 0.7*spread/2)-1) [dynamic kappa from live spread]
        d = gamma*sigma2*(T-t) + (2/gamma)*ln(1+gamma/k) [optimal spread]
        bid_quote = r - d/2
        ask_quote = r + d/2

    Key design decisions:
    - get_order_book(GLOBAL_BID/ASK) reads ONLY historical flow, never
      our own quotes. Prevents the recursive spiral where we requote
      inside our own previous quotes, collapsing spread to zero.
    - Dynamic kappa calibrated to 70% of historical spread so our quotes
      beat other MM teams who naively quote at 80-90%.
    - gamma=0.5 gives strong inventory skew — when long, our ask drops
      aggressively to attract buyers and prevent one-sided accumulation.
    """

    def __init__(
        self,
        trader:         shift.Trader,
        risk_manager:   RiskManager,
        ticker:         str,
        session_start:  datetime,
        session_end:    datetime,
        gamma:          float = 0.5,
        kappa:          float = 1.5,
        max_inventory:  int   = 3,
        vol_window:     int   = 30,
        quote_refresh:  float = 3.0,
        min_spread_pct: float = 0.002,
        check_freq:     float = 1.0,
    ):
        self.trader         = trader
        self.risk_manager   = risk_manager
        self.ticker         = ticker
        self.session_start  = session_start
        self.session_end    = session_end
        self.gamma          = gamma
        self.kappa          = kappa
        self.max_inventory  = max_inventory
        self.vol_window     = vol_window
        self.quote_refresh  = quote_refresh
        self.min_spread_pct = min_spread_pct
        self.check_freq     = check_freq

        self.price_history   = []
        self.trade_count     = 0
        self.running         = False
        self.last_quote_time = 0.0
        self.last_bid        = 0.0
        self.last_ask        = 0.0

    # ------------------------------------------------------------------ #
    #  Logging                                                            #
    # ------------------------------------------------------------------ #
    def log(self, msg: str):
        log("AS", self.ticker, msg)

    # ------------------------------------------------------------------ #
    #  Time remaining in session [0.0 → 1.0]                            #
    # ------------------------------------------------------------------ #
    def time_remaining(self) -> float:
        now     = self.trader.get_last_trade_time()
        total   = (self.session_end - self.session_start).total_seconds()
        elapsed = (now - self.session_start).total_seconds()
        if total <= 0:
            return 0.0
        return max(0.0, min(1.0, 1.0 - elapsed / total))

    # ------------------------------------------------------------------ #
    #  Rolling volatility                                                 #
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
    #  Read HISTORICAL (global) book only — never our own quotes         #
    # ------------------------------------------------------------------ #
    def get_global_spread(self) -> tuple[float, float] | None:
        """
        Uses GLOBAL_BID/GLOBAL_ASK to read only historical order flow.
        get_best_prices() returns combined local+global, which means our
        own limit orders appear as the "market" once posted — causing a
        recursive spiral where we requote inside ourselves each tick until
        spread → 0. This method prevents that entirely.
        """
        try:
            bid_book = self.trader.get_order_book(
                self.ticker, shift.OrderBookType.GLOBAL_BID, 1
            )
            ask_book = self.trader.get_order_book(
                self.ticker, shift.OrderBookType.GLOBAL_ASK, 1
            )
            if not bid_book or not ask_book:
                return None
            return float(bid_book[0].price), float(ask_book[0].price)
        except Exception:
            return None

    # ------------------------------------------------------------------ #
    #  A-S quote computation                                             #
    # ------------------------------------------------------------------ #
    def compute_quotes(
        self,
        mid:       float,
        bid_mkt:   float,
        ask_mkt:   float,
        inventory: float,
    ) -> tuple[float, float, float, float, float]:
        """Returns (bid_quote, ask_quote, reservation_price, spread, kappa)."""
        sigma2        = self.estimate_sigma2()
        t_rem         = self.time_remaining()
        actual_spread = max(ask_mkt - bid_mkt, 0.01)

        # Dynamic kappa: target 70% of historical spread so we beat
        # other MM teams posting at 80-90% — we become the best bid/ask
        target = actual_spread * 0.70
        kappa  = self.gamma / (np.exp(self.gamma * target / 2) - 1)
        kappa  = max(kappa, 0.01)

        # Reservation price: skews quotes to offload inventory
        # gamma=0.5 means at inv=+3, ask drops by 0.5*spread*T-t per lot
        # e.g. HELE spread=0.9, T-t=0.8: ask drops 3*0.5*0.9*0.8 = $1.08
        # This forces buyers to cross our cheaper ask and clear inventory
        r = mid - inventory * self.gamma * actual_spread * t_rem

        # Optimal spread (A-S formula)
        term1  = self.gamma * sigma2 * t_rem
        term2  = (2.0 / self.gamma) * np.log(1.0 + self.gamma / kappa)
        spread = max(term1 + term2, mid * self.min_spread_pct)

        bid_q = round(r - spread / 2, 2)
        ask_q = round(r + spread / 2, 2)
        return bid_q, ask_q, r, spread, kappa

    # ------------------------------------------------------------------ #
    #  Price moved enough to warrant requoting?                          #
    # ------------------------------------------------------------------ #
    def price_moved(self, bid: float, ask: float) -> bool:
        return (
            abs(bid - self.last_bid) > 0.02 or
            abs(ask - self.last_ask) > 0.02
        )

    # ------------------------------------------------------------------ #
    #  Post quotes                                                       #
    # ------------------------------------------------------------------ #
    def post_quotes(
        self,
        bid_q:     float,
        ask_q:     float,
        r:         float,
        spread:    float,
        kappa:     float,
        inventory: float,
    ):
        posted_bid = False
        posted_ask = False

        # Only post bid if below max long inventory
        if inventory < self.max_inventory:
            if self.risk_manager.can_trade(self.ticker, True, bid_q):
                submit_limit_buy(self.trader, self.ticker, 1, bid_q)
                self.trade_count += 1
                posted_bid = True

        # Only post ask if above max short inventory
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
                f"r={r:.2f} spread={spread:.3f} κ={kappa:.2f} | "
                f"inv={int(inventory)} | trades={self.trade_count}"
            )
            self.last_bid        = bid_q
            self.last_ask        = ask_q
            self.last_quote_time = time()

    # ------------------------------------------------------------------ #
    #  Close all inventory at end of session                             #
    # ------------------------------------------------------------------ #
    def close_inventory(self, reason: str):
        cancel_orders_for(self.trader, self.ticker)
        sleep(0.3)
        position = get_position(self.trader, self.ticker)
        if position > 0:
            lots = int(position / 100)
            submit_market_sell(self.trader, self.ticker, lots)
            self.trade_count += 1
            self.log(
                f"CLOSE ({reason}) MARKET_SELL {lots}L | "
                f"trades={self.trade_count}"
            )
        elif position < 0:
            lots = int(-position / 100)
            submit_market_buy(self.trader, self.ticker, lots)
            self.trade_count += 1
            self.log(
                f"CLOSE ({reason}) MARKET_BUY {lots}L | "
                f"trades={self.trade_count}"
            )

    # ------------------------------------------------------------------ #
    #  Main loop                                                         #
    # ------------------------------------------------------------------ #
    def run(self, end_time):
        self.running = True
        self.log("Strategy started")
        self.log(
            f"Config | γ={self.gamma} | "
            f"max_inv=±{self.max_inventory} | "
            f"refresh={self.quote_refresh}s | "
            f"target_spread=70% | "
            f"min_spread={self.min_spread_pct*100:.2f}%"
        )

        self.trader.sub_order_book(self.ticker)
        sleep(0.5)

        # Warmup: collect price history for sigma2 estimation
        # (uses get_best_prices — fine here, we're not trading yet)
        self.log(f"Warming up — need {self.vol_window} samples...")
        while len(self.price_history) < self.vol_window:
            if self.risk_manager.all_halted:
                return
            prices = get_best_prices(self.trader, self.ticker)
            if prices:
                self.update_price((prices[0] + prices[1]) / 2)
            sleep(self.check_freq)
        self.log("Warmup complete — quoting started")

        while (
            self.running and
            self.trader.get_last_trade_time() < end_time
        ):
            if self.risk_manager.all_halted:
                self.log("Global halt")
                break

            try:
                # CRITICAL: use GLOBAL book to avoid the recursive spiral
                global_prices = self.get_global_spread()
                if global_prices is None:
                    sleep(self.check_freq)
                    continue

                bid_mkt, ask_mkt = global_prices
                mid               = (bid_mkt + ask_mkt) / 2
                self.update_price(mid)

                position  = get_position(self.trader, self.ticker)
                inventory = position / 100

                bid_q, ask_q, r, spread, kappa = self.compute_quotes(
                    mid, bid_mkt, ask_mkt, inventory
                )

                self.log(
                    f"mid={mid:.2f} | "
                    f"hist=[{bid_mkt:.2f}/{ask_mkt:.2f}] | "
                    f"our=[{bid_q:.2f}/{ask_q:.2f}] | "
                    f"r={r:.2f} spread={spread:.3f} κ={kappa:.2f} | "
                    f"inv={int(inventory)} T-t={self.time_remaining():.2f}"
                )

                # Repost on timer or when historical price moves
                time_expired  = (
                    time() - self.last_quote_time >= self.quote_refresh
                )
                price_shifted = self.price_moved(bid_mkt, ask_mkt)

                if time_expired or price_shifted:
                    cancel_orders_for(self.trader, self.ticker)
                    sleep(0.2)

                    position  = get_position(self.trader, self.ticker)
                    inventory = position / 100

                    bid_q, ask_q, r, spread, kappa = self.compute_quotes(
                        mid, bid_mkt, ask_mkt, inventory
                    )
                    self.post_quotes(bid_q, ask_q, r, spread, kappa, inventory)

                sleep(self.check_freq)

            except KeyboardInterrupt:
                break
            except Exception as e:
                self.log(f"Error: {e}")
                sleep(self.check_freq)

        self.log("End time — closing inventory")
        self.close_inventory("end_of_session")
        self.trader.unsub_order_book(self.ticker)

        final_pl = self.trader.get_portfolio_item(self.ticker).get_realized_pl()
        self.log(
            f"DONE | P&L: ${final_pl:.2f} | trades={self.trade_count}"
        )
        self.running = False