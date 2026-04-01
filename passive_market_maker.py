import shift
from time import sleep, time
from utils import (
    log, get_position,
    submit_limit_buy, submit_limit_sell,
    cancel_orders_for, close_position
)
from risk_manager import RiskManager


class PassiveMarketMaker:
    """
    Lightweight market maker — order count engine for calm periods.

    Posts a 1-lot limit buy just inside the bid and a 1-lot limit sell
    just inside the ask every repost_interval seconds. Earns the $0.002
    rebate on each fill. Cancels and reposts to stay at top of book.

    Automatically skips quoting if a directional position is already open
    (e.g. from SpikeFadeStrategy on the same ticker), so the two strategies
    do not fight each other.

    At 15-second repost intervals on 2 tickers this generates ~240 order
    submissions in a 5-hour session — enough to satisfy the 200-order minimum
    even on a day with few spikes.
    """

    def __init__(
        self,
        trader: shift.Trader,
        risk_manager: RiskManager,
        ticker: str,
        order_size: int = 1,
        repost_interval: float = 15.0,  # seconds between cancel-and-repost cycles
        check_freq: float = 1.0,
        max_position_lots: int = 2,     # pause quoting above this position size
    ):
        self.trader           = trader
        self.risk_manager     = risk_manager
        self.ticker           = ticker
        self.order_size       = order_size
        self.repost_interval  = repost_interval
        self.check_freq       = check_freq
        self.max_position_lots = max_position_lots

        self.trade_count      = 0
        self.last_post_time   = 0
        self.running          = False

    # ------------------------------------------------------------------ #
    #  Logging                                                             #
    # ------------------------------------------------------------------ #
    def log(self, message: str):
        log("PMM", self.ticker, message)

    # ------------------------------------------------------------------ #
    #  State helpers                                                       #
    # ------------------------------------------------------------------ #
    def has_pending_orders(self) -> bool:
        return any(
            o.symbol == self.ticker
            for o in self.trader.get_waiting_list()
        )

    def get_position(self) -> int:
        return get_position(self.trader, self.ticker)

    def cancel_pending(self):
        cancel_orders_for(self.trader, self.ticker)
        sleep(0.3)

    # ------------------------------------------------------------------ #
    #  Quote posting                                                       #
    # ------------------------------------------------------------------ #
    def post_quotes(self, bid: float, ask: float):
        """Post limit buy just inside bid, limit sell just inside ask."""
        buy_price  = round(bid + 0.01, 2)
        sell_price = round(ask - 0.01, 2)

        if sell_price <= buy_price:
            self.log("Spread too tight to quote — skipping")
            return

        if not self.risk_manager.can_trade(self.ticker, True, buy_price):
            return
        if not self.risk_manager.can_trade(self.ticker, False, sell_price):
            return

        r_buy  = submit_limit_buy(
            self.trader, self.ticker, self.order_size, buy_price
        )
        r_sell = submit_limit_sell(
            self.trader, self.ticker, self.order_size, sell_price
        )

        submitted = (1 if r_buy else 0) + (1 if r_sell else 0)
        if submitted > 0:
            self.trade_count   += submitted
            self.last_post_time = time()
            self.log(
                f"QUOTES posted | buy@{buy_price:.2f} sell@{sell_price:.2f} | "
                f"total_orders={self.trade_count}"
            )

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
            f"Passive market maker started | "
            f"repost_interval={self.repost_interval:.0f}s | "
            f"size={self.order_size} lot"
        )
        self.trader.sub_order_book(self.ticker)
        sleep(1)

        while self.running and self.trader.get_last_trade_time() < end_time:

            if self.risk_manager.all_halted:
                self.log("Global halt — stopping PMM")
                break

            try:
                position = self.get_position()

                # Pause quoting if a larger directional position is open
                if abs(position) >= self.max_position_lots * 100:
                    if self.has_pending_orders():
                        self.cancel_pending()
                    sleep(self.check_freq)
                    continue

                # Not time to repost yet
                if (time() - self.last_post_time) < self.repost_interval:
                    sleep(self.check_freq)
                    continue

                # Cancel stale quotes before reposting
                if self.has_pending_orders():
                    self.cancel_pending()

                best = self.trader.get_best_price(self.ticker)
                bid  = best.get_bid_price()
                ask  = best.get_ask_price()

                if bid <= 0 or ask <= 0:
                    sleep(self.check_freq)
                    continue

                self.post_quotes(bid, ask)
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
        self.log(f"DONE | Total orders submitted: {self.trade_count}")
        self.running = False
