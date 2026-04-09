import shift
from time import sleep, time
from datetime import datetime
from utils import (
    log, get_position,
    submit_limit_buy, submit_limit_sell,
    submit_market_buy, submit_market_sell,
    cancel_orders_for, close_position
)
from risk_manager import RiskManager


class OrderBookArbStrategy:
    def __init__(
        self,
        trader: shift.Trader,
        risk_manager: RiskManager,
        ticker: str,
        order_size: int = 2,
        min_arb_spread: float = 0.02,
        min_pressure_gap: float = 0.05,
        check_freq: float = 1.0,
        cooldown: float = 5.0,
        max_history: int = 50,
    ):
        self.trader           = trader
        self.risk_manager     = risk_manager
        self.ticker           = ticker
        self.order_size       = order_size
        self.min_arb_spread   = min_arb_spread
        self.min_pressure_gap = min_pressure_gap
        self.check_freq       = check_freq
        self.cooldown         = cooldown
        self.max_history      = max_history

        self.trade_count      = 0
        self.last_trade_time  = 0
        self.arb_count        = 0
        self.pressure_count   = 0
        self.running          = False

        # ---- Local book tracking — NEW ---- #
        self.local_book_active     = False
        self.local_book_first_seen = None   # timestamp when local book first appeared
        self.local_bid_history     = []     # track local bid over time
        self.local_ask_history     = []     # track local ask over time

    # ------------------------------------------------------------------ #
    #  Logging                                                             #
    # ------------------------------------------------------------------ #
    def log(self, message: str):
        log("ARB", self.ticker, message)

    # ------------------------------------------------------------------ #
    #  Cooldown                                                            #
    # ------------------------------------------------------------------ #
    def is_cooling_down(self) -> bool:
        return (time() - self.last_trade_time) < self.cooldown

    # ------------------------------------------------------------------ #
    #  Pending orders                                                      #
    # ------------------------------------------------------------------ #
    def has_pending_orders(self) -> bool:
        return any(
            o.symbol == self.ticker
            for o in self.trader.get_waiting_list()
        )

    # ------------------------------------------------------------------ #
    #  Order book data                                                     #
    # ------------------------------------------------------------------ #
    def get_book_prices(self) -> dict:
        """
        Fetch top-of-book from both GLOBAL and LOCAL order books.
        Returns dict with global_bid, global_ask, local_bid, local_ask.
        Any value can be None if book is empty.
        """
        result = {
            "global_bid": None,
            "global_ask": None,
            "local_bid":  None,
            "local_ask":  None,
        }

        try:
            g_bid = self.trader.get_order_book(
                self.ticker, shift.OrderBookType.GLOBAL_BID, 1
            )
            g_ask = self.trader.get_order_book(
                self.ticker, shift.OrderBookType.GLOBAL_ASK, 1
            )
            l_bid = self.trader.get_order_book(
                self.ticker, shift.OrderBookType.LOCAL_BID, 1
            )
            l_ask = self.trader.get_order_book(
                self.ticker, shift.OrderBookType.LOCAL_ASK, 1
            )

            if g_bid and g_bid[0].price > 0:
                result["global_bid"] = g_bid[0].price
            if g_ask and g_ask[0].price > 0:
                result["global_ask"] = g_ask[0].price
            if l_bid and l_bid[0].price > 0:
                result["local_bid"] = l_bid[0].price
            if l_ask and l_ask[0].price > 0:
                result["local_ask"] = l_ask[0].price

        except Exception as e:
            self.log(f"Error fetching order book: {e}")

        return result

    # ------------------------------------------------------------------ #
    #  Local book monitoring — NEW                                         #
    # ------------------------------------------------------------------ #
    def check_local_book_activity(self, prices: dict):
        """
        Monitors local book and logs when competition trading begins.
        Tracks local bid/ask history for context.
        """
        l_bid = prices["local_bid"]
        l_ask = prices["local_ask"]
        g_bid = prices["global_bid"]
        g_ask = prices["global_ask"]

        # First time we see local book activity
        if not self.local_book_active and (l_bid or l_ask):
            self.local_book_active     = True
            self.local_book_first_seen = datetime.now().strftime("%H:%M:%S")
            self.log(
                f"🔔 LOCAL BOOK NOW ACTIVE — "
                f"competition trading detected! | "
                f"local_bid={l_bid:.2f if l_bid else 'N/A'} | "
                f"local_ask={l_ask:.2f if l_ask else 'N/A'} | "
                f"global_bid={g_bid:.2f if g_bid else 'N/A'} | "
                f"global_ask={g_ask:.2f if g_ask else 'N/A'}"
            )

        # Track local book history
        if l_bid:
            self.local_bid_history.append(l_bid)
            if len(self.local_bid_history) > self.max_history:
                self.local_bid_history = self.local_bid_history[-self.max_history:]

        if l_ask:
            self.local_ask_history.append(l_ask)
            if len(self.local_ask_history) > self.max_history:
                self.local_ask_history = self.local_ask_history[-self.max_history:]

    # ------------------------------------------------------------------ #
    #  Signal detection                                                    #
    # ------------------------------------------------------------------ #
    def detect_arb(self, prices: dict) -> tuple[str, float] | tuple[None, None]:
        """
        Detect true cross-book arbitrage.

        LOCAL_BID > GLOBAL_ASK:
            Competitor buying above market ask
            → SELL to them at their high bid
            → Profit = local_bid - global_ask

        LOCAL_ASK < GLOBAL_BID:
            Competitor selling below market bid
            → BUY from them at their low ask
            → Profit = global_bid - local_ask
        """
        g_bid = prices["global_bid"]
        g_ask = prices["global_ask"]
        l_bid = prices["local_bid"]
        l_ask = prices["local_ask"]

        if l_bid and g_ask:
            profit = l_bid - g_ask
            if profit > self.min_arb_spread:
                return "SELL_TO_LOCAL", profit

        if l_ask and g_bid:
            profit = g_bid - l_ask
            if profit > self.min_arb_spread:
                return "BUY_FROM_LOCAL", profit

        return None, None

    def detect_pressure(self, prices: dict) -> tuple[str, float] | tuple[None, None]:
        """
        Detect competitor pressure signals.

        LOCAL_BID >> GLOBAL_BID:
            Competitors bidding aggressively above market
            → Price going UP → BUY ahead of them

        LOCAL_ASK << GLOBAL_ASK:
            Competitors asking aggressively below market
            → Price going DOWN → SELL ahead of them
        """
        g_bid = prices["global_bid"]
        g_ask = prices["global_ask"]
        l_bid = prices["local_bid"]
        l_ask = prices["local_ask"]

        if l_bid and g_bid:
            premium = l_bid - g_bid
            if premium > self.min_pressure_gap:
                return "BUY_PRESSURE", premium

        if l_ask and g_ask:
            discount = g_ask - l_ask
            if discount > self.min_pressure_gap:
                return "SELL_PRESSURE", discount

        return None, None

    # ------------------------------------------------------------------ #
    #  Trade execution                                                     #
    # ------------------------------------------------------------------ #
    def execute_arb(self, signal: str, prices: dict, profit: float):
        """Execute a true cross-book arbitrage"""
        g_bid = prices["global_bid"]
        g_ask = prices["global_ask"]
        l_bid = prices["local_bid"]
        l_ask = prices["local_ask"]

        if signal == "SELL_TO_LOCAL":
            if not self.risk_manager.can_trade(self.ticker, False, l_bid):
                return

            result = submit_limit_sell(
                self.trader, self.ticker,
                self.order_size,
                round(l_bid, 2)
            )
            if result:
                self.trade_count     += 1
                self.arb_count       += 1
                self.last_trade_time  = time()
                self.log(
                    f"ARB SELL_TO_LOCAL | "
                    f"SELL @ {l_bid:.2f} | "
                    f"global_ask={g_ask:.2f} | "
                    f"profit/share=${profit:.4f} | "
                    f"arb#{self.arb_count}"
                )

        elif signal == "BUY_FROM_LOCAL":
            if not self.risk_manager.can_trade(self.ticker, True, l_ask):
                return

            result = submit_limit_buy(
                self.trader, self.ticker,
                self.order_size,
                round(l_ask, 2)
            )
            if result:
                self.trade_count     += 1
                self.arb_count       += 1
                self.last_trade_time  = time()
                self.log(
                    f"ARB BUY_FROM_LOCAL | "
                    f"BUY @ {l_ask:.2f} | "
                    f"global_bid={g_bid:.2f} | "
                    f"profit/share=${profit:.4f} | "
                    f"arb#{self.arb_count}"
                )

    def execute_pressure(self, signal: str, prices: dict, gap: float):
        """Execute a competitor pressure trade"""
        g_bid = prices["global_bid"]
        g_ask = prices["global_ask"]

        if signal == "BUY_PRESSURE":
            if not g_ask or not self.risk_manager.can_trade(
                self.ticker, True, g_ask
            ):
                return

            result = submit_limit_buy(
                self.trader, self.ticker,
                self.order_size,
                round(g_ask - 0.01, 2)
            )
            if result:
                self.trade_count    += 1
                self.pressure_count += 1
                self.last_trade_time = time()
                self.log(
                    f"PRESSURE BUY | "
                    f"@ {g_ask-0.01:.2f} | "
                    f"local_premium=${gap:.4f} | "
                    f"pressure#{self.pressure_count}"
                )

        elif signal == "SELL_PRESSURE":
            if not g_bid or not self.risk_manager.can_trade(
                self.ticker, False, g_bid
            ):
                return

            result = submit_limit_sell(
                self.trader, self.ticker,
                self.order_size,
                round(g_bid + 0.01, 2)
            )
            if result:
                self.trade_count    += 1
                self.pressure_count += 1
                self.last_trade_time = time()
                self.log(
                    f"PRESSURE SELL | "
                    f"@ {g_bid+0.01:.2f} | "
                    f"local_discount=${gap:.4f} | "
                    f"pressure#{self.pressure_count}"
                )

    # ------------------------------------------------------------------ #
    #  Position closing                                                    #
    # ------------------------------------------------------------------ #
    def close_all(self):
        cancel_orders_for(self.trader, self.ticker)
        sleep(2)
        close_position(self.trader, self.ticker)

    # ------------------------------------------------------------------ #
    #  Main strategy loop                                                  #
    # ------------------------------------------------------------------ #
    def run(self, end_time):
        self.running = True
        self.log("Strategy started")
        self.log(
            f"Waiting for LOCAL book activity — "
            f"arb inactive until competition starts"
        )

        self.trader.sub_order_book(self.ticker)
        sleep(1)

        while self.running and self.trader.get_last_trade_time() < end_time:

            # ---- Check global halt first ---- #
            if self.risk_manager.all_halted:
                self.log("Global halt detected — stopping")
                break

            try:
                prices = self.get_book_prices()

                g_bid = prices["global_bid"]
                g_ask = prices["global_ask"]
                l_bid = prices["local_bid"]
                l_ask = prices["local_ask"]

                # ---- Monitor local book activity ---- #
                self.check_local_book_activity(prices)

                # Format prices for logging
                g_bid_str = f"{g_bid:.2f}" if g_bid else "N/A"
                g_ask_str = f"{g_ask:.2f}" if g_ask else "N/A"
                l_bid_str = f"{l_bid:.2f}" if l_bid else "N/A"
                l_ask_str = f"{l_ask:.2f}" if l_ask else "N/A"

                self.log(
                    f"GLOBAL bid={g_bid_str} ask={g_ask_str} | "
                    f"LOCAL  bid={l_bid_str} ask={l_ask_str}"
                )

                # Skip if no global prices
                if not g_bid or not g_ask:
                    sleep(self.check_freq)
                    continue

                # Skip if cooling down
                if self.is_cooling_down():
                    sleep(self.check_freq)
                    continue

                # Skip if pending orders
                if self.has_pending_orders():
                    sleep(self.check_freq)
                    continue

                # ---- Only trade if local book is active ---- #
                if not self.local_book_active:
                    sleep(self.check_freq)
                    continue

                # ---- Priority 1: True arbitrage ---- #
                arb_signal, arb_profit = self.detect_arb(prices)
                if arb_signal:
                    self.log(
                        f"ARB SIGNAL: {arb_signal} | "
                        f"profit/share=${arb_profit:.4f}"
                    )
                    self.execute_arb(arb_signal, prices, arb_profit)
                    sleep(self.check_freq)
                    continue

                # ---- Priority 2: Pressure signal ---- #
                position = get_position(self.trader, self.ticker)
                if position == 0:
                    pressure_signal, pressure_gap = self.detect_pressure(prices)
                    if pressure_signal:
                        self.log(
                            f"PRESSURE SIGNAL: {pressure_signal} | "
                            f"gap=${pressure_gap:.4f}"
                        )
                        self.execute_pressure(pressure_signal, prices, pressure_gap)

                # ---- Close pressure position when gap closes ---- #
                elif position != 0:
                    pressure_signal, _ = self.detect_pressure(prices)
                    if pressure_signal is None:
                        self.log(
                            f"Pressure gap closed — closing position "
                            f"(pos={position})"
                        )
                        cancel_orders_for(self.trader, self.ticker)
                        sleep(1)
                        close_position(self.trader, self.ticker)

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

        # Final summary
        self.log("="*40)
        self.log(f"DONE | Total trades:    {self.trade_count}")
        self.log(f"DONE | Arb trades:      {self.arb_count}")
        self.log(f"DONE | Pressure trades: {self.pressure_count}")
        if self.local_book_first_seen:
            self.log(f"DONE | Local book first active: {self.local_book_first_seen}")
        else:
            self.log(f"DONE | Local book never became active (pre-competition)")
        self.log("="*40)
        self.running = False