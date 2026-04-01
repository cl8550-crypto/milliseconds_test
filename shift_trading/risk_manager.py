import shift
from datetime import datetime
from utils import get_buying_power, get_position, log


class RiskManager:
    def __init__(
        self,
        trader: shift.Trader,
        total_tickers: list[str],
        max_bp_usage: float = 0.80,
        max_position_lots: int = 5,
        max_loss_per_ticker: float = -2000.0,
        max_total_loss: float = -20000.0,
    ):
        self.trader              = trader
        self.total_tickers       = total_tickers
        self.max_bp_usage        = max_bp_usage
        self.max_position_lots   = max_position_lots
        self.max_loss_per_ticker = max_loss_per_ticker
        self.max_total_loss      = max_total_loss

        self.initial_bp      = None
        self.halted_tickers  = set()
        self.all_halted      = False

    # ------------------------------------------------------------------ #
    #  Initialization                                                      #
    # ------------------------------------------------------------------ #
    def initialize(self):
        self.initial_bp = get_buying_power(self.trader)
        log("RISK", "INIT", f"Starting buying power:     ${self.initial_bp:,.2f}")
        log("RISK", "INIT", f"Max BP usage:              {self.max_bp_usage*100:.0f}%")
        log("RISK", "INIT", f"Max position per ticker:   {self.max_position_lots} lots")
        log("RISK", "INIT", f"Max loss per ticker:       ${self.max_loss_per_ticker:,.2f}")
        log("RISK", "INIT", f"Max total loss:            ${self.max_total_loss:,.2f}")

    # ------------------------------------------------------------------ #
    #  Dynamic lot sizing                                                  #
    # ------------------------------------------------------------------ #
    def get_safe_lot_size(self, price: float, max_lots: int = 5) -> int:
        """
        Calculate max affordable lot size based on price and available BP.
        Prevents high-priced stocks from failing risk checks entirely.
        """
        if price <= 0 or self.initial_bp is None:
            return 0

        current_bp   = get_buying_power(self.trader)
        bp_floor     = self.initial_bp * (1 - self.max_bp_usage)
        available_bp = max(0, current_bp - bp_floor)

        max_affordable = int(available_bp / (price * 100))
        safe_lots      = max(0, min(max_affordable, max_lots))

        if safe_lots == 0:
            log("RISK", "LOTS", f"No affordable lots at price=${price:.2f}")

        return safe_lots

    # ------------------------------------------------------------------ #
    #  Buying power checks                                                 #
    # ------------------------------------------------------------------ #
    def has_buying_power(self, estimated_cost: float) -> bool:
        if self.initial_bp is None:
            return False

        current_bp = get_buying_power(self.trader)
        bp_floor   = self.initial_bp * (1 - self.max_bp_usage)

        if current_bp - estimated_cost < bp_floor:
            log(
                "RISK", "BP",
                f"Insufficient BP: current=${current_bp:,.2f} | "
                f"floor=${bp_floor:,.2f} | "
                f"cost=${estimated_cost:,.2f}"
            )
            return False
        return True

    def get_bp_usage_pct(self) -> float:
        if self.initial_bp is None or self.initial_bp == 0:
            return 0.0
        current_bp = get_buying_power(self.trader)
        return (self.initial_bp - current_bp) / self.initial_bp

    # ------------------------------------------------------------------ #
    #  Position size checks                                                #
    # ------------------------------------------------------------------ #
    def can_open_long(self, ticker: str) -> bool:
        position = get_position(self.trader, ticker)
        if position >= self.max_position_lots * 100:
            log("RISK", ticker, f"Max long position reached: {position} shares")
            return False
        return True

    def can_open_short(self, ticker: str) -> bool:
        position = get_position(self.trader, ticker)
        if position <= -self.max_position_lots * 100:
            log("RISK", ticker, f"Max short position reached: {position} shares")
            return False
        return True

    # ------------------------------------------------------------------ #
    #  Loss limit checks                                                   #
    # ------------------------------------------------------------------ #
    def check_ticker_loss(self, ticker: str) -> bool:
        if ticker in self.halted_tickers:
            return False
        try:
            pl         = self.trader.get_portfolio_item(ticker).get_realized_pl()
            unrealized = self.trader.get_unrealized_pl(ticker)
            total_pl   = pl + unrealized

            if total_pl < self.max_loss_per_ticker:
                log(
                    "RISK", ticker,
                    f"Loss limit breached: ${total_pl:,.2f} < "
                    f"${self.max_loss_per_ticker:,.2f} — halting {ticker}"
                )
                self.halted_tickers.add(ticker)
                return False
        except Exception as e:
            log("RISK", ticker, f"Error checking ticker loss: {e}")
            return False
        return True

    def check_total_loss(self) -> bool:
        if self.all_halted:
            return False
        try:
            total_pl = self.trader.get_portfolio_summary().get_total_realized_pl()
            if total_pl < self.max_total_loss:
                log(
                    "RISK", "PORTFOLIO",
                    f"Total loss limit breached: ${total_pl:,.2f} — "
                    f"HALTING ALL TRADING"
                )
                self.all_halted = True
                return False
        except Exception as e:
            log("RISK", "PORTFOLIO", f"Error checking total loss: {e}")
            return False
        return True

    # ------------------------------------------------------------------ #
    #  Master check — for OPENING new positions                            #
    # ------------------------------------------------------------------ #
    def can_trade(self, ticker: str, is_buy: bool, price: float) -> bool:
        """
        Full check before opening any new position.
        Returns True only if ALL conditions are met.
        """
        # 1. Price sanity
        if price <= 0:
            log("RISK", ticker, f"Invalid price {price} — rejecting")
            return False

        # 2. Global halt
        if self.all_halted:
            return False

        # 3. Ticker halted
        if ticker in self.halted_tickers:
            return False

        # 4. Total loss check
        if not self.check_total_loss():
            return False

        # 5. Ticker loss check
        if not self.check_ticker_loss(ticker):
            return False

        # 6. Buying power — cost of 1 lot
        if not self.has_buying_power(price * 100):
            return False

        # 7. Position size
        if is_buy and not self.can_open_long(ticker):
            return False
        if not is_buy and not self.can_open_short(ticker):
            return False

        return True

    # ------------------------------------------------------------------ #
    #  Lighter check — for CLOSING existing positions                      #
    # ------------------------------------------------------------------ #
    def can_close(self, ticker: str, price: float) -> bool:
        """
        Lightweight check just for closing positions.
        Bypasses halt flags — closing must ALWAYS be allowed.
        Only checks price validity and basic buying power.
        """
        # Price sanity only
        if price <= 0:
            log("RISK", ticker, f"Invalid close price {price} — rejecting")
            return False

        # Buying power — need at least enough for 1 lot
        if not self.has_buying_power(price * 100):
            log("RISK", ticker, "Insufficient BP to close — forcing market close")
            return False

        return True

    # ------------------------------------------------------------------ #
    #  Status report                                                       #
    # ------------------------------------------------------------------ #
    def print_status(self):
        bp_usage   = self.get_bp_usage_pct()
        current_bp = get_buying_power(self.trader)

        try:
            total_pl = self.trader.get_portfolio_summary().get_total_realized_pl()
        except Exception:
            total_pl = 0.0

        print("\n" + "-"*50, flush=True)
        print(
            f"[RISK STATUS @ {datetime.now().strftime('%H:%M:%S')}]",
            flush=True
        )
        print(f"  Buying Power:    ${current_bp:,.2f} ({bp_usage*100:.1f}% used)", flush=True)
        print(f"  Total P&L:       ${total_pl:,.2f}", flush=True)
        print(f"  Max Total Loss:  ${self.max_total_loss:,.2f}", flush=True)
        print(
            f"  Halted Tickers:  "
            f"{self.halted_tickers if self.halted_tickers else 'None'}",
            flush=True
        )
        print(f"  All Halted:      {self.all_halted}", flush=True)
        print("-"*50 + "\n", flush=True)

    # ------------------------------------------------------------------ #
    #  Manual controls                                                     #
    # ------------------------------------------------------------------ #
    def halt_ticker(self, ticker: str):
        self.halted_tickers.add(ticker)
        log("RISK", ticker, "Manually halted")

    def resume_ticker(self, ticker: str):
        self.halted_tickers.discard(ticker)
        log("RISK", ticker, "Manually resumed")

    def halt_all(self):
        """Emergency halt — stops all NEW trading but allows closing"""
        self.all_halted = True
        log("RISK", "ALL", "Emergency halt — no new positions allowed")



