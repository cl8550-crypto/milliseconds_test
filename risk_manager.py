import shift
from datetime import datetime
from utils import get_buying_power, get_position, log


class RiskManager:
    def __init__(
        self,
        trader: shift.Trader,
        total_tickers: list[str],
        max_bp_usage: float = 0.60,
        max_position_lots: int = 3,
        max_loss_per_ticker: float = -2000.0,
        max_total_loss: float = -20000.0,
        max_concurrent_positions: int = 12,    # raised from 6 — BP is real throttle
        max_bp_per_trade: float = 100000.0,    # raised from 50k — allows GS/CAT
    ):
        self.trader                   = trader
        self.total_tickers            = total_tickers
        self.max_bp_usage             = max_bp_usage
        self.max_position_lots        = max_position_lots
        self.max_loss_per_ticker      = max_loss_per_ticker
        self.max_total_loss           = max_total_loss
        self.max_concurrent_positions = max_concurrent_positions
        self.max_bp_per_trade         = max_bp_per_trade

        self.initial_bp          = None
        self.halted_tickers      = set()
        self.all_halted          = False
        self.ticker_strategy_map = {}

    # ------------------------------------------------------------------ #
    #  Initialization                                                      #
    # ------------------------------------------------------------------ #
    def initialize(self):
        self.initial_bp = get_buying_power(self.trader)
        log("RISK", "INIT", f"Starting BP:               ${self.initial_bp:,.2f}")
        log("RISK", "INIT", f"Max BP usage:              {self.max_bp_usage*100:.0f}%")
        log("RISK", "INIT", f"Max position per ticker:   {self.max_position_lots} lots")
        log("RISK", "INIT", f"Max loss per ticker:       ${self.max_loss_per_ticker:,.2f}")
        log("RISK", "INIT", f"Max total loss:            ${self.max_total_loss:,.2f}")
        log("RISK", "INIT", f"Max concurrent positions:  {self.max_concurrent_positions}")
        log("RISK", "INIT", f"Max BP per trade:          ${self.max_bp_per_trade:,.2f}")

    # ------------------------------------------------------------------ #
    #  Dynamic lot sizing                                                  #
    # ------------------------------------------------------------------ #
    def get_safe_lot_size(self, price: float, max_lots: int = 3) -> int:
        """
        Calculate affordable lot size based on:
        1. Available buying power
        2. Per-trade BP cap ($100k — allows GS $92k, CAT $63k)
        3. Requested max lots
        """
        if price <= 0 or self.initial_bp is None:
            return 0

        current_bp   = get_buying_power(self.trader)
        bp_floor     = self.initial_bp * (1 - self.max_bp_usage)
        available_bp = max(0, current_bp - bp_floor)

        # Cap by available BP
        max_from_bp  = int(available_bp / (price * 100))

        # Cap by per-trade limit
        max_from_cap = int(self.max_bp_per_trade / (price * 100))

        safe_lots = max(0, min(max_from_bp, max_from_cap, max_lots))

        if safe_lots == 0:
            log("RISK", "LOTS", f"No affordable lots at ${price:.2f}")

        return safe_lots

    def get_dynamic_order_size(self, price: float) -> int:
        """
        Scale order size inversely with price.

        GS  $927 × 1 lot = $92,700  → 1 lot
        CAT $631 × 1 lot = $63,100  → 1 lot
        JPM $308 × 2 lots = $61,600 → 2 lots
        AAPL$259 × 2 lots = $51,800 → 2 lots
        BA  $240 × 2 lots = $48,000 → 2 lots
        NVDA$181 × 2 lots = $36,200 → 2 lots
        NKE  $67 × 3 lots = $20,100 → 3 lots
        MMM $169 × 3 lots = $50,700 → 3 lots
        """
        if price > 500:
            return 1
        elif price > 200:
            return 2
        else:
            return 3

    # ------------------------------------------------------------------ #
    #  Concurrent position check                                           #
    # ------------------------------------------------------------------ #
    def get_open_position_count(self) -> int:
        count = 0
        for ticker in self.total_tickers:
            try:
                if get_position(self.trader, ticker) != 0:
                    count += 1
            except Exception:
                pass
        return count

    def can_open_new_position(self) -> bool:
        count = self.get_open_position_count()
        if count >= self.max_concurrent_positions:
            log(
                "RISK", "CONCURRENT",
                f"Max concurrent positions reached: "
                f"{count}/{self.max_concurrent_positions}"
            )
            return False
        return True

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
        return (self.initial_bp - get_buying_power(self.trader)) / self.initial_bp

    def get_available_bp(self) -> float:
        if self.initial_bp is None:
            return 0.0
        bp_floor = self.initial_bp * (1 - self.max_bp_usage)
        return max(0, get_buying_power(self.trader) - bp_floor)

    # ------------------------------------------------------------------ #
    #  Position size checks                                                #
    # ------------------------------------------------------------------ #
    def can_open_long(self, ticker: str) -> bool:
        position = get_position(self.trader, ticker)
        if position >= self.max_position_lots * 100:
            log("RISK", ticker, f"Max long reached: {position} shares")
            return False
        return True

    def can_open_short(self, ticker: str) -> bool:
        position = get_position(self.trader, ticker)
        if position <= -self.max_position_lots * 100:
            log("RISK", ticker, f"Max short reached: {position} shares")
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
                    f"Loss limit: ${total_pl:,.2f} < "
                    f"${self.max_loss_per_ticker:,.2f} — halting"
                )
                self.halted_tickers.add(ticker)
                return False
        except Exception as e:
            log("RISK", ticker, f"Error checking loss: {e}")
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
                    f"Total loss breached: ${total_pl:,.2f} — HALTING ALL"
                )
                self.all_halted = True
                return False
        except Exception as e:
            log("RISK", "PORTFOLIO", f"Error: {e}")
            return False
        return True

    # ------------------------------------------------------------------ #
    #  Master trade check                                                  #
    # ------------------------------------------------------------------ #
    def can_trade(self, ticker: str, is_buy: bool, price: float) -> bool:
        if price <= 0:
            return False
        if self.all_halted:
            return False
        if ticker in self.halted_tickers:
            return False
        if not self.check_total_loss():
            return False
        if not self.check_ticker_loss(ticker):
            return False
        if not self.can_open_new_position():
            return False
        if not self.has_buying_power(price * 100):
            return False
        if is_buy and not self.can_open_long(ticker):
            return False
        if not is_buy and not self.can_open_short(ticker):
            return False
        return True

    # ------------------------------------------------------------------ #
    #  Close check — bypasses halt                                         #
    # ------------------------------------------------------------------ #
    def can_close(self, ticker: str, price: float) -> bool:
        if price <= 0:
            return False
        if not self.has_buying_power(price * 100):
            return False
        return True

    # ------------------------------------------------------------------ #
    #  Status report                                                       #
    # ------------------------------------------------------------------ #
    def print_status(self):
        bp_usage   = self.get_bp_usage_pct()
        current_bp = get_buying_power(self.trader)
        available  = self.get_available_bp()
        open_pos   = self.get_open_position_count()

        try:
            total_pl = self.trader.get_portfolio_summary().get_total_realized_pl()
        except Exception:
            total_pl = 0.0

        print("\n" + "-"*50, flush=True)
        print(f"[RISK @ {datetime.now().strftime('%H:%M:%S')}]", flush=True)
        print(f"  Buying Power:      ${current_bp:,.2f} ({bp_usage*100:.1f}% used)", flush=True)
        print(f"  Available BP:      ${available:,.2f}", flush=True)
        print(f"  Total P&L:         ${total_pl:,.2f}", flush=True)
        print(f"  Open Positions:    {open_pos}/{self.max_concurrent_positions}", flush=True)
        print(f"  Max Total Loss:    ${self.max_total_loss:,.2f}", flush=True)
        print(f"  Halted:            {self.halted_tickers if self.halted_tickers else 'None'}", flush=True)
        print(f"  All Halted:        {self.all_halted}", flush=True)
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
        self.all_halted = True
        log("RISK", "ALL", "Emergency halt")

    def register_strategy(self, ticker: str, strategy_name: str):
        self.ticker_strategy_map[ticker] = strategy_name

    def get_strategy_for_ticker(self, ticker: str) -> str:
        return self.ticker_strategy_map.get(ticker, "UNKNOWN")