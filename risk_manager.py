import shift
from time import time
from utils import log, get_position


class RiskManager:
    def __init__(
        self,
        trader: shift.Trader,
        total_tickers: list[str],
        max_bp_usage: float          = 0.60,
        max_position_lots: int       = 2,
        max_loss_per_ticker: float   = -2000.0,
        max_total_loss: float        = -20000.0,
        max_concurrent_positions: int = 12,
        max_bp_per_trade: float      = 65000.0,
    ):
        self.trader                   = trader
        self.total_tickers            = total_tickers
        self.max_bp_usage             = max_bp_usage
        self.max_position_lots        = max_position_lots
        self.max_loss_per_ticker      = max_loss_per_ticker
        self.max_total_loss           = max_total_loss
        self.max_concurrent_positions = max_concurrent_positions
        self.max_bp_per_trade         = max_bp_per_trade

        self.starting_bp   = 0.0
        self.halted_tickers: set[str] = set()
        self.all_halted    = False

    # ------------------------------------------------------------------ #
    #  Initialise                                                          #
    # ------------------------------------------------------------------ #
    def initialize(self):
        self.starting_bp = self.trader.get_portfolio_summary().get_total_bp()
        log("RISK", "INIT",
            f"Starting BP:               ${self.starting_bp:,.2f}")
        log("RISK", "INIT",
            f"Max BP usage:              {int(self.max_bp_usage*100)}%")
        log("RISK", "INIT",
            f"Max position per ticker:   {self.max_position_lots} lots")
        log("RISK", "INIT",
            f"Max loss per ticker:       ${self.max_loss_per_ticker:,.2f}")
        log("RISK", "INIT",
            f"Max total loss:            ${self.max_total_loss:,.2f}")
        log("RISK", "INIT",
            f"Max concurrent positions:  {self.max_concurrent_positions}")
        log("RISK", "INIT",
            f"Max BP per trade:          ${self.max_bp_per_trade:,.2f}")

    # ------------------------------------------------------------------ #
    #  Lot sizing                                                          #
    # ------------------------------------------------------------------ #
    def get_dynamic_order_size(self, price: float) -> int:
        """
        Week 3 illiquid stock sizing — more conservative than Week 1/2.

        Why smaller lots for illiquid stocks:
        - Bid/ask book depths are often just 1 lot on each side
        - Placing 3-lot orders on stocks with 1-lot books causes market impact
        - Smaller positions = more trades possible within BP limits
        - $7 BGS gets 1 lot ($700) not 3 lots ($2,100) to avoid being the whole market

        Tiers (based on Week 3 price observations):
          > $200  →  1 lot   SAM $324, WING $327, WDFC $270
          > $80   →  2 lots  CROX $110, CAR $100, SHAK $134, TXRH $192
          > $10   →  2 lots  HELE $73, COLM $90, JACK $48, PZZA $49,
                              SHOO $44, YETI $43, ENR $38
          ≤ $10   →  1 lot   BGS $7 — tiny stock, 1 lot avoids being
                              the entire bid side
        """
        if price > 200:
            return 1   # was 2 — SAM/WING/WDFC have thin books at $270-$327
        if price > 10:
            return 2   # was 3 — mid-range illiquid stocks
        return 1       # was 3 — micro-price stocks like BGS

    def get_safe_lot_size(self, price: float, requested_lots: int) -> int:
        """
        Cap requested lots by:
        1. max_position_lots global ceiling
        2. max_bp_per_trade cost ceiling
        Returns 0 if neither constraint can be satisfied.
        """
        if price <= 0:
            return 0

        lots = min(requested_lots, self.max_position_lots)

        # Reduce until cost fits within per-trade BP limit
        while lots > 0:
            cost = price * lots * 100  # lots × 100 shares
            if cost <= self.max_bp_per_trade:
                return lots
            lots -= 1

        return 0

    # ------------------------------------------------------------------ #
    #  Trade gate                                                          #
    # ------------------------------------------------------------------ #
    def can_trade(
        self, ticker: str, is_buy: bool, price: float
    ) -> bool:
        """
        Returns True only if ALL of the following pass:
        1. Global halt not active
        2. Ticker not individually halted
        3. Buying power floor not breached
        4. Per-ticker loss limit not breached
        5. Total loss limit not breached
        6. Concurrent position limit not breached
        """
        if self.all_halted:
            log("RISK", ticker, "Global halt active — rejecting trade")
            return False

        if ticker in self.halted_tickers:
            log("RISK", ticker, "Ticker halted — rejecting trade")
            return False

        # Check BP floor
        try:
            current_bp  = self.trader.get_portfolio_summary().get_total_bp()
            bp_floor    = self.starting_bp * (1 - self.max_bp_usage)
            trade_cost  = price * 100  # 1 lot minimum cost estimate
            if current_bp - trade_cost < bp_floor:
                log("RISK", "BP",
                    f"Insufficient BP: current=${current_bp:,.2f} | "
                    f"floor=${bp_floor:,.2f} | cost=${trade_cost:,.2f}")
                return False
        except Exception:
            pass

        # Check per-ticker P&L
        try:
            ticker_pl = self.trader.get_portfolio_item(ticker).get_realized_pl()
            if ticker_pl <= self.max_loss_per_ticker:
                log("RISK", ticker,
                    f"Ticker loss limit hit: ${ticker_pl:,.2f} — halting")
                self.halted_tickers.add(ticker)
                return False
        except Exception:
            pass

        # Check total P&L
        try:
            total_pl = self.trader.get_portfolio_summary().get_total_realized_pl()
            if total_pl <= self.max_total_loss:
                log("RISK", "ALL",
                    f"Total loss limit hit: ${total_pl:,.2f} — halting all")
                self.halt_all()
                return False
        except Exception:
            pass

        # Check concurrent positions
        try:
            open_count = self._count_open_positions()
            if open_count >= self.max_concurrent_positions:
                log("RISK", "CONCURRENT",
                    f"Max concurrent positions reached "
                    f"({open_count}/{self.max_concurrent_positions})")
                return False
        except Exception:
            pass

        return True

    def can_close(self, ticker: str, price: float) -> bool:
        """
        Closing positions is always allowed except on global halt.
        Individual ticker halts don't block closes — we want to exit.
        """
        if self.all_halted:
            return False
        return True

    def can_open_new_position(self) -> bool:
        return (
            not self.all_halted and
            self._count_open_positions() < self.max_concurrent_positions
        )

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #
    def _count_open_positions(self) -> int:
        count = 0
        for ticker in self.total_tickers:
            try:
                item  = self.trader.get_portfolio_item(ticker)
                long  = item.get_long_shares()
                short = item.get_short_shares()
                if long > 0 or short > 0:
                    count += 1
            except Exception:
                pass
        return count

    # ------------------------------------------------------------------ #
    #  Halt controls                                                       #
    # ------------------------------------------------------------------ #
    def halt_ticker(self, ticker: str):
        self.halted_tickers.add(ticker)
        log("RISK", ticker, "Ticker halted")

    def halt_all(self):
        self.all_halted = True
        log("RISK", "ALL", "Emergency halt — all strategies stopped")

    # ------------------------------------------------------------------ #
    #  Status reporting                                                    #
    # ------------------------------------------------------------------ #
    def print_status(self):
        try:
            summary    = self.trader.get_portfolio_summary()
            current_bp = summary.get_total_bp()
            total_pl   = summary.get_total_realized_pl()
            bp_used_pct = (
                (self.starting_bp - current_bp) / self.starting_bp * 100
                if self.starting_bp > 0 else 0
            )
            open_pos = self._count_open_positions()

            print("\n" + "-" * 50, flush=True)
            print(
                f"[RISK @ "
                f"{__import__('datetime').datetime.now().strftime('%H:%M:%S')}]",
                flush=True
            )
            print(
                f"  Buying Power:      ${current_bp:,.2f} "
                f"({bp_used_pct:.1f}% used)",
                flush=True
            )
            print(
                f"  Available BP:      "
                f"${current_bp - self.starting_bp*(1-self.max_bp_usage):,.2f}",
                flush=True
            )
            print(f"  Total P&L:         ${total_pl:,.2f}", flush=True)
            print(
                f"  Open Positions:    {open_pos}/{self.max_concurrent_positions}",
                flush=True
            )
            print(
                f"  Max Total Loss:    ${self.max_total_loss:,.2f}",
                flush=True
            )
            print(
                f"  Halted:            "
                f"{', '.join(self.halted_tickers) if self.halted_tickers else 'None'}",
                flush=True
            )
            print(f"  All Halted:        {self.all_halted}", flush=True)
            print("-" * 50 + "\n", flush=True)

        except Exception as e:
            log("RISK", "STATUS", f"Error: {e}")