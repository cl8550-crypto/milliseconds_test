"""
run.py — Week 3: ILLIQUID

Theme: small-cap / micro-cap stocks with wide spreads and low liquidity.

Strategy: pure two-sided market making on dynamically-discovered tickers.

Why market making maximises Sharpe
-----------------------------------
Sharpe = mean(returns) / std(returns).  Market making earns
  spread + $0.004/share rebate
on every round trip.  That income is highly consistent (low variance) as
long as inventory is kept flat via quote skewing.  No directional bets →
no large drawdowns → high Sharpe.

Dynamic ticker selection
-------------------------
We call trader.get_stock_list() at startup, subscribe to every ticker,
sample bid-ask spreads for SPREAD_RANK_SECS seconds, then trade only the
top MM_MAX_TICKERS by average spread.  Wider spread = more edge per trade.

Sharpe tracker
---------------
A background thread samples total realized P&L every SHARPE_SAMPLE_SECS
seconds and reports the running annualised intraday Sharpe.  This is logged
in real time so we can see how the strategy is performing.
"""

import shift
import signal
import sys
import numpy as np
from time import sleep
from datetime import datetime, timedelta
from threading import Thread, Lock

from utils import (
    init_log_file, log,
    print_portfolio_summary,
    close_all_positions, cancel_all_orders,
    get_best_prices,
)
from risk_manager import RiskManager
from illiquid_market_maker_ting import IlliquidMarketMaker


# ------------------------------------------------------------------ #
#  Configuration                                                       #
# ------------------------------------------------------------------ #
USERNAME = "sub-millisecond"
PASSWORD = "XqklHgpv"
CONFIG   = "initiator.cfg"

# TEST_MODE = True  → replay mode
# TEST_MODE = False → competition (live)
TEST_MODE   = True
REPLAY_DATE = "2025-04-07"  # update to the Week 3 replay date when known

# ---- Market-maker parameters ---- #
MM_LOT_SIZE       = 1       # 1 lot = 100 shares (stay small in illiquid markets)
MM_MAX_POS_LOTS   = 2       # ±200 shares max per ticker
MM_MIN_SPREAD     = 0.06    # absolute floor — never quote below this
MM_VOL_WINDOW     = 20      # ticks used to compute per-tick volatility
MM_VOL_MULTIPLIER = 2.0     # spread must be >= vol_multiplier * per-tick-vol
MM_QUOTE_OFFSET   = 0.01    # improve best quote by 1 tick ($0.01)
MM_SKEW_TICKS     = 1       # 1-tick extra aggression when carrying inventory
MM_QUOTE_AGE      = 3.0     # cancel/repost after 3 seconds
MM_MOVE_PCT       = 0.003   # cancel/repost on 0.3% mid-price move
MM_WARMUP_TICKS   = 20      # price samples per ticker before first quote (raised for vol estimate)
MM_MAX_TICKERS    = 10      # trade at most 10 tickers simultaneously

# ---- Phase-switch controls ---- #
MIN_REQUIRED_TRADES   = 200   # competition minimum
TRADE_SAFETY_BUFFER   = 60    # don't switch exactly at the minimum

# Volume mode: prioritize safely clearing the trade-count hurdle
MM_VOLUME_MIN_SPREAD     = 0.03
MM_VOLUME_VOL_MULTIPLIER = 1.25
MM_VOLUME_QUOTE_AGE      = 1.0
MM_VOLUME_MOVE_PCT       = 0.006
MM_VOLUME_QUOTE_OFFSET   = 0.01

# ---- Session timing ---- #
WARMUP_MINUTES       = 3    # observe market before starting to quote
CLOSE_BEFORE_MINUTES = 15   # close all positions this many minutes early
                             # (illiquid EOD close is expensive — start early)
SPREAD_RANK_SECS     = 30   # seconds spent sampling spreads during warmup

# ---- Sharpe tracking ---- #
SHARPE_SAMPLE_SECS   = 30   # sample portfolio P&L every 30 seconds

# ---- Risk limits (conservative for illiquid stocks) ---- #
MAX_BP_USAGE         = 0.30   # never use more than 30% of buying power
MAX_LOSS_PER_TICKER  = -500.0
MAX_TOTAL_LOSS       = -5000.0
MAX_BP_PER_TRADE     = 5000.0  # small-cap / micro-cap stocks are cheap

# ------------------------------------------------------------------ #
#  Globals                                                             #
# ------------------------------------------------------------------ #
_trader_ref: shift.Trader | None = None
_risk_ref:   RiskManager  | None = None
_all_tickers: list[str]          = []


def handle_interrupt(sig, frame):
    print("\n[INTERRUPT] Ctrl+C — shutting down...", flush=True)
    if _trader_ref is not None:
        try:
            if _risk_ref:
                _risk_ref.halt_all()
            sleep(1)
            cancel_all_orders(_trader_ref)
            sleep(5)
            close_all_positions(_trader_ref, _all_tickers)
            sleep(25)
            _trader_ref.disconnect()
        except Exception as e:
            print(f"[INTERRUPT] Error: {e}", flush=True)
    sys.exit(0)


# ------------------------------------------------------------------ #
#  Dynamic ticker discovery                                            #
# ------------------------------------------------------------------ #
def discover_tickers(trader: shift.Trader) -> list[str]:
    """
    Ask SHIFT for all available tickers.
    Returns an empty list (and logs clearly) if the API is unavailable.
    """
    try:
        stock_list = trader.get_stock_list()
        if stock_list:
            tickers = sorted(str(t).upper() for t in stock_list)
            log("DISCOVER", "ALL",
                f"get_stock_list() returned {len(tickers)} tickers")
            return tickers
    except AttributeError:
        log("DISCOVER", "ALL",
            "get_stock_list() not available in this SHIFT build")
    except Exception as e:
        log("DISCOVER", "ALL", f"get_stock_list() error: {e}")
    return []


def rank_tickers_by_spread(
    trader: shift.Trader,
    tickers: list[str],
    sample_secs: float = 30.0,
) -> list[tuple[str, float]]:
    """
    Subscribe to all tickers, wait sample_secs, collect bid-ask spread
    samples, return a list of (ticker, avg_spread) sorted descending.
    Wider spread = more profit per round trip = prefer for market making.
    """
    log("DISCOVER", "ALL",
        f"Sampling spreads for {len(tickers)} tickers "
        f"over {sample_secs:.0f}s...")

    spread_sums:   dict[str, float] = {t: 0.0 for t in tickers}
    spread_counts: dict[str, int]   = {t: 0   for t in tickers}

    elapsed = 0.0
    while elapsed < sample_secs:
        for t in tickers:
            prices = get_best_prices(trader, t)
            if prices:
                spread_sums[t]   += prices[1] - prices[0]
                spread_counts[t] += 1
        sleep(1.0)
        elapsed += 1.0

    ranked: list[tuple[str, float]] = []
    for t in tickers:
        avg = (
            spread_sums[t] / spread_counts[t]
            if spread_counts[t] > 0 else 0.0
        )
        ranked.append((t, avg))
        log("DISCOVER", t,
            f"avg_spread=${avg:.4f} ({spread_counts[t]} samples)")

    ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked


# ------------------------------------------------------------------ #
#  Sharpe tracker                                                      #
# ------------------------------------------------------------------ #


class TradePhaseManager:
    """
    Shared state across all market-maker threads.

    We use submitted passive orders as a practical live proxy for trade-count
    progress.  Once the team safely clears the requirement buffer, all threads
    switch from volume mode to Sharpe-selective mode.
    """

    def __init__(self, min_required_orders: int, safety_buffer: int = 60):
        self.min_required_orders = int(min_required_orders)
        self.safety_buffer       = int(safety_buffer)
        self.switch_orders       = self.min_required_orders + self.safety_buffer
        self._orders_submitted   = 0
        self._phase              = "volume"
        self._lock               = Lock()

    def record_orders(self, count: int = 1):
        with self._lock:
            self._orders_submitted += count
            if self._phase == "volume" and self._orders_submitted >= self.switch_orders:
                self._phase = "sharpe"
                log(
                    "PHASE", "ALL",
                    f"Switching to SHARPE mode | submitted_orders={self._orders_submitted} | "
                    f"requirement={self.min_required_orders} | buffer={self.safety_buffer}"
                )

    def current_phase(self) -> str:
        with self._lock:
            return self._phase

    def total_orders(self) -> int:
        with self._lock:
            return self._orders_submitted


class SharpeTracker:
    """
    Samples realized P&L every sample_interval seconds and computes a
    running annualised intraday Sharpe ratio.

    Annualisation factor:
        √( 252 days/year × (390 min/day × 60s/min) / sample_interval )
    """

    def __init__(
        self,
        trader: shift.Trader,
        initial_pl: float,
        sample_interval: float = 30.0,
        starting_capital: float = 1_000_000.0,
    ):
        self.trader           = trader
        self.sample_interval  = sample_interval
        self.starting_capital = starting_capital
        self.pl_series:  list[float] = [initial_pl]
        self.returns:    list[float] = []

    def compute_sharpe(self) -> float | None:
        if len(self.returns) < 5:
            return None
        arr = np.array(self.returns)
        if arr.std() == 0:
            return None
        n_per_day  = (390 * 60) / self.sample_interval
        annualizer = np.sqrt(252 * n_per_day)
        return float(arr.mean() / arr.std() * annualizer)

    def run(self, end_time):
        log("SHARPE", "INIT",
            f"Sampling P&L every {self.sample_interval:.0f}s")
        while self.trader.get_last_trade_time() < end_time:
            sleep(self.sample_interval)
            try:
                pl    = (self.trader.get_portfolio_summary()
                         .get_total_realized_pl())
                delta = pl - self.pl_series[-1]
                ret   = delta / self.starting_capital
                self.pl_series.append(pl)
                self.returns.append(ret)
                sharpe = self.compute_sharpe()
                log(
                    "SHARPE", "ALL",
                    f"P&L=${pl:,.2f} | "
                    f"Δ=${delta:+,.2f} | "
                    f"Sharpe={'N/A' if sharpe is None else f'{sharpe:.3f}'} | "
                    f"n={len(self.returns)}"
                )
            except Exception as e:
                log("SHARPE", "ERR", str(e))


# ------------------------------------------------------------------ #
#  Thread runners                                                      #
# ------------------------------------------------------------------ #
def run_market_maker(
    trader: shift.Trader,
    risk_manager: RiskManager,
    ticker: str,
    end_time,
    phase_manager: TradePhaseManager,
):
    try:
        mm = IlliquidMarketMaker(
            trader            = trader,
            risk_manager      = risk_manager,
            ticker            = ticker,
            lot_size          = MM_LOT_SIZE,
            max_position_lots = MM_MAX_POS_LOTS,
            min_spread        = MM_MIN_SPREAD,
            vol_window        = MM_VOL_WINDOW,
            vol_multiplier    = MM_VOL_MULTIPLIER,
            quote_offset      = MM_QUOTE_OFFSET,
            skew_ticks        = MM_SKEW_TICKS,
            quote_max_age     = MM_QUOTE_AGE,
            price_move_pct    = MM_MOVE_PCT,
            check_freq        = 1.0,
            warmup_ticks      = MM_WARMUP_TICKS,
            phase_manager     = phase_manager,
            volume_min_spread = MM_VOLUME_MIN_SPREAD,
            volume_vol_multiplier = MM_VOLUME_VOL_MULTIPLIER,
            volume_quote_max_age  = MM_VOLUME_QUOTE_AGE,
            volume_price_move_pct = MM_VOLUME_MOVE_PCT,
            volume_quote_offset   = MM_VOLUME_QUOTE_OFFSET,
        )
        mm.run(end_time)
    except Exception as e:
        log("MM", ticker, f"Thread crashed: {e}")


def run_sharpe_tracker(
    trader: shift.Trader,
    initial_pl: float,
    end_time,
):
    tracker = SharpeTracker(
        trader          = trader,
        initial_pl      = initial_pl,
        sample_interval = SHARPE_SAMPLE_SECS,
    )
    try:
        tracker.run(end_time)
    except Exception as e:
        log("SHARPE", "ERR", f"Tracker crashed: {e}")
    # Final session Sharpe
    final_sharpe = tracker.compute_sharpe()
    log(
        "SHARPE", "FINAL",
        f"Session Sharpe = "
        f"{'N/A' if final_sharpe is None else f'{final_sharpe:.4f}'} | "
        f"n_samples={len(tracker.returns)}"
    )


def monitor_portfolio(
    trader: shift.Trader,
    risk_manager: RiskManager,
    end_time,
    tickers: list[str],
    interval: int = 30,
):
    while trader.get_last_trade_time() < end_time:
        try:
            print_portfolio_summary(trader, tickers)
            risk_manager.print_status()
        except Exception as e:
            log("MONITOR", "ALL", f"Error: {e}")
        sleep(interval)


# ------------------------------------------------------------------ #
#  Main                                                                #
# ------------------------------------------------------------------ #
def main():
    global _trader_ref, _risk_ref, _all_tickers

    signal.signal(signal.SIGINT, handle_interrupt)
    log_path = init_log_file()

    with shift.Trader(USERNAME) as trader:
        _trader_ref = trader

        trader.connect(CONFIG, PASSWORD)
        sleep(2)
        log("MAIN", "INIT", f"Connected:  {trader.is_connected()}")
        log("MAIN", "INIT", f"Log:        {log_path}")
        log("MAIN", "INIT",
            f"Mode:       "
            f"{'TEST (Replay)' if TEST_MODE else 'COMPETITION (Live)'}")
        log("MAIN", "INIT", f"Strategy:   IlliquidMarketMaker (Week 3)")

        # ---- Discover tickers ---- #
        tickers = discover_tickers(trader)
        if not tickers:
            log("MAIN", "INIT",
                "FATAL: No tickers found. "
                "Ensure Week 3 data is loaded on the SHIFT server and that "
                "trader.get_stock_list() is available.  Exiting.")
            return

        _all_tickers = tickers
        log("MAIN", "INIT",
            f"Discovered {len(tickers)} tickers: {tickers}")

        # Subscribe to all tickers so we can sample their order books
        log("MAIN", "INIT",
            f"Subscribing to {len(tickers)} order books...")
        for t in tickers:
            trader.sub_order_book(t)
            sleep(0.05)
        sleep(2)

        # ---- Timing ---- #
        current_time = trader.get_last_trade_time()
        market_open  = current_time.replace(
            hour=9, minute=30, second=0, microsecond=0
        )
        market_close = current_time.replace(
            hour=15, minute=50, second=0, microsecond=0
        )
        start_time   = market_open  + timedelta(minutes=WARMUP_MINUTES)
        end_time     = market_close - timedelta(minutes=CLOSE_BEFORE_MINUTES)

        log("MAIN", "INIT",
            f"Market:     {market_open.strftime('%H:%M')} — "
            f"{market_close.strftime('%H:%M')}")
        log("MAIN", "INIT",
            f"Trading:    {start_time.strftime('%H:%M:%S')} — "
            f"{end_time.strftime('%H:%M:%S')}  "
            f"({CLOSE_BEFORE_MINUTES} min early close)")

        # ---- Wait for market open if needed ---- #
        now = trader.get_last_trade_time()
        if now < market_open:
            log("MAIN", "WAIT",
                f"Waiting for market open "
                f"({market_open.strftime('%H:%M:%S')})...")
            while trader.get_last_trade_time() < market_open:
                sleep(5)

        # ---- Warmup: rank tickers by spread ---- #
        log("MAIN", "WARMUP",
            f"Warmup phase — sampling spreads for "
            f"{SPREAD_RANK_SECS}s then waiting until "
            f"{start_time.strftime('%H:%M:%S')}...")

        ranked = rank_tickers_by_spread(
            trader, tickers, sample_secs=SPREAD_RANK_SECS
        )

        # Filter to tickers with usable spreads
        qualified = [
            (t, sp) for t, sp in ranked if sp >= MM_MIN_SPREAD
        ]
        log("MAIN", "WARMUP",
            f"{len(qualified)}/{len(ranked)} tickers meet "
            f"spread threshold (${MM_MIN_SPREAD:.2f})")

        selected = [t for t, _ in qualified[:MM_MAX_TICKERS]]
        if not selected:
            # Fallback: take the top-N regardless of spread (trade something)
            log("MAIN", "WARMUP",
                "No tickers meet spread threshold — "
                "falling back to top-N by raw spread")
            selected = [t for t, _ in ranked[:MM_MAX_TICKERS]]

        if not selected:
            log("MAIN", "WARMUP", "FATAL: No tradeable tickers. Exiting.")
            return

        _all_tickers = selected
        log("MAIN", "WARMUP",
            f"Selected {len(selected)} tickers for market making:")
        for i, (t, sp) in enumerate(
            [(t, sp) for t, sp in ranked if t in selected]
        ):
            log("MAIN", "WARMUP",
                f"  [{i+1}] {t:<8} avg_spread=${sp:.4f}")

        # Unsubscribe unused tickers to save resources
        for t in tickers:
            if t not in selected:
                try:
                    trader.unsub_order_book(t)
                except Exception:
                    pass

        # Wait for start_time (spread sampling may finish early)
        while trader.get_last_trade_time() < start_time:
            sleep(2)

        # ---- Risk manager ---- #
        risk_manager = RiskManager(
            trader                   = trader,
            total_tickers            = selected,
            max_bp_usage             = MAX_BP_USAGE,
            max_position_lots        = MM_MAX_POS_LOTS,
            max_loss_per_ticker      = MAX_LOSS_PER_TICKER,
            max_total_loss           = MAX_TOTAL_LOSS,
            max_concurrent_positions = len(selected),
            max_bp_per_trade         = MAX_BP_PER_TRADE,
        )
        _risk_ref = risk_manager
        risk_manager.initialize()

        phase_manager = TradePhaseManager(
            min_required_orders = MIN_REQUIRED_TRADES,
            safety_buffer       = TRADE_SAFETY_BUFFER,
        )

        initial_pl = (
            trader.get_portfolio_summary().get_total_realized_pl()
        )
        initial_bp = trader.get_portfolio_summary().get_total_bp()

        log("MAIN", "START", "=" * 55)
        log("MAIN", "START", "WEEK 3 — ILLIQUID — MARKET MAKING STARTED")
        log("MAIN", "START", "=" * 55)
        log("MAIN", "START", f"Initial BP:  ${initial_bp:,.2f}")
        log("MAIN", "START", f"Initial P&L: ${initial_pl:,.2f}")
        log("MAIN", "START", f"Tickers ({len(selected)}): {selected}")
        log("MAIN", "START",
            f"Lot size:    {MM_LOT_SIZE} ({MM_LOT_SIZE * 100} shares)")
        log("MAIN", "START",
            f"Max pos:     ±{MM_MAX_POS_LOTS * 100} shares per ticker")
        log("MAIN", "START",
            f"Min spread:  ${MM_MIN_SPREAD:.2f}")
        log("MAIN", "START",
            f"Rebate edge: +${0.002 * 2:.3f}/share per round trip")
        log("MAIN", "START",
            f"Trade requirement proxy: {MIN_REQUIRED_TRADES} submitted passive orders")
        log("MAIN", "START",
            f"Phase switch: volume → sharpe at {MIN_REQUIRED_TRADES + TRADE_SAFETY_BUFFER} orders")

        # ---- Launch threads ---- #
        threads: list[Thread] = []

        for t in selected:
            threads.append(Thread(
                target = run_market_maker,
                args   = (trader, risk_manager, t, end_time, phase_manager),
                name   = f"MM_{t}",
                daemon = True,
            ))

        sharpe_thread = Thread(
            target = run_sharpe_tracker,
            args   = (trader, initial_pl, end_time),
            name   = "SHARPE",
            daemon = True,
        )
        monitor_thread = Thread(
            target = monitor_portfolio,
            args   = (trader, risk_manager, end_time, selected, 30),
            name   = "MONITOR",
            daemon = True,
        )

        log("MAIN", "START",
            f"Launching {len(threads)} MM threads + Sharpe tracker + monitor")

        for t in threads:
            t.start()
            sleep(0.3)
        sharpe_thread.start()
        monitor_thread.start()

        log("MAIN", "START", "All threads live — market making in progress")

        for t in threads:
            t.join()
        sharpe_thread.join(timeout=5)
        monitor_thread.join(timeout=2)

        # ---- Final cleanup ---- #
        log("MAIN", "END", "=" * 55)
        log("MAIN", "END", "ALL STRATEGIES DONE — FINAL CLEANUP")
        log("MAIN", "END", "=" * 55)

        cancel_all_orders(trader)
        sleep(5)
        close_all_positions(trader, selected)
        sleep(25)

        # ---- Final report ---- #
        final_pl     = (
            trader.get_portfolio_summary().get_total_realized_pl()
        )
        final_bp     = trader.get_portfolio_summary().get_total_bp()
        total_orders = phase_manager.total_orders()
        net_pl       = final_pl - initial_pl

        log("MAIN", "RESULT", "=" * 55)
        log("MAIN", "RESULT", f"Final P&L:     ${final_pl:,.2f}")
        log("MAIN", "RESULT", f"Net P&L:       ${net_pl:+,.2f}")
        log("MAIN", "RESULT", f"Final BP:      ${final_bp:,.2f}")
        log("MAIN", "RESULT", f"Total orders:  {total_orders}")
        log("MAIN", "RESULT",
            f"200-order min: "
            f"{'MET' if total_orders >= 200 else 'MISSED'}")
        log("MAIN", "RESULT", "=" * 55)

        log("MAIN", "RESULT", "Per-ticker breakdown:")
        for t in sorted(selected):
            try:
                item  = trader.get_portfolio_item(t)
                pl    = item.get_realized_pl()
                long  = item.get_long_shares()
                short = item.get_short_shares()
                log("MAIN", t,
                    f"P&L=${pl:,.2f} | long={long} short={short}")
            except Exception as e:
                log("MAIN", t, f"Error: {e}")


if __name__ == "__main__":
    main()
