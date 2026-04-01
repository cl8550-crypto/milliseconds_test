import shift
import signal
import sys
from time import sleep
from datetime import datetime, timedelta
from threading import Thread

from utils import (
    log, print_portfolio_summary,
    close_all_positions, cancel_all_orders
)
from risk_manager import RiskManager
from mean_reversion import MeanReversionStrategy
from pairs_strategy import PairsStrategy
from order_book_arb import OrderBookArbStrategy


# ------------------------------------------------------------------ #
#  Configuration                                                       #
# ------------------------------------------------------------------ #
USERNAME = "sub-millisecond"
PASSWORD = "XqkIHgpvD"
CONFIG   = "initiator.cfg"

# ---- Mode ----
TEST_MODE             = False
TEST_DURATION_MINUTES = 30

# ---- Pairs ----
PAIRS = [
    ("JPM",  "GS"),
    ("NVDA", "AMZN"),
    ("MRK",  "JNJ"),
    ("WMT",  "PG"),
    ("CSCO", "IBM"),
]

# ---- Mean reversion tickers ----
MR_TICKERS = ["BA", "CAT", "DIS", "MMM"]

# ---- Order book arb tickers ----
ARB_TICKERS = ["AAPL", "MSFT", "NVDA", "JPM", "GS"]

# ---- All tickers combined ----
ALL_TICKERS = list(
    {t for pair in PAIRS for t in pair}
    | set(MR_TICKERS)
    | set(ARB_TICKERS)
)

# ---- Timing ----
WARMUP_MINUTES       = 5
CLOSE_BEFORE_MINUTES = 10


# ------------------------------------------------------------------ #
#  Global references for signal handler                                #
# ------------------------------------------------------------------ #
_trader_ref = None
_risk_ref   = None


def handle_interrupt(sig, frame):
    print(
        "\n[INTERRUPT] Ctrl+C detected — shutting down gracefully...",
        flush=True
    )
    if _trader_ref is not None:
        try:
            if _risk_ref is not None:
                _risk_ref.halt_all()
            sleep(1)
            cancel_all_orders(_trader_ref)
            sleep(5)
            close_all_positions(_trader_ref, ALL_TICKERS)
            sleep(15)   # ← increased from 5
            _trader_ref.disconnect()
        except Exception as e:
            print(f"[INTERRUPT] Error during cleanup: {e}", flush=True)
    sys.exit(0)


# ------------------------------------------------------------------ #
#  Thread runners                                                      #
# ------------------------------------------------------------------ #
def run_mean_reversion(trader, risk_manager, ticker, end_time):
    try:
        strategy = MeanReversionStrategy(
            trader=trader,
            risk_manager=risk_manager,
            ticker=ticker,
            order_size=3,        # ← reduced from 5
            window=20,
            entry_threshold=1.5,
            exit_threshold=0.5,
            check_freq=1.0,
            cooldown=10.0,
            max_history=100,
        )
        strategy.run(end_time)
    except Exception as e:
        log("MR", ticker, f"Thread crashed: {e}")


def run_pairs(trader, risk_manager, ticker1, ticker2, end_time):
    try:
        strategy = PairsStrategy(
            trader=trader,
            risk_manager=risk_manager,
            ticker1=ticker1,
            ticker2=ticker2,
            order_size=2,        # ← reduced from 3
            window=30,
            entry_threshold=2.0,
            exit_threshold=0.5,
            check_freq=1.0,
            cooldown=15.0,
            max_history=100,
        )
        strategy.run(end_time)
    except Exception as e:
        log("PAIRS", f"{ticker1}/{ticker2}", f"Thread crashed: {e}")


def run_order_book_arb(trader, risk_manager, ticker, end_time):
    try:
        strategy = OrderBookArbStrategy(
            trader=trader,
            risk_manager=risk_manager,
            ticker=ticker,
            order_size=2,
            min_arb_spread=0.02,
            min_pressure_gap=0.05,
            check_freq=1.0,
            cooldown=5.0,
        )
        strategy.run(end_time)
    except Exception as e:
        log("ARB", ticker, f"Thread crashed: {e}")


# ------------------------------------------------------------------ #
#  Portfolio monitor                                                   #
# ------------------------------------------------------------------ #
def monitor_portfolio(trader, risk_manager, end_time, interval=30):
    while trader.get_last_trade_time() < end_time:
        try:
            print_portfolio_summary(trader, ALL_TICKERS)
            risk_manager.print_status()
        except Exception as e:
            log("MONITOR", "ALL", f"Monitor error: {e}")
        sleep(interval)


# ------------------------------------------------------------------ #
#  Main                                                                #
# ------------------------------------------------------------------ #
def main():
    global _trader_ref, _risk_ref

    signal.signal(signal.SIGINT, handle_interrupt)

    with shift.Trader(USERNAME) as trader:
        _trader_ref = trader

        # Connect
        trader.connect(CONFIG, PASSWORD)
        sleep(2)
        log("MAIN", "INIT", f"Connected: {trader.is_connected()}")

        # Subscribe only to tickers we trade
        log("MAIN", "INIT", f"Subscribing to {len(ALL_TICKERS)} tickers...")
        for ticker in ALL_TICKERS:
            trader.sub_order_book(ticker)
            sleep(0.1)
        log("MAIN", "INIT", f"Subscribed: {sorted(ALL_TICKERS)}")
        sleep(2)

        # Get timing
        current_time = trader.get_last_trade_time()

        if TEST_MODE:
            log("MAIN", "INIT", f"TEST MODE — {TEST_DURATION_MINUTES} min session")
            start_time = current_time + timedelta(minutes=WARMUP_MINUTES)
            end_time   = current_time + timedelta(
                minutes=WARMUP_MINUTES + TEST_DURATION_MINUTES
            )
        else:
            market_open  = current_time.replace(hour=9,  minute=30, second=0)
            market_close = current_time.replace(hour=15, minute=50, second=0)
            start_time   = market_open  + timedelta(minutes=WARMUP_MINUTES)
            end_time     = market_close - timedelta(minutes=CLOSE_BEFORE_MINUTES)

        log("MAIN", "INIT", f"Mode:           {'TEST' if TEST_MODE else 'FULL'}")
        log("MAIN", "INIT", f"Strategy start: {start_time.strftime('%H:%M:%S')}")
        log("MAIN", "INIT", f"Strategy end:   {end_time.strftime('%H:%M:%S')}")

        # Initialize risk manager
        risk_manager = RiskManager(
            trader=trader,
            total_tickers=ALL_TICKERS,
            max_bp_usage=0.80,
            max_position_lots=5,
            max_loss_per_ticker=-2000.0,
            max_total_loss=-20000.0,
        )
        _risk_ref = risk_manager
        risk_manager.initialize()

        # Wait for market open (full session only)
        if not TEST_MODE:
            market_open = current_time.replace(hour=9, minute=30, second=0)
            while trader.get_last_trade_time() < market_open:
                log("MAIN", "WAIT", "Waiting for market open...")
                sleep(5)

        # Warmup
        log("MAIN", "WARMUP", f"Warming up for {WARMUP_MINUTES} minutes...")
        while trader.get_last_trade_time() < start_time:
            log("MAIN", "WARMUP", "Collecting price history...")
            sleep(10)

        log("MAIN", "START", "="*50)
        log("MAIN", "START", "TRADING STARTED")
        log("MAIN", "START", "="*50)

        # Record initial state
        initial_pl = trader.get_portfolio_summary().get_total_realized_pl()
        initial_bp = trader.get_portfolio_summary().get_total_bp()
        log("MAIN", "START", f"Initial BP:    ${initial_bp:,.2f}")
        log("MAIN", "START", f"Initial P&L:   ${initial_pl:,.2f}")

        # ---- Launch all threads ---- #
        threads = []

        for ticker1, ticker2 in PAIRS:
            t = Thread(
                target=run_pairs,
                args=(trader, risk_manager, ticker1, ticker2, end_time),
                name=f"PAIRS_{ticker1}_{ticker2}",
                daemon=True
            )
            threads.append(t)

        for ticker in MR_TICKERS:
            t = Thread(
                target=run_mean_reversion,
                args=(trader, risk_manager, ticker, end_time),
                name=f"MR_{ticker}",
                daemon=True
            )
            threads.append(t)

        for ticker in ARB_TICKERS:
            t = Thread(
                target=run_order_book_arb,
                args=(trader, risk_manager, ticker, end_time),
                name=f"ARB_{ticker}",
                daemon=True
            )
            threads.append(t)

        monitor_thread = Thread(
            target=monitor_portfolio,
            args=(trader, risk_manager, end_time, 30),
            name="MONITOR",
            daemon=True
        )

        log("MAIN", "START", f"Launching {len(threads)} threads...")
        log("MAIN", "START", f"  {len(PAIRS)} pairs strategies")
        log("MAIN", "START", f"  {len(MR_TICKERS)} mean reversion strategies")
        log("MAIN", "START", f"  {len(ARB_TICKERS)} order book arb strategies")

        for t in threads:
            t.start()
            sleep(0.5)
        monitor_thread.start()
        log("MAIN", "START", "All threads launched")

        for t in threads:
            t.join()
        monitor_thread.join(timeout=2)

        # ---- Final cleanup ---- #
        log("MAIN", "END", "="*50)
        log("MAIN", "END", "ALL STRATEGIES DONE — FINAL CLEANUP")
        log("MAIN", "END", "="*50)

        cancel_all_orders(trader)
        sleep(5)
        close_all_positions(trader, ALL_TICKERS)
        sleep(15)   # ← increased from 5 — gives market orders time to fill

        # ---- Final report ---- #
        final_pl     = trader.get_portfolio_summary().get_total_realized_pl()
        final_bp     = trader.get_portfolio_summary().get_total_bp()
        net_pl       = final_pl - initial_pl
        total_trades = trader.get_submitted_orders_size()

        log("MAIN", "RESULT", "="*50)
        log("MAIN", "RESULT", f"Final BP:        ${final_bp:,.2f}")
        log("MAIN", "RESULT", f"Net P&L:         ${net_pl:,.2f}")
        log("MAIN", "RESULT", f"Total trades:    {total_trades}")
        log("MAIN", "RESULT", f"Target trades:   200")
        log("MAIN", "RESULT", f"Target met:      {'✓ YES' if total_trades >= 200 else '✗ NO'}")
        log("MAIN", "RESULT", "="*50)

        print("\nPer-ticker breakdown:", flush=True)
        for ticker in sorted(ALL_TICKERS):
            try:
                item  = trader.get_portfolio_item(ticker)
                pl    = item.get_realized_pl()
                long  = item.get_long_shares()
                short = item.get_short_shares()
                pos   = long - short
                log(
                    "MAIN", ticker,
                    f"P&L: ${pl:,.2f} | "
                    f"pos: {pos} | "
                    f"long: {long} | "
                    f"short: {short}"
                )
            except Exception as e:
                log("MAIN", ticker, f"Error: {e}")


if __name__ == "__main__":
    main()