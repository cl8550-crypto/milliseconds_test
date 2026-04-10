import shift
import signal
import sys
from time import sleep
from datetime import timedelta
from threading import Thread

from utils import (
    init_log_file, log,
    print_portfolio_summary,
    close_all_positions, cancel_all_orders
)
from risk_manager import RiskManager
from market_maker_as import AvellanedaStoikov


# ================================================================== #
#  CREDENTIALS — fill these in before competition day                #
# ================================================================== #
USERNAME = "sub-millisecond"   # ← replace
PASSWORD = "XqkIHgpv"   # ← replace
CONFIG   = "initiator.cfg"

# ================================================================== #
#  Week 3 tickers (all 15 illiquid names)                           #
# ================================================================== #
AS_TICKERS = [
    "BGS", "CAR", "COLM", "CROX", "ENR",
    "HELE", "JACK", "PZZA", "SAM", "SHAK",
    "SHOO", "TXRH", "WDFC", "WING", "YETI",
]
ALL_TICKERS = AS_TICKERS

# ================================================================== #
#  Timing                                                            #
# ================================================================== #
WARMUP_MINUTES       = 2    # wait after market open before quoting
CLOSE_BEFORE_MINUTES = 10   # stop quoting and flatten this many mins early

# ================================================================== #
#  Avellaneda-Stoikov parameters                                     #
#                                                                    #
#  gamma = 0.5  — risk aversion / inventory skew strength           #
#    At inv=+3, HELE spread=0.9, T-t=0.8:                          #
#    ask drops by 3 × 0.5 × 0.9 × 0.8 = $1.08 below mid           #
#    Forces buyers to cross our ask and clear inventory              #
#    (gamma=0.1 only moved it $0.22 — not enough on trending days)  #
#                                                                    #
#  max_inventory = 3  — tighter risk, forces faster round trips     #
#    Quotes go single-sided when inventory hits ±3 lots             #
#                                                                    #
#  quote_refresh = 3.0s  — faster requoting with real participants  #
#                                                                    #
#  target_spread = 70%  — post deeper inside the spread than        #
#    other MM teams, ensuring we are always the best bid/ask        #
# ================================================================== #
GAMMA          = 0.5
MAX_INVENTORY  = 3
QUOTE_REFRESH  = 3.0
MIN_SPREAD_PCT = 0.002   # 0.2% floor — keeps BGS spread sensible

_trader_ref = None
_risk_ref   = None


def handle_interrupt(sig, frame):
    print("\n[INTERRUPT] Ctrl+C — shutting down cleanly...", flush=True)
    if _trader_ref is not None:
        try:
            if _risk_ref is not None:
                _risk_ref.halt_all()
            sleep(1)
            cancel_all_orders(_trader_ref)
            sleep(5)
            close_all_positions(_trader_ref, ALL_TICKERS)
            sleep(25)
            _trader_ref.disconnect()
        except Exception as e:
            print(f"[INTERRUPT] Error during shutdown: {e}", flush=True)
    sys.exit(0)


# ------------------------------------------------------------------ #
#  Thread runner — one per ticker                                    #
# ------------------------------------------------------------------ #
def run_as(trader, rm, ticker, session_start, session_end):
    try:
        AvellanedaStoikov(
            trader         = trader,
            risk_manager   = rm,
            ticker         = ticker,
            session_start  = session_start,
            session_end    = session_end,
            gamma          = GAMMA,
            max_inventory  = MAX_INVENTORY,
            vol_window     = 30,
            quote_refresh  = QUOTE_REFRESH,
            min_spread_pct = MIN_SPREAD_PCT,
            check_freq     = 1.0,
        ).run(session_end)
    except Exception as e:
        log("AS", ticker, f"Thread crashed: {e}")


def monitor_portfolio(trader, rm, end_time, interval=30):
    while trader.get_last_trade_time() < end_time:
        try:
            print_portfolio_summary(trader, ALL_TICKERS)
            rm.print_status()
        except Exception as e:
            log("MONITOR", "ALL", f"Error: {e}")
        sleep(interval)


# ------------------------------------------------------------------ #
#  Main                                                              #
# ------------------------------------------------------------------ #
def main():
    global _trader_ref, _risk_ref
    signal.signal(signal.SIGINT, handle_interrupt)
    log_path = init_log_file()

    with shift.Trader(USERNAME) as trader:
        _trader_ref = trader
        trader.connect(CONFIG, PASSWORD)
        sleep(2)
        log("MAIN", "INIT", f"Connected: {trader.is_connected()}")
        log("MAIN", "INIT", f"Log file:  {log_path}")

        # Subscribe all tickers
        log("MAIN", "INIT", f"Subscribing to {len(ALL_TICKERS)} tickers...")
        for ticker in ALL_TICKERS:
            trader.sub_order_book(ticker)
            sleep(0.1)
        log("MAIN", "INIT", f"Subscribed: {sorted(ALL_TICKERS)}")
        sleep(2)

        # Wait for market clock (competition admin starts the replay)
        log("MAIN", "WAIT",
            "Waiting for market clock (9:00-15:50)... "
            "→ Start replay in admin panel now.")
        while True:
            t = trader.get_last_trade_time()
            if 9 <= t.hour < 16:
                log("MAIN", "WAIT", f"Market clock active — {t.strftime('%H:%M:%S')}")
                break
            log("MAIN", "WAIT",
                f"Market time = {t.strftime('%H:%M:%S')} — waiting...")
            sleep(2)

        # Compute session window
        current_time = trader.get_last_trade_time()
        market_open  = current_time.replace(
            hour=9, minute=30, second=0, microsecond=0)
        market_close = current_time.replace(
            hour=15, minute=50, second=0, microsecond=0)
        start_time   = market_open  + timedelta(minutes=WARMUP_MINUTES)
        end_time     = market_close - timedelta(minutes=CLOSE_BEFORE_MINUTES)

        if current_time >= end_time:
            log("MAIN", "ERROR",
                "Already past end_time — restart replay and re-run.")
            trader.disconnect()
            sys.exit(1)

        log("MAIN", "INIT",
            f"Trading window: {start_time.strftime('%H:%M:%S')} — "
            f"{end_time.strftime('%H:%M:%S')}")
        log("MAIN", "INIT",
            f"A-S params: γ={GAMMA} | max_inv=±{MAX_INVENTORY} | "
            f"refresh={QUOTE_REFRESH}s | target_spread=70%")

        # Risk manager
        rm = RiskManager(
            trader                   = trader,
            total_tickers            = ALL_TICKERS,
            max_bp_usage             = 0.60,
            max_position_lots        = MAX_INVENTORY,
            max_loss_per_ticker      = -2000.0,
            max_total_loss           = -20000.0,
            max_concurrent_positions = 15,
            max_bp_per_trade         = 65000.0,
        )
        _risk_ref = rm
        rm.initialize()

        # Wait for market open
        if trader.get_last_trade_time() < market_open:
            log("MAIN", "WAIT",
                f"Waiting for market open "
                f"{market_open.strftime('%H:%M:%S')}...")
            while trader.get_last_trade_time() < market_open:
                sleep(2)

        # Warmup period
        log("MAIN", "WARMUP",
            f"Warming up until {start_time.strftime('%H:%M:%S')}...")
        while trader.get_last_trade_time() < start_time:
            sleep(2)

        log("MAIN", "START", "=" * 55)
        log("MAIN", "START", "WEEK 3 — AVELLANEDA-STOIKOV MM — STARTED")
        log("MAIN", "START", "=" * 55)

        initial_pl = trader.get_portfolio_summary().get_total_realized_pl()
        initial_bp = trader.get_portfolio_summary().get_total_bp()
        log("MAIN", "START", f"Initial BP:  ${initial_bp:,.2f}")
        log("MAIN", "START", f"Initial P&L: ${initial_pl:,.2f}")

        # Launch one A-S thread per ticker
        threads = []
        for ticker in AS_TICKERS:
            threads.append(Thread(
                target  = run_as,
                args    = (trader, rm, ticker, start_time, end_time),
                name    = f"AS_{ticker}",
                daemon  = True,
            ))

        monitor_thread = Thread(
            target = monitor_portfolio,
            args   = (trader, rm, end_time, 30),
            name   = "MONITOR",
            daemon = True,
        )

        log("MAIN", "START",
            f"Launching {len(threads)} threads — one per ticker")
        for t in threads:
            t.start()
            sleep(0.3)
        monitor_thread.start()
        log("MAIN", "START", "All threads live — market making active")

        # Wait for all threads to finish
        for t in threads:
            t.join()
        monitor_thread.join(timeout=2)

        # ---- Final cleanup ---- #
        log("MAIN", "END", "=" * 55)
        log("MAIN", "END", "ALL DONE — FINAL CLEANUP")
        log("MAIN", "END", "=" * 55)

        cancel_all_orders(trader)
        sleep(5)
        close_all_positions(trader, ALL_TICKERS)
        sleep(25)

        final_pl     = trader.get_portfolio_summary().get_total_realized_pl()
        final_bp     = trader.get_portfolio_summary().get_total_bp()
        total_trades = trader.get_submitted_orders_size()

        log("MAIN", "RESULT", "=" * 55)
        log("MAIN", "RESULT", f"Final BP:     ${final_bp:,.2f}")
        log("MAIN", "RESULT", f"Net P&L:      ${final_pl - initial_pl:,.2f}")
        log("MAIN", "RESULT", f"Total trades: {total_trades}")
        log("MAIN", "RESULT",
            f"Eligible:     "
            f"{'✓ YES (≥200)' if total_trades >= 200 else f'✗ NO ({total_trades}/200)'}")
        log("MAIN", "RESULT", "=" * 55)

        for ticker in sorted(ALL_TICKERS):
            try:
                item = trader.get_portfolio_item(ticker)
                log("MAIN", ticker,
                    f"P&L: ${item.get_realized_pl():,.2f} | "
                    f"long: {item.get_long_shares()} | "
                    f"short: {item.get_short_shares()}")
            except Exception as e:
                log("MAIN", ticker, f"Error: {e}")


if __name__ == "__main__":
    main()