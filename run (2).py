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
from mean_reversion_zi import MeanReversionZI


# ================================================================== #
#  CREDENTIALS                                                       #
# ================================================================== #
USERNAME = "sub-millisecond"
PASSWORD = "XqkIHgpv"
CONFIG   = "initiator.cfg"

ALL_TICKERS = ["CS1", "CS2", "CS3"]

WARMUP_MINUTES       = 2
CLOSE_BEFORE_MINUTES = 10

# ================================================================== #
#  ⚠️  TEST RUN — lower threshold to validate signal fires          #
#  Before competition day, change ENTRY = 0.0025                    #
# ================================================================== #
ENTRY         = 0.0025  # 0.25% — competition day threshold
HOLD          = 5       # exit faster — reversion is quick
STOP          = 0.003   # 0.30% stop loss
TAKE_PROFIT   = 0.0008  # 0.08% take profit
LOTS          = 10      # 10 lots per trade
COOLDOWN      = 8.0     # 8s between trades per ticker

# ================================================================== #
#  Timing                                                            #
# ================================================================== #
LOOKBACK = 5

_trader_ref = None
_risk_ref   = None


def handle_interrupt(sig, frame):
    print("\n[INTERRUPT] shutting down...", flush=True)
    if _trader_ref:
        try:
            if _risk_ref:
                _risk_ref.halt_all()
            sleep(1)
            cancel_all_orders(_trader_ref)
            sleep(3)
            close_all_positions(_trader_ref, ALL_TICKERS)
            sleep(25)
            _trader_ref.disconnect()
        except Exception as e:
            print(f"[INTERRUPT] {e}", flush=True)
    sys.exit(0)


def run_mr(trader, rm, ticker, end_time):
    try:
        MeanReversionZI(
            trader           = trader,
            risk_manager     = rm,
            ticker           = ticker,
            lookback         = LOOKBACK,
            entry_threshold  = ENTRY,
            hold_ticks       = HOLD,
            stop_pct         = STOP,
            take_profit_pct  = TAKE_PROFIT,
            lots             = LOTS,
            cooldown         = COOLDOWN,
            check_freq       = 1.0,
        ).run(end_time)
    except Exception as e:
        log("MR", ticker, f"Thread crashed: {e}")


def monitor(trader, rm, end_time):
    while trader.get_last_trade_time() < end_time:
        try:
            print_portfolio_summary(trader, ALL_TICKERS)
            rm.print_status()
        except Exception as e:
            log("MONITOR", "ALL", f"Error: {e}")
        sleep(30)


def main():
    global _trader_ref, _risk_ref
    signal.signal(signal.SIGINT, handle_interrupt)
    log_path = init_log_file()

    with shift.Trader(USERNAME) as trader:
        _trader_ref = trader
        trader.connect(CONFIG, PASSWORD)
        sleep(2)
        log("MAIN", "INIT", f"Connected: {trader.is_connected()}")
        log("MAIN", "INIT", f"Log: {log_path}")
        log("MAIN", "INIT", "Week 4 — ZI Mean Reversion — CS1/CS2/CS3")
        log("MAIN", "INIT",
            f"MODE: entry={ENTRY*100:.2f}% | lots={LOTS} | hold={HOLD}t | cooldown={COOLDOWN}s")

        for t in ALL_TICKERS:
            trader.sub_order_book(t)
            sleep(0.1)
        log("MAIN", "INIT", f"Subscribed: {ALL_TICKERS}")
        sleep(2)

        log("MAIN", "WAIT", "Waiting for market clock...")
        while True:
            t = trader.get_last_trade_time()
            if 9 <= t.hour < 16:
                log("MAIN", "WAIT", f"Active — {t.strftime('%H:%M:%S')}")
                break
            sleep(2)

        ct           = trader.get_last_trade_time()
        market_open  = ct.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = ct.replace(hour=15, minute=50, second=0, microsecond=0)
        start_time   = market_open  + timedelta(minutes=WARMUP_MINUTES)
        end_time     = market_close - timedelta(minutes=CLOSE_BEFORE_MINUTES)

        if ct >= end_time:
            log("MAIN", "ERROR", "Past end_time — restart.")
            trader.disconnect()
            sys.exit(1)

        log("MAIN", "INIT",
            f"Window: {start_time.strftime('%H:%M:%S')} — "
            f"{end_time.strftime('%H:%M:%S')}")
        log("MAIN", "INIT",
            f"entry={ENTRY*100:.2f}% | hold={HOLD}t | "
            f"stop={STOP*100:.2f}% | tp={TAKE_PROFIT*100:.2f}% | "
            f"lots={LOTS} | cooldown={COOLDOWN}s")

        rm = RiskManager(
            trader                   = trader,
            total_tickers            = ALL_TICKERS,
            max_bp_usage             = 0.60,
            max_position_lots        = LOTS,
            max_loss_per_ticker      = -4000.0,
            max_total_loss           = -20000.0,
            max_concurrent_positions = 9,   # 3 tickers × 3 re-entries each — not the binding constraint
            max_bp_per_trade         = 120000.0,
        )
        _risk_ref = rm
        rm.initialize()

        if trader.get_last_trade_time() < market_open:
            log("MAIN", "WAIT", "Waiting for open...")
            while trader.get_last_trade_time() < market_open:
                sleep(2)

        log("MAIN", "WARMUP",
            f"Warming up until {start_time.strftime('%H:%M:%S')}...")
        while trader.get_last_trade_time() < start_time:
            sleep(2)

        log("MAIN", "START", "=" * 55)
        log("MAIN", "START", "WEEK 4 — ZI MEAN REVERSION — STARTED")
        log("MAIN", "START", "=" * 55)

        initial_pl = trader.get_portfolio_summary().get_total_realized_pl()
        initial_bp = trader.get_portfolio_summary().get_total_bp()
        log("MAIN", "START", f"Initial BP:  ${initial_bp:,.2f}")
        log("MAIN", "START", f"Initial P&L: ${initial_pl:,.2f}")

        threads = [
            Thread(target=run_mr,
                   args=(trader, rm, t, end_time),
                   name=f"MR_{t}", daemon=True)
            for t in ALL_TICKERS
        ]
        monitor_t = Thread(
            target=monitor, args=(trader, rm, end_time),
            name="MONITOR", daemon=True
        )

        log("MAIN", "START", "Launching 3× MeanReversionZI")
        for t in threads:
            t.start()
            sleep(0.5)
        monitor_t.start()
        log("MAIN", "START", "All threads live")

        for t in threads:
            t.join()
        monitor_t.join(timeout=2)

        log("MAIN", "END", "=" * 55)
        log("MAIN", "END", "DONE — FINAL CLEANUP")
        log("MAIN", "END", "=" * 55)

        cancel_all_orders(trader)
        sleep(3)
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
            f"Eligible: "
            f"{'✓ YES' if total_trades >= 200 else f'✗ NO ({total_trades}/200)'}")
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
