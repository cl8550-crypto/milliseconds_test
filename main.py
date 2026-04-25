"""
Week 5 — Production Market Maker (MM ONLY)
Per-ticker parameter tuning. No momentum layer.

Verified strategy:
  Diagnostic run: $111,788 P&L in ~70 min (CS1 $0.05, CS2 $0.05)
  Bug confirmed: momentum layer caused $128k drawdown when RL spikes
                 triggered false trend signals at price inflections.
  This version: keep the proven MM, drop momentum entirely.
                CS2 spread bumped to $0.10 (proportional to higher price).
"""
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
from market_maker import MarketMaker


USERNAME = "sub-millisecond"
PASSWORD = "XqkIHgpv"
CONFIG   = "initiator.cfg"

ALL_TICKERS = ["CS1", "CS2"]

WARMUP_MINUTES       = 2
CLOSE_BEFORE_MINUTES = 10

# ================================================================== #
#  Per-ticker MM parameters                                          #
#                                                                    #
#  Same proportional spread (0.05% half-spread on each):            #
#    CS1 (~$100):  $0.05 half-spread                                #
#    CS2 (~$200):  $0.10 half-spread                                #
#                                                                    #
#  Same lot size and inventory cap on both.                         #
# ================================================================== #
MM_CONFIG = {
    "CS1": {"lots": 2, "half_spread": 0.05},
    "CS2": {"lots": 2, "half_spread": 0.10},
}
MM_MAX_INVENTORY  = 6
MM_QUOTE_REFRESH  = 3.0


# Global refs for Ctrl+C handler
_trader_ref         = None
_risk_ref           = None
_strategy_instances = []


def handle_interrupt(sig, frame):
    print("\n" + "="*60, flush=True)
    print("[INTERRUPT] Ctrl+C — squaring off all positions", flush=True)
    print("="*60, flush=True)

    for s in _strategy_instances:
        try:
            s.shutdown()
        except Exception:
            pass

    if _risk_ref:
        try:
            _risk_ref.halt_all()
        except Exception:
            pass

    sleep(1)

    if _trader_ref:
        try:
            print("[INTERRUPT] Cancelling pending orders...", flush=True)
            cancel_all_orders(_trader_ref)
            sleep(3)

            print("[INTERRUPT] Closing positions...", flush=True)
            close_all_positions(_trader_ref, ALL_TICKERS)
            sleep(5)

            print("[INTERRUPT] Verifying flat...", flush=True)
            waited   = 0
            all_flat = False
            while waited < 20:
                all_flat = True
                for t in ALL_TICKERS:
                    try:
                        item = _trader_ref.get_portfolio_item(t)
                        pos = item.get_long_shares() - item.get_short_shares()
                        if pos != 0:
                            all_flat = False
                            print(f"  {t}: pos={pos}", flush=True)
                    except Exception:
                        pass
                if all_flat:
                    print("[INTERRUPT] All flat ✓", flush=True)
                    break
                sleep(2)
                waited += 2

            if not all_flat:
                print("[INTERRUPT] Retry close orders", flush=True)
                close_all_positions(_trader_ref, ALL_TICKERS)
                sleep(10)

            try:
                summary = _trader_ref.get_portfolio_summary()
                print(f"[INTERRUPT] Final BP:  ${summary.get_total_bp():,.2f}", flush=True)
                print(f"[INTERRUPT] Realized: ${summary.get_total_realized_pl():,.2f}", flush=True)
            except Exception:
                pass

            sleep(2)
            _trader_ref.disconnect()
            print("[INTERRUPT] Disconnected", flush=True)
        except Exception as e:
            print(f"[INTERRUPT] Error: {e}", flush=True)

    sys.exit(0)


def run_strategy(strategy, end_time):
    try:
        strategy.run(end_time)
    except Exception as e:
        log("STRAT", strategy.ticker, f"Thread crashed: {e}")


def monitor(trader, rm, end_time):
    while trader.get_last_trade_time() < end_time:
        try:
            print_portfolio_summary(trader, ALL_TICKERS)
            rm.print_status()
        except Exception as e:
            log("MONITOR", "ALL", f"Error: {e}")
        sleep(30)


def main():
    global _trader_ref, _risk_ref, _strategy_instances
    signal.signal(signal.SIGINT, handle_interrupt)
    log_path = init_log_file()

    with shift.Trader(USERNAME) as trader:
        _trader_ref = trader
        trader.connect(CONFIG, PASSWORD)
        sleep(2)
        log("MAIN", "INIT", f"Connected: {trader.is_connected()}")
        log("MAIN", "INIT", f"Log: {log_path}")
        log("MAIN", "INIT", "WEEK 5 — MM ONLY (no momentum)")
        log("MAIN", "INIT",
            f"CS1: lots={MM_CONFIG['CS1']['lots']} "
            f"half_spread=${MM_CONFIG['CS1']['half_spread']:.2f}")
        log("MAIN", "INIT",
            f"CS2: lots={MM_CONFIG['CS2']['lots']} "
            f"half_spread=${MM_CONFIG['CS2']['half_spread']:.2f}")

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

        rm = RiskManager(
            trader                   = trader,
            total_tickers            = ALL_TICKERS,
            max_bp_usage             = 0.50,
            max_position_lots        = MM_MAX_INVENTORY,
            max_loss_per_ticker      = -10000.0,
            max_total_loss           = -20000.0,
            max_concurrent_positions = 4,
            max_bp_per_trade         = 100000.0,
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
        log("MAIN", "START", "MARKET MAKER — LIVE")
        log("MAIN", "START", "=" * 55)

        initial_pl = trader.get_portfolio_summary().get_total_realized_pl()
        initial_bp = trader.get_portfolio_summary().get_total_bp()
        log("MAIN", "START", f"Initial BP:  ${initial_bp:,.2f}")
        log("MAIN", "START", f"Initial P&L: ${initial_pl:,.2f}")

        # Create per-ticker MM instances
        _strategy_instances = [
            MarketMaker(
                trader        = trader,
                risk_manager  = rm,
                ticker        = t,
                lots          = MM_CONFIG[t]["lots"],
                half_spread   = MM_CONFIG[t]["half_spread"],
                max_inventory = MM_MAX_INVENTORY,
                quote_refresh = MM_QUOTE_REFRESH,
            )
            for t in ALL_TICKERS
        ]

        threads = [
            Thread(target=run_strategy, args=(mm, end_time),
                   name=f"MM_{mm.ticker}", daemon=True)
            for mm in _strategy_instances
        ]
        monitor_t = Thread(
            target=monitor, args=(trader, rm, end_time),
            name="MONITOR", daemon=True
        )

        log("MAIN", "START", f"Launching {len(threads)} MM threads")
        for t in threads:
            t.start()
            sleep(1.0)
        monitor_t.start()
        log("MAIN", "START", "All threads live")

        for t in threads:
            t.join()
        monitor_t.join(timeout=2)

        log("MAIN", "END", "Normal end — final cleanup")
        cancel_all_orders(trader)
        sleep(3)
        close_all_positions(trader, ALL_TICKERS)
        sleep(10)

        for t in ALL_TICKERS:
            try:
                item = trader.get_portfolio_item(t)
                pos = item.get_long_shares() - item.get_short_shares()
                if pos != 0:
                    log("MAIN", t, f"WARNING not flat: pos={pos}")
            except Exception:
                pass

        final_pl     = trader.get_portfolio_summary().get_total_realized_pl()
        final_bp     = trader.get_portfolio_summary().get_total_bp()
        total_orders = trader.get_submitted_orders_size()

        log("MAIN", "RESULT", "=" * 55)
        log("MAIN", "RESULT", f"Final BP:     ${final_bp:,.2f}")
        log("MAIN", "RESULT", f"Net P&L:      ${final_pl - initial_pl:,.2f}")
        log("MAIN", "RESULT", f"Total orders: {total_orders}")
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