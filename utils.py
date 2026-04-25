"""
utils.py — SHIFT trading helpers, rate-limit aware.

Token bucket rate limit (enforced in SHIFT API as of Week 4):
  - 5 tokens/sec refill, 10 token burst capacity
  - Each submit_order() OR submit_cancellation() costs 1 token
  - Both methods now return True/False

Safe message budget for 3 tickers:
  quote_refresh=3.0s → 12 msg/3.0s = 4.0 msg/s (under 5/s limit)

Key changes from pre-rate-limit version:
  cancel_orders_for()  — uses get_waiting_list() + individual cancels
                         with 0.25s sleep, NOT cancel_all_pending_orders()
  cancel_all_orders()  — same, iterates all tickers sequentially
  submit_*()           — check bool return, retry once with 0.5s backoff
"""

import shift
import os
import logging
from datetime import datetime
from time import sleep, time


# ------------------------------------------------------------------ #
#  Logging                                                            #
# ------------------------------------------------------------------ #
_log_path = None

def init_log_file() -> str:
    global _log_path
    os.makedirs("logs", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    _log_path = f"logs/run_{ts}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s]%(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.FileHandler(_log_path),
            logging.StreamHandler(),
        ],
    )
    logging.info(f"[LOGGING] Writing to {os.path.abspath(_log_path)}")
    return _log_path

def log(strategy: str, ticker: str, msg: str):
    logging.info(f"[{strategy}][{ticker}] {msg}")


# ------------------------------------------------------------------ #
#  Position helpers                                                   #
# ------------------------------------------------------------------ #
def get_position(trader: shift.Trader, ticker: str) -> int:
    """Returns signed share count (+ve long, -ve short)."""
    try:
        item = trader.get_portfolio_item(ticker)
        return item.get_long_shares() - item.get_short_shares()
    except Exception:
        return 0

def get_best_prices(
    trader: shift.Trader, ticker: str
) -> tuple[float, float] | None:
    """Returns (best_bid, best_ask) or None if book is empty."""
    try:
        bid = trader.get_best_price(ticker).get_bid_price()
        ask = trader.get_best_price(ticker).get_ask_price()
        if bid <= 0 or ask <= 0 or ask <= bid:
            return None
        return float(bid), float(ask)
    except Exception:
        return None


# ------------------------------------------------------------------ #
#  Rate-limit-safe order submission                                   #
# ------------------------------------------------------------------ #
_SUBMIT_RETRY_SLEEP = 0.25   # seconds to wait before retry

def _submit_with_retry(order: shift.Order, trader: shift.Trader) -> bool:
    """
    Submit an order, retrying once if rate-limited (returns False).
    Returns True if successfully submitted.
    """
    result = trader.submit_order(order)
    if not result:
        sleep(_SUBMIT_RETRY_SLEEP)
        result = trader.submit_order(order)
    return result

def submit_limit_buy(
    trader: shift.Trader, ticker: str, lots: int, price: float
) -> bool:
    order = shift.Order(
        shift.Order.Type.LIMIT_BUY, ticker, lots, price
    )
    return _submit_with_retry(order, trader)

def submit_limit_sell(
    trader: shift.Trader, ticker: str, lots: int, price: float
) -> bool:
    order = shift.Order(
        shift.Order.Type.LIMIT_SELL, ticker, lots, price
    )
    return _submit_with_retry(order, trader)

def submit_market_buy(
    trader: shift.Trader, ticker: str, lots: int
) -> bool:
    order = shift.Order(
        shift.Order.Type.MARKET_BUY, ticker, lots
    )
    return _submit_with_retry(order, trader)

def submit_market_sell(
    trader: shift.Trader, ticker: str, lots: int
) -> bool:
    order = shift.Order(
        shift.Order.Type.MARKET_SELL, ticker, lots
    )
    return _submit_with_retry(order, trader)


# ------------------------------------------------------------------ #
#  Rate-limit-safe cancellations                                      #
# ------------------------------------------------------------------ #
_CANCEL_SLEEP = 0.25   # seconds between cancellations

def cancel_orders_for(trader: shift.Trader, ticker: str):
    """
    Cancel all pending orders for a specific ticker.

    Uses get_waiting_list() → filter by ticker → individual
    submit_cancellation() with sleep between each.

    IMPORTANT: Do NOT use cancel_all_pending_orders(ticker) —
    that method fires cancellations in a rapid-fire loop and will
    exhaust the token bucket immediately (rate limit spec, Week 4).
    """
    try:
        waiting = trader.get_waiting_list()
        to_cancel = [
            o for o in waiting if o.symbol == ticker
        ]
        if not to_cancel:
            return

        cancelled = 0
        for order in to_cancel:
            result = trader.submit_cancellation(order)
            if not result:
                sleep(_CANCEL_SLEEP)
                trader.submit_cancellation(order)   # one retry
            cancelled += 1
            if cancelled < len(to_cancel):
                sleep(_CANCEL_SLEEP)

        log("ORDER", ticker, f"Cancelled {len(to_cancel)} pending orders")
    except Exception as e:
        log("ORDER", ticker, f"Cancel error: {e}")


def cancel_all_orders(trader: shift.Trader):
    """
    Cancel ALL pending orders across all tickers.

    Iterates get_waiting_list() and cancels one at a time with
    sleep between each to respect the 5 msg/s token bucket limit.
    """
    try:
        waiting = trader.get_waiting_list()
        if not waiting:
            log("ORDER", "ALL", "No pending orders to cancel")
            return

        cancelled = 0
        for order in waiting:
            result = trader.submit_cancellation(order)
            if not result:
                sleep(_CANCEL_SLEEP)
                trader.submit_cancellation(order)
            cancelled += 1
            sleep(_CANCEL_SLEEP)

        log("ORDER", "ALL", f"Cancelled {cancelled} pending orders")
    except Exception as e:
        log("ORDER", "ALL", f"Cancel all error: {e}")


# ------------------------------------------------------------------ #
#  End-of-session position flattening                                #
# ------------------------------------------------------------------ #
def close_all_positions(
    trader: shift.Trader,
    tickers: list[str],
    max_wait: float = 30.0
):
    """
    Close all open positions with market orders, one ticker at a time.
    Waits for confirmation before proceeding to next ticker.
    """
    log("CLOSE", "ALL", "Cancelling all pending orders...")
    cancel_all_orders(trader)
    sleep(2.0)   # let cancellations process

    log("CLOSE", "ALL", "Closing all open positions...")
    for ticker in tickers:
        try:
            position = get_position(trader, ticker)
            if position == 0:
                continue
            if position > 0:
                lots = int(position / 100)
                submit_market_sell(trader, ticker, lots)
                log("CLOSE", ticker, f"MARKET_SELL {lots}L")
            elif position < 0:
                lots = int(-position / 100)
                submit_market_buy(trader, ticker, lots)
                log("CLOSE", ticker, f"MARKET_BUY {lots}L")
            sleep(0.3)   # spacing between tickers
        except Exception as e:
            log("CLOSE", ticker, f"Error: {e}")

    # Wait for positions to clear
    sleep(3.0)
    log("CLOSE", "ALL", "Verifying all positions are flat...")
    for ticker in tickers:
        try:
            position = get_position(trader, ticker)
            status = "Confirmed flat" if position == 0 else f"WARNING: pos={position}"
            log("CLOSE", ticker, status)
        except Exception as e:
            log("CLOSE", ticker, f"Verify error: {e}")
    log("CLOSE", "ALL", "All positions processed")


# ------------------------------------------------------------------ #
#  Portfolio summary                                                  #
# ------------------------------------------------------------------ #
def print_portfolio_summary(
    trader: shift.Trader, tickers: list[str]
):
    try:
        summary = trader.get_portfolio_summary()
        bp      = summary.get_total_bp()
        pl      = summary.get_total_realized_pl()
        print(f"\n{'='*50}")
        print(f"PORTFOLIO SUMMARY @ {datetime.now().strftime('%H:%M:%S')}")
        print(f"  Buying Power:  ${bp:,.2f}")
        print(f"  Total P&L:     ${pl:,.2f}")
        print(f"{'-'*50}")
        for ticker in tickers:
            try:
                item = trader.get_portfolio_item(ticker)
                pos  = item.get_long_shares() - item.get_short_shares()
                upl  = item.get_unrealized_pl()
                print(
                    f"  {ticker:<6} | position={pos:>6} | "
                    f"unrealized P&L=${upl:.2f}"
                )
            except Exception:
                pass
        print(f"{'='*50}\n")
    except Exception as e:
        log("MONITOR", "ALL", f"Portfolio summary error: {e}")