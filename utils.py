import shift
from time import sleep, time
from datetime import datetime
import os

# Global log file handle
_log_file = None


# ------------------------------------------------------------------ #
#  Logging                                                             #
# ------------------------------------------------------------------ #
def init_log_file():
    global _log_file
    os.makedirs(os.path.expanduser("~/shift_trading/logs"), exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path  = os.path.expanduser(
        f"~/shift_trading/logs/run_{timestamp}.log"
    )
    _log_file = open(log_path, "w", buffering=1)
    print(f"[LOGGING] Writing to {log_path}", flush=True)
    return log_path


def log(strategy: str, ticker: str, message: str):
    ts   = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}][{strategy}][{ticker}] {message}"
    print(line, flush=True)
    if _log_file:
        try:
            _log_file.write(line + "\n")
        except Exception:
            pass


# ------------------------------------------------------------------ #
#  Buying power                                                        #
# ------------------------------------------------------------------ #
def get_buying_power(trader: shift.Trader) -> float:
    try:
        return trader.get_portfolio_summary().get_total_bp()
    except Exception:
        return 0.0


# ------------------------------------------------------------------ #
#  Position                                                            #
# ------------------------------------------------------------------ #
def get_position(trader: shift.Trader, ticker: str) -> int:
    try:
        item  = trader.get_portfolio_item(ticker)
        long  = item.get_long_shares()
        short = item.get_short_shares()
        return long - short
    except Exception:
        return 0


# ------------------------------------------------------------------ #
#  Best prices                                                         #
# ------------------------------------------------------------------ #
def get_best_prices(
    trader: shift.Trader,
    ticker: str
) -> tuple[float, float] | None:
    try:
        best = trader.get_best_price(ticker)
        bid  = best.get_bid_price()
        ask  = best.get_ask_price()
        if bid > 0 and ask > 0 and ask > bid:
            return bid, ask
        return None
    except Exception:
        return None


def get_spread(trader: shift.Trader, ticker: str) -> float | None:
    prices = get_best_prices(trader, ticker)
    if prices is None:
        return None
    return round(prices[1] - prices[0], 4)


# ------------------------------------------------------------------ #
#  Order submission                                                    #
# ------------------------------------------------------------------ #
def submit_limit_buy(
    trader: shift.Trader,
    ticker: str,
    lots: int,
    price: float
) -> bool:
    try:
        order = shift.Order(
            shift.Order.Type.LIMIT_BUY,
            ticker, lots, round(price, 2)
        )
        trader.submit_order(order)
        return True
    except Exception as e:
        log("ORDER", ticker, f"LIMIT_BUY failed: {e}")
        return False


def submit_limit_sell(
    trader: shift.Trader,
    ticker: str,
    lots: int,
    price: float
) -> bool:
    try:
        order = shift.Order(
            shift.Order.Type.LIMIT_SELL,
            ticker, lots, round(price, 2)
        )
        trader.submit_order(order)
        return True
    except Exception as e:
        log("ORDER", ticker, f"LIMIT_SELL failed: {e}")
        return False


def submit_market_buy(
    trader: shift.Trader,
    ticker: str,
    lots: int
) -> bool:
    try:
        order = shift.Order(
            shift.Order.Type.MARKET_BUY,
            ticker, lots
        )
        trader.submit_order(order)
        return True
    except Exception as e:
        log("ORDER", ticker, f"MARKET_BUY failed: {e}")
        return False


def submit_market_sell(
    trader: shift.Trader,
    ticker: str,
    lots: int
) -> bool:
    try:
        order = shift.Order(
            shift.Order.Type.MARKET_SELL,
            ticker, lots
        )
        trader.submit_order(order)
        return True
    except Exception as e:
        log("ORDER", ticker, f"MARKET_SELL failed: {e}")
        return False


# ------------------------------------------------------------------ #
#  Order cancellation — FIXED                                          #
#  submit_cancellation(order) takes the ORDER OBJECT not an ID        #
#  cancel_all_pending_orders() cancels everything at once             #
# ------------------------------------------------------------------ #
def cancel_orders_for(trader: shift.Trader, ticker: str):
    """Cancel all pending orders for a specific ticker"""
    try:
        waiting = trader.get_waiting_list()
        cancelled = 0
        for order in waiting:
            if order.symbol == ticker:
                trader.submit_cancellation(order)
                cancelled += 1
        if cancelled > 0:
            log("ORDER", ticker, f"Cancelled {cancelled} pending orders")
    except Exception as e:
        log("ORDER", ticker, f"Cancel failed: {e}")


def cancel_all_orders(trader: shift.Trader):
    """Cancel all pending orders across all tickers"""
    try:
        trader.cancel_all_pending_orders()
        log("ORDER", "ALL", "Cancelled all pending orders")
    except Exception as e:
        log("ORDER", "ALL", f"Cancel all failed: {e}")


# ------------------------------------------------------------------ #
#  Position closing                                                    #
# ------------------------------------------------------------------ #
def close_position(trader: shift.Trader, ticker: str):
    """Close a single ticker's position with market order"""
    try:
        item         = trader.get_portfolio_item(ticker)
        long_shares  = item.get_long_shares()
        short_shares = item.get_short_shares()

        if long_shares > 0:
            lots = int(long_shares / 100)
            submit_market_sell(trader, ticker, lots)
            log("CLOSE", ticker,
                f"MARKET_SELL {long_shares} shares ({lots} lots)")

        elif short_shares > 0:
            lots = int(short_shares / 100)
            submit_market_buy(trader, ticker, lots)
            log("CLOSE", ticker,
                f"MARKET_BUY {short_shares} shares ({lots} lots)")

    except Exception as e:
        log("CLOSE", ticker, f"close_position error: {e}")


def close_all_positions(trader: shift.Trader, tickers: list[str]):
    """
    Close all positions.
    Cancel all orders first, then read position ONCE per ticker
    and submit a single close. sleep(1) between tickers prevents
    race conditions between multiple strategy threads.
    """
    log("CLOSE", "ALL", "Cancelling all pending orders...")
    cancel_all_orders(trader)
    sleep(3)

    log("CLOSE", "ALL", "Closing all open positions...")
    for ticker in tickers:
        try:
            item         = trader.get_portfolio_item(ticker)
            long_shares  = item.get_long_shares()
            short_shares = item.get_short_shares()

            if long_shares > 0:
                lots = int(long_shares / 100)
                submit_market_sell(trader, ticker, lots)
                log("CLOSE", ticker,
                    f"MARKET_SELL {long_shares} shares ({lots} lots)")
                sleep(1)

            elif short_shares > 0:
                lots = int(short_shares / 100)
                submit_market_buy(trader, ticker, lots)
                log("CLOSE", ticker,
                    f"MARKET_BUY {short_shares} shares ({lots} lots)")
                sleep(1)

        except Exception as e:
            log("CLOSE", ticker, f"Error: {e}")

    sleep(10)

    # Verification pass
    log("CLOSE", "ALL", "Verifying all positions are flat...")
    still_open = []
    for ticker in tickers:
        try:
            item  = trader.get_portfolio_item(ticker)
            long  = item.get_long_shares()
            short = item.get_short_shares()
            if long == 0 and short == 0:
                log("CLOSE", ticker, "Confirmed flat")
            else:
                log("CLOSE", ticker,
                    f"Still open! long={long} short={short} — retrying")
                still_open.append(ticker)
        except Exception as e:
            log("CLOSE", ticker, f"Verify error: {e}")

    # Retry with fresh cancel + market order
    for ticker in still_open:
        try:
            cancel_orders_for(trader, ticker)
            sleep(1)
            item  = trader.get_portfolio_item(ticker)
            long  = item.get_long_shares()
            short = item.get_short_shares()
            if long > 0:
                submit_market_sell(trader, ticker, int(long / 100))
                log("CLOSE", ticker,
                    f"RETRY MARKET_SELL {long} shares")
            elif short > 0:
                submit_market_buy(trader, ticker, int(short / 100))
                log("CLOSE", ticker,
                    f"RETRY MARKET_BUY {short} shares")
            sleep(5)
        except Exception as e:
            log("CLOSE", ticker, f"Retry error: {e}")

    log("CLOSE", "ALL", "All positions processed")


# ------------------------------------------------------------------ #
#  Portfolio reporting                                                 #
# ------------------------------------------------------------------ #
def print_portfolio_summary(trader: shift.Trader, tickers: list[str]):
    try:
        summary  = trader.get_portfolio_summary()
        bp       = summary.get_total_bp()
        total_pl = summary.get_total_realized_pl()

        print("\n" + "="*50, flush=True)
        print(
            f"PORTFOLIO SUMMARY @ "
            f"{datetime.now().strftime('%H:%M:%S')}",
            flush=True
        )
        print(f"  Buying Power:  ${bp:,.2f}", flush=True)
        print(f"  Total P&L:     ${total_pl:,.2f}", flush=True)
        print("-"*50, flush=True)

        for ticker in sorted(tickers):
            try:
                item       = trader.get_portfolio_item(ticker)
                position   = (
                    item.get_long_shares() - item.get_short_shares()
                )
                unrealized = trader.get_unrealized_pl(ticker)
                print(
                    f"  {ticker:<6} | "
                    f"position={position:>6} | "
                    f"unrealized P&L=${unrealized:,.2f}",
                    flush=True
                )
            except Exception:
                pass

        print("="*50 + "\n", flush=True)

    except Exception as e:
        log("MONITOR", "ALL", f"Portfolio summary error: {e}")