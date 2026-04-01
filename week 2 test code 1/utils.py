import shift
import numpy as np
from time import sleep
from datetime import datetime


# ------------------------------------------------------------------ #
#  Logging                                                             #
# ------------------------------------------------------------------ #
def log(strategy: str, ticker: str, message: str):
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}][{strategy}][{ticker}] {message}", flush=True)


# ------------------------------------------------------------------ #
#  Price helpers                                                       #
# ------------------------------------------------------------------ #
def get_mid_price(trader: shift.Trader, ticker: str) -> float | None:
    best = trader.get_best_price(ticker)
    bid = best.get_bid_price()
    ask = best.get_ask_price()
    if bid > 0 and ask > 0:
        return round((bid + ask) / 2, 4)
    return None


def get_spread(trader: shift.Trader, ticker: str) -> float | None:
    best = trader.get_best_price(ticker)
    bid = best.get_bid_price()
    ask = best.get_ask_price()
    if bid > 0 and ask > 0:
        return round(ask - bid, 4)
    return None


def get_zscore(price_history: list, window: int) -> float | None:
    if len(price_history) < window:
        return None
    series = np.array(price_history[-window:])
    mean = np.mean(series)
    std = np.std(series)
    if std == 0:
        return None
    return round((price_history[-1] - mean) / std, 4)


# ------------------------------------------------------------------ #
#  Position helpers                                                    #
# ------------------------------------------------------------------ #
def get_position(trader: shift.Trader, ticker: str) -> int:
    """Returns net position: positive = long, negative = short"""
    item = trader.get_portfolio_item(ticker)
    return item.get_long_shares() - item.get_short_shares()


def get_buying_power(trader: shift.Trader) -> float:
    return trader.get_portfolio_summary().get_total_bp()


def get_total_pl(trader: shift.Trader) -> float:
    return trader.get_portfolio_summary().get_total_realized_pl()


# ------------------------------------------------------------------ #
#  Validation helpers                                                  #
# ------------------------------------------------------------------ #
def is_valid_price(price: float) -> bool:
    return price is not None and price > 0


def is_valid_bid_ask(bid: float, ask: float) -> bool:
    return bid > 0 and ask > 0 and ask > bid


# ------------------------------------------------------------------ #
#  Order helpers                                                       #
# ------------------------------------------------------------------ #
def cancel_orders_for(trader: shift.Trader, ticker: str):
    for order in trader.get_waiting_list():
        if order.symbol == ticker:
            trader.submit_cancellation(order)


def cancel_all_orders(trader: shift.Trader):
    for order in trader.get_waiting_list():
        trader.submit_cancellation(order)


def submit_limit_buy(
    trader: shift.Trader,
    ticker: str,
    size: int,
    price: float
) -> shift.Order | None:
    if not is_valid_price(price):
        log("UTILS", ticker, f"Invalid price {price} — skipping LIMIT_BUY")
        return None
    if size <= 0:
        log("UTILS", ticker, f"Invalid size {size} — skipping LIMIT_BUY")
        return None
    order = shift.Order(
        shift.Order.Type.LIMIT_BUY,
        ticker,
        size,
        round(price, 2)
    )
    trader.submit_order(order)
    return order


def submit_limit_sell(
    trader: shift.Trader,
    ticker: str,
    size: int,
    price: float
) -> shift.Order | None:
    if not is_valid_price(price):
        log("UTILS", ticker, f"Invalid price {price} — skipping LIMIT_SELL")
        return None
    if size <= 0:
        log("UTILS", ticker, f"Invalid size {size} — skipping LIMIT_SELL")
        return None
    order = shift.Order(
        shift.Order.Type.LIMIT_SELL,
        ticker,
        size,
        round(price, 2)
    )
    trader.submit_order(order)
    return order


def submit_market_buy(
    trader: shift.Trader,
    ticker: str,
    size: int
) -> shift.Order | None:
    if size <= 0:
        log("UTILS", ticker, f"Invalid size {size} — skipping MARKET_BUY")
        return None
    order = shift.Order(
        shift.Order.Type.MARKET_BUY,
        ticker,
        size
    )
    trader.submit_order(order)
    return order


def submit_market_sell(
    trader: shift.Trader,
    ticker: str,
    size: int
) -> shift.Order | None:
    if size <= 0:
        log("UTILS", ticker, f"Invalid size {size} — skipping MARKET_SELL")
        return None
    order = shift.Order(
        shift.Order.Type.MARKET_SELL,
        ticker,
        size
    )
    trader.submit_order(order)
    return order


# ------------------------------------------------------------------ #
#  Position closing                                                    #
# ------------------------------------------------------------------ #
def close_position(trader: shift.Trader, ticker: str):
    """Cancel all pending orders then market close any open position"""
    cancel_orders_for(trader, ticker)
    sleep(3)  # wait for cancellations to process

    item = trader.get_portfolio_item(ticker)
    long_shares  = item.get_long_shares()
    short_shares = item.get_short_shares()

    if long_shares > 0:
        lots = int(long_shares / 100)
        if lots > 0:
            submit_market_sell(trader, ticker, lots)
            log("CLOSE", ticker, f"MARKET_SELL {long_shares} shares ({lots} lots)")
            sleep(2)

    if short_shares > 0:
        lots = int(short_shares / 100)
        if lots > 0:
            submit_market_buy(trader, ticker, lots)
            log("CLOSE", ticker, f"MARKET_BUY {short_shares} shares ({lots} lots)")
            sleep(2)


def close_all_positions(trader: shift.Trader, tickers: list[str]):
    """Close all positions and verify everything is flat"""
    cancel_all_orders(trader)
    sleep(3)

    # First pass — close all positions
    for ticker in tickers:
        item = trader.get_portfolio_item(ticker)
        long_shares  = item.get_long_shares()
        short_shares = item.get_short_shares()

        if long_shares > 0 or short_shares > 0:
            close_position(trader, ticker)
            log("CLOSE", ticker, "Position closed — first pass")

    # Wait for all market orders to process
    sleep(5)

    # Second pass — verify and retry any still-open positions
    log("CLOSE", "ALL", "Verifying all positions are flat...")
    for ticker in tickers:
        try:
            item = trader.get_portfolio_item(ticker)
            long_shares  = item.get_long_shares()
            short_shares = item.get_short_shares()

            if long_shares > 0 or short_shares > 0:
                log(
                    "CLOSE", ticker,
                    f"Still open after first pass! "
                    f"long={long_shares} short={short_shares} — retrying"
                )
                close_position(trader, ticker)
                sleep(3)
            else:
                log("CLOSE", ticker, "Confirmed flat")

        except Exception as e:
            log("CLOSE", ticker, f"Error verifying position: {e}")

    sleep(3)
    log("CLOSE", "ALL", "All positions processed")


# ------------------------------------------------------------------ #
#  Portfolio summary                                                   #
# ------------------------------------------------------------------ #
def print_portfolio_summary(trader: shift.Trader, tickers: list[str]):
    print("\n" + "="*50, flush=True)
    print(
        f"PORTFOLIO SUMMARY @ {datetime.now().strftime('%H:%M:%S')}",
        flush=True
    )
    print(f"  Buying Power:  ${get_buying_power(trader):,.2f}", flush=True)
    print(f"  Total P&L:     ${get_total_pl(trader):,.2f}", flush=True)
    print("-"*50, flush=True)
    for ticker in tickers:
        try:
            item = trader.get_portfolio_item(ticker)
            pl   = trader.get_unrealized_pl(ticker)
            pos  = get_position(trader, ticker)
            print(
                f"  {ticker:<6} | position={pos:>6} | "
                f"unrealized P&L=${pl:,.2f}",
                flush=True
            )
        except Exception as e:
            print(f"  {ticker:<6} | error: {e}", flush=True)
    print("="*50 + "\n", flush=True)