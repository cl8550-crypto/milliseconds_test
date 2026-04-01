import shift
import time
from datetime import datetime

class BasicStrategy:
    def __init__(self, trader: shift.Trader, symbol: str):
        self.trader = trader
        self.symbol = symbol
        self.running = False

    def get_mid_price(self):
        """Calculate mid price from best bid/ask"""
        best = self.trader.getBestPrice(self.symbol)
        bid = best.getBidPrice()
        ask = best.getAskPrice()
        if bid > 0 and ask > 0:
            return (bid + ask) / 2
        return None

    def log(self, message: str):
        """Simple logger with timestamp"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] {message}")

    def print_portfolio(self):
        """Print current portfolio state"""
        summary = self.trader.getPortfolioSummary()
        item = self.trader.getPortfolioItem(self.symbol)

        self.log(f"--- Portfolio ---")
        self.log(f"  Buying Power:    {summary.getTotalBP():.2f}")
        self.log(f"  Total P&L:       {summary.getTotalRealizedPL():.2f}")
        self.log(f"  {self.symbol} Shares:   {item.getShares()}")
        self.log(f"  {self.symbol} Price:    {item.getPrice():.2f}")
        self.log(f"  Unrealized P&L:  {self.trader.getUnrealizedPL(self.symbol):.2f}")

    def run(self, duration_seconds=60):
        """
        Simple market monitoring loop.
        Runs for duration_seconds and prints market data.
        No orders placed yet — just observation.
        """
        self.running = True
        self.log(f"Starting strategy on {self.symbol} for {duration_seconds}s")

        # Subscribe to order book
        self.trader.subOrderBook(self.symbol)
        time.sleep(1)

        start_time = time.time()

        while self.running and (time.time() - start_time) < duration_seconds:
            try:
                best = self.trader.getBestPrice(self.symbol)
                mid = self.get_mid_price()

                self.log(
                    f"{self.symbol} | "
                    f"Bid: {best.getBidPrice():.2f} x {best.getBidSize()} | "
                    f"Ask: {best.getAskPrice():.2f} x {best.getAskSize()} | "
                    f"Mid: {mid:.2f if mid else 'N/A'}"
                )

                # Print portfolio every 10 seconds
                elapsed = time.time() - start_time
                if int(elapsed) % 10 == 0:
                    self.print_portfolio()

                time.sleep(1)

            except KeyboardInterrupt:
                self.log("Interrupted by user.")
                break

        self.running = False
        self.trader.unsubOrderBook(self.symbol)
        self.log("Strategy stopped.")


def main():
    trader = shift.Trader("YOUR_USERNAME")

    try:
        trader.connect("config/trader.cfg", "YOUR_PASSWORD")
        print(f"Connected: {trader.isConnected()}")
        time.sleep(2)

        stocks = trader.getStockList()
        print(f"Symbols: {stocks}")

        if not stocks:
            print("No symbols available.")
            return

        # Run on first available symbol
        strategy = BasicStrategy(trader, stocks[0])
        strategy.run(duration_seconds=60)

    except Exception as e:
        print(f"Error: {e}")

    finally:
        trader.cancelAllPendingOrders()
        trader.disconnect()
        print("Disconnected.")

if __name__ == "__main__":
    main()
