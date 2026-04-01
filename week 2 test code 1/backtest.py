"""
Backtest: Spike Fade Strategy
Simulates spike_fade.py logic on local 1-minute OHLCV CSV data.

Data files expected in: /Users/liucanxin/Desktop/run 2 test data/
  - 20100506_1min.csv  (Flash Crash, May 6 2010)
  - 20150824_1min.csv  (China Crash, Aug 24 2015)
  - 20250409_1min.csv  (Tariff Rally, Apr 9 2025)
"""

import pandas as pd
import numpy as np
from pathlib import Path


# ------------------------------------------------------------------ #
#  Configuration — mirrors spike_fade.py parameters                   #
# ------------------------------------------------------------------ #
DATA_DIR           = Path("/Users/liucanxin/Desktop/run 2 test data")
STARTING_CAPITAL   = 1_000_000
ORDER_SIZE_SHARES  = 300        # 3 lots x 100 shares
LIMIT_REBATE       = 0.002      # $0.002/share earned on limit order fills
MARKET_FEE         = 0.003      # $0.003/share cost on market order fills
BASELINE_WINDOW    = 30         # bars for rolling median baseline (~30 min)
SPIKE_THRESHOLD    = 0.015      # 1.5% deviation triggers entry
RECOVERY_THRESHOLD = 0.005      # 0.5% from baseline = take profit
STOP_LOSS_MULT     = 2.0        # stop if move reaches 2x threshold (3%)
TIME_STOP_BARS     = 3          # abandon position after 3 bars with no recovery


# ------------------------------------------------------------------ #
#  Data loading                                                        #
# ------------------------------------------------------------------ #
def load_csv(filepath: Path) -> pd.DataFrame:
    df = pd.read_csv(filepath)

    # Normalize column names across the two different CSV formats
    df.columns = [c.lower().strip() for c in df.columns]
    if "timestamp" in df.columns:
        df = df.rename(columns={"timestamp": "date"})

    df["date"] = pd.to_datetime(df["date"], utc=False)
    df = df.sort_values("date").reset_index(drop=True)
    return df[["date", "open", "high", "low", "close", "volume"]]


# ------------------------------------------------------------------ #
#  Core backtest logic                                                 #
# ------------------------------------------------------------------ #
def run_backtest(df: pd.DataFrame, label: str) -> dict:
    prices   = df["close"].values
    highs    = df["high"].values
    lows     = df["low"].values
    n        = len(prices)

    cash       = STARTING_CAPITAL
    position   = 0         # shares: positive = long, negative = short
    entry_price = 0.0
    entry_bar   = 0

    trades     = []        # list of closed trade records
    baseline_prices = []   # running list for median calculation

    for i in range(n):
        price = prices[i]

        # ---- Update baseline only when not in a spike ---- #
        # (mirrors the freeze logic in spike_fade.py)
        if len(baseline_prices) >= 2:
            baseline = float(np.median(baseline_prices[-BASELINE_WINDOW:]))
            deviation = (price - baseline) / baseline
        else:
            baseline = price
            deviation = 0.0

        if abs(deviation) < SPIKE_THRESHOLD * 0.5:
            baseline_prices.append(price)

        # Need enough history before trading
        if len(baseline_prices) < BASELINE_WINDOW // 2:
            baseline_prices.append(price)
            continue

        baseline  = float(np.median(baseline_prices[-BASELINE_WINDOW:]))
        deviation = (price - baseline) / baseline
        stop_dev  = SPIKE_THRESHOLD * STOP_LOSS_MULT

        # ---- Manage open position ---- #
        if position != 0:
            bars_held = i - entry_bar

            if position > 0:
                # LONG — bought the dip, waiting for recovery
                take_profit = deviation > -RECOVERY_THRESHOLD
                stop_loss   = deviation < -stop_dev
                time_stop   = bars_held >= TIME_STOP_BARS

                if take_profit or stop_loss or time_stop:
                    exit_price = price
                    is_limit   = take_profit
                    fee        = -LIMIT_REBATE if is_limit else MARKET_FEE
                    pnl        = (exit_price - entry_price) * position - fee * position
                    cash      += pnl

                    reason = ("TAKE_PROFIT" if take_profit
                              else "STOP_LOSS" if stop_loss
                              else "TIME_STOP")
                    trades.append({
                        "bar":        i,
                        "time":       df["date"].iloc[i],
                        "side":       "LONG",
                        "entry":      entry_price,
                        "exit":       exit_price,
                        "bars_held":  bars_held,
                        "pnl":        round(pnl, 2),
                        "reason":     reason,
                        "deviation":  round(deviation * 100, 2),
                    })
                    position = 0

            elif position < 0:
                # SHORT — sold the spike, waiting for recovery
                take_profit = deviation < RECOVERY_THRESHOLD
                stop_loss   = deviation > stop_dev
                time_stop   = bars_held >= TIME_STOP_BARS

                if take_profit or stop_loss or time_stop:
                    exit_price = price
                    is_limit   = take_profit
                    fee        = -LIMIT_REBATE if is_limit else MARKET_FEE
                    pnl        = (entry_price - exit_price) * abs(position) - fee * abs(position)
                    cash      += pnl

                    reason = ("TAKE_PROFIT" if take_profit
                              else "STOP_LOSS" if stop_loss
                              else "TIME_STOP")
                    trades.append({
                        "bar":        i,
                        "time":       df["date"].iloc[i],
                        "side":       "SHORT",
                        "entry":      entry_price,
                        "exit":       exit_price,
                        "bars_held":  bars_held,
                        "pnl":        round(pnl, 2),
                        "reason":     reason,
                        "deviation":  round(deviation * 100, 2),
                    })
                    position = 0

        # ---- Look for new entry (flat only) ---- #
        if position == 0:
            if deviation > SPIKE_THRESHOLD:
                # Spike UP — enter short
                entry_price = price - 0.01   # limit sell just inside ask
                position    = -ORDER_SIZE_SHARES
                entry_bar   = i
                # Entry rebate credited immediately
                cash       += LIMIT_REBATE * ORDER_SIZE_SHARES

            elif deviation < -SPIKE_THRESHOLD:
                # Dip DOWN — enter long
                entry_price = price + 0.01   # limit buy just inside bid
                position    = ORDER_SIZE_SHARES
                entry_bar   = i
                # Entry rebate credited immediately
                cash       += LIMIT_REBATE * ORDER_SIZE_SHARES

    # ---- Force close any remaining position at end of day ---- #
    if position != 0:
        exit_price = prices[-1]
        if position > 0:
            pnl = (exit_price - entry_price) * position - MARKET_FEE * position
        else:
            pnl = (entry_price - exit_price) * abs(position) - MARKET_FEE * abs(position)
        cash += pnl
        trades.append({
            "bar":        n - 1,
            "time":       df["date"].iloc[-1],
            "side":       "LONG" if position > 0 else "SHORT",
            "entry":      entry_price,
            "exit":       exit_price,
            "bars_held":  n - 1 - entry_bar,
            "pnl":        round(pnl, 2),
            "reason":     "EOD_CLOSE",
            "deviation":  0.0,
        })
        position = 0

    net_pnl    = cash - STARTING_CAPITAL
    trades_df  = pd.DataFrame(trades)

    return {
        "label":      label,
        "net_pnl":    round(net_pnl, 2),
        "trades":     trades_df,
        "final_cash": round(cash, 2),
    }


# ------------------------------------------------------------------ #
#  Report printer                                                      #
# ------------------------------------------------------------------ #
def print_report(result: dict):
    label  = result["label"]
    pnl    = result["net_pnl"]
    trades = result["trades"]

    print("\n" + "=" * 60)
    print(f"  {label}")
    print("=" * 60)

    if trades.empty:
        print("  No trades triggered.")
        print(f"  Net P&L: $0.00")
        return

    wins   = trades[trades["pnl"] > 0]
    losses = trades[trades["pnl"] <= 0]

    print(f"  Net P&L:        ${pnl:>10,.2f}")
    print(f"  Total trades:   {len(trades)}")
    print(f"  Winners:        {len(wins)}  (avg ${wins['pnl'].mean():.2f})" if not wins.empty else "  Winners:        0")
    print(f"  Losers:         {len(losses)}  (avg ${losses['pnl'].mean():.2f})" if not losses.empty else "  Losers:         0")
    print(f"  Win rate:       {len(wins)/len(trades)*100:.1f}%")
    print(f"  Best trade:     ${trades['pnl'].max():,.2f}")
    print(f"  Worst trade:    ${trades['pnl'].min():,.2f}")
    print(f"  Avg bars held:  {trades['bars_held'].mean():.1f} min")

    print("\n  Exit reasons:")
    for reason, count in trades["reason"].value_counts().items():
        print(f"    {reason:<15} {count} trades")

    print("\n  Trade log:")
    print(f"  {'Time':<25} {'Side':<6} {'Entry':>8} {'Exit':>8} "
          f"{'Bars':>5} {'Dev%':>6} {'P&L':>10} {'Reason'}")
    print("  " + "-" * 80)
    for _, t in trades.iterrows():
        print(
            f"  {str(t['time']):<25} {t['side']:<6} "
            f"{t['entry']:>8.2f} {t['exit']:>8.2f} "
            f"{int(t['bars_held']):>5} {t['deviation']:>6.2f}% "
            f"${t['pnl']:>9,.2f}  {t['reason']}"
        )


# ------------------------------------------------------------------ #
#  Main                                                                #
# ------------------------------------------------------------------ #
def main():
    FILES = {
        "Flash Crash — May 6, 2010":    DATA_DIR / "20100506_1min.csv",
        "China Crash — Aug 24, 2015":   DATA_DIR / "20150824_1min.csv",
        "Tariff Rally — Apr 9, 2025":   DATA_DIR / "20250409_1min.csv",
    }

    print("\nSPIKE FADE BACKTEST")
    print(f"Spike threshold:    {SPIKE_THRESHOLD*100:.1f}%")
    print(f"Recovery threshold: {RECOVERY_THRESHOLD*100:.1f}%")
    print(f"Stop loss:          {SPIKE_THRESHOLD*STOP_LOSS_MULT*100:.1f}%")
    print(f"Time stop:          {TIME_STOP_BARS} bars")
    print(f"Order size:         {ORDER_SIZE_SHARES} shares ({ORDER_SIZE_SHARES//100} lots)")

    total_pnl = 0.0
    for label, filepath in FILES.items():
        if not filepath.exists():
            print(f"\n[SKIP] File not found: {filepath}")
            continue
        df = load_csv(filepath)
        result = run_backtest(df, label)
        print_report(result)
        total_pnl += result["net_pnl"]

    print("\n" + "=" * 60)
    print(f"  TOTAL NET P&L ACROSS ALL 3 DAYS:  ${total_pnl:,.2f}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
