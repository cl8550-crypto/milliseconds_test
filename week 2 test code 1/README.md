# Week 2 Strategy: Spike Fade

## Theme
"Spikes and Dips" — days where a major exogenous shock causes a sudden large price move that **recovers within the same day**. The rest of the day is calm.

## Core Idea
We **fade the move** — trade against the spike or dip, and profit when the price snaps back to normal.

- Price **spikes up** → we **sell** (short it), close when it comes back down
- Price **dips down** → we **buy** (long it), close when it recovers

---

## Changes from Week 1

### Removed
| Component | Reason |
|---|---|
| `PairsStrategy` | Pairs trading profits from sustained divergence between two correlated stocks. On a spike-and-recover day the move is sudden and short-lived — not enough time for a pairs trade to develop and close cleanly. |
| `OrderBookArbStrategy` | Order book arb exploits pricing gaps between the local and global book. The day is calm outside of spikes, so the order book stays tight with almost no arb opportunities. |
| `MeanReversionStrategy` | This used a 20-tick z-score window (~20 seconds) which is too slow. By the time a z-score signals an entry the spike may already be half-recovered, leaving little profit on the table. |

### Added
| Component | Purpose |
|---|---|
| `SpikeFadeStrategy` | Primary strategy — detects sudden large price moves and fades them with limit orders, exiting on recovery |
| `PassiveMarketMaker` | Order count engine — posts small limit buy/sell pairs during calm periods to hit the 200 orders/day minimum and earn rebates |

---

## SpikeFadeStrategy — How It Works

Maintains a **rolling median baseline** of the last 60 price ticks (~30 seconds of price history at a 0.5-second sampling rate). When the live price deviates more than **1.5%** from that baseline, a spike or dip is flagged and a contra-directional limit order is placed.

**Why median and not average?**
The spike itself is included in the price history. If we used a simple average, a 5% spike would pull the baseline up by ~0.1% per tick, making the strategy think the price has "always been here" and never triggering an entry. Median ignores outliers, so the baseline stays anchored to where prices were before the shock.

**Why 1.5%?**
We looked at the real historical events that match this theme:

| Event | Ticker | Total Intraday Move | Same-Day Recovery |
|---|---|---|---|
| Flash Crash, May 2010 | AAPL | -22% | ~84% reversed |
| China crash, Aug 2015 | AAPL | -13% | ~80% reversed |
| Fake tariff news, Apr 2025 | AAPL | +9.5% spike | Fully reversed in ~2 hrs |
| Yen carry unwind, Aug 2024 | NVDA | -15.5% | ~59% reversed |

These spikes total 10–22% from the prior close, but the move develops over minutes from the morning trading level. A threshold below 1% would fire on normal intraday noise (bid/ask bounce, routine order flow). A threshold above 3% means we wait too long and enter after most of the recovery has already happened. **1.5% is the earliest point where the signal is clearly a spike and not noise**, leaving 8–20% of recovery still ahead of us.

**Entry:** Limit order placed just inside the bid/ask to earn the $0.002/share rebate.

**Exit — first condition to trigger wins:**
| Condition | Action |
|---|---|
| Price recovers within 0.5% of baseline | Limit order close — take profit and earn rebate again |
| Price moves 3% further against us (2x the entry threshold) | Market order — cut the loss immediately |
| Position held more than 3 minutes with no recovery | Market order — time stop, do not stay stuck if the shock is a genuine repricing |

---

## PassiveMarketMaker — How It Works

During the calm stretches of the day when no spikes are happening, the spike fade strategy sits idle. We still need to submit **200+ orders per day** to meet the competition requirement.

The passive market maker runs on AAPL and MSFT and posts a 1-lot limit buy just inside the best bid and a 1-lot limit sell just inside the best ask. Every 15 seconds it cancels both quotes and reposts at the updated prices to stay at the top of the book. Each cancel-and-repost cycle submits 2 new orders. Over a 5-hour session this generates approximately **240 order submissions** — enough to clear the 200-order minimum even on a day with very few spikes.

It automatically pauses on a ticker if a directional position of 2 or more lots is already open (e.g. from the spike fade), so the two strategies do not conflict with each other.

---

## Tickers
| Strategy | Tickers |
|---|---|
| Spike Fade | AAPL, MSFT, NVDA, JPM, GS |
| Passive Market Maker | AAPL, MSFT |

---

## Files
| File | Description |
|---|---|
| `run.py` | Main entry point — launches all strategy threads |
| `spike_fade.py` | Primary strategy — detects and fades spikes/dips |
| `passive_market_maker.py` | Order count engine — light market making during calm periods |
| `risk_manager.py` | Position limits, loss limits, buying power checks |
| `utils.py` | Shared helpers — order submission, price fetching, logging |
| `initiator.cfg` | FIX protocol connection config |
