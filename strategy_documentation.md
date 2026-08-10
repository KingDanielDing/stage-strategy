# Stage-Pattern Trading Strategy — Documentation

> **Version**: `daily_stage.py` (standalone, yfinance-based)  
> **Date**: 2026-08-06

---

## 1. Core Concept

This strategy trades based on **Fair Value Gap (FVG) Stage transitions**.  
Every daily bar is assigned an integer **Stage ∈ {1, 2, 3, 4, 5, 6}** that encodes where the closing price sits relative to a price band defined by the most recent FVG.

| Stage | Position relative to FVG band |
|:-----:|:-----------------------------|
| 1 | **Below** the band (FVG gap oriented upward) |
| 2 | **Below** the band (FVG gap oriented downward) |
| 3 | **Inside** the band (FVG gap oriented upward) |
| 4 | **Inside** the band (FVG gap oriented downward) |
| 5 | **Above** the band (FVG gap oriented upward) |
| 6 | **Above** the band (FVG gap oriented downward) |

- **Stages 1–2**: Weakest — price broken below the gap  
- **Stages 3–4**: Transition / inside the gap  
- **Stages 5–6**: Strongest — price above the gap  

---

## 2. BUY Signal Patterns

A BUY is triggered when the **deduplicated** stage sequence ends with one of:

| Pattern | Meaning |
|---------|---------|
| `(1, 5)` | Stage 1 → Stage 5 |
| `(3, 5)` | Stage 3 → Stage 5 |
| `(1, 3, 5)` | Stage 1 → Stage 3 → Stage 5 |
| `(2, 4)` | Stage 2 → Stage 4 |
| `(1, 4)` | Stage 1 → Stage 4 |
| `(1, 2, 4)` | Stage 1 → Stage 2 → Stage 4 |

> **Priority**: Longer patterns take precedence when multiple match simultaneously (e.g. `(1, 2, 4)` > `(1, 4)` > `(2, 4)`; `(1, 3, 5)` > `(1, 5)` / `(3, 5)`).

### Duration Filter

A BUY only fires if the **last stage** of the matched pattern has persisted for **≥ `MIN_STAGE_DURATION`** calendar days in the raw undeduplicated daily data. If the duration is insufficient, the signal shows as HOLD with reason `"pattern (X, Y) (duration Dd < Nd)"`.

| Parameter | Current value |
|-----------|:--:|
| `MIN_STAGE_DURATION` | **1** (effectively no filter) |

> Tunable at the top of section 5 in `daily_stage.py`.

---

## 3. SELL & HOLD Logic

After a BUY, the **first distinct stage** determines the action:

### 3a. If the next stage ∈ {4, 5, 6} → HOLD (enter rally)

The strategy enters a **rally state** and continues holding through any sequence of stages 4, 5, and 6.

> **Rationale**: Stage 4 (inside the gap, gap oriented downward) is still a valuable position.  
> Selling on a `6→4` or `5→4` transition cuts rallies too early.

### 3b. While in rally → SELL when stage drops to {1, 2, 3}

The moment the stage changes to anything **outside** {4, 5, 6}, a SELL is triggered.

| Transition | Action |
|------------|--------|
| Buy → 4/5/6 | HOLD |
| 4/5/6 → 4/5/6 | HOLD (stay in rally) |
| 4/5/6 → **1/2/3** | **SELL** |

### 3c. If post-BUY stage ∈ {1, 2, 3} → SELL immediately

The pattern "failed" — price didn't enter the rally zone. Exit at that bar.

---

## 4. Position Sizing & Backtest

- **100% of available cash** per entry (no fractional sizing)
- If the position is still open at the end of the date range, it is **closed out** at the last available daily close (creates a proper buy/sell pair with PnL)
- Initial capital: **$10,000**
- Transaction costs: not modeled

---

## 5. Pipeline Summary

```
yfinance daily OHLCV
    ↓
FVG detection (3-candle, 1% buffer)
    ↓
Forward-fill Bench / Benchmark levels
    ↓
Stage assignment (1–6) per bar
    ↓
Deduplication (keep only first of consecutive same-stage rows)
    ↓
Pattern matching + duration filter → BUY / HOLD / SELL signals
    ↓
Backtest (round-trip simulation)
    ↓
Chart with performance stats box
```

---

## 6. Usage

```bash
python3 daily_stage.py AAPL                    # default: 2022-01-01 → today
python3 daily_stage.py NVDA 2023-01-01 2026-06-30  # custom range
```

---

## 7. Known Limitations

- yfinance intraday data limited to ~59 days; daily data uses up to ~3 years
- Transaction costs not modeled (no slippage, commission)
- FVG detection uses a hard-coded 1% buffer
- No short-selling — long-only strategy
- Chart saved as `{TICKER}_trades.png`

---

## 8. Evolution Log

| Version | Change |
|---------|--------|
| v1 | Initial build — FVG detection + stage assignment |
| v2 | Deduplicated stages, BUY on `(1,5)`, `(3,5)`, `(1,3,5)` |
| v3 | Added `(1,3)` as BUY- (weak) signal with half-position penalty |
| v4 | Removed `(1,3)` — low quality |
| v5 | Added both BUY- and half-position sizing for BUY-after-BUY- |
| v6 | Removed all BUY- / half-position logic |
| v7 | SELL only on stages outside {5,6} (stage 5 treated as hold) |
| v8 | SELL only on stages outside {4,5,6} (stage 4 treated as hold) |
| v9 | End-of-data closeout uses raw daily close (not signal-row close) |
| v10 | Performance stats box on chart |
| v11 | `(1,3)` removed; BUY: `(1,5)`, `(3,5)`, `(1,3,5)`; HOLD: {4,5,6}; SELL: {1,2,3} |
| v12 | Added MIN_STAGE_DURATION filter (tunable); default 3 days |
| v13 | Added stage-2/4 patterns: `(2,4)`, `(1,4)`, `(1,2,4)` (6 patterns total) |
| v14 | Changed MIN_STAGE_DURATION to 5, then back to 3, then to 1 |
| **current** | 6 BUY patterns; MIN_STAGE_DURATION=1; HOLD: {4,5,6}; SELL: {1,2,3} |