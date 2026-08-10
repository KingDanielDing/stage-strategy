"""
daily_stage_v4.py — Extended version with MACD momentum filter for BUY signals.

New in v4:
  - MACD (12/26/9) computed on daily close prices
  - Three selectable MACD filter modes gating BUY signals:
      "off"               — no filter (baseline = v2 behaviour)
      "histogram_positive" — MACD Histogram > 0
      "histogram_rising"   — MACD Histogram > 0 AND Histogram[today] > Histogram[yesterday]
      "line_crossover"     — MACD Line > Signal Line
  - Comparison mode: runs all four modes side-by-side on the same ticker

Usage:
    python daily_stage_v4.py AAPL
    python daily_stage_v4.py NVDA 2022-01-01 2026-12-31
"""

import sys
import yfinance as yf
import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# 1. Data retrieval
# ---------------------------------------------------------------------------
def retrieve_data(filename: str, end_date: str, start_date: str) -> pd.DataFrame:
    """
    Fetch daily OHLCV from yfinance.
    Returns DataFrame with columns [Date, Open, High, Low, Close, Volume].
    """
    df = yf.download(filename, start=start_date, end=end_date, interval="1d")
    df = df.reset_index()

    if "Datetime" in df.columns:
        df = df.rename(columns={"Datetime": "Date"})
    if df.index.name == "Date":
        df = df.reset_index()

    df = df[["Date", "Open", "High", "Low", "Close", "Volume"]]
    # Drop the multi-level column header yfinance sometimes adds
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)

    # Drop zero-volume rows
    df = df[df["Volume"] != 0].copy()

    # Round prices to 3 decimal places
    df.loc[:, ["Open", "High", "Low", "Close"]] = (
        df[["Open", "High", "Low", "Close"]].round(3)
    )
    return df


# ---------------------------------------------------------------------------
# 2. Fair Value Gap (FVG) detection
# ---------------------------------------------------------------------------
def find_fvgs(mystock: pd.DataFrame) -> pd.DataFrame:
    """
    Scan a 3-candle window for Fair Value Gaps with a 1 % buffer.
    Returns a DataFrame with columns [Date, FVG_Type, Bench, Benchmark].
    """
    df = mystock.sort_values("Date").reset_index(drop=True)
    bullish_fvg = []
    bearish_fvg = []

    for i in range(2, len(df)):
        prev_high = df.loc[i - 2, "High"]
        prev_low = df.loc[i - 2, "Low"]
        current_high = df.loc[i, "High"]
        current_low = df.loc[i, "Low"]

        # Bullish FVG: candle i-2 low > 1% above candle i high
        if prev_low / current_high > 1.01:
            bullish_fvg.append({
                "Date": df.loc[i, "Date"],
                "FVG_Type": "Bullish",
                "Prev_Low": prev_low,
                "Current_High": current_high,
            })

        # Bearish FVG: candle i low > 1% above candle i-2 high
        if current_low / prev_high > 1.01:
            bearish_fvg.append({
                "Date": df.loc[i, "Date"],
                "FVG_Type": "Bearish",
                "Prev_High": prev_high,
                "Current_Low": current_low,
            })

    fvg_df = pd.DataFrame(bullish_fvg + bearish_fvg)

    if fvg_df.empty:
        # No FVGs found — return an empty frame with expected columns
        fvg_df["Bench"] = pd.Series(dtype=float)
        fvg_df["Benchmark"] = pd.Series(dtype=float)
        return fvg_df

    fvg_df["Bench"] = np.where(
        fvg_df["FVG_Type"] == "Bullish",
        fvg_df["Current_High"],
        fvg_df["Current_Low"],
    )
    fvg_df["Benchmark"] = np.where(
        fvg_df["FVG_Type"] == "Bullish",
        fvg_df["Prev_Low"],
        fvg_df["Prev_High"],
    )
    return fvg_df


# ---------------------------------------------------------------------------
# 3. MACD computation
# ---------------------------------------------------------------------------
def compute_macd(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """
    Compute MACD indicators and append them as columns to the DataFrame.

    Adds columns:
      - EMA_Fast        : {fast}-day EMA of Close
      - EMA_Slow        : {slow}-day EMA of Close
      - MACD_Line       : EMA_Fast - EMA_Slow
      - MACD_Signal     : {signal}-day EMA of MACD_Line
      - MACD_Histogram  : MACD_Line - MACD_Signal
    """
    result = df.copy()
    result["EMA_Fast"] = result["Close"].ewm(span=fast, adjust=False).mean()
    result["EMA_Slow"] = result["Close"].ewm(span=slow, adjust=False).mean()
    result["MACD_Line"] = result["EMA_Fast"] - result["EMA_Slow"]
    result["MACD_Signal"] = result["MACD_Line"].ewm(span=signal, adjust=False).mean()
    result["MACD_Histogram"] = result["MACD_Line"] - result["MACD_Signal"]
    return result


# MAP: mode constant → human-readable label (used in comparison table & reason strings)
MACD_MODE_LABEL = {
    "off":                "off",
    "histogram_positive": "Histogram > 0",
    "histogram_rising":   "Histogram > 0 & rising",
    "line_crossover":     "MACD Line > Signal",
}


def check_macd_filter(row: dict, df: pd.DataFrame, idx: int, mode: str) -> bool:
    """
    Evaluate the MACD filter condition for a candidate BUY bar.

    Parameters
    ----------
    row : dict
        The candidate BUY signal row (must contain 'Date').
    df : pd.DataFrame
        The full (undeduplicated) DataFrame containing MACD columns.  Must
        have 'Date' as a column and be sorted chronologically.
    idx : int
        The index within `df` of the candidate BUY row's date.
    mode : str
        One of {"off", "histogram_positive", "histogram_rising", "line_crossover"}.

    Returns
    -------
    bool
        True if the MACD condition is satisfied (or mode == "off").
    """
    if mode == "off":
        return True

    # Ensure the DataFrame carries the needed columns
    if "MACD_Histogram" not in df.columns or "MACD_Line" not in df.columns:
        return True

    macd_hist = df["MACD_Histogram"].iloc[idx]
    macd_line = df["MACD_Line"].iloc[idx]
    macd_signal = df["MACD_Signal"].iloc[idx]

    if mode == "histogram_positive":
        return macd_hist > 0

    if mode == "histogram_rising":
        if idx == 0:
            return macd_hist > 0
        prev_hist = df["MACD_Histogram"].iloc[idx - 1]
        return macd_hist > 0 and macd_hist > prev_hist

    if mode == "line_crossover":
        return macd_line > macd_signal

    return True


# ---------------------------------------------------------------------------
# 4. Merge levels with OHLCV and forward-fill
# ---------------------------------------------------------------------------
def generate_finals(
    mystock: pd.DataFrame,
    fvg_df: pd.DataFrame,
    end_trade_date: str,
    start_trade_date: str = "2022-01-01",
) -> pd.DataFrame:
    """
    Merge stock data with FVG levels and forward-fill Bench / Benchmark so
    every row carries the most recent FVG's levels.
    """
    df = mystock.copy()
    fvg = fvg_df.copy()

    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
    fvg["Date"] = pd.to_datetime(fvg["Date"]).dt.tz_localize(None)

    # Keep only the columns needed for stage assignment
    merged = pd.merge(df, fvg, on="Date", how="left")[
        ["Date", "Close", "High", "Low", "Open", "Benchmark", "Bench"]
    ]

    merged = merged[
        (merged["Date"] >= start_trade_date) & (merged["Date"] <= end_trade_date)
    ].reset_index(drop=True)

    # Forward-fill the levels — they persist until a new FVG appears
    merged["Benchmark"] = merged["Benchmark"].ffill()
    merged["Bench"] = merged["Bench"].ffill()

    return merged


# ---------------------------------------------------------------------------
# 5. Stage assignment
# ---------------------------------------------------------------------------
def stage_assessment(close: float, bench: float, benchmark: float) -> int:
    """
    Map (close, bench, benchmark) → integer Stage ∈ {1, …, 6}.

    Encoding (orientation × close position):

      Position      | bench ≤ benchmark | benchmark ≤ bench
      --------------|:------------------|:------------------
      below band    |        1          |        2
      inside band   |        3          |        4
      above band    |        5          |        6
    """
    if close < bench <= benchmark:
        return 1
    if close <= benchmark <= bench:
        return 2
    if benchmark <= bench < close:
        return 6
    if bench < benchmark <= close:
        return 5
    if bench <= close < benchmark:
        return 3
    if benchmark < close <= bench:
        return 4
    # Fallback (shouldn't normally be reached)
    return 0


def generate_pure_stages(df: pd.DataFrame) -> pd.DataFrame:
    """
    Attach a 'Stage' column to a DataFrame that contains
    [Date, Close, Benchmark, Bench].
    """
    result = df[["Date", "Close", "Benchmark", "Bench"]].copy()
    stages = [0]

    for i in range(1, len(result)):
        stage = stage_assessment(
            close=result["Close"].iloc[i],
            bench=result["Bench"].iloc[i],
            benchmark=result["Benchmark"].iloc[i],
        )
        stages.append(stage)

    result["Stage"] = stages
    return result


def deduplicate_stages(stages_df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only rows where the Stage changes from the previous row.
    Returns a DataFrame with the first row and every subsequent row where
    Stage differs from its predecessor.
    """
    df = stages_df.copy()
    # Keep row if it's the first row OR the stage differs from the previous row
    mask = (df["Stage"].shift() != df["Stage"])
    return df[mask].reset_index(drop=True)


# ---------------------------------------------------------------------------
# 6. Trading strategy — stage-pattern BUY / HOLD / SELL  (with MACD filter)
# ---------------------------------------------------------------------------
#
# ── BUY SIGNALS ──────────────────────────────────────────────────────────
# A BUY fires when the deduplicated stage sequence ends with one of:
#
#   Pattern     Meaning
#   ──────────  ──────────────────────────────
#   (1, 5)      Stage 1 → Stage 5            (below→above, gap up)
#   (3, 5)      Stage 3 → Stage 5            (inside→above, gap up)
#   (2, 4)      Stage 2 → Stage 4            (below→inside, gap down)
#   (1, 6)      Stage 1 → Stage 6            (below(gap up)→above(gap down))
#   (1, 3)      Stage 1 → Stage 3            (below→inside, gap up)
#   (2, 6)      Stage 2 → Stage 6            (below(gap down)→above(gap down))
#   (3, 6)      Stage 3 → Stage 6            (inside(gap up)→above(gap down))
#
# Duration filter:
#   - Default: the LAST stage must have lasted ≥ MIN_STAGE_DURATION days.
#   - Per-pattern overrides via PATTERN_DURATION:
#       (1, 3): checks PRIOR stage (stage 1) duration instead (≥3 days),
#               since stage 3 is naturally multi-day and offers no filter.
#
# MACD filter (NEW in v4):
#   - Only applied to weak "inside-band" patterns in MACD_SELECTIVE_PATTERNS.
#     Strong "above-band" patterns — (1,5), (3,5), (1,6), (2,6), (3,6) —
#     pass through without MACD check.
#   - If a weak pattern matches AND MACD fails → HOLD with reason
#     "pattern (X, Y) (MACD {mode}: failed)"
#
# ── SELL / HOLD LOGIC (after a BUY) ──────────────────────────────────────
#
#   Post-BUY stage | Action
#   ───────────────┼─────────
#   4, 5, 6        | HOLD — enter / stay in rally
#   1, 2, 3        | SELL immediately — pattern failed
#
#   While in rally:
#     - Stay in {4,5,6} → continue HOLDING
#     - Drop to {1,2,3} → SELL (rally ended)
#
#   End-of-data closeout:
#     - If still holding → auto-SELL at last available daily close
#
# ── TUNABLE PARAMETERS ──────────────────────────────────────────────────

BUY_PATTERNS = [
    (1, 5), (3, 5),
    (2, 4),
    (1, 6), (1, 3),
    (2, 6), (3, 6),
]
MIN_STAGE_DURATION = 1  # last stage of pattern must persist ≥ N days (default)

# Per-pattern duration override — (1, 3) noise comes from brief dips below
# the band; requiring the PRIOR stage (stage 1) to last ≥ N days filters whipsaws.
PATTERN_DURATION = {
    (1, 3): {"check_prior": True, "min_days": 3},
}

# ── MACD FILTER (NEW in v4) ──────────────────────────────────────────────
# Change this to test different filter modes:
#   "off"               — no MACD filter (baseline)
#   "histogram_positive" — MACD Histogram > 0
#   "histogram_rising"   — MACD Histogram > 0 AND rising
#   "line_crossover"     — MACD Line > Signal Line
MACD_FILTER_MODE = "histogram_positive"

# ── SELECTIVE MACD (v4) ───────────────────────────────────────────────────
# Only apply MACD to these weaker "inside-band" patterns:
#   (1,3) — below→inside, gap up      (price didn't clear the band)
#   (2,4) — below→inside, gap down    (price didn't clear the band)
#
# Stronger "above-band" patterns pass through without MACD:
#   (1,5), (3,5), (1,6), (2,6), (3,6) — price crossed above the band
MACD_SELECTIVE_PATTERNS = {(1, 3), (2, 4)}


def check_pattern(tail: list, pattern: tuple) -> bool:
    """Return True if `tail` ends with `pattern`."""
    tail_tuple = tuple(tail[-len(pattern):])
    return tail_tuple == pattern


def _stage_duration_days(full_stages: pd.DataFrame, stage_date: pd.Timestamp, stage_value: int) -> int:
    """
    Count how many consecutive calendar days the full (undeduplicated) daily data
    stays at `stage_value` starting from `stage_date`.
    """
    full = full_stages.copy()
    full["Date"] = pd.to_datetime(full["Date"])
    mask = (full["Date"] >= stage_date) & (full["Stage"] == stage_value)
    return int(mask.sum())


def _find_date_index(df: pd.DataFrame, target_date) -> int:
    """
    Find the integer row index in `df` where Date == target_date.
    Returns -1 if not found.
    """
    target = pd.to_datetime(target_date)
    matches = df.index[df["Date"] == target]
    if len(matches) == 0:
        return -1
    return int(matches[0])


def generate_trade_signals(
    deduped: pd.DataFrame,
    full_stages: pd.DataFrame,   # undeduplicated daily bars [Date, Close, ..., Stage]
    macd_mode: str = "off",      # NEW: MACD filter mode
) -> pd.DataFrame:
    """
    Walk through the deduplicated stage sequence and produce trade signals.

    A BUY only fires when:
      1. The deduplicated stage tail matches a pattern, AND
      2. The LAST stage of that pattern has lasted ≥ MIN_STAGE_DURATION days
         in the full (undeduplicated) data, AND
      3. (NEW in v4) The MACD filter condition is satisfied on the BUY date.

    Returns a DataFrame with columns:
      Date, Close, Stage, Signal (BUY / HOLD / SELL), Reason
    """
    stages = deduped["Stage"].astype(int).tolist()
    dates  = deduped["Date"].tolist()
    closes = deduped["Close"].tolist()
    n = len(stages)

    records = []          # list of dicts for output rows
    state = "SEARCHING"   # SEARCHING → IN_POSITION → SEARCHING

    i = 0
    while i < n:
        row = {
            "Date":   dates[i],
            "Close":  closes[i],
            "Stage":  stages[i],
            "Signal": "HOLD",
            "Reason": "",
        }

        if state == "SEARCHING":
            # Build the tail of stages seen so far (up to current index)
            tail = stages[:i + 1]

            # --- check buy patterns ---
            matched = None
            for pat in BUY_PATTERNS:
                if len(tail) >= len(pat) and check_pattern(tail, pat):
                    # longer patterns take priority when multiple match simultaneously
                    if matched is None or len(pat) > len(matched):
                        matched = pat

            if matched is not None:
                # ── Duration filter ──
                dur_config = PATTERN_DURATION.get(matched)

                if dur_config is not None and dur_config.get("check_prior"):
                    # Check the PRIOR stage (second-to-last in pattern)
                    prior_stage_value = matched[-2]
                    prior_stage_date = pd.to_datetime(dates[i - 1])
                    duration = _stage_duration_days(full_stages, prior_stage_date, prior_stage_value)
                    min_dur = dur_config["min_days"]
                    dur_label = f"prior stage {prior_stage_value}"
                else:
                    # Default: check the LAST stage
                    last_stage_value = matched[-1]
                    last_stage_date = pd.to_datetime(dates[i])
                    duration = _stage_duration_days(full_stages, last_stage_date, last_stage_value)
                    min_dur = dur_config["min_days"] if dur_config else MIN_STAGE_DURATION
                    dur_label = f"stage {last_stage_value}"

                if duration >= min_dur:
                    # ── MACD filter (NEW in v4) ──
                    # Only apply to weaker patterns; strong patterns skip MACD
                    if macd_mode != "off" and matched in MACD_SELECTIVE_PATTERNS:
                        date_idx = _find_date_index(full_stages, dates[i])
                        if not check_macd_filter(row, full_stages, date_idx, macd_mode):
                            mode_label = MACD_MODE_LABEL.get(macd_mode, macd_mode)
                            row["Signal"] = "HOLD"
                            row["Reason"] = f"pattern {matched} (MACD {mode_label}: failed)"
                            records.append(row)
                            i += 1
                            continue

                    # ── All filters passed → BUY ──
                    row["Signal"] = "BUY"
                    row["Reason"] = f"pattern {matched}"
                    state = "IN_POSITION"
                    records.append(row)
                    i += 1
                    continue
                else:
                    row["Signal"] = "HOLD"
                    row["Reason"] = f"pattern {matched} ({dur_label} duration {duration}d < {min_dur}d)"
                    records.append(row)
                    i += 1
                    continue

            # No pattern matched
            row["Signal"] = "HOLD"
            row["Reason"] = "searching"
            records.append(row)
            i += 1
            continue

        # ----------------------------------------------------------------
        # state == "IN_POSITION"  — we already bought, now manage the trade
        # ----------------------------------------------------------------
        current_stage = stages[i]

        # Stages 4, 5, 6 are "strong" stages — keep holding
        if current_stage in (4, 5, 6):
            # Rally started — HOLD
            row["Signal"] = "HOLD"
            row["Reason"] = f"in rally (stage {current_stage})"
            records.append(row)

            # advance while stage stays 4, 5 or 6
            i += 1
            while i < n and stages[i] in (4, 5, 6):
                records.append({
                    "Date":   dates[i],
                    "Close":  closes[i],
                    "Stage":  stages[i],
                    "Signal": "HOLD",
                    "Reason": f"in rally (stage {stages[i]})",
                })
                i += 1

            # First stage outside {4,5,6} after the rally (or end of data)
            if i < n:
                records.append({
                    "Date":   dates[i],
                    "Close":  closes[i],
                    "Stage":  stages[i],
                    "Signal": "SELL",
                    "Reason": f"rally ended (stage {stages[i]})",
                })
                state = "SEARCHING"
                i += 1
                continue
            else:
                # End of data while still in rally — position stays open
                state = "SEARCHING"
                break
        else:
            # Next stage is 1,2,3 — pattern failed; SELL immediately
            row["Signal"] = "SELL"
            row["Reason"] = f"pattern failed (stage {current_stage} ∉ {{4,5,6}})"
            records.append(row)
            state = "SEARCHING"
            i += 1
            continue

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# 7. Backtest — simulate trades and compute performance
# ---------------------------------------------------------------------------
def backtest_trades(
    signal_df: pd.DataFrame,
    raw_df: pd.DataFrame,              # full daily bars (for closeout price)
    initial_capital: float = 10000.0,
):
    """
    Walk through signal rows and simulate BUY/SELL round-trips.
    All entries use 100% of available cash.

    Returns a list of trade dicts and the final cash balance.
    """
    trades = []
    cash = initial_capital
    shares = 0.0
    buy_price = 0.0
    buy_date = None

    for _, row in signal_df.iterrows():
        sig = row["Signal"]

        # ── ENTRY ──
        if sig == "BUY" and shares == 0:
            buy_price = row["Close"]
            buy_date = row["Date"]
            shares = cash / buy_price
            cash = 0.0
            continue

        # ── EXIT ──
        if sig == "SELL" and shares > 0:
            sell_price = row["Close"]
            sell_date = row["Date"]
            cash = shares * sell_price
            pnl_pct = (sell_price / buy_price - 1) * 100

            trades.append({
                "Entry Date":   buy_date,
                "Entry Price":  round(buy_price, 2),
                "Exit Date":    sell_date,
                "Exit Price":   round(sell_price, 2),
                "PnL %":        round(pnl_pct, 2),
                "Capital":      round(cash, 2),
            })
            shares = 0.0
            buy_price = 0.0
            buy_date = None
            continue

    # If still holding shares at end, close out using the LAST raw daily close
    if shares > 0:
        sell_price = raw_df["Close"].iloc[-1]
        sell_date = pd.to_datetime(raw_df["Date"].iloc[-1])
        cash = shares * sell_price
        pnl_pct = (sell_price / buy_price - 1) * 100

        trades.append({
            "Entry Date":   buy_date,
            "Entry Price":  round(buy_price, 2),
            "Exit Date":    sell_date,
            "Exit Price":   round(sell_price, 2),
            "PnL %":        round(pnl_pct, 2),
            "Capital":      round(cash, 2),
        })
        shares = 0.0
        buy_price = 0.0
        buy_date = None

    return trades, cash


# ---------------------------------------------------------------------------
# 8. Chart — price with BUY / SELL annotations
# ---------------------------------------------------------------------------
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def plot_trades(
    raw_df: pd.DataFrame,          # full daily OHLCV (not deduped)
    signals: pd.DataFrame,        # trade signals DataFrame
    trades: list[dict],           # backtest trades
    ticker: str,
    stats: Optional[dict] = None,    # {initial_capital, final_capital, total_return, ...}
    macd_mode: str = "off",        # NEW: for chart title suffix
):
    """
    Draw a closing-price chart with BUY / SELL markers overlaid and
    a performance-stats box in the lower-right corner.
    """
    # Prepare data
    price = raw_df.copy()
    price["Date"] = pd.to_datetime(price["Date"])

    sig = signals.copy()
    sig["Date"] = pd.to_datetime(sig["Date"])

    buys  = sig[sig["Signal"] == "BUY"]
    sells = sig[sig["Signal"] == "SELL"]

    fig, ax = plt.subplots(figsize=(16, 7))

    # ── Price line ──
    ax.plot(price["Date"], price["Close"], linewidth=0.8, color="#2c3e50", label="Close")

    # ── BUY markers (▲ green) ──
    if not buys.empty:
        ax.scatter(
            buys["Date"], buys["Close"],
            marker="^", s=100, color="#27ae60", edgecolors="white",
            linewidths=0.5, zorder=5, label="BUY",
        )

    # ── SELL markers (▼ red) ──
    if not sells.empty:
        ax.scatter(
            sells["Date"], sells["Close"],
            marker="v", s=100, color="#e74c3c", edgecolors="white",
            linewidths=0.5, zorder=5, label="SELL",
        )

    # ── Connecting lines between matched entry/exit pairs ──
    for t in trades:
        entry_date = pd.to_datetime(t["Entry Date"])
        exit_date = pd.to_datetime(t["Exit Date"])
        entry_price = t["Entry Price"]
        exit_price = t["Exit Price"]
        color = "#27ae60" if t["PnL %"] > 0 else "#e74c3c"
        ax.plot(
            [entry_date, exit_date],
            [entry_price, exit_price],
            linestyle="--", linewidth=0.6, color=color, alpha=0.5,
        )

    # ── Performance stats box ──
    if stats is not None:
        lines = [
            f"Initial capital : ${stats['initial']:,.2f}",
            f"Final capital   : ${stats['final']:,.2f}",
            f"Total return    : {stats['total_return']:.2f}%",
            f"Total trades    : {stats['n_trades']}",
            f"Wins            : {stats['wins']}",
            f"Losses          : {stats['losses']}",
            f"Win rate        : {stats['win_rate']:.1f}%",
            f"Avg win         : {stats['avg_win']:.2f}%",
            f"Avg loss        : {stats['avg_loss']:.2f}%",
        ]
        text = "\n".join(lines)

        # Place box at lower-right, anchored to (0.98, 0.02) in axes coords
        props = dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="#7f8c8d",
                     alpha=0.92)
        ax.text(
            0.98, 0.02, text, transform=ax.transAxes, fontsize=9,
            fontfamily="monospace", verticalalignment="bottom",
            horizontalalignment="right", bbox=props,
        )

    # ── Styling ──
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    mode_suffix = f" [MACD: {MACD_MODE_LABEL.get(macd_mode, macd_mode)}]" if macd_mode != "off" else ""
    ax.set_title(f"{ticker}  —  Stage-Pattern Strategy (v4){mode_suffix}", fontsize=14, fontweight="bold")
    ax.set_ylabel("Price (USD)")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    return fig


# ---------------------------------------------------------------------------
# 9. Main pipeline
# ---------------------------------------------------------------------------
def get_daily_stage(ticker: str, start_date: str, end_date: str):
    """
    Full pipeline: fetch data → find FVGs → compute MACD → assign daily stages.

    Returns a DataFrame with:
      [Date, Close, Benchmark, Bench, EMA_Fast, EMA_Slow, MACD_Line,
       MACD_Signal, MACD_Histogram, Stage]
    """
    print(f"Fetching daily data for {ticker} ({start_date} → {end_date}) …")
    raw = retrieve_data(ticker, end_date, start_date)

    if raw.empty:
        print("⚠️  No data returned from yfinance.")
        return None

    print(f"  → {len(raw)} daily bars")
    print("Detecting Fair Value Gaps …")
    fvgs = find_fvgs(raw)
    print(f"  → {len(fvgs)} FVGs found")

    final_df = generate_finals(raw, fvgs, end_date, start_date)

    # NEW: Compute MACD before stage assignment (uses Close prices)
    print("Computing MACD (12/26/9) …")
    final_df = compute_macd(final_df)

    stages_df = generate_pure_stages(final_df)

    # Carry MACD columns into the stages DataFrame
    for col in ["EMA_Fast", "EMA_Slow", "MACD_Line", "MACD_Signal", "MACD_Histogram"]:
        stages_df[col] = final_df[col].values

    return stages_df


# ---------------------------------------------------------------------------
# 10. Helper — compute stats dict from backtest output
# ---------------------------------------------------------------------------
def _compute_stats(trades: list, final_capital: float, initial_capital: float = 10000.0) -> dict:
    """Build a stats dict from the backtest result."""
    total_return = (final_capital / initial_capital - 1) * 100
    win_count = sum(1 for t in trades if t["PnL %"] > 0)
    loss_count = sum(1 for t in trades if t["PnL %"] <= 0)
    avg_win = (
        sum(t["PnL %"] for t in trades if t["PnL %"] > 0) / win_count
        if win_count else 0
    )
    avg_loss = (
        sum(t["PnL %"] for t in trades if t["PnL %"] <= 0) / loss_count
        if loss_count else 0
    )
    return {
        "initial":      initial_capital,
        "final":        final_capital,
        "total_return": total_return,
        "n_trades":     len(trades),
        "wins":         win_count,
        "losses":       loss_count,
        "win_rate":     win_count / len(trades) * 100 if trades else 0,
        "avg_win":      avg_win,
        "avg_loss":     avg_loss,
    }


# ---------------------------------------------------------------------------
# 11. Main entry point  —  Comparison Mode
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    ticker = sys.argv[1].upper()
    start = sys.argv[2] if len(sys.argv) > 2 else "2022-01-01"
    end = sys.argv[3] if len(sys.argv) > 3 else pd.Timestamp.today().strftime("%Y-%m-%d")

    # -- Run the pipeline *once* (stages + MACD are shared) --
    result = get_daily_stage(ticker, start, end)

    if result is None:
        sys.exit(1)

    deduped = deduplicate_stages(result)

    print("\n" + "=" * 60)
    print(f"  Daily stage transitions for {ticker}")
    print(f"  ({len(result)} rows -> {len(deduped)} rows after dedup)")
    print("=" * 60)
    print(deduped.to_string(index=True, header=True))

    # -- Comparison: run ALL FOUR MACD modes --
    MACD_MODES = ["off", "histogram_positive", "histogram_rising", "line_crossover"]
    initial_capital = 10000.0
    all_stats = []

    print("\n" + "=" * 80)
    print(f"  MACD FILTER COMPARISON - {ticker}")
    print("=" * 80)

    for mode in MACD_MODES:
        signals = generate_trade_signals(deduped, full_stages=result, macd_mode=mode)
        trades, final_capital = backtest_trades(signals, raw_df=result)
        stats = _compute_stats(trades, final_capital, initial_capital)

        mode_label = MACD_MODE_LABEL.get(mode, mode)
        all_stats.append({"mode": mode, "label": mode_label, "stats": stats, "signals": signals, "trades": trades})

        print(f"\n  -- Mode: {mode_label} --")
        print(f"     Trades: {stats['n_trades']:>4}  |  "
              f"Win rate: {stats['win_rate']:>6.1f}%  |  "
              f"Total return: {stats['total_return']:>7.2f}%  |  "
              f"Avg win: {stats['avg_win']:>6.2f}%  |  "
              f"Avg loss: {stats['avg_loss']:>6.2f}%")

    # -- Side-by-side summary table --
    print("\n" + "=" * 80)
    print(f"  MACD FILTER COMPARISON SUMMARY - {ticker}")
    print("=" * 80)
    print(f"  {'Mode':<24s} {'Trades':>7s} {'Win Rate':>9s} {'Total Ret':>10s} "
          f"{'Avg Win':>9s} {'Avg Loss':>9s}")
    print(f"  {'-' * 24} {'-' * 7} {'-' * 9} {'-' * 10} {'-' * 9} {'-' * 9}")

    for entry in all_stats:
        s = entry["stats"]
        print(f"  {entry['label']:<24s} {s['n_trades']:>7d} {s['win_rate']:>8.1f}% "
              f"{s['total_return']:>9.2f}% {s['avg_win']:>8.2f}% {s['avg_loss']:>8.2f}%")

    # -- Analysis: which mode wins? --
    print(f"\n  {'-' * 76}")
    best_by_wr = max(all_stats, key=lambda x: x["stats"]["win_rate"])
    best_by_ret = max(all_stats, key=lambda x: x["stats"]["total_return"])
    print(f"  Best win rate   : {best_by_wr['label']:<24s} ({best_by_wr['stats']['win_rate']:.1f}%)")
    print(f"  Best total return: {best_by_ret['label']:<24s} ({best_by_ret['stats']['total_return']:.2f}%)")

    # Count how many trades were filtered by each MACD mode vs. baseline
    baseline_n = all_stats[0]["stats"]["n_trades"]
    for entry in all_stats[1:]:
        filtered = baseline_n - entry["stats"]["n_trades"]
        delta_wr = entry["stats"]["win_rate"] - all_stats[0]["stats"]["win_rate"]
        delta_ret = entry["stats"]["total_return"] - all_stats[0]["stats"]["total_return"]
        emoji_wr = "OK" if delta_wr > 0 else ("--" if delta_wr == 0 else "XX")
        emoji_ret = "OK" if delta_ret > 0 else ("--" if delta_ret == 0 else "XX")
        print(f"     {entry['label']:<20s}: filtered {filtered:>2d} trades  "
              f"dWR {emoji_wr} {delta_wr:+.1f}%  "
              f"dRet {emoji_ret} {delta_ret:+.2f}%")

    # -- Detailed trade log & chart for the ACTIVE mode (MACD_FILTER_MODE) --
    active_mode = MACD_FILTER_MODE
    print(f"\n\n{'=' * 80}")
    print(f"  DETAILED TRADE LOG - Mode: {MACD_MODE_LABEL.get(active_mode, active_mode)}")
    print(f"{'=' * 80}")

    active_entry = next((e for e in all_stats if e["mode"] == active_mode), all_stats[0])
    active_signals = active_entry["signals"]
    active_trades = active_entry["trades"]

    print(active_signals[["Date", "Stage", "Signal", "Reason"]].to_string(index=True, header=True))

    print(f"\n{'=' * 80}")
    print(f"  Backtest results for {ticker}")
    print(f"{'=' * 80}")

    if active_trades:
        trades_df = pd.DataFrame(active_trades)
        print(trades_df.to_string(index=True, header=True))
        print()

        s = active_entry["stats"]
        print(f"  Initial capital : ${s['initial']:,.2f}")
        print(f"  Final capital   : ${s['final']:,.2f}")
        print(f"  Total return    : {s['total_return']:.2f}%")
        print(f"  Total trades    : {s['n_trades']}")
        print(f"  Wins           : {s['wins']}")
        print(f"  Losses          : {s['losses']}")
        print(f"  Win rate        : {s['win_rate']:.1f}%")
        print(f"  Avg win         : {s['avg_win']:.2f}%")
        print(f"  Avg loss        : {s['avg_loss']:.2f}%")

        # -- Chart --
        chart_file = f"{ticker}_trades_v4.png"
        fig = plot_trades(result, active_signals, active_trades, ticker,
                          stats=s, macd_mode=active_mode)
        fig.savefig(chart_file, dpi=150)
        print(f"\n  Chart saved to: {chart_file}")
        # Display the chart if running interactively
        try:
            plt.show()
        except Exception:
            pass
    else:
        print("  No trades were generated during this period.")
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    ticker = sys.argv[1].upper()
    start = sys.argv[2] if len(sys.argv) > 2 else "2022-01-01"
    end = sys.argv[3] if len(sys.argv) > 3 else pd.Timestamp.today().strftime("%Y-%m-%d")

    # -- Run the pipeline *once* (stages + MACD are shared) --
    result = get_daily_stage(ticker, start, end)

    if result is None:
        sys.exit(1)

    deduped = deduplicate_stages(result)

    print("\n" + "=" * 60)
    print(f"  Daily stage transitions for {ticker}")
    print(f"  ({len(result)} rows -> {len(deduped)} rows after dedup)")
    print("=" * 60)
    print(deduped.to_string(index=True, header=True))

    # -- Comparison: run ALL FOUR MACD modes --
    MACD_MODES = ["off", "histogram_positive", "histogram_rising", "line_crossover"]
    initial_capital = 10000.0
    all_stats = []

    print("\n" + "=" * 80)
    print(f"  MACD FILTER COMPARISON - {ticker}")
    print("=" * 80)

    for mode in MACD_MODES:
        signals = generate_trade_signals(deduped, full_stages=result, macd_mode=mode)
        trades, final_capital = backtest_trades(signals, raw_df=result)
        stats = _compute_stats(trades, final_capital, initial_capital)

        mode_label = MACD_MODE_LABEL.get(mode, mode)
        all_stats.append({"mode": mode, "label": mode_label, "stats": stats, "signals": signals, "trades": trades})

        print(f"\n  -- Mode: {mode_label} --")
        print(f"     Trades: {stats['n_trades']:>4}  |  "
              f"Win rate: {stats['win_rate']:>6.1f}%  |  "
              f"Total return: {stats['total_return']:>7.2f}%  |  "
              f"Avg win: {stats['avg_win']:>6.2f}%  |  "
              f"Avg loss: {stats['avg_loss']:>6.2f}%")

    # -- Side-by-side summary table --
    print("\n" + "=" * 80)
    print(f"  MACD FILTER COMPARISON SUMMARY - {ticker}")
    print("=" * 80)
    print(f"  {'Mode':<24s} {'Trades':>7s} {'Win Rate':>9s} {'Total Ret':>10s} "
          f"{'Avg Win':>9s} {'Avg Loss':>9s}")
    print(f"  {'-' * 24} {'-' * 7} {'-' * 9} {'-' * 10} {'-' * 9} {'-' * 9}")

    for entry in all_stats:
        s = entry["stats"]
        print(f"  {entry['label']:<24s} {s['n_trades']:>7d} {s['win_rate']:>8.1f}% "
              f"{s['total_return']:>9.2f}% {s['avg_win']:>8.2f}% {s['avg_loss']:>8.2f}%")

    # -- Analysis: which mode wins? --
    print(f"\n  {'-' * 76}")
    best_by_wr = max(all_stats, key=lambda x: x["stats"]["win_rate"])
    best_by_ret = max(all_stats, key=lambda x: x["stats"]["total_return"])
    print(f"  Best win rate   : {best_by_wr['label']:<24s} ({best_by_wr['stats']['win_rate']:.1f}%)")
    print(f"  Best total return: {best_by_ret['label']:<24s} ({best_by_ret['stats']['total_return']:.2f}%)")

    # Count how many trades were filtered by each MACD mode vs. baseline
    baseline_n = all_stats[0]["stats"]["n_trades"]
    for entry in all_stats[1:]:
        filtered = baseline_n - entry["stats"]["n_trades"]
        delta_wr = entry["stats"]["win_rate"] - all_stats[0]["stats"]["win_rate"]
        delta_ret = entry["stats"]["total_return"] - all_stats[0]["stats"]["total_return"]
        emoji_wr = "OK" if delta_wr > 0 else ("--" if delta_wr == 0 else "XX")
        emoji_ret = "OK" if delta_ret > 0 else ("--" if delta_ret == 0 else "XX")
        print(f"     {entry['label']:<20s}: filtered {filtered:>2d} trades  "
              f"dWR {emoji_wr} {delta_wr:+.1f}%  "
              f"dRet {emoji_ret} {delta_ret:+.2f}%")

    # -- Detailed trade log & chart for the ACTIVE mode (MACD_FILTER_MODE) --
    active_mode = MACD_FILTER_MODE
    print(f"\n\n{'=' * 80}")
    print(f"  DETAILED TRADE LOG - Mode: {MACD_MODE_LABEL.get(active_mode, active_mode)}")
    print(f"{'=' * 80}")

    active_entry = next((e for e in all_stats if e["mode"] == active_mode), all_stats[0])
    active_signals = active_entry["signals"]
    active_trades = active_entry["trades"]

    print(active_signals[["Date", "Stage", "Signal", "Reason"]].to_string(index=True, header=True))

    print(f"\n{'=' * 80}")
    print(f"  Backtest results for {ticker}")
    print(f"{'=' * 80}")

    if active_trades:
        trades_df = pd.DataFrame(active_trades)
        print(trades_df.to_string(index=True, header=True))
        print()

        s = active_entry["stats"]
        print(f"  Initial capital : ${s['initial']:,.2f}")
        print(f"  Final capital   : ${s['final']:,.2f}")
        print(f"  Total return    : {s['total_return']:.2f}%")
        print(f"  Total trades    : {s['n_trades']}")
        print(f"  Wins           : {s['wins']}")
        print(f"  Losses          : {s['losses']}")
        print(f"  Win rate        : {s['win_rate']:.1f}%")
        print(f"  Avg win         : {s['avg_win']:.2f}%")
        print(f"  Avg loss        : {s['avg_loss']:.2f}%")

        # -- Chart --
        chart_file = f"{ticker}_trades_v4.png"
        fig = plot_trades(result, active_signals, active_trades, ticker,
                          stats=s, macd_mode=active_mode)
        fig.savefig(chart_file, dpi=150)
        print(f"\n  Chart saved to: {chart_file}")
        # Display the chart if running interactively
        try:
            plt.show()
        except Exception:
            pass
    else:
        print("  No trades were generated during this period.")
