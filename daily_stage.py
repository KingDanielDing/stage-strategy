"""
daily_stage.py — Standalone script to calculate the daily FVG Stage (1–6)
for a given stock ticker using yfinance.

Usage:
    python daily_stage.py AAPL
    python daily_stage.py NVDA 2022-01-01 2026-12-31
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

    # Ensure both gap-direction columns exist (a stock may have only one type)
    for _col in ["Current_High", "Current_Low", "Prev_High", "Prev_Low"]:
        if _col not in fvg_df.columns:
            fvg_df[_col] = np.nan

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
# 3. Merge levels with OHLCV and forward-fill
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
# 4. Stage assignment
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
# 5. Trading strategy — stage-pattern BUY / HOLD / SELL
# ---------------------------------------------------------------------------
#
# ── BUY SIGNALS ──────────────────────────────────────────────────────────
# A BUY fires when the deduplicated stage sequence ends with one of:
#
#   Pattern     Meaning
#   ──────────  ──────────────────────────────
#   (1, 5)      Stage 1 → Stage 5            (below→above, gap up)
#   (3, 5)      Stage 3 → Stage 5            (inside→above, gap up)
#   (1, 3, 5)   Stage 1 → 3 → 5             (below→inside→above)
#   (2, 4)      Stage 2 → Stage 4            (below→inside, gap down)
#   (1, 4)      Stage 1 → Stage 4            (below(gap up)→inside(gap down))
#   (1, 2, 4)   Stage 1 → 2 → 4             (below↑→below↓→inside↓)
#   (1, 6)      Stage 1 → Stage 6            (below(gap up)→above(gap down))
#   (1, 3)      Stage 1 → Stage 3            (below→inside, gap up)
#
# Priority: longer patterns win when multiple match simultaneously
#   e.g. (1, 3, 5) > (1, 5) / (3, 5);  (1, 2, 4) > (1, 4) / (2, 4)
#
# Duration filter: the LAST stage of the pattern must have lasted
# ≥ MIN_STAGE_DURATION calendar days in the undeduplicated daily data.
# If not, the signal is suppressed (shown as HOLD with reason).
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
# ── TUNABLE PARAMETER ────────────────────────────────────────────────────

BUY_PATTERNS = [(1, 5), (3, 5), (1, 3, 5), (1, 2, 4), (1, 4), (2, 4), (1, 6), (1, 3)]
MIN_STAGE_DURATION = 1  # last stage of pattern must persist ≥ N days


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


def generate_trade_signals(
    deduped: pd.DataFrame,
    full_stages: pd.DataFrame,   # undeduplicated daily bars [Date, Close, ..., Stage]
) -> pd.DataFrame:
    """
    Walk through the deduplicated stage sequence and produce trade signals.

    A BUY only fires when:
      1. The deduplicated stage tail matches a pattern, AND
      2. The LAST stage of that pattern has lasted ≥ MIN_STAGE_DURATION days
         in the full (undeduplicated) data.

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
                    # longer patterns (1,3,5) take priority over (1,5)/(3,5)
                    if matched is None or len(pat) > len(matched):
                        matched = pat

            if matched is not None:
                # Validate minimum duration of the LAST stage in the pattern
                last_stage_value = matched[-1]
                last_stage_date = pd.to_datetime(dates[i])
                duration = _stage_duration_days(full_stages, last_stage_date, last_stage_value)

                if duration >= MIN_STAGE_DURATION:
                    row["Signal"] = "BUY"
                    row["Reason"] = f"pattern {matched}"
                    state = "IN_POSITION"
                    records.append(row)
                    i += 1
                    continue
                else:
                    row["Signal"] = "HOLD"
                    row["Reason"] = f"pattern {matched} (duration {duration}d < {MIN_STAGE_DURATION}d)"
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

            # First stage outside {5,6} after the rally (or end of data)
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
# 6. Backtest — simulate trades and compute performance
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
# 7. Chart — price with BUY / SELL annotations
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
    ax.set_title(f"{ticker}  —  Stage-Pattern Strategy", fontsize=14, fontweight="bold")
    ax.set_ylabel("Price (USD)")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    return fig


# ---------------------------------------------------------------------------
# 9. Main entry point
# ---------------------------------------------------------------------------
def get_daily_stage(ticker: str, start_date: str, end_date: str):
    """
    Full pipeline: fetch data → find FVGs → assign daily stages.
    Returns a DataFrame with [Date, Close, Benchmark, Bench, Stage].
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
    stages_df = generate_pure_stages(final_df)

    return stages_df


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    ticker = sys.argv[1].upper()
    start = sys.argv[2] if len(sys.argv) > 2 else "2022-01-01"
    end = sys.argv[3] if len(sys.argv) > 3 else pd.Timestamp.today().strftime("%Y-%m-%d")

    result = get_daily_stage(ticker, start, end)

    if result is not None:
        # ── 1. Deduplicated stage transitions ──
        deduped = deduplicate_stages(result)

        print("\n" + "=" * 60)
        print(f"  Daily stage transitions for {ticker}")
        print(f"  ({len(result)} rows → {len(deduped)} rows after dedup)")
        print("=" * 60)
        print(deduped.to_string(index=True, header=True))

        # ── 2. Generate trade signals ──
        signals = generate_trade_signals(deduped, full_stages=result)

        print("\n" + "=" * 60)
        print(f"  Trade signals for {ticker}")
        print("=" * 60)
        print(signals[["Date", "Stage", "Signal", "Reason"]].to_string(index=True, header=True))

        # ── 3. Backtest ──
        trades, final_capital = backtest_trades(signals, raw_df=result)
        initial_capital = 10000.0

        print("\n" + "=" * 60)
        print(f"  Backtest results for {ticker}")
        print("=" * 60)

        if trades:
            trades_df = pd.DataFrame(trades)
            print(trades_df.to_string(index=True, header=True))
            print()

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

            print(f"  💰 Initial capital : ${initial_capital:,.2f}")
            print(f"  💰 Final capital   : ${final_capital:,.2f}")
            print(f"  📈 Total return    : {total_return:.2f}%")
            print(f"  📊 Total trades    : {len(trades)}")
            print(f"  ✅ Wins           : {win_count}")
            print(f"  ❌ Losses          : {loss_count}")
            print(f"  🎯 Win rate        : {win_count / len(trades) * 100:.1f}%")
            print(f"  📈 Avg win         : {avg_win:.2f}%")
            print(f"  📉 Avg loss        : {avg_loss:.2f}%")

            # ── 4. Chart ──
            stats = {
                "initial":      initial_capital,
                "final":        final_capital,
                "total_return": total_return,
                "n_trades":     len(trades),
                "wins":         win_count,
                "losses":       loss_count,
                "win_rate":     win_count / len(trades) * 100,
                "avg_win":      avg_win,
                "avg_loss":     avg_loss,
            }
            chart_file = f"{ticker}_trades.png"
            fig = plot_trades(result, signals, trades, ticker, stats=stats)
            fig.savefig(chart_file, dpi=150)
            print(f"\n  📊 Chart saved to: {chart_file}")
            # Display the chart if running interactively
            try:
                plt.show()
            except Exception:
                pass
        else:
            print("  No trades were generated during this period.")
