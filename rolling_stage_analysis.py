"""
rolling_stage_analysis.py — Rolling-window Return & Win rate calculator.

Computes the stage-pattern strategy's Return and Win rate over overlapping
1-year windows, stepped monthly, producing two time series that can later
be tested as potential leading indicators for price direction.

Usage:
    python rolling_stage_analysis.py

The pipeline reuses core functions from daily_stage_v4.py (same functions
that daily_stage_v6.py depends on) and mirrors the v6 configuration:
    - MACD_FILTER_MODE = "histogram_positive"
    - INITIAL_CAPITAL = 10000.0
"""

import sys
import warnings
from datetime import datetime
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from daily_stage_v4 import (
    retrieve_data,
    find_fvgs,
    generate_finals,
    compute_macd,
    generate_pure_stages,
    deduplicate_stages,
    generate_trade_signals,
    backtest_trades,
)

# ---------------------------------------------------------------------------
# Configuration (mirrors daily_stage_v6.py)
# ---------------------------------------------------------------------------
MACD_FILTER_MODE = "histogram_positive"
INITIAL_CAPITAL = 10000.0
WINDOW_YEARS = 1
STEP_MONTHS = 1
WARMUP_DAYS = 90       # extra lead-in so MACD & FVG forward-fill are accurate
MIN_BARS = 60           # minimum bars for a slice to be considered valid

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Core: run the full v6 pipeline on a pre-sliced DataFrame
# ---------------------------------------------------------------------------
def compute_window_metrics(
    raw_df: pd.DataFrame,
    initial_capital: float = INITIAL_CAPITAL,
) -> dict:
    """
    Run the full stage-pattern + backtest pipeline on *raw_df* and return
    performance metrics.

    Parameters
    ----------
    raw_df : pd.DataFrame
        Daily OHLCV data (columns: Date, Open, High, Low, Close, Volume).
    initial_capital : float

    Returns
    -------
    dict with keys:
        return_pct, win_rate, n_trades, wins, losses,
        final_capital, trades (list of dicts)
    """
    if raw_df.empty or len(raw_df) < 3:
        return _empty_result()

    raw_df = raw_df.sort_values("Date").reset_index(drop=True)

    # Ensure dates are datetime
    if not pd.api.types.is_datetime64_any_dtype(raw_df["Date"]):
        raw_df["Date"] = pd.to_datetime(raw_df["Date"])

    end_date = str(raw_df["Date"].max().date())
    start_date = str(raw_df["Date"].min().date())

    # 1. FVG detection
    fvgs = find_fvgs(raw_df)

    # 2. Merge FVG levels + forward-fill
    final_df = generate_finals(raw_df, fvgs, end_date, start_date)

    # 3. MACD
    final_df = compute_macd(final_df)

    # 4. Stage assignment
    stages_df = generate_pure_stages(final_df)
    for col in ["EMA_Fast", "EMA_Slow", "MACD_Line", "MACD_Signal", "MACD_Histogram"]:
        if col in final_df.columns:
            stages_df[col] = final_df[col].values

    # 5. Deduplicate
    deduped = deduplicate_stages(stages_df)

    # 6. Trade signals
    signals = generate_trade_signals(
        deduped, full_stages=stages_df, macd_mode=MACD_FILTER_MODE
    )

    # 7. Backtest
    trades, final_capital = backtest_trades(
        signals, raw_df=stages_df, initial_capital=initial_capital
    )

    # 8. Compute metrics
    total_return = (final_capital / initial_capital - 1) * 100

    if trades:
        win_count = sum(1 for t in trades if t["PnL %"] > 0)
        win_rate = win_count / len(trades) * 100
        loss_count = len(trades) - win_count
    else:
        win_count = 0
        loss_count = 0
        win_rate = np.nan

    return {
        "return_pct": round(total_return, 2),
        "win_rate": round(win_rate, 2) if not np.isnan(win_rate) else np.nan,
        "n_trades": len(trades),
        "wins": win_count,
        "losses": loss_count,
        "final_capital": round(final_capital, 2),
        "trades": trades,
    }


def _empty_result() -> dict:
    """Return a metrics dict representing no valid result."""
    return {
        "return_pct": 0.0,
        "win_rate": np.nan,
        "n_trades": 0,
        "wins": 0,
        "losses": 0,
        "final_capital": INITIAL_CAPITAL,
        "trades": [],
    }


# ---------------------------------------------------------------------------
# Rolling-window orchestrator
# ---------------------------------------------------------------------------
def run_rolling_analysis(
    ticker: str,
    window_years: int = WINDOW_YEARS,
    step_months: int = STEP_MONTHS,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute rolling Return & Win rate for *ticker*.

    Parameters
    ----------
    ticker : str
        Yahoo Finance ticker (e.g. "AAPL", "NVDA").
    window_years : int
        Length of each rolling window in calendar years (default 1).
    step_months : int
        Step between consecutive windows in months (default 1).
    start_date : str or None
        Earliest overall date (YYYY-MM-DD).  Default: today - 4 years.
    end_date : str or None
        Latest overall date (YYYY-MM-DD).  Default: today.

    Returns
    -------
    (results_df, raw_all_df)
        results_df : columns [Window_End, Return_%, Win_Rate_%, Trades,
                              Wins, Losses, Final_Capital]
        raw_all_df : the full fetched OHLCV data (for optional plotting)
    """
    today = pd.Timestamp.today()
    if end_date is None:
        end_date = today.strftime("%Y-%m-%d")
    if start_date is None:
        start_date = (today - pd.DateOffset(years=4)).strftime("%Y-%m-%d")

    end_dt = pd.Timestamp(end_date)
    start_dt = pd.Timestamp(start_date)

    # Fetch ALL data once (with warmup headroom)
    fetch_start = (start_dt - pd.Timedelta(days=WARMUP_DAYS + 60)).strftime("%Y-%m-%d")

    print(f"\n{'=' * 60}")
    print(f"  Rolling Stage Analysis — {ticker}")
    print(f"{'=' * 60}")
    print(f"  Window : {window_years} year(s), step : {step_months} month(s)")
    print(f"  Range  : {start_date} → {end_date}")
    print(f"  MACD filter: {MACD_FILTER_MODE}")
    print(f"\n  Fetching full data from {fetch_start} → {end_date} ...")
    raw_all = retrieve_data(ticker, end_date, fetch_start)

    if raw_all.empty:
        print("  !! No data returned.")
        return pd.DataFrame(), pd.DataFrame()

    raw_all["Date"] = pd.to_datetime(raw_all["Date"])
    raw_all = raw_all.sort_values("Date").reset_index(drop=True)
    print(f"  -> {len(raw_all)} daily bars fetched")

    # Generate window anchors (end-of-window dates)
    anchor_start = start_dt + pd.DateOffset(years=window_years)
    if anchor_start > end_dt:
        print("  !! Date range too short for even one window.")
        return pd.DataFrame(), raw_all

    anchors = pd.date_range(anchor_start, end_dt, freq="MS")
    if len(anchors) == 0:
        anchors = pd.DatetimeIndex([anchor_start])

    data_max = raw_all["Date"].max()
    anchors = anchors[anchors <= data_max]
    print(f"  Windows to compute: {len(anchors)}")

    rows = []
    skipped = 0

    for i, T in enumerate(anchors):
        window_end = T
        window_start = T - pd.DateOffset(years=window_years)
        slice_start = window_start - pd.Timedelta(days=WARMUP_DAYS)

        mask = (raw_all["Date"] >= slice_start) & (raw_all["Date"] <= window_end)
        raw_slice = raw_all.loc[mask].copy()

        if len(raw_slice) < MIN_BARS:
            skipped += 1
            continue

        result = compute_window_metrics(raw_slice)

        # Filter trades: only those whose entry date is within the actual window
        all_trades = result["trades"]
        window_trades = []
        for t in all_trades:
            ed = t["Entry Date"]
            if isinstance(ed, str):
                ed = pd.Timestamp(ed)
            if ed >= window_start:
                window_trades.append(t)

        # Recompute metrics from filtered trades
        if window_trades:
            win_count = sum(1 for t in window_trades if t["PnL %"] > 0)
            loss_count = len(window_trades) - win_count
            win_rate = win_count / len(window_trades) * 100
            cap = INITIAL_CAPITAL
            for t in window_trades:
                cap = cap * (1 + t["PnL %"] / 100)
            total_return = (cap / INITIAL_CAPITAL - 1) * 100
        else:
            win_count = 0
            loss_count = 0
            win_rate = np.nan
            total_return = 0.0
            cap = INITIAL_CAPITAL

        rows.append({
            "Window_End": window_end,
            "Return_%": round(total_return, 2),
            "Win_Rate_%": round(win_rate, 2) if not np.isnan(win_rate) else np.nan,
            "Trades": len(window_trades),
            "Wins": win_count,
            "Losses": loss_count,
            "Final_Capital": round(cap, 2),
        })

        if (i + 1) % 6 == 0 or i == len(anchors) - 1:
            r = rows[-1]
            wr = f"{r['Win_Rate_%']:.1f}%" if not np.isnan(r['Win_Rate_%']) else "N/A"
            print(f"  [{i+1}/{len(anchors)}] {window_end.strftime('%Y-%m-%d')}  "
                  f"Return: {r['Return_%']:+.2f}%  Win: {wr}  Trades: {r['Trades']}")

    results_df = pd.DataFrame(rows)
    if skipped:
        print(f"  ({skipped} window(s) skipped — too few bars)")
    return results_df, raw_all


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_rolling_results(
    results_df: pd.DataFrame,
    raw_df: pd.DataFrame,
    ticker: str,
):
    """
    Three-panel chart:
      Top  — Close price
      Mid  — Return_% bars
      Bot  — Win_Rate_% line with 50 % reference
    """
    if results_df.empty:
        print("  Nothing to plot.")
        return None

    price = raw_df.copy()
    price["Date"] = pd.to_datetime(price["Date"])

    res = results_df.copy()
    res["Window_End"] = pd.to_datetime(res["Window_End"])

    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, figsize=(16, 12),
        gridspec_kw={"height_ratios": [2, 1, 1]},
        sharex=True,
    )

    # Panel 1: Price
    ax1.plot(price["Date"], price["Close"], linewidth=0.8, color="#2c3e50")
    ax1.set_ylabel("Close Price")
    ax1.set_title(f"{ticker} — Rolling Stage-Pattern Metrics",
                  fontsize=14, fontweight="bold")
    ax1.grid(True, alpha=0.3)

    # Panel 2: Return %
    colors = ["#27ae60" if v > 0 else "#e74c3c" for v in res["Return_%"]]
    ax2.bar(res["Window_End"], res["Return_%"], width=20, color=colors, alpha=0.85)
    ax2.axhline(y=0, color="black", linewidth=0.5)
    ax2.set_ylabel("Return %")
    ax2.grid(True, alpha=0.3)

    # Panel 3: Win Rate %
    valid = res[res["Win_Rate_%"].notna()]
    ax3.plot(valid["Window_End"], valid["Win_Rate_%"],
             marker="o", markersize=4, linewidth=1.2,
             color="#2980b9", label="Win Rate %")
    ax3.axhline(y=50, color="gray", linewidth=0.8, linestyle="--", alpha=0.6)
    ax3.fill_between(valid["Window_End"], 50, valid["Win_Rate_%"],
                     where=(valid["Win_Rate_%"] >= 50),
                     color="#27ae60", alpha=0.15)
    ax3.fill_between(valid["Window_End"], 50, valid["Win_Rate_%"],
                     where=(valid["Win_Rate_%"] < 50),
                     color="#e74c3c", alpha=0.15)
    ax3.set_ylabel("Win Rate %")
    ax3.set_xlabel("Window End Date")
    ax3.legend(loc="upper left")
    ax3.grid(True, alpha=0.3)

    for ax in (ax1, ax2, ax3):
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Combined time-series chart: Price + Return% + Win_Rate%
# ---------------------------------------------------------------------------
def plot_time_series(
    results_df: pd.DataFrame,
    raw_df: pd.DataFrame,
    ticker: str,
):
    """
    Single dual-axis chart:
      - Left axis:  daily close price (line)
      - Right axis: Rolled Return_% (green bars) and Win_Rate_% (blue line)
    """
    if results_df.empty:
        print("  Nothing to plot.")
        return None

    price = raw_df[["Date", "Close"]].copy()
    price["Date"] = pd.to_datetime(price["Date"])
    price = price.sort_values("Date")

    res = results_df.copy()
    res["Window_End"] = pd.to_datetime(res["Window_End"])

    fig, ax1 = plt.subplots(figsize=(16, 7))

    # ── Left axis: price ──
    ax1.plot(price["Date"], price["Close"], linewidth=0.9, color="#2c3e50", label="Close")
    ax1.set_ylabel("Price", color="#2c3e50")
    ax1.tick_params(axis="y", labelcolor="#2c3e50")

    # ── Right axis: Return_% and Win_Rate_% ──
    ax2 = ax1.twinx()

    # Return_% as bars
    bar_colors = ["#27ae60" if v > 0 else "#e74c3c" for v in res["Return_%"]]
    ax2.bar(res["Window_End"], res["Return_%"],
            width=18, color=bar_colors, alpha=0.35, label="Return %")

    # Win_Rate_% as a stepped line
    valid = res[res["Win_Rate_%"].notna()]
    ax2.plot(valid["Window_End"], valid["Win_Rate_%"],
             marker="D", markersize=5, linewidth=2.0,
             color="#2980b9", label="Win Rate %", zorder=5)

    # 50% reference
    ax2.axhline(y=50, color="gray", linewidth=0.7, linestyle="--", alpha=0.5)

    ax2.set_ylabel("Percent", color="#2980b9")
    ax2.tick_params(axis="y", labelcolor="#2980b9")

    # ── Labels & legend ──
    ax1.set_title(f"{ticker} — Price, Rolling Return & Win Rate (1‑yr windows)",
                  fontsize=14, fontweight="bold")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax1.grid(True, alpha=0.2)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=9)

    fig.tight_layout()


# ---------------------------------------------------------------------------
# Regime dashboard: price + colour-coded Return zones
# ---------------------------------------------------------------------------
def plot_regime_dashboard(
    results_df: pd.DataFrame,
    raw_df: pd.DataFrame,
    ticker: str,
):
    """
    Price chart with background coloured by rolling Return regime:
      - Green band when Return_% > 0  (strategy working)
      - Red band  when Return_% < 0  (strategy struggling)
      - Win Rate overlaid as a dotted line on secondary axis
    """
    if results_df.empty:
        print("  Nothing to plot.")
        return None

    price = raw_df[["Date", "Close"]].copy()
    price["Date"] = pd.to_datetime(price["Date"])
    price = price.sort_values("Date")

    res = results_df.copy()
    res["Window_End"] = pd.to_datetime(res["Window_End"])

    fig, ax1 = plt.subplots(figsize=(18, 8))

    # Draw regime bands behind price
    for _, row in res.iterrows():
        end_dt = row["Window_End"]
        start_dt = end_dt - pd.DateOffset(years=1)
        ret = row["Return_%"]
        if pd.isna(ret):
            continue
        color = "#27ae60" if ret > 0 else "#e74c3c"
        ax1.axvspan(start_dt, end_dt, alpha=0.12, color=color, linewidth=0)

    # Price line
    ax1.plot(price["Date"], price["Close"], linewidth=1.0, color="#2c3e50",
             zorder=5, label="Close")

    # Win Rate on secondary axis
    ax2 = ax1.twinx()
    valid_win = res[res["Win_Rate_%"].notna()]
    ax2.plot(valid_win["Window_End"], valid_win["Win_Rate_%"],
             linestyle="dotted", linewidth=1.5, color="#2980b9",
             marker=".", markersize=4, zorder=6, label="Win Rate %")
    ax2.axhline(y=50, color="gray", linewidth=0.6, linestyle="--", alpha=0.4)
    ax2.set_ylabel("Win Rate %", color="#2980b9")
    ax2.tick_params(axis="y", labelcolor="#2980b9")
    ax2.set_ylim(0, 105)

    # Labels
    ax1.set_title(
        f"{ticker} — Regime Dashboard (green = Return > 0, red = Return < 0)",
        fontsize=14, fontweight="bold",
    )
    ax1.set_ylabel("Price", color="#2c3e50")
    ax1.tick_params(axis="y", labelcolor="#2c3e50")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax1.grid(True, alpha=0.15)

    # Legend
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color="#2c3e50", linewidth=1, label="Close"),
        Line2D([0], [0], linestyle="dotted", color="#2980b9", linewidth=1.5,
               label="Win Rate %"),
        Patch(facecolor="#27ae60", alpha=0.25, label="Return > 0 (favourable)"),
        Patch(facecolor="#e74c3c", alpha=0.25, label="Return < 0 (unfavourable)"),
    ]
    ax1.legend(handles=legend_elements, loc="upper left", fontsize=9)

    fig.tight_layout()
    return fig


    return fig

def print_summary(results_df: pd.DataFrame):
    """Print a summary statistics table for the rolling results."""
    if results_df.empty:
        return

    res = results_df.copy()
    valid = res[res["Win_Rate_%"].notna()]

    print(f"\n{'=' * 60}")
    print(f"  SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Windows computed : {len(res)}")
    print(f"  Windows w/ trades: {len(valid)}")
    print(f"  {'─' * 50}")

    if not valid.empty:
        print(f"  Return_%  →  mean: {valid['Return_%'].mean():+.2f}%  "
              f"median: {valid['Return_%'].median():+.2f}%  "
              f"min: {valid['Return_%'].min():+.2f}%  "
              f"max: {valid['Return_%'].max():+.2f}%")
        print(f"  Win_Rate_% →  mean: {valid['Win_Rate_%'].mean():.1f}%  "
              f"median: {valid['Win_Rate_%'].median():.1f}%  "
              f"min: {valid['Win_Rate_%'].min():.1f}%  "
              f"max: {valid['Win_Rate_%'].max():.1f}%")
        print(f"  Trades     →  mean: {valid['Trades'].mean():.1f}  "
              f"median: {valid['Trades'].median():.0f}  "
              f"min: {valid['Trades'].min()}  "
              f"max: {valid['Trades'].max()}")
        print(f"  {'─' * 50}")

        valid2 = valid.dropna(subset=["Return_%", "Win_Rate_%"])
        if len(valid2) > 2:
            corr = valid2["Return_%"].corr(valid2["Win_Rate_%"])
            print(f"  Corr(Return, Win Rate): {corr:+.3f}")

    display_cols = ["Window_End", "Return_%", "Win_Rate_%", "Trades", "Wins", "Losses"]
    print(f"\n  Full table (first 10 + last 5 rows):")
    head = res[display_cols].head(10)
    tail = res[display_cols].tail(5)
    print(head.to_string(index=False))
    if len(res) > 15:
        print("  ...")
        print(tail.to_string(index=False))


# ---------------------------------------------------------------------------
# __main__
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    ticker = input("Enter stock ticker: ").strip().upper()
    if not ticker:
        print("No ticker provided.")
        sys.exit(0)

    start_default = (pd.Timestamp.today() - pd.DateOffset(years=4)).strftime("%Y-%m-%d")
    end_default   = pd.Timestamp.today().strftime("%Y-%m-%d")

    start_in = input(f"Start date [{start_default}]: ").strip()
    end_in   = input(f"End date   [{end_default}]: ").strip()

    start = start_in if start_in else start_default
    end   = end_in if end_in else end_default

    # Run
    results_df, raw_all = run_rolling_analysis(
        ticker,
        window_years=WINDOW_YEARS,
        step_months=STEP_MONTHS,
        start_date=start,
        end_date=end,
    )

    if results_df.empty:
        print("\nNo results to display.")
        sys.exit(0)

    # Print summary + table
    print_summary(results_df)

    # Save CSV
    csv_file = f"{ticker}_rolling_stage.csv"
    results_df.to_csv(csv_file, index=False)
    print(f"\n  Results saved to: {csv_file}")

    # Plots
    fig1 = plot_rolling_results(results_df, raw_all, ticker)
    if fig1:
        chart_file = f"{ticker}_rolling_stage.png"
        fig1.savefig(chart_file, dpi=150)
        print(f"  Chart saved to: {chart_file}")

    fig2 = plot_time_series(results_df, raw_all, ticker)
    if fig2:
        ts_file = f"{ticker}_rolling_ts.png"
        fig2.savefig(ts_file, dpi=150)
        print(f"  Time-series chart saved to: {ts_file}")

    fig3 = plot_regime_dashboard(results_df, raw_all, ticker)
    if fig3:
        dash_file = f"{ticker}_regime_dashboard.png"
        fig3.savefig(dash_file, dpi=150)
        print(f"  Regime dashboard saved to: {dash_file}")

    try:
        plt.show()
    except Exception:
        pass


