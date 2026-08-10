"""
rolling_predictive_test.py — Phase 2: Test if rolling Return / Win Rate predict
future price direction.

For each rolling window ending at time T, computes forward N‑month price returns
and checks whether the strategy metrics (Return_%, Win_Rate_%) correlate with
future price moves.

Usage:
    python rolling_predictive_test.py           # single stock, interactive
    python rolling_predictive_test.py AAPL      # single stock, CLI args
"""

import sys
import warnings
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from rolling_stage_analysis import (
    run_rolling_analysis,
    MACD_FILTER_MODE,
    WINDOW_YEARS,
    STEP_MONTHS,
)

warnings.filterwarnings("ignore", category=FutureWarning)

# Forward horizons to test (in calendar months)
FORWARD_MONTHS = [1, 3, 6]

# Multi-stock universe for batch mode
DEFAULT_UNIVERSE = [
    "AAPL", "NVDA", "MSFT", "GOOGL", "AMZN", "META",
    "TSLA", "JPM", "XOM", "JNJ",
]

def _rank_corr(a: pd.Series, b: pd.Series) -> float:
    """Spearman rank correlation without scipy (uses pandas rank + pearson)."""
    valid = a.notna() & b.notna()
    if valid.sum() < 3:
        return np.nan
    return a[valid].rank().corr(b[valid].rank(), method="pearson")


# ---------------------------------------------------------------------------
# Forward return computation
# ---------------------------------------------------------------------------
def add_forward_returns(
    results_df: pd.DataFrame,
    raw_df: pd.DataFrame,
    forward_months: Optional[List[int]] = None,
) -> pd.DataFrame:
    """
    Append forward N-month price-return columns to *results_df*.

    For each row (window ending at T), finds the close price at T and
    the close price ~N months later; forward return = (P_future / P_T - 1) * 100.
    """
    if forward_months is None:
        forward_months = FORWARD_MONTHS

    df = results_df.copy()
    df["Window_End"] = pd.to_datetime(df["Window_End"])

    price = raw_df[["Date", "Close"]].copy()
    price["Date"] = pd.to_datetime(price["Date"])
    price = price.sort_values("Date").reset_index(drop=True)

    for months in forward_months:
        col_name = f"Fwd_{months}M_Return_%"
        fwd_returns = []

        for _, row in df.iterrows():
            T = row["Window_End"]
            mask_T = price["Date"] <= T
            if not mask_T.any():
                fwd_returns.append(np.nan)
                continue
            close_T = price.loc[mask_T, "Close"].iloc[-1]

            target_date = T + pd.DateOffset(months=months)
            mask_fwd = (price["Date"] >= T) & (price["Date"] <= target_date)
            if mask_fwd.sum() < 2:
                fwd_returns.append(np.nan)
                continue

            close_future = price.loc[mask_fwd, "Close"].iloc[-1]
            fwd_ret = (close_future / close_T - 1) * 100
            fwd_returns.append(round(fwd_ret, 2))

        df[col_name] = fwd_returns

    return df



# ---------------------------------------------------------------------------
# Predictive power analysis
# ---------------------------------------------------------------------------
def analyze_predictive_power(
    results_df: pd.DataFrame,
    raw_df: pd.DataFrame,
    forward_months: Optional[List[int]] = None,
) -> dict:
    """
    Test whether Return_% or Win_Rate_% predicts future price direction.

    Computes:
      - Spearman rank correlation between each metric and each forward horizon
      - Win-rate regime analysis: split Win into quartiles, compare avg forward return

    Returns a dict of stats for reporting / cross-stock aggregation.
    """
    if forward_months is None:
        forward_months = FORWARD_MONTHS

    df = add_forward_returns(results_df, raw_df, forward_months)

    # Drop rows where Win rate is NaN (0-trade windows)
    valid = df.dropna(subset=["Win_Rate_%"])

    metrics = ["Return_%", "Win_Rate_%"]
    fwd_cols = [f"Fwd_{m}M_Return_%" for m in forward_months]

    # ---- Spearman correlations ----
    corr_data = {}
    for metric in metrics:
        for fwd in fwd_cols:
            sub = valid.dropna(subset=[fwd])
            if len(sub) > 5:
                spear = _rank_corr(sub[metric], sub[fwd])
                pear = sub[metric].corr(sub[fwd], method="pearson")
                corr_data[f"{metric}_vs_{fwd}"] = {
                    "spearman": round(spear, 4),
                    "pearson": round(pear, 4),
                    "n": len(sub),
                }

    # ---- Quartile analysis on Win_Rate_% ----
    quartile_data = {}
    for fwd in fwd_cols:
        sub = valid.dropna(subset=[fwd])
        if len(sub) < 10:
            continue

        sub = sub.copy()
        sub["Win_Quartile"] = pd.qcut(
            sub["Win_Rate_%"], 4, labels=["Q1_low", "Q2", "Q3", "Q4_high"],
            duplicates="drop",
        )
        qstats = sub.groupby("Win_Quartile", observed=False)[fwd].agg(
            ["mean", "median", "std", "count"]
        )
        quartile_data[fwd] = qstats

    return {
        "correlations": corr_data,
        "quartiles": quartile_data,
        "enriched_df": df,
        "n_windows": len(valid),
    }



# ---------------------------------------------------------------------------
# Report printing
# ---------------------------------------------------------------------------
def print_predictive_report(analysis: dict, ticker: str):
    """Pretty-print the predictive power analysis for a single stock."""
    corr = analysis["correlations"]
    quartiles = analysis["quartiles"]
    n = analysis["n_windows"]

    print(f"\n{'=' * 70}")
    print(f"  PREDICTIVE POWER REPORT — {ticker}  ({n} valid windows)")
    print(f"{'=' * 70}")

    # ---- Correlation table ----
    print(f"\n  {'─' * 60}")
    print(f"  {'Metric':<14s} {'vs Forward':<14s} {'Spearman':>10s} {'Pearson':>10s} {'N':>6s}")
    print(f"  {'─' * 60}")
    for key, v in corr.items():
        metric, fwd = key.split("_vs_")
        print(f"  {metric:<14s} {fwd:<14s} {v['spearman']:>+10.4f} {v['pearson']:>+10.4f} {v['n']:>6d}")
    print(f"  {'─' * 60}")

    # ---- Quartile analysis ----
    for fwd, qdf in quartiles.items():
        print(f"\n  Win_Rate quartile → {fwd}:")
        print(f"  {'Quartile':<12s} {'Mean':>8s} {'Median':>8s} {'Std':>8s} {'N':>5s}")
        print(f"  {'─' * 45}")
        for idx, row in qdf.iterrows():
            print(f"  {idx:<12s} {row['mean']:>+8.2f}% {row['median']:>+8.2f}% "
                  f"{row['std']:>8.2f} {int(row['count']):>5d}")
        if len(qdf) >= 2:
            spread = qdf["mean"].iloc[-1] - qdf["mean"].iloc[0]
            print(f"  {'─' * 45}")
            print(f"  {'Q4−Q1 spread':<12s} {spread:>+8.2f}%")

    print()



# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
def plot_predictive_scatter(analysis: dict, ticker: str):
    """Scatter plots: Win_Rate_% vs forward returns for each horizon."""
    df = analysis["enriched_df"].dropna(subset=["Win_Rate_%"])
    fwd_cols = [c for c in df.columns if c.startswith("Fwd_") and c.endswith("_Return_%")]

    if not fwd_cols:
        print("  No forward return columns to plot.")
        return None

    n = len(fwd_cols)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, fwd in zip(axes, fwd_cols):
        sub = df.dropna(subset=[fwd])
        ax.scatter(sub["Win_Rate_%"], sub[fwd], alpha=0.6, edgecolors="white", s=50)

        if len(sub) > 2:
            z = np.polyfit(sub["Win_Rate_%"], sub[fwd], 1)
            p = np.poly1d(z)
            x_line = np.linspace(sub["Win_Rate_%"].min(), sub["Win_Rate_%"].max(), 50)
            ax.plot(x_line, p(x_line), "r--", linewidth=1.2, alpha=0.7)

        ax.axhline(y=0, color="gray", linewidth=0.5, linestyle=":")
        ax.set_xlabel("Win Rate %")
        ax.set_ylabel(fwd.replace("_", " "))
        ax.set_title(f"{ticker}: Win Rate vs {fwd}")
        ax.grid(True, alpha=0.3)

        if len(sub) > 2:
            spear = _rank_corr(sub["Win_Rate_%"], sub[fwd])
            ax.text(0.05, 0.95, f"Spearman = {spear:+.3f}",
                    transform=ax.transAxes, fontsize=11,
                    verticalalignment="top",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    fig.tight_layout()
    return fig



# ---------------------------------------------------------------------------
# Single-stock pipeline
# ---------------------------------------------------------------------------
def run_single_stock_analysis(
    ticker: str,
    start_date: str = "2022-01-01",
    end_date: Optional[str] = None,
    window_years: int = WINDOW_YEARS,
    step_months: int = STEP_MONTHS,
    forward_months: Optional[List[int]] = None,
    do_plot: bool = True,
) -> dict:
    """Full pipeline for one ticker: roll -> enrich -> analyse -> report -> plot."""
    if end_date is None:
        end_date = pd.Timestamp.today().strftime("%Y-%m-%d")
    if forward_months is None:
        forward_months = FORWARD_MONTHS

    print(f"\n{'█' * 70}")
    print(f"  ANALYSING: {ticker}")
    print(f"{'█' * 70}")

    results_df, raw_all = run_rolling_analysis(
        ticker, window_years=window_years, step_months=step_months,
        start_date=start_date, end_date=end_date,
    )

    if results_df.empty:
        print(f"  !! No rolling results for {ticker}")
        return {}

    analysis = analyze_predictive_power(results_df, raw_all, forward_months)
    print_predictive_report(analysis, ticker)

    if do_plot:
        fig = plot_predictive_scatter(analysis, ticker)
        if fig:
            chart_file = f"{ticker}_predictive_scatter.png"
            fig.savefig(chart_file, dpi=150)
            print(f"  Chart saved to: {chart_file}")

    return analysis


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Multi-stock batch
# ---------------------------------------------------------------------------
def run_multi_stock_analysis(
    tickers: List[str],
    start_date: str = "2022-01-01",
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """Run predictive analysis across multiple stocks and return a summary."""
    if end_date is None:
        end_date = pd.Timestamp.today().strftime("%Y-%m-%d")

    all_corrs = []

    for ticker in tickers:
        try:
            analysis = run_single_stock_analysis(
                ticker, start_date, end_date,
                window_years=1, step_months=1, do_plot=False,
            )
            if not analysis:
                continue
            for key, v in analysis["correlations"].items():
                all_corrs.append({
                    "Ticker": ticker,
                    "Metric_vs_Forward": key,
                    "Spearman": v["spearman"],
                    "Pearson": v["pearson"],
                    "N": v["n"],
                })
        except Exception as e:
            print(f"  !! {ticker}: {e}")

    if not all_corrs:
        return pd.DataFrame()

    summary_df = pd.DataFrame(all_corrs)

    print(f"\n{'=' * 70}")
    print(f"  CROSS-STOCK SUMMARY")
    print(f"{'=' * 70}")

    for fwd_label in summary_df["Metric_vs_Forward"].unique():
        sub = summary_df[summary_df["Metric_vs_Forward"] == fwd_label]
        mean_spear = sub["Spearman"].mean()
        pos_count = (sub["Spearman"] > 0).sum()
        print(f"\n  {fwd_label}:")
        print(f"    Mean Spearman = {mean_spear:+.4f}  "
              f"({pos_count}/{len(sub)} stocks positive)")
        for _, row in sub.iterrows():
            bar = "█" * max(1, int(abs(row["Spearman"]) * 50))
            sign = "+" if row["Spearman"] > 0 else " "
            print(f"    {row['Ticker']:<8s} {sign}{row['Spearman']:+.4f}  {bar}")

    return summary_df


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# __main__
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) > 1:
        tickers = [t.upper() for t in sys.argv[1:]]
    else:
        inp = input("Enter ticker(s) [space-separated, or 'batch' for 10]: ").strip()
        if not inp:
            print("No tickers provided.")
            sys.exit(0)
        if inp.lower() == "batch":
            tickers = DEFAULT_UNIVERSE
        else:
            tickers = [t.upper() for t in inp.split()]

    if len(tickers) == 1:
        run_single_stock_analysis(tickers[0])
    else:
        summary = run_multi_stock_analysis(tickers)
        if not summary.empty:
            csv_file = "multi_stock_predictive_summary.csv"
            summary.to_csv(csv_file, index=False)
            print(f"\n  Cross-stock summary saved to: {csv_file}")

