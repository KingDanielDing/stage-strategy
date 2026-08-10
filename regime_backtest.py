"""
regime_backtest.py — Backtest "Return > 0" as a simple regime filter.

Rule: at each monthly window-end T, if rolling 1-year Return_% > 0,
      go long the next month; otherwise stay in cash.

Compares against: Buy & hold, 12-month price momentum.
"""

import sys
import warnings
from typing import Optional, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from rolling_stage_analysis import run_rolling_analysis

warnings.filterwarnings("ignore", category=FutureWarning)

DEFAULT_UNIVERSE = [
    "AAPL", "NVDA", "MSFT", "GOOGL", "AMZN", "META",
    "TSLA", "JPM", "XOM", "JNJ",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _compute_stats(monthly_rets: pd.Series, label: str) -> dict:
    """Return performance stats dict for a monthly return series."""
    v = monthly_rets.dropna()
    if len(v) == 0:
        return {"label": label, "n_months": 0}
    cum = np.prod(1 + v.values / 100) - 1
    wr = (v > 0).sum() / len(v) * 100
    ann_ret = ((1 + cum) ** (12 / len(v)) - 1) * 100 if len(v) > 0 else 0
    ann_vol = v.std() * np.sqrt(12) if len(v) > 1 else np.nan
    sharpe = ann_ret / ann_vol if ann_vol and ann_vol > 0 else np.nan
    eq = (1 + v / 100).cumprod()
    peak = eq.expanding().max()
    dd = (eq / peak - 1) * 100
    max_dd = dd.min()
    return {
        "label": label,
        "n_months": len(v),
        "cum_return": round(cum * 100, 2),
        "ann_return": round(ann_ret, 2),
        "ann_vol": round(ann_vol, 2),
        "sharpe": round(sharpe, 2) if not np.isnan(sharpe) else np.nan,
        "win_rate": round(wr, 1),
        "max_dd": round(max_dd, 2),
        "mean_monthly": round(v.mean(), 2),
    }



# ---------------------------------------------------------------------------
# Single-stock regime backtest
# ---------------------------------------------------------------------------
def backtest_regime(
    ticker: str,
    start_date: str = "2022-01-01",
    end_date: Optional[str] = None,
) -> dict:
    """Backtest the 'Return > 0' rule on one stock."""
    if end_date is None:
        end_date = pd.Timestamp.today().strftime("%Y-%m-%d")

    results_df, raw_all = run_rolling_analysis(
        ticker, window_years=1, step_months=1,
        start_date=start_date, end_date=end_date,
    )
    if results_df.empty:
        return {}

    price = raw_all[["Date", "Close"]].copy()
    price["Date"] = pd.to_datetime(price["Date"])
    price = price.sort_values("Date").reset_index(drop=True)

    res = results_df.copy()
    res["Window_End"] = pd.to_datetime(res["Window_End"])
    res = res.dropna(subset=["Return_%"])

    fwd_rets, signals, price_mom = [], [], []

    for _, row in res.iterrows():
        T = row["Window_End"]
        ret = row["Return_%"]

        mask_T = price["Date"] <= T
        if not mask_T.any():
            fwd_rets.append(np.nan); signals.append(np.nan); price_mom.append(np.nan)
            continue
        close_T = price.loc[mask_T, "Close"].iloc[-1]

        target = T + pd.DateOffset(months=1)
        mask_f = (price["Date"] >= T) & (price["Date"] <= target)
        if mask_f.sum() < 2:
            fwd_rets.append(np.nan); signals.append(np.nan); price_mom.append(np.nan)
            continue
        close_fwd = price.loc[mask_f, "Close"].iloc[-1]
        fwd_rets.append((close_fwd / close_T - 1) * 100)
        signals.append(1 if ret > 0 else 0)

        lb = T - pd.DateOffset(years=1)
        mask_lb = (price["Date"] >= lb) & (price["Date"] <= T)
        if mask_lb.sum() >= 2:
            close_lb = price.loc[mask_lb, "Close"].iloc[0]
            price_mom.append((close_T / close_lb - 1) * 100)
        else:
            price_mom.append(np.nan)

    res["Fwd_1M"] = fwd_rets
    res["Signal"] = signals
    res["Price_12M_Mom"] = price_mom

    valid = res.dropna(subset=["Fwd_1M", "Signal"])
    long_mask = valid["Signal"] == 1
    cash_mask = valid["Signal"] == 0

    # Strategy: long when signal=1, 0% when signal=0
    strat_rets = valid["Fwd_1M"].copy()
    strat_rets[cash_mask] = 0.0

    # Momentum benchmark
    mom_valid = valid.dropna(subset=["Price_12M_Mom"])
    mom_long = mom_valid["Price_12M_Mom"] > 0
    mom_rets = mom_valid["Fwd_1M"].copy()
    mom_rets[~mom_long] = 0.0

    # Buy & hold
    bh_rets = valid["Fwd_1M"]

    long_rets = valid.loc[long_mask, "Fwd_1M"]
    cash_actual = valid.loc[cash_mask, "Fwd_1M"]

    return {
        "ticker": ticker,
        "strategy": _compute_stats(strat_rets, "Return>0 Rule"),
        "buyhold": _compute_stats(bh_rets, "Buy & Hold"),
        "momentum": _compute_stats(mom_rets, "12M Mom>0"),
        "regime": {
            "long_months": len(long_rets),
            "cash_months": len(cash_actual),
            "long_avg": round(long_rets.mean(), 2),
            "cash_avg": round(cash_actual.mean(), 2),
            "long_win": round((long_rets > 0).sum() / len(long_rets) * 100, 1) if len(long_rets) else 0,
            "cash_win": round((cash_actual > 0).sum() / len(cash_actual) * 100, 1) if len(cash_actual) else 0,
            "pct_long": round(len(long_rets) / len(valid) * 100, 1),
        },
    }



# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def print_regime_report(result: dict):
    t = result["ticker"]
    s = result["strategy"]
    b = result["buyhold"]
    m = result["momentum"]
    r = result["regime"]
    print(f"\n{'=' * 65}")
    print(f"  REGIME BACKTEST — {t}")
    print(f"{'=' * 65}")
    print(f"\n  {'Strategy':<18s} {'Cum':>8s} {'Ann':>8s} {'Sharpe':>7s} "
          f"{'Win%':>6s} {'MaxDD':>7s} {'N':>5s}")
    print(f"  {'─' * 60}")
    for st in [s, b, m]:
        print(f"  {st['label']:<18s} {st['cum_return']:>+7.2f}% {st['ann_return']:>+7.2f}% "
              f"{st['sharpe']:>6.2f} {st['win_rate']:>5.1f}% {st['max_dd']:>+7.2f}% "
              f"{st['n_months']:>4d}")
    print(f"\n  Regime breakdown:")
    print(f"    Long ({r['long_months']:>3d} months, {r['pct_long']:.0f}% of time): "
          f"avg {r['long_avg']:+.2f}%/mo, win {r['long_win']:.0f}%")
    print(f"    Cash ({r['cash_months']:>3d} months): "
          f"avg {r['cash_avg']:+.2f}%/mo, win {r['cash_win']:.0f}%")
    print(f"    Spread (Long - Cash): {r['long_avg'] - r['cash_avg']:+.2f}%/mo")


# ---------------------------------------------------------------------------
# Multi-stock
# ---------------------------------------------------------------------------
def run_multi_regime(tickers, start_date="2022-01-01", end_date=None):
    rows = []
    for ticker in tickers:
        try:
            result = backtest_regime(ticker, start_date, end_date)
            if not result:
                continue
            print_regime_report(result)
            s = result["strategy"]
            b = result["buyhold"]
            m = result["momentum"]
            r = result["regime"]
            rows.append({
                "Ticker": ticker, "Rule_Cum%": s["cum_return"],
                "Rule_Sharpe": s["sharpe"], "Rule_Win%": s["win_rate"],
                "Rule_MaxDD": s["max_dd"], "Rule_LongAvg": r["long_avg"],
                "Rule_CashAvg": r["cash_avg"],
                "Rule_Spread": round(r["long_avg"] - r["cash_avg"], 2),
                "Rule_%Long": r["pct_long"],
                "BH_Cum%": b["cum_return"], "BH_Sharpe": b["sharpe"],
                "BH_MaxDD": b["max_dd"], "Mom_Cum%": m["cum_return"],
                "Mom_Sharpe": m["sharpe"], "N_Months": s["n_months"],
            })
        except Exception as e:
            print(f"  !! {ticker}: {e}")
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    print(f"\n{'=' * 65}")
    print(f"  CROSS-STOCK SUMMARY ({len(df)} stocks)")
    print(f"{'=' * 65}")
    print(f"  {'Metric':<16s} {'Rule':>12s} {'B&H':>12s} {'12M Mom':>12s}")
    print(f"  {'─' * 55}")
    for label, cr, cb, cm in [
        ("Cum Return %", "Rule_Cum%", "BH_Cum%", "Mom_Cum%"),
        ("Sharpe", "Rule_Sharpe", "BH_Sharpe", "Mom_Sharpe"),
        ("Win Rate %", "Rule_Win%", "", ""),
        ("Max DD %", "Rule_MaxDD", "BH_MaxDD", ""),
    ]:
        rv = df[cr].mean()
        bv = df[cb].mean() if cb else float("nan")
        mv = df[cm].mean() if cm else float("nan")
        print(f"  {label:<16s} {rv:>+11.2f}  {bv:>+11.2f}  {mv:>+11.2f}")
    nb = (df["Rule_Cum%"] > df["BH_Cum%"]).sum()
    nm = (df["Rule_Cum%"] > df["Mom_Cum%"]).sum()
    print(f"\n  Rule beats B&H on {nb}/{len(df)} stocks")
    print(f"  Rule beats 12M Mom on {nm}/{len(df)} stocks")
    return df


# ---------------------------------------------------------------------------
# __main__
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) > 1:
        tickers = [t.upper() for t in sys.argv[1:]]
    else:
        inp = input("Ticker(s) [space-sep, or 'batch']: ").strip()
        if not inp:
            print("No tickers.")
            sys.exit(0)
        if inp.lower() == "batch":
            tickers = DEFAULT_UNIVERSE
        else:
            tickers = [t.upper() for t in inp.split()]
    if len(tickers) == 1:
        result = backtest_regime(tickers[0])
        if result:
            print_regime_report(result)
    else:
        df = run_multi_regime(tickers)
        if not df.empty:
            df.to_csv("regime_backtest_summary.csv", index=False)
            print("\n  Saved to: regime_backtest_summary.csv")
