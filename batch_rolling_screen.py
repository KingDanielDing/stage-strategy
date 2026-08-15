"""
batch_rolling_screen.py — Screen a universe of A-shares for "recovery inflection"
candidates using the rolling stage-pattern Return_% series.

For each ticker we reuse rolling_stage_analysis.run_rolling_analysis() and keep
the most recent RECENT_MONTHS monthly Return_% values (chronological). A stock
PASSES when its recent returns are in an upward trend:

    upward trend → latest > prev
    (optionally also require positive linear slope when SLOPE_CHECK = True)

Note: "still below zero" and "near zero" are still computed as informational
columns, but they no longer gate the PASS result.

Usage:
    python batch_rolling_screen.py
"""

import io
import contextlib
import numpy as np
import pandas as pd

from rolling_stage_analysis import run_rolling_analysis

# ---------------------------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------------------------
# --- A-share watchlist (kept for reference; set STOCKS = A_SHARE_STOCKS to use) ---
A_SHARE_STOCKS = {
    "300308.SZ": "中际旭创",
    "300469.SZ": "信息发展",
    "300502.SZ": "新易盛",
    "300666.SZ": "江丰电子",
    "300751.SZ": "迈为股份",
    "300759.SZ": "康龙化成",
    "300812.SZ": "易天股份 (waiting)",
    "300913.SZ": "兆龙互连 (waiting)",
    "301217.SZ": "铜冠铜箔",
    "301312.SZ": "智立方",
    "000938.SZ": "紫光股份",
    "000977.SZ": "浪潮信息",
    "000988.SZ": "华工科技",
    "001309.SZ": "德明利",
    "002008.SZ": "大族激光",
    "002185.SZ": "华天科技",
    "002281.SZ": "光迅科技",
    "002371.SZ": "北方华创",
    "002409.SZ": "雅克科技",
    "002460.SZ": "赣锋锂业 (star)",
    "002472.SZ": "双环传动",
    "002747.SZ": "埃斯顿",
    "002885.SZ": "京泉华",
    "600183.SS": "生益科技",
    "600460.SS": "士兰微",
    "600498.SS": "烽火通信",
    "600522.SS": "中天科技",
    "600584.SS": "长电科技",
    "600667.SS": "太极实业",
    "603005.SS": "晶方科技",
    "603773.SS": "沃格光电",
    "603993.SS": "洛阳钼业",
    "603688.SS": "石英股份",
    "603986.SS": "兆易创新",
    "605111.SS": "新洁能",
    "688012.SS": "中微公司",
    "688017.SS": "绿地谐波",
    "688141.SS": "华杰特",
    "688195.SS": "腾景科技",
    "688256.SS": "寒武纪",
    "688300.SS": "联瑞新材",
    "688507.SS": "索晨科技",
    "688525.SS": "百维存储",
    "688700.SS": "东威科技",
}

# --- US watchlist (active) ---
US_STOCKS = {
    "AAPL": "Apple / Consumer Tech",
    "MSFT": "Microsoft / Software & Cloud",
    "NVDA": "NVIDIA / AI Chips & Semis",
    "GOOGL": "Alphabet / Internet & Ads",
    "AMZN": "Amazon / E-commerce & Cloud",
    "META": "Meta Platforms / Social & Ads",
    "TSLA": "Tesla / EV & Auto",
    "NFLX": "Netflix / Streaming",
    "JPM": "JPMorgan Chase / Banking",
    "V": "Visa / Payments",
    "JNJ": "Johnson & Johnson / Pharma",
    "UNH": "UnitedHealth / Health Insurance",
    "XOM": "Exxon Mobil / Oil & Gas",
    "PG": "Procter & Gamble / Consumer Staples",
    "KO": "Coca-Cola / Consumer Staples",
    "WMT": "Walmart / Retail",
    "HD": "Home Depot / Retail",
    "MCD": "McDonald's / Consumer Discretionary",
    "BA": "Boeing / Aerospace & Defense",
    "CAT": "Caterpillar / Industrials",
}

STOCKS = A_SHARE_STOCKS

START_DATE = (pd.Timestamp.today() - pd.DateOffset(years=4)).strftime("%Y-%m-%d")
END_DATE = pd.Timestamp.today().strftime("%Y-%m-%d")
DATE_RANGE = (START_DATE, END_DATE)

RECENT_MONTHS = 6          # how many recent monthly windows to inspect
MIN_LATEST = -2.0          # informational only (near-zero flag; no longer gates PASS)
MAX_LATEST = None          # informational only (optional upper cap for the flag)
SLOPE_CHECK = True         # optional: also require positive linear slope to PASS
VERBOSE = False            # print the rolling engine's own progress


# ---------------------------------------------------------------------------
# 2. Screening logic
# ---------------------------------------------------------------------------
def _run_rolling(ticker: str, start: str, end: str):
    if VERBOSE:
        return run_rolling_analysis(ticker, start_date=start, end_date=end)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        return run_rolling_analysis(ticker, start_date=start, end_date=end)


def evaluate_rolling(
    results_df: pd.DataFrame,
    recent_months: int = RECENT_MONTHS,
    min_latest: float = MIN_LATEST,
    max_latest=None,
    slope_check: bool = SLOPE_CHECK,
) -> dict:
    """Evaluate a rolling-analysis results DataFrame and return a pass/fail verdict.

    PASS = upward trend: latest monthly return > previous, and (if slope_check)
    a positive linear slope over the most recent ``recent_months`` windows.
    """
    base = {
        "Windows": np.nan, "Latest": np.nan, "Prev": np.nan, "Slope": np.nan,
        "UpTrend": False, "StillBelow0": False, "NearZero": False,
        "Pass": False, "Reason": "", "Error": "",
    }

    if results_df is None or results_df.empty:
        base["Error"] = "no data"
        base["Reason"] = "no data"
        return base

    df = results_df.dropna(subset=["Return_%"]).reset_index(drop=True)
    if len(df) < 2:
        base["Error"] = "insufficient windows"
        base["Reason"] = "insufficient windows"
        return base

    recent = df.tail(recent_months)
    returns = recent["Return_%"].astype(float).tolist()
    latest = returns[-1]
    prev = returns[-2]
    slope = float(np.polyfit(range(len(returns)), returns, 1)[0]) if len(returns) >= 2 else np.nan

    up_trend = latest > prev
    still_below = prev < 0
    near_zero = latest > min_latest
    if max_latest is not None:
        near_zero = near_zero and (latest <= max_latest)
    if slope_check:
        up_trend = up_trend and (slope > 0)

    base.update({
        "Windows": len(df), "Latest": round(latest, 2), "Prev": round(prev, 2),
        "Slope": round(slope, 3) if not np.isnan(slope) else np.nan,
        "UpTrend": bool(up_trend), "StillBelow0": bool(still_below),
        "NearZero": bool(near_zero),
    })
    base["Pass"] = bool(up_trend)

    if up_trend:
        base["Reason"] = "OK"
    else:
        reasons = []
        if latest <= prev:
            reasons.append("latest <= prev")
        if slope_check and (np.isnan(slope) or slope <= 0):
            reasons.append("negative slope")
        base["Reason"] = ", ".join(reasons) if reasons else "not uptrend"
    return base


def screen_ticker(ticker: str, name: str, start: str, end: str) -> dict:
    """Run rolling analysis for one ticker and apply the pass/fail filter."""
    try:
        results_df, _ = _run_rolling(ticker, start, end)
    except Exception as e:
        return {
            "Ticker": ticker, "Name": name, "Windows": np.nan,
            "Latest": np.nan, "Prev": np.nan, "Slope": np.nan,
            "UpTrend": False, "StillBelow0": False, "NearZero": False,
            "Pass": False, "Reason": "exception", "Error": str(e),
        }

    verdict = evaluate_rolling(results_df)
    verdict["Ticker"] = ticker
    verdict["Name"] = name
    return verdict


# ---------------------------------------------------------------------------
# 3. Batch runner
# ---------------------------------------------------------------------------
def run_batch() -> list:
    start, end = DATE_RANGE
    results = []
    total = len(STOCKS)
    print(f"Scanning {total} stocks  ({start} -> {end})")
    print("=" * 70)
    for i, (ticker, name) in enumerate(STOCKS.items(), 1):
        print(f"[{i:>2}/{total}] {ticker} {name} ...", end=" ", flush=True)
        r = screen_ticker(ticker, name, start, end)
        if r["Error"]:
            print(f"ERROR: {r['Error']}")
        elif r["Pass"]:
            print(f"PASS  latest={r['Latest']:+.2f}%  prev={r['Prev']:+.2f}%")
        else:
            print(f"      latest={r['Latest']:+.2f}%  prev={r['Prev']:+.2f}%")
        results.append(r)
    return results


# ---------------------------------------------------------------------------
# 4. Reporting
# ---------------------------------------------------------------------------
def report(results: list):
    df = pd.DataFrame(results)

    passes = df[df["Pass"]].sort_values("Latest", ascending=False)
    print("\n" + "=" * 70)
    print(f"  PASS CANDIDATES: {len(passes)}")
    print("=" * 70)
    if not passes.empty:
        cols = ["Ticker", "Name", "Latest", "Prev", "Slope", "Windows"]
        print(passes[cols].to_string(index=False))
    else:
        print("  (none matched the filter)")

    all_cols = ["Ticker", "Name", "Windows", "Latest", "Prev", "Slope",
                "UpTrend", "StillBelow0", "NearZero", "Pass", "Reason", "Error"]
    df[all_cols].to_csv("rolling_screen_all.csv", index=False)
    passes[all_cols].to_csv("rolling_screen_passes.csv", index=False)
    print(f"\n  Saved: rolling_screen_all.csv ({len(df)} rows)")
    print(f"  Saved: rolling_screen_passes.csv ({len(passes)} rows)")


if __name__ == "__main__":
    report(run_batch())


