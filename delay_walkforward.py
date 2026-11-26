"""
delay_walkforward.py — Out-of-sample validation of the entry-delay idea.

The (default) 2-year window is split in half:
  FIT  (year 1): sweep entry delays 0..MAX_DELAY, pick the best by total return
  TEST (year 2): evaluate baseline (0d), the fitted delay, fixed rules
                 (1d/2d/3d) and the "oracle" best-on-test (in-sample reference)

Method notes:
  - Signals are generated once on the full window. The engine is causal
    (3-candle FVG lookback, recursive EMAs from the same start date,
    forward-walking stage state machine), so fit-period signals match a
    year-1-only run.
  - Fit trades still open at the split are closed at the split bar's close,
    mimicking a standalone year-1 run.
  - Trades are assigned to FIT/TEST by entry date.

Usage:
    python3 delay_walkforward.py [max_delay=10]

Results are printed to screen and saved to delay_walkforward.csv.
"""

import contextlib
import io
import sys

import pandas as pd

from trade_pairs import default_dates, run_pipeline
from trade_pairs_delay import repriced_pairs, summarize
from batch_delay_screen import parse_watchlist

MAX_DELAY = 10
FIXED_RULES = (1, 2, 3)      # "always delay N days" rules tested out-of-sample
OUT_CSV = "delay_walkforward.csv"


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------
def split_trades(trades, stages_df, split):
    """Split trades into fit (entry < split) and test (entry >= split).
    Fit trades still open past the split are closed at the split bar."""
    dates = pd.to_datetime(stages_df["Date"]).reset_index(drop=True)
    split_bars = dates[dates <= split]
    if split_bars.empty:
        return [], list(trades)
    split_bar = split_bars.iloc[-1]

    fit, test = [], []
    for t in trades:
        entry = pd.to_datetime(t["Entry Date"])
        exit_ = pd.to_datetime(t["Exit Date"])
        if entry >= split:
            test.append(t)
            continue
        t = dict(t)
        if exit_ > split_bar:
            t["Exit Date"] = split_bar      # cap exit at the split bar
        fit.append(t)
    return fit, test


def sweep(trades, stages_df, max_delay):
    """Return {delay: summary} for every delay 0..max_delay."""
    out = {}
    for n in range(max_delay + 1):
        pairs, cap = repriced_pairs(stages_df, trades, n)
        out[n] = summarize(pairs, cap)
    return out


def evaluate(ticker, start, end, split, max_delay):
    """Fit delay on year 1, evaluate on year 2. Returns a result row dict."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        result = run_pipeline(ticker, start, end)
    if result is None:
        return {"Ticker": ticker, "Error": "no data"}
    stages_df, signals, trades, _ = result
    if not trades:
        return {"Ticker": ticker, "Error": "no trades"}

    fit_trades, test_trades = split_trades(trades, stages_df, split)
    if not fit_trades:
        return {"Ticker": ticker, "Error": "no fit trades"}

    fit = sweep(fit_trades, stages_df, max_delay)
    fit_base = fit[0]
    fit_best_n = max(fit, key=lambda n: fit[n]["total_return"])

    row = {
        "Ticker": ticker,
        "Fit_Trades": fit_base["n"],
        "Fit_Base_%": round(fit_base["total_return"], 2),
        "Fit_Delay": fit_best_n,
        "Fit_Best_%": round(fit[fit_best_n]["total_return"], 2),
        "Error": "",
    }

    if not test_trades:
        row.update({"Test_Trades": 0, "Test_Base_%": None, "Test_Fit_%": None,
                    "Test_Delta_%": None, "Oracle_Delay": None, "Oracle_%": None})
        for k in FIXED_RULES:
            row[f"Test_{k}d_%"] = None
        return row

    test = sweep(test_trades, stages_df, max_delay)
    test_base = test[0]
    test_fit = test[fit_best_n]
    oracle_n = max(test, key=lambda n: test[n]["total_return"])
    row.update({
        "Test_Trades": test_base["n"],
        "Test_Base_%": round(test_base["total_return"], 2),
        "Test_Fit_%": round(test_fit["total_return"], 2),
        "Test_Delta_%": round(test_fit["total_return"] - test_base["total_return"], 2),
        "Oracle_Delay": oracle_n,
        "Oracle_%": round(test[oracle_n]["total_return"], 2),
    })
    for k in FIXED_RULES:
        row[f"Test_{k}d_%"] = round(test[k]["total_return"], 2)
    return row

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    max_delay = int(sys.argv[1]) if len(sys.argv) > 1 else MAX_DELAY
    start, end = default_dates()
    split = pd.Timestamp(start) + pd.Timedelta(days=365)
    split_str = split.strftime("%Y-%m-%d")
    tickers = parse_watchlist()
    total = len(tickers)
    print(f"Walk-forward validation: {total} tickers", flush=True)
    print(f"  FIT  {start} -> {split_str}   (pick best delay 0-{max_delay}d)", flush=True)
    print(f"  TEST {split_str} -> {end}   (out-of-sample)\n", flush=True)

    rows = []
    for i, (ticker, name, market) in enumerate(tickers, 1):
        try:
            r = evaluate(ticker, start, end, split, max_delay)
        except Exception as e:
            r = {"Ticker": ticker, "Error": str(e)}
        r["Name"] = name
        r["Market"] = market or "?"
        rows.append(r)
        if r["Error"]:
            print(f"[{i:>3}/{total}] {ticker:<10s} {name}  !! {r['Error']}",
                  flush=True)
        elif not r.get("Test_Trades"):
            print(f"[{i:>3}/{total}] {ticker:<10s} {name}  "
                  f"fit: best {r['Fit_Delay']}d  |  test: no trades", flush=True)
        else:
            print(f"[{i:>3}/{total}] {ticker:<10s} {name}  "
                  f"fit: best {r['Fit_Delay']}d ({r['Fit_Best_%']:+.1f}% vs "
                  f"{r['Fit_Base_%']:+.1f}%)  |  test: {r['Test_Fit_%']:+.1f}% "
                  f"vs base {r['Test_Base_%']:+.1f}%  "
                  f"(oracle {r['Oracle_Delay']}d {r['Oracle_%']:+.1f}%)",
                  flush=True)

    # ------------------------------------------------------------------
    # Per-ticker table
    # ------------------------------------------------------------------
    valid = [r for r in rows if not r["Error"] and r.get("Test_Trades")]
    print()
    print("=" * 100)
    print(f"  WALK-FORWARD RESULTS   fit {start} -> {split_str}   |   "
          f"test {split_str} -> {end}")
    print("=" * 100)
    print(f"  {'Mkt':<4s} {'Ticker':<10s} {'Name':<14s}"
          f" {'FitBase':>8s} {'FitDly':>6s} {'FitBest':>8s}"
          f" {'TstTr':>5s} {'TstBase':>8s} {'TstFit':>8s} {'Delta':>8s}"
          f" {'Orcl':>4s} {'Oracle':>8s}")
    print("  " + "-" * 96)
    for r in rows:
        if r["Error"]:
            print(f"  {r['Market']:<4s} {r['Ticker']:<10s} {r['Name'][:14]:<14s}"
                  f"  -- {r['Error'][:24]}")
        elif not r.get("Test_Trades"):
            print(f"  {r['Market']:<4s} {r['Ticker']:<10s} {r['Name'][:14]:<14s}"
                  f" {r['Fit_Base_%']:>+7.2f}% {r['Fit_Delay']:>5d}d"
                  f" {r['Fit_Best_%']:>+7.2f}% {'0':>5s}  -- no test trades --")
        else:
            print(f"  {r['Market']:<4s} {r['Ticker']:<10s} {r['Name'][:14]:<14s}"
                  f" {r['Fit_Base_%']:>+7.2f}% {r['Fit_Delay']:>5d}d"
                  f" {r['Fit_Best_%']:>+7.2f}% {r['Test_Trades']:>5d}"
                  f" {r['Test_Base_%']:>+7.2f}% {r['Test_Fit_%']:>+7.2f}%"
                  f" {r['Test_Delta_%']:>+7.2f}% {r['Oracle_Delay']:>4d}"
                  f" {r['Oracle_%']:>+7.2f}%")
    print("  " + "-" * 96)

    # ------------------------------------------------------------------
    # Aggregate summary (the verdict)
    # ------------------------------------------------------------------
    if valid:
        n = len(valid)

        def stats(col):
            vals = [r[col] for r in valid]
            return sum(vals) / n, sorted(vals)[n // 2]

        base_avg, base_med = stats("Test_Base_%")
        fit_avg, fit_med = stats("Test_Fit_%")
        orc_avg, _ = stats("Oracle_%")
        fit_wins = sum(1 for r in valid if r["Test_Delta_%"] > 0)

        print(f"  Tickers with fit AND test trades: {n}"
              f"   (errors/no-trades excluded: {len(rows) - n})")
        print()
        print(f"  FIT side (in-sample, year 1):")
        fit_base_avg = sum(r["Fit_Base_%"] for r in valid) / n
        fit_best_avg = sum(r["Fit_Best_%"] for r in valid) / n
        print(f"    avg baseline {fit_base_avg:+.2f}%  ->  avg best-delay "
              f"{fit_best_avg:+.2f}%   (what the in-sample fit promises)")
        print()
        print(f"  TEST side (out-of-sample, year 2):")
        print(f"    baseline (0d)      avg {base_avg:+.2f}%   median {base_med:+.2f}%")
        for k in FIXED_RULES:
            k_avg, k_med = stats(f"Test_{k}d_%")
            k_wins = sum(1 for r in valid if r[f"Test_{k}d_%"] > r["Test_Base_%"])
            print(f"    fixed rule {k}d      avg {k_avg:+.2f}%   median {k_med:+.2f}%"
                  f"   beats baseline on {k_wins}/{n}")
        print(f"    fitted delay       avg {fit_avg:+.2f}%   median {fit_med:+.2f}%"
              f"   beats baseline on {fit_wins}/{n}")
        print(f"    oracle (best OOS)  avg {orc_avg:+.2f}%   <- upper bound,"
              f" not achievable live")
        print()
        d = fit_avg - base_avg
        verdict = ("PASSES" if d > 0 else "FAILS")
        print(f"  >> Fitted delay {verdict} out-of-sample: "
              f"avg delta {d:+.2f}% vs baseline")

    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    print(f"\n  Saved: {OUT_CSV} ({len(rows)} rows)")


if __name__ == "__main__":
    main()

