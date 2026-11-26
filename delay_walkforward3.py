"""
delay_walkforward3.py — 3-window walk-forward validation of the
"weak stocks get a short entry delay" rule.

Windows (approx. calendar years, computed in Beijing time):
    A (design)   : start     -> start+1y    rule is designed ONLY on A
    B (validate) : start+1y  -> start+2y    untouched check 1
    C (test)     : start+2y  -> end         untouched check 2

Rule form:
    if stock's SOURCE-window baseline (0d delay) return < THETA
        -> delay entry by K trading days in the TARGET window
    else -> no delay (0d)

(THETA, K) are selected by maximizing average delta on window A only;
windows B and C never influence the selection.

Variants reported:
    V1 static   : weakness always assessed on window A
    V2 adaptive : C-window weakness assessed on window B (yearly re-check)

Usage:
    python3 delay_walkforward3.py [max_delay=10]

Results printed to screen and saved to delay_walkforward3.csv.
"""

import contextlib
import io
import sys

import pandas as pd

from trade_pairs import run_pipeline
from trade_pairs_delay import repriced_pairs, summarize
from batch_delay_screen import parse_watchlist

MAX_DELAY = 10
RULE_DELAYS = (1, 2, 3)
OUT_CSV = "delay_walkforward3.csv"


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------
def window_trades(trades, stages_df, lo, hi):
    """Trades with entry in [lo, hi); exits capped at the last bar <= hi.
    lo=None -> from the first bar; hi=None -> to the last bar."""
    dates = pd.to_datetime(stages_df["Date"]).reset_index(drop=True)
    if hi is not None:
        hi_bars = dates[dates <= hi]
        if hi_bars.empty:
            return []
        hi_bar = hi_bars.iloc[-1]
    else:
        hi_bar = dates.iloc[-1]
    out = []
    for t in trades:
        e = pd.to_datetime(t["Entry Date"])
        if lo is not None and e < lo:
            continue
        if hi is not None and e >= hi:
            continue
        t = dict(t)
        if pd.to_datetime(t["Exit Date"]) > hi_bar:
            t["Exit Date"] = hi_bar
        out.append(t)
    return out


def sweep(trades, stages_df, max_delay):
    """Return {delay: total_return} for delays 0..max_delay."""
    out = {}
    for n in range(max_delay + 1):
        pairs, cap = repriced_pairs(stages_df, trades, n)
        out[n] = summarize(pairs, cap)["total_return"]
    return out


def evaluate(ticker, bounds, max_delay):
    """One pipeline run over the full 3y; per-window delay->return maps."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        result = run_pipeline(ticker, bounds["fetch_start"], bounds["end"])
    if result is None:
        return {"Ticker": ticker, "Error": "no data"}
    stages_df, signals, trades, _ = result
    if not trades:
        return {"Ticker": ticker, "Error": "no trades"}

    row = {"Ticker": ticker, "Error": ""}
    for w in ("A", "B", "C"):
        lo, hi = bounds[w]
        tw = window_trades(trades, stages_df, lo, hi)
        row[f"{w}_trades"] = len(tw)
        if tw:
            sw = sweep(tw, stages_df, max_delay)
            row[f"{w}_base"] = round(sw[0], 2)
            for k in RULE_DELAYS:
                row[f"{w}_{k}d"] = round(sw[k], 2)
        else:
            row[f"{w}_base"] = None
            for k in RULE_DELAYS:
                row[f"{w}_{k}d"] = None
    return row


# ---------------------------------------------------------------------------
# Rule machinery
# ---------------------------------------------------------------------------
def apply_rule(rows, src, dst, theta_kind, k):
    """Evaluate one rule on the cohort of stocks with trades in BOTH the
    source and destination windows.

    theta_kind: "none" (universal delay), "zero" (weak iff src base < 0),
                "median" (weak iff src base < cohort median).
    Returns dict with averages, or None if the cohort is empty.
    """
    cohort = [r for r in rows
              if not r["Error"]
              and r[f"{src}_trades"] > 0 and r[f"{dst}_trades"] > 0]
    if not cohort:
        return None
    src_bases = [r[f"{src}_base"] for r in cohort]
    if theta_kind == "none":
        weak = [True] * len(cohort)
    else:
        thr = 0.0 if theta_kind == "zero" else pd.Series(src_bases).median()
        weak = [b < thr for b in src_bases]
    base = [r[f"{dst}_base"] for r in cohort]
    rule = [r[f"{dst}_{k}d"] if w else r[f"{dst}_base"]
            for r, w in zip(cohort, weak)]
    deltas = [a - b for a, b in zip(rule, base)]
    n = len(cohort)
    return {"n": n, "n_weak": sum(weak),
            "avg_base": sum(base) / n, "avg_rule": sum(rule) / n,
            "avg_delta": sum(deltas) / n,
            "med_delta": sorted(deltas)[n // 2],
            "wins": sum(1 for d in deltas if d > 0)}


RULE_LABELS = {"none": "universal", "zero": "weak(<0%)", "median": "weak(<med)"}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    max_delay = int(sys.argv[1]) if len(sys.argv) > 1 else MAX_DELAY
    now = pd.Timestamp.now(tz="Asia/Shanghai")
    end = (now + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    start = (now - pd.Timedelta(days=3 * 365)).strftime("%Y-%m-%d")
    b1 = pd.Timestamp(start) + pd.Timedelta(days=365)
    b2 = pd.Timestamp(start) + pd.Timedelta(days=730)
    bounds = {"fetch_start": start, "end": end,
              "A": (None, b1), "B": (b1, b2), "C": (b2, None)}

    tickers = parse_watchlist()
    total = len(tickers)
    print(f"3-window walk-forward: {total} tickers", flush=True)
    print(f"  A (design)   {start} -> {b1.date()}", flush=True)
    print(f"  B (validate) {b1.date()} -> {b2.date()}", flush=True)
    print(f"  C (test)     {b2.date()} -> {end}\n", flush=True)

    rows = []
    for i, (ticker, name, market) in enumerate(tickers, 1):
        try:
            r = evaluate(ticker, bounds, max_delay)
        except Exception as e:
            r = {"Ticker": ticker, "Error": str(e)}
        r["Name"] = name
        r["Market"] = market or "?"
        rows.append(r)
        if r["Error"]:
            print(f"[{i:>3}/{total}] {ticker:<10s} {name}  !! {r['Error']}",
                  flush=True)
        else:
            print(f"[{i:>3}/{total}] {ticker:<10s} {name}  "
                  f"A:{r['A_trades']}t {r['A_base'] if r['A_base'] is not None else '--'}  "
                  f"B:{r['B_trades']}t  C:{r['C_trades']}t", flush=True)

    # ------------------------------------------------------------------
    # DESIGN PHASE — window A only
    # ------------------------------------------------------------------
    print()
    print("=" * 84)
    print(f"  DESIGN PHASE (window A in-sample, {start} -> {b1.date()})")
    print("=" * 84)
    candidates = []
    for theta in ("none", "zero", "median"):
        for k in RULE_DELAYS:
            res = apply_rule(rows, "A", "A", theta, k)
            if res:
                candidates.append((theta, k, res))
    print(f"  {'rule':<18s} {'n':>4s} {'weak':>5s} {'avg_base':>9s}"
          f" {'avg_rule':>9s} {'avg_delta':>10s} {'wins':>6s}")
    print("  " + "-" * 66)
    for theta, k, res in candidates:
        print(f"  {RULE_LABELS[theta] + ' ' + str(k) + 'd':<18s} {res['n']:>4d}"
              f" {res['n_weak']:>5d} {res['avg_base']:>+8.2f}%"
              f" {res['avg_rule']:>+8.2f}% {res['avg_delta']:>+9.2f}%"
              f" {res['wins']:>3d}/{res['n']}")
    theta, k, sel = max(candidates, key=lambda c: c[2]["avg_delta"])
    print(f"\n  >> SELECTED RULE: {RULE_LABELS[theta]} -> {k}d"
          f"   (window-A avg delta {sel['avg_delta']:+.2f}pp)")


    # ------------------------------------------------------------------
    # VALIDATION PHASE — windows B and C (never used for selection)
    # ------------------------------------------------------------------
    print()
    print("=" * 84)
    print(f"  VALIDATION (untouched windows) — selected rule: "
          f"{RULE_LABELS[theta]} -> {k}d")
    print("=" * 84)
    evals = [
        ("B  (weak from A)", apply_rule(rows, "A", "B", theta, k),
         apply_rule(rows, "A", "B", "none", 1)),
        ("C  (weak from A)", apply_rule(rows, "A", "C", theta, k),
         apply_rule(rows, "A", "C", "none", 1)),
        ("C  (weak from B)", apply_rule(rows, "B", "C", theta, k),
         apply_rule(rows, "B", "C", "none", 1)),
    ]
    print(f"  {'window':<18s} {'n':>4s} {'weak':>5s} {'baseline':>9s}"
          f" {'rule':>9s} {'delta':>8s} {'med_d':>8s} {'wins':>7s}"
          f"   {'univ 1d':>9s}")
    print("  " + "-" * 84)
    for label, res, ref1d in evals:
        if res is None:
            print(f"  {label:<18s}  -- empty cohort --")
            continue
        ref = f"{ref1d['avg_rule']:>+8.2f}%" if ref1d else "--"
        print(f"  {label:<18s} {res['n']:>4d} {res['n_weak']:>5d}"
              f" {res['avg_base']:>+8.2f}% {res['avg_rule']:>+8.2f}%"
              f" {res['avg_delta']:>+8.2f}% {res['med_delta']:>+7.2f}%"
              f" {res['wins']:>3d}/{res['n']}   {ref}")
    print("  " + "-" * 84)

    b_res, c_res = evals[0][1], evals[1][1]
    if b_res and c_res:
        ok_b = b_res["avg_delta"] > 0
        ok_c = c_res["avg_delta"] > 0
        verdict = ("PASSES both untouched windows" if ok_b and ok_c
                   else "FAILS (no out-of-sample edge)")
        print(f"  >> Selected rule {verdict}:"
              f"  B {b_res['avg_delta']:+.2f}pp, C {c_res['avg_delta']:+.2f}pp")

    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    print(f"\n  Saved: {OUT_CSV} ({len(rows)} rows)")


if __name__ == "__main__":
    main()

