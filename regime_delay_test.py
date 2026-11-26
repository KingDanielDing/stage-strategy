"""
regime_delay_test.py — Regime-conditional entry delay.

Hypothesis: whipsaw losses cluster in down/choppy market regimes.
Rule: delay the BUY by K trading days ONLY when the trailing 12-month
regime (A/B/C/D, from rolling_stage_analysis) at the entry date belongs
to a chosen set (e.g. {C,D}); otherwise enter at the signal close.

Discipline (lessons from delay_walkforward / delay_walkforward3):
  - candidate rules are selected on window A only
  - validated on untouched windows B and C
  - regime lookup is strictly causal: latest window ending strictly
    BEFORE the entry date (bisect_left)

Usage:
    python3 regime_delay_test.py

Saved: regime_delay_test.csv
"""

import bisect
import contextlib
import io
import sys

import pandas as pd

from daily_stage_v6 import INITIAL_CAPITAL
from trade_pairs import run_pipeline
from trade_pairs_delay import repriced_pairs, summarize
from rolling_stage_analysis import run_rolling_analysis
from regime_screen import classify
from delay_walkforward3 import window_trades
from batch_delay_screen import parse_watchlist

RULE_DELAYS = (1, 2, 3)
OUT_CSV = "regime_delay_test.csv"


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------
def regime_at(dt, ends, codes):
    """Regime code as of the latest window ending strictly before dt."""
    i = bisect.bisect_left(ends, dt) - 1
    return codes[i] if i >= 0 else None


def conditional_pairs(stages_df, trades, delay, regime_set, ends, codes):
    """Like repriced_pairs, but the entry delay is applied per trade:
    `delay` days when the regime at entry is in `regime_set`, else 0.
    Returns (pairs, final_capital, n_delayed)."""
    full = stages_df.reset_index(drop=True)
    dates = pd.to_datetime(full["Date"])
    closes = full["Close"].tolist()
    idx_of = {d: i for i, d in enumerate(dates)}
    capital = INITIAL_CAPITAL
    n_delayed = 0
    pairs = []
    for t in trades:
        entry = pd.to_datetime(t["Entry Date"])
        exit_ = pd.to_datetime(t["Exit Date"])
        i0 = idx_of.get(entry)
        if i0 is None:
            continue
        d = delay if regime_at(entry, ends, codes) in regime_set else 0
        if d:
            n_delayed += 1
        i1 = i0 + d
        if i1 >= len(full):
            continue
        buy_date = dates.iloc[i1]
        if buy_date >= exit_:
            continue
        j1 = idx_of.get(exit_)
        buy_price = closes[i1]
        sell_price = closes[j1] if j1 is not None else t["Exit Price"]
        capital = capital / buy_price * sell_price
        pairs.append({"ret": (sell_price / buy_price - 1) * 100,
                      "days": (exit_ - buy_date).days})
    return pairs, capital, n_delayed


def evaluate(ticker, bounds, candidates):
    """Trade pipeline (3y data) + regime series (4y rolling windows);
    per-window total return for every candidate rule."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        result = run_pipeline(ticker, bounds["fetch_start"], bounds["end"])
        results_df, _ = run_rolling_analysis(
            ticker, window_years=1, step_months=1,
            start_date=bounds["regime_start"], end_date=bounds["end"])

    if result is None:
        return {"Ticker": ticker, "Error": "no data"}
    stages_df, signals, trades, _ = result
    if not trades:
        return {"Ticker": ticker, "Error": "no trades"}
    if results_df is None or results_df.empty:
        return {"Ticker": ticker, "Error": "no regimes"}

    reg = classify(results_df)
    if reg.empty:
        return {"Ticker": ticker, "Error": "no regimes"}
    ends = [pd.Timestamp(x) for x in reg["Window_End"]]
    codes = [str(c) for c in reg["Code"]]

    row = {"Ticker": ticker, "Error": "", "n_regimes": len(codes)}
    for w in ("A", "B", "C"):
        lo, hi = bounds[w]
        tw = window_trades(trades, stages_df, lo, hi)
        row[f"{w}_trades"] = len(tw)
        if not tw:
            row[f"{w}_base"] = None
            for cid in candidates:
                row[f"{w}_{cid}"] = None
                row[f"{w}_{cid}_n"] = None
            continue
        pairs, cap = repriced_pairs(stages_df, tw, 0)
        row[f"{w}_base"] = round(summarize(pairs, cap)["total_return"], 2)
        for cid, (rset, k) in candidates.items():
            p2, c2, nd = conditional_pairs(stages_df, tw, k, rset, ends, codes)
            row[f"{w}_{cid}"] = round(summarize(p2, c2)["total_return"], 2)
            row[f"{w}_{cid}_n"] = nd
    return row


def eval_candidate(rows, w, cid):
    """Aggregate stats of one candidate rule on window w (cohort: trades > 0)."""
    cohort = [r for r in rows if not r["Error"] and r[f"{w}_trades"] > 0]
    if not cohort:
        return None
    deltas = [r[f"{w}_{cid}"] - r[f"{w}_base"] for r in cohort]
    n = len(cohort)
    return {"n": n,
            "avg_base": sum(r[f"{w}_base"] for r in cohort) / n,
            "avg_rule": sum(r[f"{w}_{cid}"] for r in cohort) / n,
            "avg_delta": sum(deltas) / n,
            "med_delta": sorted(deltas)[n // 2],
            "wins": sum(1 for d in deltas if d > 0),
            "fired": sum(r[f"{w}_{cid}_n"] for r in cohort)}



# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    now = pd.Timestamp.now(tz="Asia/Shanghai")
    end = (now + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    start = (now - pd.Timedelta(days=3 * 365)).strftime("%Y-%m-%d")
    regime_start = (now - pd.Timedelta(days=4 * 365)).strftime("%Y-%m-%d")
    b1 = pd.Timestamp(start) + pd.Timedelta(days=365)
    b2 = pd.Timestamp(start) + pd.Timedelta(days=730)
    bounds = {"fetch_start": start, "end": end, "regime_start": regime_start,
              "A": (None, b1), "B": (b1, b2), "C": (b2, None)}

    candidates = {}
    for rname, rset in [("CD", {"C", "D"}), ("D", {"D"}), ("C", {"C"}),
                        ("BCD", {"B", "C", "D"})]:
        for k in RULE_DELAYS:
            candidates[f"{rname}_{k}d"] = (rset, k)
    # references: universal delay (applies in every regime, incl. unknown)
    candidates["ALL_1d"] = ({"A", "B", "C", "D", None}, 1)
    candidates["ALL_2d"] = ({"A", "B", "C", "D", None}, 2)
    cond_ids = [c for c in candidates if not c.startswith("ALL")]

    tickers = parse_watchlist()
    total = len(tickers)
    print(f"Regime-conditional delay test: {total} tickers", flush=True)
    print(f"  A (design)   {start} -> {b1.date()}", flush=True)
    print(f"  B (validate) {b1.date()} -> {b2.date()}", flush=True)
    print(f"  C (test)     {b2.date()} -> {end}\n", flush=True)

    rows = []
    for i, (ticker, name, market) in enumerate(tickers, 1):
        try:
            r = evaluate(ticker, bounds, candidates)
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
                  f"A:{r['A_trades']}t B:{r['B_trades']}t C:{r['C_trades']}t"
                  f"  regimes:{r['n_regimes']}", flush=True)

    # ------------------------------------------------------------------
    # DESIGN PHASE — window A only
    # ------------------------------------------------------------------
    print()
    print("=" * 88)
    print(f"  DESIGN PHASE (window A in-sample, {start} -> {b1.date()})")
    print("=" * 88)
    print(f"  {'rule':<10s} {'n':>4s} {'avg_base':>9s} {'avg_rule':>9s}"
          f" {'avg_delta':>10s} {'med_d':>8s} {'wins':>7s} {'fired':>6s}")
    print("  " + "-" * 72)
    scored = []
    for cid in cond_ids:
        res = eval_candidate(rows, "A", cid)
        if res:
            scored.append((cid, res))
    for cid, res in sorted(scored, key=lambda x: -x[1]["avg_delta"]):
        print(f"  {cid:<10s} {res['n']:>4d} {res['avg_base']:>+8.2f}%"
              f" {res['avg_rule']:>+8.2f}% {res['avg_delta']:>+9.2f}%"
              f" {res['med_delta']:>+7.2f}% {res['wins']:>3d}/{res['n']}"
              f" {res['fired']:>6d}")
    sel, sel_a = max(scored, key=lambda c: c[1]["avg_delta"])
    print(f"\n  >> SELECTED RULE: {sel}   (window-A avg delta "
          f"{sel_a['avg_delta']:+.2f}pp, fired on {sel_a['fired']} entries)")

    # ------------------------------------------------------------------
    # VALIDATION PHASE — windows B and C
    # ------------------------------------------------------------------
    print()
    print("=" * 88)
    print(f"  VALIDATION (untouched windows) — selected rule: {sel}")
    print("=" * 88)
    ref_ids = [sel, "CD_1d", "CD_2d", "ALL_1d"]
    deltas = {}
    for w, label in (("B", "validate"), ("C", "test")):
        print(f"\n  Window {w} ({label}):")
        print(f"  {'rule':<10s} {'n':>4s} {'avg_base':>9s} {'avg_rule':>9s}"
              f" {'avg_delta':>10s} {'med_d':>8s} {'wins':>7s} {'fired':>6s}")
        print("  " + "-" * 72)
        for cid in ref_ids:
            if cid not in candidates:
                continue
            res = eval_candidate(rows, w, cid)
            if res is None:
                continue
            tag = " <- selected" if cid == sel else ""
            print(f"  {cid:<10s} {res['n']:>4d} {res['avg_base']:>+8.2f}%"
                  f" {res['avg_rule']:>+8.2f}% {res['avg_delta']:>+9.2f}%"
                  f" {res['med_delta']:>+7.2f}% {res['wins']:>3d}/{res['n']}"
                  f" {res['fired']:>6d}{tag}")
            deltas[w] = eval_candidate(rows, w, sel)["avg_delta"]

    print()
    if deltas.get("B") is not None and deltas.get("C") is not None:
        ok = deltas["B"] > 0 and deltas["C"] > 0
        verdict = ("PASSES both untouched windows" if ok
                   else "FAILS (no consistent out-of-sample edge)")
        print(f"  >> Selected rule {verdict}:"
              f"  B {deltas['B']:+.2f}pp, C {deltas['C']:+.2f}pp")

    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    print(f"\n  Saved: {OUT_CSV} ({len(rows)} rows)")


if __name__ == "__main__":
    main()

