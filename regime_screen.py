"""
regime_screen.py — Reusable regime screener for the watchlist.

Computes each watchlist ticker's monthly regime (A/B/C/D) over a fixed date
range, caches the result, and answers repeated questions like:
    - who is currently in regime A?
    - who just flipped C -> A?

Regime definition (same as app.py / rolling_stage_analysis.py):
    A = up + calm    B = up + volatile    C = down + calm    D = down + volatile
    "calm vs volatile" = volatility drag vs the ticker's own median over the range.

Usage:
    python3 regime_screen.py                        # stocks currently in A
    python3 regime_screen.py --regime B             # stocks currently in B
    python3 regime_screen.py --regime A,B           # stocks currently in A or B
    python3 regime_screen.py --all                  # full table (every ticker)
    python3 regime_screen.py --flip C-A             # stocks that JUST flipped C->A
    python3 regime_screen.py --refresh              # ignore cache and re-fetch
    python3 regime_screen.py --start 2023-01-01 --end 2026-08-30

Results are cached to regime_screen_cache.csv. Re-runs are instant unless
--refresh is used. The cache is keyed by --start/--end.
"""

import argparse
import os
import warnings
from collections import Counter

import numpy as np
import pandas as pd

from rolling_stage_analysis import run_rolling_analysis

warnings.filterwarnings("ignore")

DEFAULT_START = "2023-01-01"
DEFAULT_END = "2026-08-30"
HERE = os.path.dirname(os.path.abspath(__file__))
WATCHLIST = os.path.join(HERE, "watchlist.yml")
CACHE = os.path.join(HERE, "regime_screen_cache.csv")


def parse_watchlist(path=WATCHLIST):
    """Return list of (ticker, name, market), deduped by ticker."""
    items, group = [], None
    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.endswith(":") and ":" not in line[:-1]:
                group = line[:-1].strip().lower()
                continue
            if ":" in line:
                left, right = line.split(":", 1)
                t = left.strip().upper()
                name = right.split("#")[0].strip()
                if t:
                    items.append((t, name, group))
    seen = {}
    for t, n, g in items:
        seen.setdefault(t, (t, n, g))
    return list(seen.values())


def classify(results_df):
    """Return DataFrame[Window_End, Code] in ascending order (0-trade dropped)."""
    wr = results_df["Win_Rate_%"] / 100.0
    aw = results_df["Avg_Win_%"].fillna(0.0)
    al = results_df["Avg_Loss_%"].fillna(0.0)
    arith = aw * wr + al * (1.0 - wr)
    trades = results_df["Trades"].replace(0, np.nan)
    geo = ((1.0 + results_df["Return_%"] / 100.0) ** (1.0 / trades) - 1.0) * 100.0
    drag = arith - geo
    med = drag.median()
    up = arith >= 0.0
    vol = drag > med
    df = results_df.copy()
    df["Code"] = np.select([up & ~vol, up & vol, ~up & ~vol, ~up & vol],
                           ["A", "B", "C", "D"], default="")
    df = df[df["Code"] != ""].sort_values("Window_End")
    return df[["Window_End", "Code"]]


def detect_transition(codes, from_r, to_r, to_run_max=2, from_run_min=1):
    """Return (ok, from_run_len, to_run_len) for a trailing to_r run of length
    1..to_run_max preceded by a from_r run of length >= from_run_min."""
    if not codes or codes[-1] != to_r:
        return False, 0, 0
    to_run = 0
    i = len(codes) - 1
    while i >= 0 and codes[i] == to_r:
        to_run += 1
        i -= 1
    if to_run > to_run_max:
        return False, 0, to_run
    if i < 0 or codes[i] != from_r:
        return False, 0, to_run
    from_run = 0
    j = i
    while j >= 0 and codes[j] == from_r:
        from_run += 1
        j -= 1
    return from_run >= from_run_min, from_run, to_run
def load_cache(start, end):
    """Return {ticker: (window_ends, codes)} for the given date range."""
    if not os.path.exists(CACHE):
        return {}
    df = pd.read_csv(CACHE, dtype={"ticker": str})
    df = df[(df["start"] == start) & (df["end"] == end)]
    out = {}
    for ticker, g in df.groupby("ticker"):
        g = g.sort_values("window_end")
        out[ticker] = (g["window_end"].tolist(), g["code"].tolist())
    return out


def save_cache(cache, start, end):
    rows = []
    for ticker, (ends, codes) in cache.items():
        for w, c in zip(ends, codes):
            rows.append({"ticker": ticker, "start": start, "end": end,
                         "window_end": w, "code": c})
    pd.DataFrame(rows).to_csv(CACHE, index=False)


def compute(ticker, start, end):
    """Return (window_ends, codes) for one ticker, or None."""
    results_df, _ = run_rolling_analysis(
        ticker, window_years=1, step_months=1, start_date=start, end_date=end
    )
    if results_df.empty:
        return None
    sub = classify(results_df)
    ends = [str(pd.Timestamp(w).date()) for w in sub["Window_End"]]
    return ends, sub["Code"].tolist()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--regime", default=None,
                    help="comma-separated list, e.g. A or A,B (default: A)")
    ap.add_argument("--all", action="store_true",
                    help="show every ticker regardless of regime")
    ap.add_argument("--flip", default=None,
                    help="detect a transition, e.g. C-A")
    ap.add_argument("--refresh", action="store_true",
                    help="ignore cache and re-fetch every ticker")
    ap.add_argument("--start", default=DEFAULT_START)
    ap.add_argument("--end", default=DEFAULT_END)
    args = ap.parse_args()

    tickers = parse_watchlist()
    cache = {} if args.refresh else load_cache(args.start, args.end)

    for t, name, market in tickers:
        if t in cache:
            continue
        print(f"computing {t} ...", flush=True)
        try:
            r = compute(t, args.start, args.end)
        except Exception as e:
            print(f"  !! {t}: {e}", flush=True)
            r = None
        cache[t] = r or ([], [])

    save_cache(cache, args.start, args.end)

    # current-regime view
    rows = []
    for t, name, market in tickers:
        ends, codes = cache.get(t, ([], []))
        cur = codes[-1] if codes else "?"
        rows.append((market or "?", t, name, cur, codes, ends))

    if args.flip:
        from_r, to_r = args.flip.upper().split("-")
        print(f"\nStocks that JUST flipped {from_r} -> {to_r}:\n")
        found = 0
        for market, t, name, cur, codes, ends in rows:
            ok, from_run, to_run = detect_transition(codes, from_r, to_r)
            if ok:
                found += 1
                last = ",".join(codes[-8:])
                print(f"  [{market}] {t:<12s} {name}")
                print(f"       {from_r}x{from_run} -> {to_r}x{to_run}   recent=[{last}]")
        print(f"\n  matches: {found}")
        return

    want = set(args.regime.split(",")) if args.regime else {"A"}
    dist = Counter(r[3] for r in rows)
    print(f"\nCurrent regime distribution: "
          + "  ".join(f"{k}={dist.get(k, 0)}" for k in ("A", "B", "C", "D")) + "\n")

    show = rows if args.all else [r for r in rows if r[3] in want]
    for market, t, name, cur, codes, ends in show:
        tail = ",".join(codes[-6:]) if codes else ""
        print(f"  [{market}] {t:<12s} {name:<14s} regime={cur}   recent=[{tail}]")
    print(f"\n  shown: {len(show)} / {len(rows)}")


if __name__ == "__main__":
    main()

