"""
batch_delay_screen.py — Find the best BUY-entry delay for every watchlist stock.

For each ticker in watchlist.yml, runs the stage-pattern pipeline once
(default 2-year window, same as trade_pairs.py), sweeps entry delays
0..MAX_DELAY trading days (re-pricing logic from trade_pairs_delay.py),
and reports the delay that maximizes total return.

Usage:
    python3 batch_delay_screen.py [max_delay=10]

Results are printed to screen and saved to delay_screen_all.csv.
Note: best delay is fitted on the same window — an in-sample optimum.
"""

import contextlib
import io
import sys

import pandas as pd

from trade_pairs import default_dates, run_pipeline
from trade_pairs_delay import repriced_pairs, summarize

MAX_DELAY = 10
WATCHLIST = "watchlist.yml"
OUT_CSV = "delay_screen_all.csv"


def parse_watchlist(path=WATCHLIST):
    """Return list of (ticker, name, market), deduped by ticker.
    (Same hand-rolled parser as regime_screen.py — no PyYAML needed.)"""
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


def evaluate_ticker(ticker, start, end, max_delay):
    """Pipeline + delay sweep for one ticker. Returns a result row dict."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        result = run_pipeline(ticker, start, end)
    if result is None:
        return {"Ticker": ticker, "Error": "no data"}
    stages_df, signals, trades, _ = result
    if not trades:
        return {"Ticker": ticker, "Error": "no trades"}

    rows = []
    for n in range(max_delay + 1):
        pairs, cap = repriced_pairs(stages_df, trades, n)
        rows.append((n, summarize(pairs, cap)))

    base = rows[0][1]
    best_n, best = max(rows, key=lambda r: r[1]["total_return"])
    return {
        "Ticker": ticker,
        "Trades": base["n"],
        "Base_Ret_%": round(base["total_return"], 2),
        "Base_WR_%": round(base["win_rate"], 1),
        "Best_Delay": best_n,
        "Best_Trades": best["n"],
        "Best_Ret_%": round(best["total_return"], 2),
        "Best_WR_%": round(best["win_rate"], 1) if best["n"] else None,
        "Delta_%": round(best["total_return"] - base["total_return"], 2),
        "Error": "",
    }

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    max_delay = int(sys.argv[1]) if len(sys.argv) > 1 else MAX_DELAY
    start, end = default_dates()
    tickers = parse_watchlist()
    total = len(tickers)
    print(f"Scanning {total} watchlist tickers  ({start} -> {end},"
          f"  delays 0-{max_delay}d)\n", flush=True)

    rows = []
    for i, (ticker, name, market) in enumerate(tickers, 1):
        try:
            r = evaluate_ticker(ticker, start, end, max_delay)
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
                  f"base {r['Base_Ret_%']:+.1f}%  ->  "
                  f"best {r['Best_Delay']}d {r['Best_Ret_%']:+.1f}%  "
                  f"(delta {r['Delta_%']:+.1f})", flush=True)

    # ------------------------------------------------------------------
    # Full table (watchlist order)
    # ------------------------------------------------------------------
    ok = [r for r in rows if not r["Error"]]
    print()
    print("=" * 96)
    print(f"  BEST ENTRY DELAY PER STOCK   ({start} -> {end})")
    print("=" * 96)
    print(f"  {'Mkt':<4s} {'Ticker':<10s} {'Name':<14s} {'Trades':>6s}"
          f"  {'BaseRet':>9s} {'BaseWR':>7s} {'Delay':>5s} {'BestTr':>6s}"
          f"  {'BestRet':>9s} {'BestWR':>7s} {'Delta':>8s}")
    print("  " + "-" * 92)
    for r in rows:
        if r["Error"]:
            print(f"  {r['Market']:<4s} {r['Ticker']:<10s} {r['Name'][:14]:<14s}"
                  f"  {'--':>6s}  {'--':>9s} {'--':>7s} {'--':>5s} {'--':>6s}"
              f"  {'--':>9s} {'--':>7s}  {r['Error'][:20]}")
        else:
            bwr = (f"{r['Best_WR_%']:.1f}%" if r["Best_WR_%"] is not None
                   else "--")
            print(f"  {r['Market']:<4s} {r['Ticker']:<10s} {r['Name'][:14]:<14s}"
                  f" {r['Trades']:>6d}  {r['Base_Ret_%']:>+8.2f}%"
                  f" {r['Base_WR_%']:>6.1f}% {r['Best_Delay']:>4d}d"
                  f" {r['Best_Trades']:>6d}  {r['Best_Ret_%']:>+8.2f}%"
                  f" {bwr:>7s} {r['Delta_%']:>+7.2f}%")
    print("  " + "-" * 92)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    if ok:
        dist = {}
        for r in ok:
            dist[r["Best_Delay"]] = dist.get(r["Best_Delay"], 0) + 1
        improved = sum(1 for r in ok if r["Delta_%"] > 0)
        deltas = [r["Delta_%"] for r in ok]
        print(f"  Evaluated: {len(ok)}   Errors: {len(rows) - len(ok)}")
        print(f"  Improved by delaying: {improved}/{len(ok)}"
              f"   avg delta {sum(deltas) / len(deltas):+.2f}%"
              f"   median delta {sorted(deltas)[len(deltas) // 2]:+.2f}%")
        print("  Best-delay distribution: "
              + "  ".join(f"{d}d x{c}" for d, c in sorted(dist.items())))
        print("  (in-sample optima on this 2y window — validate out-of-sample)")

    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    print(f"\n  Saved: {OUT_CSV} ({len(rows)} rows)")


if __name__ == "__main__":
    main()

