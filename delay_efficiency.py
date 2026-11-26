"""
delay_efficiency.py — Per-trade efficiency sweep over entry delays.

For one stock, sweep entry delay 0..max_delay trading days and report
per-trade efficiency statistics instead of total return.

Rationale: total return conflates per-trade edge with trade count and
compounding sequence. Per-trade Sharpe (mean/std) isolates the quality
of a single deployment of capital, and t = Sharpe * sqrt(n) quantifies
statistical significance given the small trade counts.

Columns
-------
  dly      entry delay in trading days
  n        surviving trades (whipsaws are deleted as delay grows)
  mean     mean return per taken trade, %
  std      std dev of per-trade returns, %
  sharpe   mean / std
  t        sharpe * sqrt(n)     (* = |t| >= 1.96, ~95% significant)
  med      median per-trade return, %
  skew     moment skewness of per-trade returns
  mean/sig mean return per ORIGINAL signal, % (deleted trades count 0% —
           the opportunity-cost-honest view)
  days     mean calendar days held
  tot_ret  compounded total return over the window, % (for reference)

Usage:
    python3 delay_efficiency.py 300806.SZ
    python3 delay_efficiency.py AAPL 2024-11-26 2026-11-27 10
"""

import sys

import pandas as pd

from daily_stage_v6 import INITIAL_CAPITAL
from trade_pairs import default_dates, run_pipeline
from trade_pairs_delay import repriced_pairs, per_trade_stats

MAX_DELAY = 10


def print_table(rows, ticker, start, end):
    print()
    print("=" * 88)
    print(f"  {ticker} — per-trade efficiency by entry delay   {start} -> {end}")
    print("=" * 88)
    print(f"  {'dly':>3s} {'n':>3s} {'mean':>8s} {'std':>7s} {'sharpe':>7s}"
          f" {'t':>6s} {'sig':>3s} {'med':>7s} {'skew':>6s} {'mean/sig':>8s}"
          f" {'days':>5s} {'tot_ret':>9s}")
    print("  " + "-" * 84)
    for s in rows:
        if s["n"] == 0:
            print(f"  {s['delay']:>3d} {'0':>3s}   -- no trades survive --")
            continue
        sig = "*" if abs(s["t"]) >= 1.96 else ""
        mps = (f"{s['mean_per_signal']:>+7.2f}%"
               if s["mean_per_signal"] is not None else "--")
        print(f"  {s['delay']:>3d} {s['n']:>3d} {s['mean']:>+7.2f}%"
              f" {s['std']:>6.2f}% {s['sharpe']:>+7.3f} {s['t']:>+5.2f} {sig:>3s}"
              f" {s['med']:>+6.2f}% {s['skew']:>+6.2f} {mps}"
              f" {s['avg_days']:>4.0f}d {s['tot_ret']:>+8.2f}%")
    print("  " + "-" * 84)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        sys.exit(1)

    ticker = args[0].upper()
    def_start, def_end = default_dates()
    start = args[1] if len(args) > 1 else def_start
    end = args[2] if len(args) > 2 else def_end
    try:
        max_delay = int(args[3]) if len(args) > 3 else MAX_DELAY
    except ValueError:
        print(f"  !! max_delay must be an integer, got {args[3]!r}")
        sys.exit(1)

    print(f"\nFetching {ticker} ({start} -> {end}) ...")
    result = run_pipeline(ticker, start, end)
    if result is None:
        print(f"  !! No data returned for {ticker}. Check ticker symbol and date range.")
        sys.exit(1)
    stages_df, signals, trades, _ = result

    if not trades:
        print("  No BUY -> SELL pairs were generated in this period.")
        return
    n_signals = len(trades)

    rows = []
    for d in range(max_delay + 1):
        pairs, cap = repriced_pairs(stages_df, trades, d)
        s = per_trade_stats(pairs, n_signals=n_signals)
        s["delay"] = d
        s["tot_ret"] = (cap / INITIAL_CAPITAL - 1) * 100
        rows.append(s)

    print_table(rows, ticker, start, end)

    # ---------------- best delay by the two objectives ----------------
    valid = [r for r in rows if r["n"] >= 2]
    base = rows[0]
    if valid:
        best_sh = max(valid, key=lambda r: r["sharpe"])
        print(f"  >> best Sharpe : {best_sh['delay']}d"
              f"  (Sharpe {best_sh['sharpe']:+.3f} vs 0d {base['sharpe']:+.3f},"
              f"  t = {best_sh['t']:+.2f})")
        if base["mean_per_signal"] is not None:
            best_ms = max(valid, key=lambda r: r["mean_per_signal"])
            print(f"  >> best mean/sig: {best_ms['delay']}d"
                  f"  ({best_ms['mean_per_signal']:+.2f}%"
                  f" vs 0d {base['mean_per_signal']:+.2f}%)")
        print()
        print("  Notes:")
        print("   - best-of-11 selection is in-sample: the winner embeds luck.")
        print("     Trust the * marks (|t| >= 1.96) and consistency across")
        print("     neighbouring delays more than the single best row.")
        print("   - 'mean' improves mechanically as delay deletes losers;")
        print("     'mean/sig' (deleted = 0%) is the deletion-honest view.")
        print()


if __name__ == "__main__":
    main()

