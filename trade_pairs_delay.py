"""
trade_pairs_delay.py — Sweep BUY-entry delays for the stage-pattern strategy.

Idea: most losing pairs are short "whipsaw" trades. Delaying the BUY by N
trading days re-prices the entry at the close N bars after the signal bar;
pairs whose SELL arrives on/before the delayed entry are deleted entirely.
The engine's signals (BUY/SELL dates) don't depend on the delay, so the
pipeline runs once and each trade is simply re-priced per delay value.

Compounding (100% of cash per trade) is recomputed over surviving pairs.

Usage:
    python3 trade_pairs_delay.py 300806.SZ
    python3 trade_pairs_delay.py AAPL 2024-01-01 2026-11-27 10
        args: <ticker> [start] [end] [max_delay_trading_days=10]
"""

import sys

import pandas as pd

from daily_stage_v6 import MACD_FILTER_MODE, INITIAL_CAPITAL
from trade_pairs import default_dates, run_pipeline, is_auto_closed


# ---------------------------------------------------------------------------
# Core: re-price trades with a delayed entry
# ---------------------------------------------------------------------------
def repriced_pairs(stages_df, trades, delay):
    """Re-price every trade with the BUY delayed by `delay` trading days.

    Entry moves from the signal bar's close to the close `delay` bars later.
    A pair is deleted when the delayed entry lands on/after its exit bar.
    Returns (pairs, final_capital).
    """
    full = stages_df.reset_index(drop=True)
    dates = pd.to_datetime(full["Date"])
    closes = full["Close"].tolist()
    idx_of = {d: i for i, d in enumerate(dates)}

    capital = INITIAL_CAPITAL
    pairs = []
    for t in trades:
        entry = pd.to_datetime(t["Entry Date"])
        exit_ = pd.to_datetime(t["Exit Date"])
        i0 = idx_of.get(entry)
        if i0 is None:
            continue
        i1 = i0 + delay
        if i1 >= len(full):
            continue                    # no bar available to buy at
        buy_date = dates.iloc[i1]
        if buy_date >= exit_:
            continue                    # pair deleted — whipsaw filtered out
        buy_price = closes[i1]
        j1 = idx_of.get(exit_)
        sell_price = closes[j1] if j1 is not None else t["Exit Price"]
        capital = capital / buy_price * sell_price
        pairs.append({
            "#": len(pairs) + 1,
            "buy_date": buy_date.strftime("%Y-%m-%d"),
            "buy_price": round(buy_price, 2),
            "sell_date": exit_.strftime("%Y-%m-%d"),
            "sell_price": sell_price,
            "ret": (sell_price / buy_price - 1) * 100,
            "days": (exit_ - buy_date).days,
        })
    return pairs, capital


def summarize(pairs, final_capital):
    """Stats dict for one delay setting (same metrics as trade_pairs.py)."""
    total_return = (final_capital / INITIAL_CAPITAL - 1) * 100
    if not pairs:
        return {"n": 0, "wins": 0, "win_rate": float("nan"),
                "total_return": total_return, "avg_win": 0.0,
                "avg_loss": 0.0, "avg_days": 0.0}
    wins = [p for p in pairs if p["ret"] > 0]
    losses = [p for p in pairs if p["ret"] <= 0]
    return {
        "n": len(pairs),
        "wins": len(wins),
        "win_rate": len(wins) / len(pairs) * 100,
        "total_return": total_return,
        "avg_win": sum(p["ret"] for p in wins) / len(wins) if wins else 0.0,
        "avg_loss": sum(p["ret"] for p in losses) / len(losses) if losses else 0.0,
        "avg_days": sum(p["days"] for p in pairs) / len(pairs),
    }


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------
def print_sweep(rows, ticker, start, end):
    """Comparison table: one row per delay value."""
    print()
    print("=" * 84)
    print(f"  {ticker} — entry-delay sweep   {start} -> {end}   (MACD: {MACD_FILTER_MODE})")
    print("=" * 84)
    print(f"  {'Delay':>5s}  {'Trades':>6s}  {'Wins':>4s}  {'WinRate':>7s}"
          f"  {'TotalRet':>9s}  {'AvgWin':>8s}  {'AvgLoss':>8s}"
          f"  {'AvgDays':>7s}  {'FinalCap':>12s}")
    print("  " + "-" * 80)
    for s in rows:
        wr = f"{s['win_rate']:.1f}%" if s["n"] else "--"
        print(f"  {s['delay']:>4d}d  {s['n']:>6d}  {s['wins']:>4d}  {wr:>7s}"
              f"  {s['total_return']:>+8.2f}%  {s['avg_win']:>+7.2f}%"
              f"  {s['avg_loss']:>+7.2f}%  {s['avg_days']:>6.0f}d"
              f"  {s['final_capital']:>12,.2f}")
    print("  " + "-" * 80)


def print_pairs(pairs, delay, auto_closed, last_bar):
    """Pairs table for one delay value (same layout as trade_pairs.py)."""
    print()
    print("=" * 78)
    print(f"  Recalculated Buy/Sell pairs — entry delayed by {delay} trading day(s)")
    print("=" * 78)
    if not pairs:
        print("  No pairs survive this delay.")
        return
    print(f"  {'#':>3s}  {'Buy Date':<10s}  {'Buy Price':>10s}  {'Sell Date':<10s}"
          f"  {'Sell Price':>10s}  {'Return':>9s}  {'Days':>5s}")
    print("  " + "-" * 74)
    for p in pairs:
        note = " *" if (auto_closed and p["sell_date"] == last_bar) else ""
        print(f"  {p['#']:>3d}  {p['buy_date']:<10s}  {p['buy_price']:>10.2f}"
              f"  {p['sell_date']:<10s}  {p['sell_price']:>10.2f}"
              f"  {p['ret']:>+8.2f}%  {p['days']:>4d}d{note}")
    print("  " + "-" * 74)


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
        max_delay = int(args[3]) if len(args) > 3 else 10
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

    auto_closed = is_auto_closed(stages_df, signals, trades)
    last_bar = pd.to_datetime(stages_df["Date"].iloc[-1]).strftime("%Y-%m-%d")

    # ---------------- sweep delay = 0..max_delay ----------------
    rows = []
    for n in range(max_delay + 1):
        pairs, cap = repriced_pairs(stages_df, trades, n)
        s = summarize(pairs, cap)
        s.update({"delay": n, "pairs": pairs, "final_capital": cap})
        rows.append(s)

    print_sweep(rows, ticker, start, end)

    best = max(rows, key=lambda s: s["total_return"])
    base = rows[0]
    print(f"  >> Best delay: {best['delay']}d  ->  total return {best['total_return']:+.2f}%"
          f"  (baseline 0d: {base['total_return']:+.2f}%,"
          f"  delta {best['total_return'] - base['total_return']:+.2f}%)")
    print("  (best delay is fitted on this same window — in-sample result,")
    print("   validate on out-of-sample data before trading it)")

    # ---------------- recalculated pairs for the best delay ----------------
    pairs = best["pairs"]
    print_pairs(pairs, best["delay"], auto_closed, last_bar)
    if pairs:
        wins = [p for p in pairs if p["ret"] > 0]
        losses = [p for p in pairs if p["ret"] <= 0]
        days = [p["days"] for p in pairs]
        print(f"  Trades: {len(pairs)}   Wins: {len(wins)}   Losses: {len(losses)}"
              f"   Win rate: {best['win_rate']:.1f}%")
        print(f"  Total return: {best['total_return']:+.2f}%"
              f"   (capital {INITIAL_CAPITAL:,.0f} -> {best['final_capital']:,.2f})")
        print(f"  Avg win: {best['avg_win']:+.2f}%   Avg loss: {best['avg_loss']:+.2f}%")
        print(f"  Duration: avg {sum(days) / len(days):.0f}d"
              f"   min {min(days)}d   max {max(days)}d")
        if auto_closed and pairs[-1]["sell_date"] == last_bar:
            print("  * still-open position — closed at the last available daily close")
        print()


if __name__ == "__main__":
    main()

