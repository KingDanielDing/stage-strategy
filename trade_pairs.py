"""
trade_pairs.py — List every BUY → SELL pair the stage-pattern strategy
produces for one stock over a (default) two-year window.

Uses the exact same engine that powers app.py "Single Stock (v6)" mode:
    daily_stage_v4 — retrieve_data, find_fvgs, generate_finals, compute_macd,
                     generate_pure_stages, deduplicate_stages,
                     generate_trade_signals, backtest_trades
    daily_stage_v6 — MACD_FILTER_MODE, INITIAL_CAPITAL

Entries are delayed by ENTRY_DELAY_DAYS trading days (rule of thumb: 1)
to skip whipsaw trades — see delay_walkforward.py / regime_delay_test.py.

Usage:
    python3 trade_pairs.py 300806.SZ
    python3 trade_pairs.py AAPL 2024-11-26 2026-11-27 0   # 4th arg: delay (0=off)
"""

import sys

import pandas as pd

from daily_stage_v4 import (
    retrieve_data,
    find_fvgs,
    generate_finals,
    compute_macd,
    generate_pure_stages,
    deduplicate_stages,
    generate_trade_signals,
    backtest_trades,
)
from daily_stage_v6 import MACD_FILTER_MODE, INITIAL_CAPITAL

ENTRY_DELAY_DAYS = 1   # rule-of-thumb entry delay (trading days); 0 = off


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def default_dates():
    """(start, end) matching app.py: last 2 years, computed in Beijing time."""
    now = pd.Timestamp.now(tz="Asia/Shanghai")
    end = (now + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    start = (now - pd.Timedelta(days=730)).strftime("%Y-%m-%d")
    return start, end


def run_pipeline(ticker, start, end):
    """fetch -> FVG -> MACD -> stages -> dedup -> signals -> backtest.

    Returns (stages_df, signals, trades, final_capital) or None on no data.
    """
    raw = retrieve_data(ticker, end, start)
    if raw.empty:
        return None
    print(f"  -> {len(raw)} daily bars")

    fvgs = find_fvgs(raw)
    print(f"  -> {len(fvgs)} FVGs found")

    final_df = generate_finals(raw, fvgs, end, start)
    final_df = compute_macd(final_df)

    stages_df = generate_pure_stages(final_df)
    for col in ["EMA_Fast", "EMA_Slow", "MACD_Line", "MACD_Signal", "MACD_Histogram"]:
        stages_df[col] = final_df[col].values

    deduped = deduplicate_stages(stages_df)
    signals = generate_trade_signals(deduped, full_stages=stages_df,
                                     macd_mode=MACD_FILTER_MODE)
    trades, final_capital = backtest_trades(signals, raw_df=stages_df,
                                            initial_capital=INITIAL_CAPITAL)
    return stages_df, signals, trades, final_capital


def build_pairs(trades):
    """Turn backtest trade dicts into display rows with duration in days."""
    pairs = []
    for i, t in enumerate(trades, 1):
        entry = pd.to_datetime(t["Entry Date"])
        exit_ = pd.to_datetime(t["Exit Date"])
        pairs.append({
            "#": i,
            "buy_date": entry.strftime("%Y-%m-%d"),
            "buy_price": t["Entry Price"],
            "sell_date": exit_.strftime("%Y-%m-%d"),
            "sell_price": t["Exit Price"],
            "ret": t["PnL %"],
            "days": (exit_ - entry).days,
        })
    return pairs


def is_auto_closed(stages_df, signals, trades):
    """True when the last pair is a still-open position the backtest force-closed
    at the final daily close (i.e. no SELL signal ever fired for it)."""
    if not trades or signals.empty:
        return False
    last_bar = pd.to_datetime(stages_df["Date"].iloc[-1])
    last_sig = str(signals.iloc[-1]["Signal"])
    last_exit = pd.to_datetime(trades[-1]["Exit Date"])
    return last_sig != "SELL" and last_exit >= last_bar



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
        delay = int(args[3]) if len(args) > 3 else ENTRY_DELAY_DAYS
    except ValueError:
        print(f"  !! delay must be an integer, got {args[3]!r}")
        sys.exit(1)

    print(f"\nFetching {ticker} ({start} -> {end}) ...")
    result = run_pipeline(ticker, start, end)
    if result is None:
        print(f"  !! No data returned for {ticker}. Check ticker symbol and date range.")
        sys.exit(1)

    stages_df, signals, trades, final_capital = result

    if delay > 0:
        # Rule-of-thumb entry delay: re-price entries N bars later and drop
        # pairs whose exit arrives first (whipsaws). Lazy import avoids the
        # circular dependency trade_pairs <-> trade_pairs_delay.
        from trade_pairs_delay import repriced_pairs
        _, cap_base = repriced_pairs(stages_df, trades, 0)
        base_ret = (cap_base / INITIAL_CAPITAL - 1) * 100
        pairs, final_capital = repriced_pairs(stages_df, trades, delay)
    else:
        base_ret = None
        pairs = build_pairs(trades)

    auto_closed = is_auto_closed(stages_df, signals, trades)
    last_bar = pd.to_datetime(stages_df["Date"].iloc[-1]).strftime("%Y-%m-%d")

    # ------------------------------------------------------------------
    # Pairs table
    # ------------------------------------------------------------------
    print()
    print("=" * 78)
    print(f"  {ticker} — Buy/Sell pairs   {start} -> {end}"
          f"   (MACD: {MACD_FILTER_MODE}, entry delay: {delay}d)")
    print("=" * 78)

    if not pairs:
        print("  No BUY -> SELL pairs were generated in this period.")
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

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    wins = [p for p in pairs if p["ret"] > 0]
    losses = [p for p in pairs if p["ret"] <= 0]
    win_rate = len(wins) / len(pairs) * 100
    total_return = (final_capital / INITIAL_CAPITAL - 1) * 100
    avg_win = sum(p["ret"] for p in wins) / len(wins) if wins else 0.0
    avg_loss = sum(p["ret"] for p in losses) / len(losses) if losses else 0.0
    days = [p["days"] for p in pairs]

    print(f"  Trades: {len(pairs)}   Wins: {len(wins)}   Losses: {len(losses)}"
          f"   Win rate: {win_rate:.1f}%")
    print(f"  Total return: {total_return:+.2f}%"
          f"   (capital {INITIAL_CAPITAL:,.0f} -> {final_capital:,.2f})")
    if base_ret is not None:
        print(f"  Baseline (no delay): {base_ret:+.2f}%"
              f"   ->  delay edge: {total_return - base_ret:+.2f}pp")
    print(f"  Avg win: {avg_win:+.2f}%   Avg loss: {avg_loss:+.2f}%")
    print(f"  Duration: avg {sum(days) / len(days):.0f}d"
          f"   min {min(days)}d   max {max(days)}d")
    if auto_closed:
        print("  * still-open position — closed at the last available daily close")
    print()


if __name__ == "__main__":
    main()
