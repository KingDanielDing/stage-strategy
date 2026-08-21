"""
daily_stage_v6.py — Interactive single-stock scanner with full backtest & chart.

Usage:
    python daily_stage_v6.py

Strategy: stage-pattern BUY signals with selective MACD (histogram > 0)
          applied only to weak patterns (1,3) & (2,4).
"""

import sys
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from daily_stage_v4 import (
    retrieve_data, find_fvgs, generate_finals, compute_macd,
    generate_pure_stages, deduplicate_stages, generate_trade_signals,
    backtest_trades,
)

MACD_FILTER_MODE = "histogram_positive"
INITIAL_CAPITAL = 10000.0
DEFAULT_START = (pd.Timestamp.today() - pd.Timedelta(days=730)).strftime("%Y-%m-%d")
DEFAULT_END   = pd.Timestamp.today().strftime("%Y-%m-%d")


def run(ticker, start, end):
    """Full pipeline: fetch -> FVG -> MACD -> stages -> signals -> backtest -> chart."""
    print(f"\nFetching {ticker} ({start} -> {end}) …")
    print(f"Fetching daily data for {ticker} ({start} -> {end}) …")

    raw = retrieve_data(ticker, end, start)
    if raw.empty:
        print("  !! No data returned.")
        return
    print(f"  -> {len(raw)} daily bars")

    print("Detecting Fair Value Gaps …")
    fvgs = find_fvgs(raw)
    print(f"  -> {len(fvgs)} FVGs found")

    print("Computing MACD (12/26/9) …")
    final_df = generate_finals(raw, fvgs, end, start)
    final_df = compute_macd(final_df)

    stages_df = generate_pure_stages(final_df)
    for col in ["EMA_Fast", "EMA_Slow", "MACD_Line", "MACD_Signal", "MACD_Histogram"]:
        stages_df[col] = final_df[col].values

    deduped = deduplicate_stages(stages_df)

    print()
    print("=" * 60)
    print(f"  Stages ({len(stages_df)}->{len(deduped)} deduped)")
    print("=" * 60)
    print(deduped[["Date", "Close", "Benchmark", "Bench", "Stage"]].to_string(index=True))

    signals = generate_trade_signals(deduped, full_stages=stages_df, macd_mode=MACD_FILTER_MODE)
    print()
    print("=" * 60)
    print("  Trade signals")
    print("=" * 60)
    print(signals[["Date", "Stage", "Signal", "Reason"]].to_string(index=True))

    trades, final_capital = backtest_trades(signals, raw_df=stages_df, initial_capital=INITIAL_CAPITAL)

    print()
    print("=" * 60)
    print("  Backtest")
    print("=" * 60)
    if trades:
        trades_df = pd.DataFrame(trades)
        print(trades_df.to_string(index=True))
        print()

        total_return = (final_capital / INITIAL_CAPITAL - 1) * 100
        win_count = sum(1 for t in trades if t["PnL %"] > 0)
        loss_count = sum(1 for t in trades if t["PnL %"] <= 0)
        avg_win = sum(t["PnL %"] for t in trades if t["PnL %"] > 0) / win_count if win_count else 0
        avg_loss = sum(t["PnL %"] for t in trades if t["PnL %"] <= 0) / loss_count if loss_count else 0

        print(f"  Initial: ${INITIAL_CAPITAL:,.2f}  Final: ${final_capital:,.2f}")
        print(f"  Return: {total_return:.2f}%  Trades: {len(trades)}  Win: {win_count/len(trades)*100:.1f}%")
        print(f"  Avg win: {avg_win:.2f}%  Avg loss: {avg_loss:.2f}%")

        stats = {
            "initial": INITIAL_CAPITAL, "final": final_capital,
            "total_return": total_return, "n_trades": len(trades),
            "wins": win_count, "losses": loss_count,
            "win_rate": win_count / len(trades) * 100,
            "avg_win": avg_win, "avg_loss": avg_loss,
            "efficiency": total_return / len(trades),
        }
        chart_file = f"{ticker}_trades_v6.png"
        fig = _plot(stages_df, signals, trades, ticker, stats)
        fig.savefig(chart_file, dpi=150)
        print(f"\n  Chart saved to: {chart_file}")
        try:
            plt.show()
        except Exception:
            pass

        # ── Current Signal Assessment ──
        _print_signal_assessment(signals, stages_df)
    else:
        print("  No trades generated.")
        _print_signal_assessment(signals, stages_df)


def _print_signal_assessment(signals, stages_df):
    """Print the current signal analysis like v5."""
    import ast
    MACD_SELECTIVE = {(1, 3), (2, 4)}

    last_sig = signals.iloc[-1]
    sig_str = str(last_sig["Signal"])
    reason  = str(last_sig["Reason"])

    # Parse pattern
    if reason.startswith("pattern "):
        parts = reason.split(" (MACD ")
        pattern = parts[0].replace("pattern ", "")
    elif reason == "searching":
        pattern = "searching"
    else:
        pattern = "-"

    # MACD status
    if sig_str == "BUY":
        try:
            pat = ast.literal_eval(pattern)
        except (ValueError, SyntaxError):
            pat = None
        macd_status = "✅ passed" if pat in MACD_SELECTIVE else "⏭️  skipped (strong pattern)"
    elif "MACD" in reason:
        macd_status = "❌ failed"
    else:
        macd_status = "-"

    macd_hist = None
    if "MACD_Histogram" in stages_df.columns:
        macd_hist = round(float(stages_df["MACD_Histogram"].iloc[-1]), 4)

    icon = {"BUY": "🔥", "SELL": "🔻", "HOLD": "⏸️"}.get(sig_str, "❓")

    print()
    print("  -- Signal Assessment --")
    print(f"  Pattern:       {pattern}")
    if macd_hist is not None:
        print(f"  MACD Hist:     {macd_hist:+.4f}")
    print(f"  MACD Filter:   {macd_status}")
    print()
    if sig_str == "BUY":
        print(f"  >> SUGGESTION: BUY {icon} - {reason}")
    else:
        print(f"  >> SUGGESTION: {sig_str} {icon} - {reason}")


def _plot(raw_df, signals, trades, ticker, stats=None):
    price = raw_df.copy()
    price["Date"] = pd.to_datetime(price["Date"])
    sig = signals.copy()
    sig["Date"] = pd.to_datetime(sig["Date"])

    buys  = sig[sig["Signal"] == "BUY"]
    sells = sig[sig["Signal"] == "SELL"]

    fig, ax = plt.subplots(figsize=(16, 7))
    ax.plot(price["Date"], price["Close"], linewidth=0.8, color="#2c3e50", label="Close")

    if not buys.empty:
        ax.scatter(buys["Date"], buys["Close"], marker="^", s=100,
                   color="#27ae60", edgecolors="white", linewidths=0.5, zorder=5, label="BUY")
    if not sells.empty:
        ax.scatter(sells["Date"], sells["Close"], marker="v", s=100,
                   color="#e74c3c", edgecolors="white", linewidths=0.5, zorder=5, label="SELL")

    for t in trades:
        ed = pd.to_datetime(t["Entry Date"]); xd = pd.to_datetime(t["Exit Date"])
        ep = t["Entry Price"]; xp = t["Exit Price"]
        color = "#27ae60" if t["PnL %"] > 0 else "#e74c3c"
        ax.plot([ed, xd], [ep, xp], linestyle="--", linewidth=0.6, color=color, alpha=0.5)

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.set_title(f"{ticker}  -  Stage-Pattern Strategy (v6)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Price")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    ticker = input("Enter stock ticker: ").strip().upper()
    if not ticker:
        print("No ticker provided.")
        sys.exit(0)

    start_in = input(f"Start date [{DEFAULT_START}]: ").strip()
    end_in   = input(f"End date   [{DEFAULT_END}]: ").strip()
    start = start_in if start_in else DEFAULT_START
    end   = end_in if end_in else DEFAULT_END

    run(ticker, start, end)