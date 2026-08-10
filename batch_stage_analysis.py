"""
batch_stage_analysis.py — Run daily_stage_v2 pipeline on 30 A-shares and
analyse which BUY patterns produce profitable (PnL% > 10%) trades.

Usage:
    python batch_stage_analysis.py
"""

import sys
import pandas as pd
import numpy as np
from collections import defaultdict

# Import the pipeline functions from daily_stage_v2
from daily_stage_v2 import (
    get_daily_stage,
    deduplicate_stages,
    generate_trade_signals,
    BUY_PATTERNS,
    MIN_STAGE_DURATION,
    PATTERN_DURATION,
)

# ──────────────────────────────────────────────────────────────────────────────
# 1. Stock universe — 30 carefully selected A-shares
# ──────────────────────────────────────────────────────────────────────────────
STOCKS = {
    # ── Finance ──
    "600036.SS": "China Merchants Bank / Banking / Large",
    "601318.SS": "Ping An Insurance / Insurance / Large",
    "601878.SS": "Zheshang Securities / Brokerage / Mid",
    # ── Consumer & Home ──
    "000651.SZ": "Gree Electric / Home Appliances / Large",
    "000568.SZ": "Luzhou Laojiao / Baijiu / Large",
    # ── Auto & New Energy ──
    "601633.SS": "Great Wall Motors / Auto / Mid",
    "002460.SZ": "Ganfeng Lithium / Lithium / Mid",
    "688599.SS": "Trina Solar / Solar / Mid",
    # ── Medical & Pharma ──
    "300760.SZ": "Mindray Medical / Medical Devices / Large",
    "603259.SS": "WuXi AppTec / CRO Pharma / Mid",
    # ── Heavy Industry & Materials ──
    "600019.SS": "Baosteel / Steel / Mid",
    "600028.SS": "Sinopec / Oil & Gas / Large",
    "600309.SS": "Wanhua Chemical / Chemicals / Mid",
    # ── Infrastructure & Transport ──
    "601390.SS": "China Railway Group / Infrastructure / Mid",
    "601111.SS": "Air China / Aviation / Mid",
    # ── Real Estate & Agriculture ──
    "600048.SS": "Poly Developments / Real Estate / Mid",
    "002714.SZ": "Muyuan Foods / Agriculture / Mid",
    # ── Defense ──
    "600760.SS": "AVIC Shenyang / Defense Aero / Mid",
    # ── TECH (13 stocks) ──
    "000063.SZ": "ZTE / Telecom Equipment / Mid",
    "002049.SZ": "Unigroup Guoxin / FPGA Chips / Mid",
    "002230.SZ": "iFlytek / AI NLP / Mid",
    "002371.SZ": "NAURA / Semi Equipment / Mid",
    "002415.SZ": "Hikvision / Security Tech AI / Mid",
    "300033.SZ": "East Money / FinTech Internet / Mid",
    "300454.SZ": "Sangfor / Cloud Cyber / Mid",
    "603160.SS": "Goodix / Sensors Chips / Mid",
    "603501.SS": "Will Semiconductor / Chips CIS / Mid",
    "688187.SS": "CRRC Times Electric / Rail IGBT / Mid",
    "688256.SS": "Cambricon / AI Chips / Mid",
    "688561.SS": "Qi An Xin / Cybersecurity / Mid",
    "688981.SS": "SMIC / Semiconductor Mfg / Large",
}

DATE_RANGE = ("2022-01-01", "2026-08-06")


# ──────────────────────────────────────────────────────────────────────────────
# 2. Pattern-aware backtest
# ──────────────────────────────────────────────────────────────────────────────
def pattern_aware_backtest(
    signal_df: pd.DataFrame,
    raw_df: pd.DataFrame,
    initial_capital: float = 10000.0,
):
    """
    Walk through signals and simulate BUY/SELL round-trips, recording the BUY
    pattern that triggered each trade.

    Returns (trades_list, final_cash).
    Each trade dict includes: Entry Date, Entry Price, Exit Date, Exit Price,
    PnL %, Capital, Pattern.
    """
    trades = []
    cash = initial_capital
    shares = 0.0
    buy_price = 0.0
    buy_date = None
    buy_pattern = None

    for _, row in signal_df.iterrows():
        sig = row["Signal"]

        # ── ENTRY ──
        if sig == "BUY" and shares == 0:
            buy_price = row["Close"]
            buy_date = row["Date"]
            # Extract pattern from Reason field e.g. "pattern (1, 5)" → "(1, 5)"
            reason = str(row.get("Reason", ""))
            if reason.startswith("pattern "):
                buy_pattern = reason[len("pattern "):].strip()
            else:
                buy_pattern = reason
            shares = cash / buy_price
            cash = 0.0
            continue

        # ── EXIT ──
        if sig == "SELL" and shares > 0:
            sell_price = row["Close"]
            sell_date = row["Date"]
            cash = shares * sell_price
            pnl_pct = (sell_price / buy_price - 1) * 100
            trades.append({
                "Entry Date":   buy_date,
                "Entry Price":  round(buy_price, 2),
                "Exit Date":    sell_date,
                "Exit Price":   round(sell_price, 2),
                "PnL %":        round(pnl_pct, 2),
                "Capital":      round(cash, 2),
                "Pattern":      buy_pattern,
            })
            shares = 0.0
            buy_price = 0.0
            buy_date = None
            buy_pattern = None
            continue

    # Close-out any open position at end of data
    if shares > 0:
        sell_price = raw_df["Close"].iloc[-1]
        sell_date = pd.to_datetime(raw_df["Date"].iloc[-1])
        cash = shares * sell_price
        pnl_pct = (sell_price / buy_price - 1) * 100
        trades.append({
            "Entry Date":   buy_date,
            "Entry Price":  round(buy_price, 2),
            "Exit Date":    sell_date,
            "Exit Price":   round(sell_price, 2),
            "PnL %":        round(pnl_pct, 2),
            "Capital":      round(cash, 2),
            "Pattern":      buy_pattern,
        })

    return trades, cash


# ──────────────────────────────────────────────────────────────────────────────
# 3. Batch runner — process all stocks
# ──────────────────────────────────────────────────────────────────────────────
def run_batch():
    start_date, end_date = DATE_RANGE
    all_trades = []
    stock_summaries = []

    ticker_order = list(STOCKS.keys())

    for idx, ticker in enumerate(ticker_order, 1):
        label = STOCKS[ticker]
        industry = label.split(" / ")[1]
        print(f"\n[{idx:02d}/30] {ticker} — {label}")
        print("-" * 50)

        result = get_daily_stage(ticker, start_date, end_date)
        if result is None:
            print(f"  ⚠️  SKIPPED — no data")
            stock_summaries.append({
                "Ticker": ticker, "Name": label.split(" / ")[0],
                "Industry": industry, "MktCap": label.split(" / ")[2],
                "Trades": 0, "Wins": 0, "Losses": 0, "WinRate": 0,
                "AvgPnL": 0, "MaxPnL": 0, "MinPnL": 0,
                "PnL>10%": 0, "TotalReturn%": 0,
            })
            continue

        deduped = deduplicate_stages(result)
        signals = generate_trade_signals(deduped, full_stages=result)
        trades, final_cash = pattern_aware_backtest(signals, raw_df=result)

        for t in trades:
            t["Ticker"] = ticker
            t["Name"] = label.split(" / ")[0]
            t["Industry"] = industry
            t["MktCap"] = label.split(" / ")[2]

        all_trades.extend(trades)

        if trades:
            pnls = [t["PnL %"] for t in trades]
            wins = sum(1 for p in pnls if p > 0)
            losses = sum(1 for p in pnls if p <= 0)
            big_wins = sum(1 for p in pnls if p > 10)
            total_return = (final_cash / 10000 - 1) * 100

            stock_summaries.append({
                "Ticker": ticker, "Name": label.split(" / ")[0],
                "Industry": industry, "MktCap": label.split(" / ")[2],
                "Trades": len(trades), "Wins": wins, "Losses": losses,
                "WinRate": round(wins / len(trades) * 100, 1) if trades else 0,
                "AvgPnL": round(np.mean(pnls), 2),
                "MaxPnL": round(max(pnls), 2),
                "MinPnL": round(min(pnls), 2),
                "PnL>10%": big_wins,
                "TotalReturn%": round(total_return, 2),
            })

            print(f"  ✅ {len(trades)} trades | Win: {wins} | PnL>10%: {big_wins} | "
                  f"Return: {total_return:.1f}%")
        else:
            stock_summaries.append({
                "Ticker": ticker, "Name": label.split(" / ")[0],
                "Industry": industry, "MktCap": label.split(" / ")[2],
                "Trades": 0, "Wins": 0, "Losses": 0, "WinRate": 0,
                "AvgPnL": 0, "MaxPnL": 0, "MinPnL": 0,
                "PnL>10%": 0, "TotalReturn%": 0,
            })
            print(f"  ⚪ No trades")

    return all_trades, stock_summaries



# ──────────────────────────────────────────────────────────────────────────────
# 4. Analysis & reporting
# ──────────────────────────────────────────────────────────────────────────────
def analyse(all_trades, stock_summaries):
    if not all_trades:
        print("\n❌ No trades generated across any stocks. Cannot analyse.")
        return

    trades_df = pd.DataFrame(all_trades)
    stocks_df = pd.DataFrame(stock_summaries)

    # ── 4a. Pattern stats ──
    print("\n" + "=" * 80)
    print("  📊 PATTERN ANALYSIS — All 30 Stocks Combined")
    print("=" * 80)

    pattern_order = [str(p) for p in BUY_PATTERNS]  # e.g., "(1, 5)" — match Reason format
    pattern_stats = []
    for pat in pattern_order:
        subset = trades_df[trades_df["Pattern"] == pat]
        count = len(subset)
        if count == 0:
            pattern_stats.append({
                "Pattern": pat, "Count": 0, "WinCount": 0, "LossCount": 0,
                "WinRate": 0, "AvgPnL": 0, "MaxPnL": 0, "MinPnL": 0,
                "PnL>10%_Count": 0, "PnL>10%_Rate": 0,
            })
            continue

        pnls = subset["PnL %"].values
        win_count = int(np.sum(pnls > 0))
        loss_count = int(np.sum(pnls <= 0))
        big_win_count = int(np.sum(pnls > 10))
        pattern_stats.append({
            "Pattern": pat,
            "Count": count, "WinCount": win_count, "LossCount": loss_count,
            "WinRate": round(win_count / count * 100, 1),
            "AvgPnL": round(np.mean(pnls), 2),
            "MaxPnL": round(np.max(pnls), 2),
            "MinPnL": round(np.min(pnls), 2),
            "PnL>10%_Count": big_win_count,
            "PnL>10%_Rate": round(big_win_count / count * 100, 1),
        })

    pattern_df = pd.DataFrame(pattern_stats)
    pattern_df = pattern_df.sort_values(["PnL>10%_Count", "WinRate"], ascending=[False, False])
    print(pattern_df.to_string(index=False))

    # ── 4b. Top profitable trades (>10% PnL) ──
    profitable = trades_df[trades_df["PnL %"] > 10].sort_values("PnL %", ascending=False)
    print("\n" + "=" * 80)
    print(f"  ⭐ TOP PROFITABLE TRADES (PnL > 10%) — {len(profitable)} trades")
    print("=" * 80)
    if len(profitable) > 0:
        print(profitable[["Ticker", "Name", "Pattern", "PnL %", "Entry Date", "Exit Date"]]
              .to_string(index=True))

    # ── 4c. Per-stock summary ──
    print("\n" + "=" * 80)
    print("  📈 PER-STOCK SUMMARY")
    print("=" * 80)
    print(stocks_df.sort_values("TotalReturn%", ascending=False).to_string(index=False))

    # ── 4d. Industry analysis ──
    print("\n" + "=" * 80)
    print("  🏭 INDUSTRY-LEVEL ANALYSIS")
    print("=" * 80)

    industry_stats = []
    for ind in sorted(trades_df["Industry"].unique()):
        subset = trades_df[trades_df["Industry"] == ind]
        pnls = subset["PnL %"].values
        big_wins = int(np.sum(pnls > 10))
        n_stocks = int(stocks_df[stocks_df["Industry"] == ind]["Ticker"].nunique())
        industry_stats.append({
            "Industry": ind,
            "Stocks": n_stocks,
            "Trades": len(subset),
            "WinRate": round(np.sum(pnls > 0) / len(pnls) * 100, 1) if len(pnls) > 0 else 0,
            "AvgPnL": round(np.mean(pnls), 2),
            "PnL>10%_Count": big_wins,
            "PnL>10%_Rate": round(big_wins / len(pnls) * 100, 1) if len(pnls) > 0 else 0,
        })

    industry_df = pd.DataFrame(industry_stats)
    industry_df = industry_df.sort_values("PnL>10%_Count", ascending=False)
    print(industry_df.to_string(index=False))

    # ── 4e. Per-pattern breakdown by industry ──
    print("\n" + "=" * 80)
    print("  🔬 PATTERN × INDUSTRY CROSS-TABLE (>10% PnL count)")
    print("=" * 80)
    cross = trades_df[trades_df["PnL %"] > 10].groupby(["Pattern", "Industry"]).size().unstack(fill_value=0)
    if not cross.empty:
        print(cross.to_string())

    # ── 4f. Key conclusions ──
    print("\n" + "=" * 80)
    print("  🎯 KEY CONCLUSIONS")
    print("=" * 80)

    best_by_bigwin = pattern_df[pattern_df["Count"] >= 3].sort_values("PnL>10%_Rate", ascending=False)
    if len(best_by_bigwin) > 0:
        top = best_by_bigwin.iloc[0]
        print(f"  🥇 Best pattern (>10% PnL rate): {top['Pattern']} — "
              f"{top['PnL>10%_Rate']}% big-win rate ({top['PnL>10%_Count']}/{top['Count']}), "
              f"avg PnL: {top['AvgPnL']}%")

    all_pnls = trades_df["PnL %"].values
    overall_win_rate = np.sum(all_pnls > 0) / len(all_pnls) * 100
    overall_avg = np.mean(all_pnls)
    overall_big = np.sum(all_pnls > 10)
    active_stocks = int((stocks_df["Trades"] > 0).sum())

    print(f"\n  📊 Overall: {len(all_trades)} trades across {active_stocks} stocks")
    print(f"  📊 Overall win rate: {overall_win_rate:.1f}%")
    print(f"  📊 Overall avg PnL: {overall_avg:.2f}%")
    print(f"  📊 Trades with PnL > 10%: {overall_big} / {len(all_trades)} ({overall_big/len(all_trades)*100:.1f}%)")

    print(f"\n  🔝 Pattern frequency ranking:")
    for _, row in pattern_df.iterrows():
        bar = "█" * max(1, int(row["Count"] / max(1, pattern_df["Count"].max()) * 30))
        print(f"     {row['Pattern']:12s}  {bar}  {row['Count']} trades  "
              f"(win: {row['WinRate']}%, >10%: {row['PnL>10%_Rate']}%)")



# ──────────────────────────────────────────────────────────────────────────────
# 5. Main
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 80)
    print("  BATCH STAGE-PATTERN ANALYSIS — 30 A-Shares")
    print(f"  Date range: {DATE_RANGE[0]} → {DATE_RANGE[1]}")
    print(f"  BUY patterns: {len(BUY_PATTERNS)}")
    print(f"  MIN_STAGE_DURATION: {MIN_STAGE_DURATION}")
    print(f"  PATTERN_DURATION: {PATTERN_DURATION}")
    print("=" * 80)

    all_trades, stock_summaries = run_batch()
    analyse(all_trades, stock_summaries)
