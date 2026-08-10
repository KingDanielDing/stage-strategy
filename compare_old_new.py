"""compare_old_new.py — Compare old (no MACD) vs new (selective MACD) on a pool."""
import sys, pandas as pd
from collections import OrderedDict
from daily_stage_v4 import (
    retrieve_data, find_fvgs, generate_finals, compute_macd,
    generate_pure_stages, deduplicate_stages, generate_trade_signals, backtest_trades,
)

POOL = OrderedDict([
    ("300124.SZ","汇川技术"),("300274.SZ","阳光电源"),("300308.SZ","中际旭创"),
    ("300469.SZ","信息发展"),("300476.SZ","胜宏科技"),("300502.SZ","新易盛"),
    ("300666.SZ","江丰电子"),("300750.SZ","宁德时代"),("300751.SZ","迈为股份"),
    ("300759.SZ","康龙化成"),("300812.SZ","易天股份"),("300913.SZ","兆龙互连"),
    ("301217.SZ","铜冠铜箔"),("000581.SZ","威孚高科"),("000063.SZ","中兴通讯"),
    ("000938.SZ","紫光股份"),("000977.SZ","浪潮信息"),("001309.SZ","德明利"),
    ("002185.SZ","华天科技"),("002281.SZ","光迅科技"),("002371.SZ","北方华创"),
    ("002460.SZ","赣锋锂业"),("002472.SZ","双环传动"),("002747.SZ","埃斯顿"),
    ("002885.SZ","京泉华"),("600089.SS","特变电工"),("600183.SS","生益科技"),
    ("600362.SS","江西铜业"),("600522.SS","中天科技"),("600584.SS","长电科技"),
    ("600667.SS","太极实业"),("601888.SS","中国中免"),("603005.SS","晶方科技"),
    ("603257.SS","中国瑞林"),("603773.SS","沃格光电"),("603993.SS","洛阳钼业"),
    ("601138.SS","工业富联"),("603688.SS","石英股份"),("603986.SS","兆易创新"),
    ("688008.SS","澜起科技"),("688012.SS","中微公司"),("688016.SS","新麦医疗"),
    ("688017.SS","绿地谐波"),("688041.SS","海光信息"),("688141.SS","华杰特"),
    ("688195.SS","腾景科技"),("688256.SS","寒武纪"),("688300.SS","联瑞新材"),
    ("688507.SS","索晨科技"),("688525.SS","百维存储"),("688700.SS","东威科技"),
])

START, END, CAP = "2023-01-01", pd.Timestamp.today().strftime("%Y-%m-%d"), 10000.0
results = []

for i, (ticker, name) in enumerate(POOL.items()):
    print(f"  [{i+1}/{len(POOL)}] {ticker} {name} ...", end=" ", flush=True)
    try:
        raw = retrieve_data(ticker, END, START)
        if raw.empty: print("NO DATA"); continue
        fvgs = find_fvgs(raw)
        if fvgs.empty: print("NO FVGs"); continue
        final_df = generate_finals(raw, fvgs, END, START)
        final_df = compute_macd(final_df)
        stages_df = generate_pure_stages(final_df)
        for c in ["EMA_Fast","EMA_Slow","MACD_Line","MACD_Signal","MACD_Histogram"]:
            stages_df[c] = final_df[c].values
        deduped = deduplicate_stages(stages_df)

        s_old = generate_trade_signals(deduped, full_stages=stages_df, macd_mode="off")
        t_old, fc_old = backtest_trades(s_old, raw_df=stages_df, initial_capital=CAP)
        ret_old = (fc_old/CAP-1)*100; n_old = len(t_old)
        wr_old = sum(1 for t in t_old if t["PnL %"]>0)/n_old*100 if n_old else 0

        s_new = generate_trade_signals(deduped, full_stages=stages_df, macd_mode="histogram_positive")
        t_new, fc_new = backtest_trades(s_new, raw_df=stages_df, initial_capital=CAP)
        ret_new = (fc_new/CAP-1)*100; n_new = len(t_new)
        wr_new = sum(1 for t in t_new if t["PnL %"]>0)/n_new*100 if n_new else 0

        results.append({"ticker":ticker,"name":name,"ret_old":ret_old,"n_old":n_old,"wr_old":wr_old,
                        "ret_new":ret_new,"n_new":n_new,"wr_new":wr_new})
        print(f"{'+' if ret_new>=ret_old else '-'}{abs(ret_new-ret_old):.0f}%")
    except Exception as e:
        pass

# -- Table --
print()
print("=" * 110)
print(f"  OLD vs NEW  ({START} -> {END})  |  NEW = histogram_positive on (1,3),(2,4)")
print("=" * 110)
hdr = f"  {'Ticker':<12s} {'Name':<12s} {'Old Ret':>8s} {'New Ret':>8s} {'dRet':>8s}  {'Old T':>5s} {'New T':>5s}  {'Old WR':>6s} {'New WR':>6s} {'dWR':>6s}"
print(hdr)
print(f"  {'-'*10} {'-'*10} {'-'*8} {'-'*8} {'-'*8}  {'-'*5} {'-'*5}  {'-'*6} {'-'*6} {'-'*6}")

for r in sorted(results, key=lambda x: x["ret_new"]-x["ret_old"], reverse=True):
    d_ret = r["ret_new"]-r["ret_old"]; d_n = r["n_new"]-r["n_old"]; d_wr = r["wr_new"]-r["wr_old"]
    print(f"  {r['ticker']:<12s} {r['name']:<12s} {r['ret_old']:>7.1f}% {r['ret_new']:>7.1f}% {d_ret:>+7.1f}%  "
          f"{r['n_old']:>5d} {r['n_new']:>5d}  {r['wr_old']:>5.1f}% {r['wr_new']:>5.1f}% {d_wr:>+5.1f}%")

print(f"  {'-'*10} {'-'*10} {'-'*8} {'-'*8} {'-'*8}  {'-'*5} {'-'*5}  {'-'*6} {'-'*6} {'-'*6}")

better_new = [r for r in results if r["ret_new"]>r["ret_old"]]
better_old = [r for r in results if r["ret_new"]<r["ret_old"]]
avg_dnew = sum(r["ret_new"]-r["ret_old"] for r in better_new)/len(better_new) if better_new else 0
avg_dold = sum(r["ret_new"]-r["ret_old"] for r in better_old)/len(better_old) if better_old else 0
print(f"  Better with NEW: {len(better_new)} stocks  (avg dRet +{avg_dnew:.1f}%)")
print(f"  Better with OLD: {len(better_old)} stocks  (avg dRet {avg_dold:.1f}%)")

# -- Analysis --
print()
print("=" * 110)
print("  ANALYSIS: Why some stocks prefer OLD vs NEW?")
print("=" * 110)

for label, group in [("Better with OLD algo (no MACD)", better_old), ("Better with NEW algo (MACD)", better_new)]:
    if not group: continue
    awo=sum(r["wr_old"] for r in group)/len(group); awn=sum(r["wr_new"] for r in group)/len(group)
    ano=sum(r["n_old"] for r in group)/len(group); ann=sum(r["n_new"] for r in group)/len(group)
    aro=sum(r["ret_old"] for r in group)/len(group); arn=sum(r["ret_new"] for r in group)/len(group)
    print(f"\n  {label} ({len(group)} stocks):")
    print(f"    Avg old:  Return={aro:+.0f}%  Trades={ano:.0f}  WR={awo:.1f}%")
    print(f"    Avg new:  Return={arn:+.0f}%  Trades={ann:.0f}  WR={awn:.1f}%")
    print(f"    Trades filtered: {ano-ann:.0f}  |  WR change: {awn-awo:+.1f}%")

print("\n  Top 5 NEW-winners:")
for r in sorted(better_new, key=lambda x: x["ret_new"]-x["ret_old"], reverse=True)[:5]:
    d=r["ret_new"]-r["ret_old"]
    print(f"    {r['ticker']:<12s} {r['name']:<12s}  dRet={d:+.1f}%  Old:{r['n_old']}->New:{r['n_new']}t  WR:{r['wr_old']:.0f}->{r['wr_new']:.0f}%")

print("\n  Top 5 OLD-winners:")
for r in sorted(better_old, key=lambda x: x["ret_new"]-x["ret_old"])[:5]:
    d=r["ret_new"]-r["ret_old"]
    print(f"    {r['ticker']:<12s} {r['name']:<12s}  dRet={d:+.1f}%  Old:{r['n_old']}->New:{r['n_new']}t  WR:{r['wr_old']:.0f}->{r['wr_new']:.0f}%")

print("=" * 110)