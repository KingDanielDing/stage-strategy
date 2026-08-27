"""
daily_stage_v5.py — Live Signal Scanner with Stock Profiles

Three modes:
  Single stock:  python daily_stage_v5.py 300666.SZ 江丰电子
  Pool scan:     python daily_stage_v5.py --pool
  Refresh cache: python daily_stage_v5.py --pool --refresh

Strategy:
  - Stage-pattern BUY signals (6 patterns from v2)
  - MACD filter (histogram > 0) applied ONLY to weak patterns: (1,3) & (2,4)
  - Strong patterns (1,5), (3,5), (1,6), (2,6), (3,6) pass without MACD

Profiles (first run computes + caches; subsequent runs load from disk):
  🧹 Clean — MACD improved backtest returns → scanner signals trusted
  🌊 Trend — old algo did better → weak-pattern HOLDs may be false negatives
"""

import sys
import ast
import json
import os
import pandas as pd
from collections import OrderedDict

from daily_stage_v4 import (
    retrieve_data, find_fvgs, generate_finals, compute_macd,
    generate_pure_stages, deduplicate_stages, generate_trade_signals,
    backtest_trades,
    MACD_MODE_LABEL,
)


# ═══════════════════════════════════════════════════════════════════════════
#  Configuration
# ═══════════════════════════════════════════════════════════════════════════
MACD_FILTER_MODE = "histogram_positive"
MACD_SELECTIVE_PATTERNS = {(1, 3), (2, 4)}
LOOKBACK_DAYS = 365

TODAY = pd.Timestamp.today()
DEFAULT_START = (TODAY - pd.Timedelta(days=LOOKBACK_DAYS)).strftime("%Y-%m-%d")
DEFAULT_END   = TODAY.strftime("%Y-%m-%d")

DEFAULT_POOL: "OrderedDict[str, str]" = OrderedDict([
    ("300124.SZ", "汇川技术"),
    ("300274.SZ", "阳光电源"),
    ("300308.SZ","中际旭创"),
    ("300469.SZ","信息发展"),
    ("300476.SZ", "胜宏科技"),
    ("300502.SZ","新易盛"),
    ("300666.SZ","江丰电子"),
    ("300750.SZ", "宁德时代"),
    ("300751.SZ", "迈为股份"),
    ("300759.SZ", "康龙化成"),
    ("300812.SZ", "易天股份"),
    ("300913.SZ", "兆龙互连"),
    ("301217.SZ", "铜冠铜箔"),
    ("000581.SZ", "威孚高科"),
    ("000063.SZ", "中兴通讯"),
    ("000938.SZ", "紫光股份"),
    ("000977.SZ", "浪潮信息"),
    ("001309.SZ", "德明利" ),
    ("002185.SZ", "华天科技"),
    ("002281.SZ", "光迅科技"),
    ("002371.SZ", "北方华创"),
    ("002409.SZ", "雅克科技"),
    ("002460.SZ", "赣锋锂业"),
    ("002472.SZ", "双环传动"),
    ("002747.SZ", "埃斯顿"),
    ("002885.SZ", "京泉华"),
    ("600089.SS", "特变电工"),
    ("600183.SS", "生益科技"),
    ("600362.SS", "江西铜业"),
    ("600460.SS", "士兰微"),
    ("600522.SS", "中天科技"),
    ("600584.SS", "长电科技"),
    ("600667.SS", "太极实业"),
    ("601888.SS", "中国中免"),
    ("603005.SS", "晶方科技"),
    ("603257.SS", "中国瑞林"),
    ("603773.SS", "沃格光电"),
    ("603993.SS", "洛阳钼业"),
    ("601138.SS", "工业富联"),
    ("603688.SS", "石英股份"),
    ("603986.SS", "兆易创新"),
    ("605111.SS", "新洁能"),
    ("688008.SS", "澜起科技"),
    ("688012.SS", "中微公司"),
    ("688016.SS", "新麦医疗"),
    ("688017.SS", "绿地谐波"),
    ("688041.SS", "海光信息"),
    ("688141.SS", "华杰特"),
    ("688195.SS", "腾景科技"),
    ("688256.SS", "寒武纪"),
    ("688300.SS", "联瑞新材"),
    ("688507.SS", "索晨科技"),
    ("688525.SS", "百维存储"),
    ("688700.SS", "东威科技"),
])

# ── Profile cache ───────────────────────────────────────────────────────
PROFILE_CACHE_FILE = os.path.join(os.path.dirname(__file__), ".profile_cache.json")
PROFILE_START = "2023-01-01"
PROFILE_END   = pd.Timestamp.today().strftime("%Y-%m-%d")
PROFILE_CAP   = 10000.0


def load_profile_cache():
    if os.path.exists(PROFILE_CACHE_FILE):
        with open(PROFILE_CACHE_FILE) as f:
            return json.load(f)
    return {}


def save_profile_cache(cache):
    with open(PROFILE_CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2)


def compute_profile(ticker, name):
    """Backtest OLD vs NEW. Returns {d_ret, label, emoji} or None."""
    try:
        raw = retrieve_data(ticker, PROFILE_END, PROFILE_START)
        if raw.empty: return None
        fvgs = find_fvgs(raw)
        if fvgs.empty: return None
        final_df = generate_finals(raw, fvgs, PROFILE_END, PROFILE_START)
        final_df = compute_macd(final_df)
        stages_df = generate_pure_stages(final_df)
        for c in ["EMA_Fast","EMA_Slow","MACD_Line","MACD_Signal","MACD_Histogram"]:
            stages_df[c] = final_df[c].values
        deduped = deduplicate_stages(stages_df)

        s_old = generate_trade_signals(deduped, full_stages=stages_df, macd_mode="off")
        t_old, fc_old = backtest_trades(s_old, raw_df=stages_df, initial_capital=PROFILE_CAP)
        ret_old = (fc_old / PROFILE_CAP - 1) * 100

        s_new = generate_trade_signals(deduped, full_stages=stages_df, macd_mode="histogram_positive")
        t_new, fc_new = backtest_trades(s_new, raw_df=stages_df, initial_capital=PROFILE_CAP)
        ret_new = (fc_new / PROFILE_CAP - 1) * 100

        d_ret = float(ret_new - ret_old)
        emoji, label = ("🧹", "Clean") if d_ret > 0 else (("🌊", "Trend") if d_ret < 0 else ("➖", "Tie"))
        return {"d_ret": round(d_ret, 1), "label": label, "emoji": emoji}
    except Exception:
        return None


def get_profile(ticker, name, cache, refresh=False):
    """Get from cache or compute."""
    if ticker in cache and not refresh:
        return cache[ticker]
    p = compute_profile(ticker, name)
    if p:
        cache[ticker] = p
    return p


# ═══════════════════════════════════════════════════════════════════════════
#  Pipeline runner
# ═══════════════════════════════════════════════════════════════════════════
def run_pipeline(ticker, start, end, verbose=False):
    """fetch -> FVG -> MACD -> stages -> dedup -> signals.
    Returns (stages_df, deduped_df, signals_df) or (None, None, None)."""
    if verbose:
        print(f"  Fetching {ticker} …")

    try:
        raw = retrieve_data(ticker, end, start)
    except Exception as e:
        if verbose:
            print(f"  !! yfinance error: {e}")
        return None, None, None

    if raw.empty:
        if verbose:
            print(f"  !! No data for {ticker}")
        return None, None, None

    fvgs = find_fvgs(raw)
    if fvgs.empty:
        if verbose:
            print(f"  !! No FVGs for {ticker}")
        return None, None, None

    final_df = generate_finals(raw, fvgs, end, start)
    final_df = compute_macd(final_df)
    stages_df = generate_pure_stages(final_df)

    for col in ["EMA_Fast", "EMA_Slow", "MACD_Line", "MACD_Signal", "MACD_Histogram"]:
        stages_df[col] = final_df[col].values

    deduped = deduplicate_stages(stages_df)
    signals = generate_trade_signals(deduped, full_stages=stages_df, macd_mode=MACD_FILTER_MODE)
    return stages_df, deduped, signals


# ═══════════════════════════════════════════════════════════════════════════
#  Signal analysis
# ═══════════════════════════════════════════════════════════════════════════
def analyze_stock(ticker, name="", start=None, end=None, profile=None):
    """Run pipeline on one stock, return analysis dict."""
    s = start or DEFAULT_START
    e = end   or DEFAULT_END

    r = {
        "ticker": ticker, "name": name, "date": None, "close": 0.0,
        "stage": 0, "signal": "-", "pattern": "-", "reason": "",
        "macd_status": "-", "error": None, "signal_date": None,
        "recent_path": [], "macd_histogram": None,
        "profile": profile or {},
    }

    stages_df, deduped, signals = run_pipeline(ticker, s, e)
    if stages_df is None:
        r["error"] = f"No data or no FVGs for {ticker}"
        return r

    last = stages_df.iloc[-1]
    r["date"] = str(last["Date"].date()) if hasattr(last["Date"], "date") else str(last["Date"])
    r["close"] = round(float(last["Close"]), 2)
    r["stage"] = int(last["Stage"])
    if "MACD_Histogram" in stages_df.columns:
        r["macd_histogram"] = round(float(stages_df["MACD_Histogram"].iloc[-1]), 4)

    recent = deduped.tail(8)
    r["recent_path"] = [
        (str(d.date()) if hasattr(d, "date") else str(d), int(st))
        for d, st in zip(recent["Date"], recent["Stage"])
    ]

    last_sig = signals.iloc[-1]
    sig_str = str(last_sig["Signal"])
    reason  = str(last_sig["Reason"])
    r["signal"] = sig_str
    r["reason"] = reason

    sig_date = last_sig.get("Date")
    if sig_date is not None:
        r["signal_date"] = str(sig_date.date()) if hasattr(sig_date, "date") else str(sig_date)[:10]

    if reason.startswith("pattern "):
        parts = reason.split(" (MACD ")
        r["pattern"] = parts[0].replace("pattern ", "")
    elif reason == "searching":
        r["pattern"] = "searching"
    else:
        r["pattern"] = "-"

    if sig_str == "BUY":
        try:
            pat_tuple = ast.literal_eval(r["pattern"])
        except (ValueError, SyntaxError):
            pat_tuple = None
        r["macd_status"] = "passed" if pat_tuple in MACD_SELECTIVE_PATTERNS else "skipped (strong)"
    elif "MACD" in reason:
        r["macd_status"] = "failed"
    else:
        r["macd_status"] = "-"

    return r


# ═══════════════════════════════════════════════════════════════════════════
#  Output formatters
# ═══════════════════════════════════════════════════════════════════════════
def format_single(r):
    """Pretty-print a single-stock analysis."""
    icon = {"BUY": "🔥", "SELL": "🔻", "HOLD": "⏸️"}.get(r["signal"], "❓")
    macd_map = {"passed": "✅ passed", "skipped (strong)": "⏭️  skipped (strong pattern)", "failed": "❌ failed", "-": "-"}
    macd_str = macd_map.get(r["macd_status"], r["macd_status"])

    print()
    print("=" * 72)
    label = f"  {r['name']} ({r['ticker']})  |  {r['date']}"
    print(label)
    print("=" * 72)
    print(f"  Close: {r['close']:,.2f}  |  Stage: S{r['stage']}  |  Signal: {icon} {r['signal']}")

    if r["recent_path"]:
        print()
        print("  -- Recent Stage Path --")
        path_parts = []
        for i, (d, st) in enumerate(r["recent_path"]):
            marker = " <-- CURRENT" if i == len(r["recent_path"]) - 1 else ""
            path_parts.append(f"{d} S{st}{marker}")
        for p in path_parts:
            print(f"     {p}")

    print()
    print("  -- Signal Assessment --")
    print(f"  Pattern:       {r['pattern']}")
    if r["macd_histogram"] is not None:
        print(f"  MACD Hist:     {r['macd_histogram']:+.4f}")
    print(f"  MACD Filter:   {macd_str}")

    print()
    if r["signal"] == "BUY":
        print(f"  >> SUGGESTION: BUY {icon} - {r['reason']}")
    elif r["error"]:
        print(f"  >> ERROR: {r['error']}")
    else:
        print(f"  >> SUGGESTION: {r['signal']} - {r['reason']}")
    print("=" * 72)


def format_pool(results):
    """Pretty-print pool scan: sorted table + BUY summary."""
    mode_label = MACD_MODE_LABEL.get(MACD_FILTER_MODE, MACD_FILTER_MODE)
    sel = ", ".join(str(p) for p in sorted(MACD_SELECTIVE_PATTERNS))

    print()
    print("=" * 120)
    print(f"  STAGE SCANNER - {DEFAULT_END}  |  MACD: {mode_label} on {{{sel}}}")
    print("=" * 120)

    def _sort_key(r):
        return ({"BUY": 0, "HOLD": 1, "SELL": 2}.get(r["signal"], 9), r["ticker"])

    sorted_r = sorted(results, key=_sort_key)

    hdr = f"  {'Ticker':<14s} {'Name':<16s} {'Stg':>3s} {'Signal':>8s} {'Pattern':<10s} {'MACD':<18s} {'Profile':<18s} {'Close':>10s}"
    print(hdr)
    print(f"  {'-'*14} {'-'*16} {'-'*3} {'-'*8} {'-'*10} {'-'*18} {'-'*18} {'-'*10}")

    buy_count = 0
    for r in sorted_r:
        macd_str = r["macd_status"] if not r["error"] else "-"
        sig_str = r["signal"]
        if sig_str == "BUY":
            sig_str = "BUY 🔥"
            buy_count += 1
        elif r["error"]:
            sig_str = "ERR"

        p = r.get("profile", {}) or {}
        profile_str = f"{p.get('emoji','?')} {p.get('label','?')} ({p.get('d_ret',0):+.0f}%)" if p else "?"

        c = r["close"]
        cs = f"{c:,.2f}" if c < 1000 else f"{c:,.2f}" if c < 10000 else f"{c:,.0f}"

        print(f"  {r['ticker']:<14s} {r['name']:<16s} {r['stage']:>3d} {sig_str:>8s} "
              f"{r['pattern']:<10s} {macd_str:<18s} {profile_str:<18s} {cs:>10s}")

    total = len(sorted_r)
    err_count = sum(1 for r in sorted_r if r["error"])
    sell_count = sum(1 for r in sorted_r if r["signal"] == "SELL")
    hold_count = total - buy_count - sell_count - err_count

    print(f"  {'-'*116}")
    tail = f"  SUMMARY:  BUY={buy_count}  |  HOLD={hold_count}  |  SELL={sell_count}"
    if err_count:
        tail += f"  |  ERR={err_count}"
    print(tail)

    buys = [r for r in sorted_r if r["signal"] == "BUY"]
    if buys:
        print()
        print("=" * 120)
        print(f"  🔥 BUY SIGNALS ({len(buys)}):")
        print("=" * 120)
        print(f"  {'Ticker':<14s} {'Name':<16s} {'Date':>10s} {'Stg':>3s} {'Pattern':<10s} {'MACD':<18s} {'Profile':<18s} {'Close':>10s}  Reason")
        print(f"  {'-'*14} {'-'*16} {'-'*10} {'-'*3} {'-'*10} {'-'*18} {'-'*18} {'-'*10}  {'-'*30}")
        for r in buys:
            c = r["close"]
            cs = f"{c:,.2f}" if c < 1000 else f"{c:,.2f}" if c < 10000 else f"{c:,.0f}"
            sd = r.get("signal_date", "-") or "-"
            p = r.get("profile", {}) or {}
            profile_str = f"{p.get('emoji','?')} {p.get('label','?')} ({p.get('d_ret',0):+.0f}%)" if p else "?"
            print(f"  {r['ticker']:<14s} {r['name']:<16s} {sd:>10s} {r['stage']:>3d} "
                  f"{r['pattern']:<10s} {r['macd_status']:<18s} {profile_str:<18s} {cs:>10s}  {r['reason']}")

        # Action guidance
        clean_buys = [r for r in buys if (r.get("profile") or {}).get("label") == "Clean"]
        trend_buys = [r for r in buys if (r.get("profile") or {}).get("label") == "Trend"]
        if clean_buys or trend_buys:
            print()
            print(f"  {'─'*118}")
            if clean_buys:
                print(f"  🧹 Clean ({len(clean_buys)}): MACD trusted — follow scanner signals")
            if trend_buys:
                print(f"  🌊 Trend ({len(trend_buys)}): MACD cautious — weak patterns (1,3)/(2,4) may still be valid despite HOLD")

    print("=" * 120)
    print()


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    refresh = "--refresh" in sys.argv
    args = [a for a in sys.argv[1:] if not a.startswith("--")]

    if not args and "--pool" in sys.argv:
        # Pool scan with profiles
        pool = DEFAULT_POOL
        cache = load_profile_cache()

        # Compute missing profiles
        missing = [t for t in pool if t not in cache]
        if missing or refresh:
            print(f"\n  Computing profiles for {len(missing) if not refresh else len(pool)} stocks ...")
            for i, ticker in enumerate(missing if not refresh else list(pool.keys())):
                name = pool[ticker]
                print(f"    [{i+1}/{len(missing) if not refresh else len(pool)}] {ticker} {name} ...", end=" ", flush=True)
                p = compute_profile(ticker, name)
                if p:
                    cache[ticker] = p
                    print(f"{p['emoji']} {p['label']} ({p['d_ret']:+.0f}%)")
                else:
                    print("FAIL")
            save_profile_cache(cache)

        print(f"\nScanning {len(pool)} stocks ...")
        results = []
        for i, (ticker, name) in enumerate(pool.items()):
            print(f"  [{i+1}/{len(pool)}] {ticker} {name} ...", end=" ", flush=True)
            r = analyze_stock(ticker, name, profile=cache.get(ticker, {}))
            icon = {"BUY": "🔥", "HOLD": "⏸️", "SELL": "🔻"}.get(r["signal"], "⚠️")
            p_str = f"{r['profile'].get('emoji','?')} {r['profile'].get('label','?')}" if r.get("profile") else ""
            print(f"{icon} {r['signal']}  {p_str}")
            results.append(r)
        format_pool(results)

    elif args:
        ticker = args[0].upper()
        name = args[1] if len(args) > 1 else ""
        start = args[2] if len(args) > 2 else DEFAULT_START
        end   = args[3] if len(args) > 3 else DEFAULT_END

        r = analyze_stock(ticker, name, start, end)
        format_single(r)
    else:
        print("Usage: python daily_stage_v5.py <ticker> [name] [start] [end]")
        print("       python daily_stage_v5.py --pool [--refresh]")
    print()