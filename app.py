"""
app.py - Stage Pattern Analyzer (Streamlit)
"""
import warnings, numpy as np, pandas as pd, matplotlib.pyplot as plt, streamlit as st
from daily_stage_v4 import retrieve_data, find_fvgs, generate_finals, compute_macd, generate_pure_stages, deduplicate_stages, generate_trade_signals, backtest_trades
from daily_stage_v6 import _plot, MACD_FILTER_MODE, INITIAL_CAPITAL
from rolling_stage_analysis import run_rolling_analysis, plot_rolling_results
from batch_rolling_screen import evaluate_rolling
warnings.filterwarnings("ignore")
st.set_page_config(page_title="Stage Pattern Analyzer", layout="wide")

# Custom CSS
st.markdown('<style>.metric-card{background:#f8f9fa;border:1px solid #dee2e6;border-radius:12px;padding:14px 18px;text-align:center;margin:4px 0}.metric-card .label{font-size:.72rem;text-transform:uppercase;letter-spacing:.8px;color:#6c757d;margin-bottom:6px}.metric-card .value{font-size:1.35rem;font-weight:700;color:#212529}.metric-card.green .value{color:#198754}.metric-card.red .value{color:#dc3545}.metric-card .delta{font-size:.78rem;color:#6c757d;margin-top:2px}.signal-badge{display:inline-block;padding:4px 16px;border-radius:20px;font-weight:700;font-size:.9rem}.signal-badge.buy{background:#d1e7dd;color:#0f5132}.signal-badge.sell{background:#f8d7da;color:#842029}.signal-badge.hold{background:#fff3cd;color:#664d03}.section-header{font-size:1.05rem;font-weight:600;color:#343a40;border-bottom:2px solid #0d6efd;padding-bottom:6px;margin:24px 0 12px 0}[data-testid=stDataFrame]{border-radius:10px;overflow:hidden}.stButton>button{border-radius:10px;font-weight:600}.stTabs [data-baseweb=tab]{border-radius:8px 8px 0 0;padding:8px 20px;font-weight:500}</style>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("⚙️ Settings")
    st.markdown("---")
    mode = st.radio("**Analysis Mode**", ["📈 Single Stock (v6)", "🔄 Rolling Analysis"],
                    label_visibility="visible")
    st.markdown("---")
    ticker = st.text_input("**Ticker Symbol**", value="AAPL",
                           placeholder="e.g. AAPL, NVDA, TSLA").strip().upper()
    # Compute defaults dynamically in Beijing time (UTC+8)
    beijing_now = pd.Timestamp.now(tz="Asia/Shanghai")
    default_end = (beijing_now + pd.Timedelta(days=1)).date()
    default_start = (beijing_now - pd.Timedelta(days=730)).date()
    c1, c2 = st.columns(2)
    with c1: start_date = st.date_input("**Start**", value=default_start)
    with c2: end_date = st.date_input("**End**", value=default_end)
    st.markdown("---")
    if st.button("🚀 Run Analysis", type="primary", width="stretch"):
        st.session_state["run"] = True
        st.session_state["ticker"] = ticker
        st.session_state["start"] = start_date.strftime("%Y-%m-%d")
        st.session_state["end"] = end_date.strftime("%Y-%m-%d")

# ---------------------------------------------------------------------------
# Main Area
# ---------------------------------------------------------------------------
st.title("📊 Stage Pattern Analyzer")
st.markdown(
    "FVG + MACD momentum stage-pattern detection & backtesting engine  •  "
    f"*Powered by daily_stage_v6*"
)
st.markdown("---")

if "run" not in st.session_state:
    st.info("👈 Enter a ticker in the sidebar and click **Run Analysis** to get started.")
    st.stop()

ticker = st.session_state["ticker"]
start = st.session_state["start"]
end = st.session_state["end"]

# ===========================================================================
# Single Stock Mode
# ===========================================================================
if mode.startswith("📈"):
    st.subheader(f"🔍 Single Stock Analysis: {ticker}")
    st.caption(f"Period: {start} → {end}")

    try:
        with st.spinner(f"Fetching data for {ticker} …"):
            raw = retrieve_data(ticker, end, start)
    except Exception as e:
        st.error(f"❌ Invalid ticker **{ticker}** — could not fetch data. ({e})")
        st.stop()

    if raw.empty:
        st.error(f"❌ No data returned for **{ticker}**. Check the ticker symbol and date range.")
        st.stop()

    st.success(f"✅ Fetched {len(raw)} daily bars")

    with st.spinner("Detecting Fair Value Gaps …"):
        fvgs = find_fvgs(raw)

    with st.spinner("Computing MACD & stages …"):
        final_df = generate_finals(raw, fvgs, end, start)
        final_df = compute_macd(final_df)
        stages_df = generate_pure_stages(final_df)
        for col in ["EMA_Fast", "EMA_Slow", "MACD_Line", "MACD_Signal", "MACD_Histogram"]:
            stages_df[col] = final_df[col].values
        deduped = deduplicate_stages(stages_df)

    with st.spinner("Generating trade signals & backtesting …"):
        signals = generate_trade_signals(
            deduped, full_stages=stages_df, macd_mode=MACD_FILTER_MODE
        )
        trades, final_capital = backtest_trades(
            signals, raw_df=stages_df, initial_capital=INITIAL_CAPITAL
        )

    # ===== Custom Metric Cards (3 cols x 2 rows = no truncation) =====
    if trades:
        total_return = (final_capital / INITIAL_CAPITAL - 1) * 100
        win_count = sum(1 for t in trades if t["PnL %"] > 0)
        loss_count = sum(1 for t in trades if t["PnL %"] <= 0)
        avg_win = sum(t["PnL %"] for t in trades if t["PnL %"] > 0)/win_count if win_count else 0
        avg_loss = sum(t["PnL %"] for t in trades if t["PnL %"] <= 0)/loss_count if loss_count else 0
        win_rate = win_count / len(trades) * 100

        def metric_html(label, value, delta="", color=""):
            d = f'<div class="delta">{delta}</div>' if delta else ""
            return f'<div class="metric-card {color}"><div class="label">{label}</div><div class="value">{value}</div>{d}</div>'

        r1c1, r1c2, r1c3 = st.columns(3)
        r1c1.markdown(metric_html("Final Capital", f"${final_capital:,.2f}",
                                   f"Start: ${INITIAL_CAPITAL:,.0f}", "green" if total_return>0 else "red"), unsafe_allow_html=True)
        r1c2.markdown(metric_html("Total Return", f"{total_return:+.2f}%",
                                   "", "green" if total_return>0 else "red"), unsafe_allow_html=True)
        r1c3.markdown(metric_html("Win Rate", f"{win_rate:.1f}%",
                                   f"{win_count}W / {loss_count}L"), unsafe_allow_html=True)

        r2c1, r2c2, r2c3 = st.columns(3)
        r2c1.markdown(metric_html("Total Trades", str(len(trades))), unsafe_allow_html=True)
        r2c2.markdown(metric_html("Avg Win", f"+{avg_win:.2f}%", "", "green"), unsafe_allow_html=True)
        r2c3.markdown(metric_html("Avg Loss", f"{avg_loss:.2f}%", "", "red"), unsafe_allow_html=True)

        stats = {
            "initial": INITIAL_CAPITAL, "final": final_capital,
            "total_return": total_return, "n_trades": len(trades),
            "wins": win_count, "losses": loss_count,
            "win_rate": win_rate, "avg_win": avg_win, "avg_loss": avg_loss,
            "efficiency": total_return / len(trades),
        }

        r3c1, r3c2, r3c3 = st.columns(3)
        eff = stats["efficiency"]
        r3c3.markdown(metric_html("Efficiency", f"{eff:+.2f}%", "",
                                  "green" if eff > 0 else "red"), unsafe_allow_html=True)
    else:
        st.warning("No trades generated in this period.")
        stats = None

    st.markdown('<div class="section-header">📉 Price Chart with Signals</div>', unsafe_allow_html=True)
    fig = _plot(stages_df, signals, trades, ticker, stats=stats)
    st.pyplot(fig)
    plt.close(fig)

    # ===== Tabs for tables =====
    tab1, tab2, tab3 = st.tabs(["📋 Stages", "🔔 Signals", "💼 Trades"])

    with tab1:
        stage_cols = [c for c in ["Date","Close","Benchmark","Bench","Stage"] if c in deduped.columns]
        sd = deduped[stage_cols].copy()
        if "Date" in sd.columns: sd["Date"] = pd.to_datetime(sd["Date"]).dt.strftime("%Y-%m-%d")
        st.dataframe(sd, width="stretch", height=350)

    with tab2:
        sig_cols = [c for c in ["Date","Stage","Signal","Reason"] if c in signals.columns]
        sg = signals[sig_cols].copy()
        if "Date" in sg.columns: sg["Date"] = pd.to_datetime(sg["Date"]).dt.strftime("%Y-%m-%d")
        st.dataframe(sg, width="stretch", height=350)

    with tab3:
        if trades:
            trades_df = pd.DataFrame(trades)
            if "Entry Date" in trades_df.columns:
                trades_df = trades_df.sort_values("Entry Date", ascending=False)
            for col in ["Entry Date","Exit Date"]:
                if col in trades_df.columns:
                    trades_df[col] = pd.to_datetime(trades_df[col]).dt.strftime("%Y-%m-%d")
            st.dataframe(trades_df, width="stretch", height=350)
        else:
            st.info("No trades to display.")

    # ===== Latest Signal =====
    st.markdown('<div class="section-header">📡 Latest Signal</div>', unsafe_allow_html=True)
    last_signal = signals.iloc[-1]
    last_stage = deduped.iloc[-1]
    sig_str = last_signal["Signal"]
    badge_class = {"BUY":"buy","SELL":"sell","HOLD":"hold"}.get(sig_str,"")
    st.markdown(
        f'<span class="signal-badge {badge_class}">{sig_str}</span>'
        f'&nbsp;&nbsp;Stage: <code>{last_stage["Stage"]}</code>'
        f'&nbsp;|&nbsp;Close: <b>${last_stage["Close"]}</b>'
        f'&nbsp;|&nbsp;Reason: <i>{last_signal["Reason"]}</i>',
        unsafe_allow_html=True,
    )

# ===========================================================================
# Rolling Analysis Mode
# ===========================================================================
else:
    st.subheader(f"🔄 Rolling Window Analysis: {ticker}")
    st.caption(f"Period: {start} → {end} | Window: 1 year | Step: 1 month")

    try:
        with st.spinner(f"Running rolling analysis for {ticker} …"):
            results_df, raw_all = run_rolling_analysis(
                ticker, window_years=1, step_months=1,
                start_date=start, end_date=end,
            )

        if results_df.empty:
            st.error("❌ No results. Try a longer date range.")
            st.stop()

        valid = results_df[results_df["Trades"] > 0]
        c1, c2, c3 = st.columns(3)
        c1.metric("📊 Windows", len(results_df), f"{len(valid)} w/ trades")
        if not valid.empty:
            c2.metric("📈 Avg Return", f"{valid['Return_%'].mean():+.2f}%")
            c3.metric("🎯 Avg Win Rate", f"{valid['Win_Rate_%'].mean():.1f}%")

        # Pass/Fail test (same algorithm as batch_rolling_screen)
        verdict = evaluate_rolling(results_df)
        if verdict["Error"]:
            st.info(f"⚠️ Pass/Fail test skipped: {verdict['Error']}")
        elif verdict["Pass"]:
            st.markdown(
                '<div class="section-header">🎯 Pass/Fail Test</div>'
                '<span class="signal-badge buy">✅ PASS</span>&nbsp;&nbsp;'
                f'upward trend — latest <b>{verdict["Latest"]:+.2f}%</b> &gt; '
                f'prev <b>{verdict["Prev"]:+.2f}%</b> (slope <b>{verdict["Slope"]:+.2f}</b>)',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="section-header">🎯 Pass/Fail Test</div>'
                '<span class="signal-badge sell">❌ FAIL</span>&nbsp;&nbsp;'
                f'{verdict["Reason"]} — latest <b>{verdict["Latest"]:+.2f}%</b>, '
                f'prev <b>{verdict["Prev"]:+.2f}%</b>, slope <b>{verdict["Slope"]:+.2f}</b>',
                unsafe_allow_html=True,
            )

        st.subheader("📈 Rolling Stage")
        fig_rolling = plot_rolling_results(results_df, raw_all, ticker)
        if fig_rolling:
            st.pyplot(fig_rolling)
            plt.close(fig_rolling)

        st.subheader("📋 Rolling Results")
        display_cols = [
            c for c in ["Window_End", "Return_%", "Win_Rate_%", "Trades", "Wins", "Losses"]
            if c in results_df.columns
        ]
        rolling_display = results_df[display_cols].copy()
        if "Window_End" in rolling_display.columns:
            rolling_display["Window_End"] = pd.to_datetime(rolling_display["Window_End"])
            rolling_display = rolling_display.sort_values("Window_End", ascending=False)
            rolling_display["Window_End"] = rolling_display["Window_End"].dt.strftime("%Y-%m-%d")
        st.dataframe(rolling_display, width="stretch", height=400)
    except Exception as e:
        st.error(f"❌ Analysis failed for **{ticker}**: {e}")
