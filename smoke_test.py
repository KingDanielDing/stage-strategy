"""Smoke test: verify yfinance works on Render's cloud IPs."""
from flask import Flask, jsonify
import yfinance as yf
import traceback
import time

app = Flask(__name__)


@app.route("/health")
def health():
    return "OK", 200


@app.route("/test")
def test_yfinance():
    results = {"server": "Render", "python_yfinance": yf.__version__}

    # Test 1: NVDA download
    t1 = time.time()
    try:
        df = yf.download("NVDA", start="2026-07-01", end="2026-08-07", interval="1d")
        elapsed = round(time.time() - t1, 2)
        if df.empty:
            results["nvda"] = {"status": "❌ EMPTY", "rows": 0, "time_s": elapsed}
        else:
            results["nvda"] = {
                "status": "✅ OK",
                "rows": len(df),
                "time_s": elapsed,
                "last_close": round(float(df["Close"].iloc[-1].iloc[0]), 2),
                "last_date": str(df.index[-1].date()) if hasattr(df.index[-1], "date") else str(df.index[-1]),
            }
    except Exception as e:
        results["nvda"] = {"status": "❌ FAILED", "error": str(e), "time_s": round(time.time() - t1, 2)}

    # Test 2: A-share (600036.SS)
    t2 = time.time()
    try:
        df2 = yf.download("600036.SS", start="2026-07-01", end="2026-08-07", interval="1d")
        elapsed2 = round(time.time() - t2, 2)
        if df2.empty:
            results["a_share"] = {"status": "⚠️ EMPTY", "rows": 0, "time_s": elapsed2}
        else:
            results["a_share"] = {
                "status": "✅ OK",
                "rows": len(df2),
                "time_s": elapsed2,
                "last_close": round(float(df2["Close"].iloc[-1].iloc[0]), 2),
            }
    except Exception as e:
        results["a_share"] = {"status": "❌ FAILED", "error": str(e), "time_s": round(time.time() - t2, 2)}

    # Overall verdict
    nvda_ok = results["nvda"]["status"] == "✅ OK"
    ashare_ok = results["a_share"]["status"] == "✅ OK"
    results["verdict"] = "✅ READY" if nvda_ok and ashare_ok else ("⚠️ PARTIAL" if nvda_ok or ashare_ok else "❌ BLOCKED")

    return jsonify(results)


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
