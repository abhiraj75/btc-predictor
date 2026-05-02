"""
app.py — Streamlit Dashboard for BTC Next-Hour Prediction (Parts B + C)
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from model import fetch_btc_klines, predict_range

# ── Page config ──
st.set_page_config(page_title="BTC Predictor", page_icon="₿", layout="wide")

# ── Minimal CSS — just enough to not look like default Streamlit ──
st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; max-width: 1100px; }
    .stat-box {
        border: 1px solid #333;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        background: #111;
    }
    .stat-label { color: #999; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 4px; }
    .stat-value { font-size: 1.6rem; font-weight: 700; color: #e0e0e0; }
    .stat-value.green { color: #4ade80; }
    .prediction-box {
        border: 1px solid #2a2a3a;
        border-radius: 10px;
        padding: 1.2rem 1.5rem;
        background: #0d0d1a;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ── Persistence (Part C) ──
HISTORY_FILE = "predictions_history.json"

def load_history():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE) as f:
                return json.load(f)
        except Exception:
            return []
    return []

def save_prediction(pred, history):
    history.append(pred)
    history = history[-500:]
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f)
    return history

def update_actuals(history, df):
    close_map = {}
    for _, row in df.iterrows():
        ts = row["close_time"].strftime("%Y-%m-%d %H:00")
        close_map[ts] = float(row["close"])
    for h in history:
        if h.get("actual") is None and h.get("target_hour") in close_map:
            h["actual"] = close_map[h["target_hour"]]
            h["hit"] = h["lower"] <= h["actual"] <= h["upper"]
    return history

# ── Load backtest metrics ──
@st.cache_data(ttl=3600)
def load_backtest_metrics():
    path = Path("backtest_results.jsonl")
    if not path.exists():
        return None
    preds = [json.loads(l) for l in open(path)]
    if not preds:
        return None
    alpha = 0.05
    hits = [p["lower"] <= p["actual"] <= p["upper"] for p in preds]
    widths = [p["upper"] - p["lower"] for p in preds]
    winkler = []
    for p in preds:
        w = p["upper"] - p["lower"]
        if p["actual"] < p["lower"]:
            w += (2/alpha) * (p["lower"] - p["actual"])
        elif p["actual"] > p["upper"]:
            w += (2/alpha) * (p["actual"] - p["upper"])
        winkler.append(w)
    return {
        "coverage": float(np.mean(hits)),
        "width": float(np.mean(widths)),
        "winkler": float(np.mean(winkler)),
        "n": len(preds),
    }

# ── Live prediction ──
@st.cache_data(ttl=30)
def get_prediction():
    df = fetch_btc_klines(limit=500)
    prices = pd.Series(df["close"].values, dtype=float)
    pred = predict_range(prices, n_sims=10_000, alpha=0.05)
    last_close = df["close_time"].iloc[-1]
    pred["target_hour"] = (last_close + pd.Timedelta(hours=1)).strftime("%Y-%m-%d %H:00")
    pred["timestamp"] = datetime.now(timezone.utc).isoformat()
    return df, pred

# ────────────────────────────────────────────────
# MAIN
# ────────────────────────────────────────────────

st.title("₿ BTC Next-Hour Predictor")
st.caption("GBM Monte Carlo · FIGARCH volatility · Student-t fat tails · 10K simulations")

# ── Backtest metrics ──
metrics = load_backtest_metrics()
if metrics:
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        cov = metrics["coverage"]
        color = "green" if abs(cov - 0.95) < 0.03 else ""
        st.markdown(f"""<div class="stat-box">
            <div class="stat-label">Coverage (30d)</div>
            <div class="stat-value {color}">{cov:.1%}</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="stat-box">
            <div class="stat-label">Avg Width</div>
            <div class="stat-value">${metrics['width']:,.0f}</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="stat-box">
            <div class="stat-label">Winkler Score</div>
            <div class="stat-value">${metrics['winkler']:,.0f}</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="stat-box">
            <div class="stat-label">Predictions</div>
            <div class="stat-value">{metrics['n']}</div>
        </div>""", unsafe_allow_html=True)

st.divider()

# ── Live prediction ──
with st.spinner("Fetching live data & running simulations..."):
    df, pred = get_prediction()

price = pred["current_price"]
lo, hi = pred["lower"], pred["upper"]
width = hi - lo

col_price, col_pred = st.columns([1, 1])

with col_price:
    st.markdown(f"""<div class="prediction-box">
        <div class="stat-label">Current BTC Price</div>
        <div class="stat-value" style="font-size:2.2rem;">${price:,.2f}</div>
    </div>""", unsafe_allow_html=True)

with col_pred:
    st.markdown(f"""<div class="prediction-box">
        <div class="stat-label">95% Prediction → {pred['target_hour']} UTC</div>
        <div class="stat-value green" style="font-size:1.8rem;">${lo:,.2f} — ${hi:,.2f}</div>
        <div style="color:#666; font-size:0.8rem; margin-top:4px;">Width: ${width:,.2f}</div>
    </div>""", unsafe_allow_html=True)

# ── Save prediction (Part C) ──
history = load_history()
existing = {h["target_hour"] for h in history}
if pred["target_hour"] not in existing:
    history = save_prediction({
        "timestamp": pred["timestamp"],
        "target_hour": pred["target_hour"],
        "current_price": price,
        "lower": lo, "upper": hi,
        "actual": None, "hit": None,
    }, history)
history = update_actuals(history, df)
if history:
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f)

# ── Chart ──
st.subheader("Last 50 bars + prediction")
last50 = df.tail(50).copy()
next_time = last50["open_time"].iloc[-1] + pd.Timedelta(hours=1)

fig = go.Figure()
fig.add_trace(go.Candlestick(
    x=last50["open_time"],
    open=last50["open"], high=last50["high"],
    low=last50["low"], close=last50["close"],
    name="BTCUSDT 1H",
    increasing_line_color="#26a69a",
    decreasing_line_color="#ef5350",
))

# Prediction band
fig.add_shape(type="rect",
    x0=last50["open_time"].iloc[-1], x1=next_time,
    y0=lo, y1=hi,
    fillcolor="rgba(74, 222, 128, 0.12)",
    line=dict(color="rgba(74, 222, 128, 0.4)", width=1.5, dash="dot"),
)
fig.add_trace(go.Scatter(
    x=[next_time], y=[pred["mean"]],
    mode="markers", marker=dict(color="#4ade80", size=8, symbol="diamond"),
    name="Predicted mean",
))
fig.add_annotation(x=next_time, y=hi, text=f"${hi:,.0f}", showarrow=False,
                   font=dict(color="#4ade80", size=10), yshift=12)
fig.add_annotation(x=next_time, y=lo, text=f"${lo:,.0f}", showarrow=False,
                   font=dict(color="#4ade80", size=10), yshift=-12)

fig.update_layout(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    height=450,
    margin=dict(l=40, r=20, t=20, b=40),
    xaxis_rangeslider_visible=False,
    legend=dict(orientation="h", y=1.02, x=0),
    xaxis=dict(gridcolor="rgba(255,255,255,0.03)"),
    yaxis=dict(gridcolor="rgba(255,255,255,0.03)", tickprefix="$", tickformat=",.0f"),
)
st.plotly_chart(fig, width="stretch")

# ── Prediction history (Part C) ──
resolved = [h for h in history if h.get("actual") is not None]
pending = [h for h in history if h.get("actual") is None]

if resolved or pending:
    st.subheader("Prediction history")
    tab1, tab2 = st.tabs([f"Resolved ({len(resolved)})", f"Pending ({len(pending)})"])

    with tab1:
        if resolved:
            rows = []
            for h in reversed(resolved[-30:]):
                rows.append({
                    "Hour": h["target_hour"],
                    "Lower": f"${h['lower']:,.2f}",
                    "Upper": f"${h['upper']:,.2f}",
                    "Actual": f"${h['actual']:,.2f}",
                    "": "✓" if h.get("hit") else "✗",
                })
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        else:
            st.info("No resolved predictions yet, check back after the next hourly close.")

    with tab2:
        if pending:
            rows = [{"Hour": h["target_hour"],
                     "Lower": f"${h['lower']:,.2f}",
                     "Upper": f"${h['upper']:,.2f}"}
                    for h in reversed(pending)]
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        else:
            st.caption("No pending predictions.")

# ── Footer ──
st.divider()
st.caption(f"Last updated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} · "
           f"Source: Binance BTCUSDT 1H · AlphaI × Polaris Challenge")
