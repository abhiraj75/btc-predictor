"""
app.py — Streamlit Dashboard for BTC Next-Hour Prediction (Parts B + C)

Displays:
- Backtest metrics (coverage, width, Winkler)
- Current BTC price + predicted 95% range for next hour
- Chart of last 50 bars with prediction band
- Prediction history (Part C persistence)
"""

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from model import fetch_btc_klines, predict_range

# ────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="BTC Next-Hour Predictor",
    page_icon="₿",
    layout="wide",
)

# ────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="st-"] {
        font-family: 'Inter', sans-serif;
    }

    .main .block-container {
        padding-top: 2rem;
        max-width: 1200px;
    }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }
    .metric-label {
        color: #8892b0;
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 0.5rem;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #64ffda, #00bfa5);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-value.warn {
        background: linear-gradient(135deg, #ff6b6b, #ffa502);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    /* Price display */
    .price-display {
        background: linear-gradient(135deg, #0a0a1a 0%, #1a1a3e 100%);
        border: 1px solid rgba(100, 255, 218, 0.15);
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 12px 40px rgba(0,0,0,0.4);
    }
    .price-label {
        color: #8892b0;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    .price-value {
        font-size: 3rem;
        font-weight: 800;
        color: #ccd6f6;
        margin: 0.3rem 0;
    }
    .range-text {
        font-size: 1.1rem;
        color: #64ffda;
        font-weight: 600;
        margin-top: 0.5rem;
    }
    .range-width {
        font-size: 0.85rem;
        color: #8892b0;
        margin-top: 0.3rem;
    }

    /* Header */
    .header-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #f7931a, #ffcf40);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
        line-height: 1.2;
    }
    .header-subtitle {
        color: #8892b0;
        font-size: 0.95rem;
        margin-top: 0.3rem;
    }

    /* History table */
    .history-hit {
        color: #64ffda;
        font-weight: 700;
    }
    .history-miss {
        color: #ff6b6b;
        font-weight: 700;
    }

    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 12px;
        padding: 1rem;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #1a1a2e;
        border-radius: 8px;
        color: #8892b0;
        border: 1px solid rgba(255,255,255,0.06);
    }
</style>
""", unsafe_allow_html=True)


# ────────────────────────────────────────────────────────────────
# PERSISTENCE (Part C)
# ────────────────────────────────────────────────────────────────

HISTORY_FILE = "predictions_history.json"


def load_history() -> list[dict]:
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE) as f:
                return json.load(f)
        except Exception:
            return []
    return []


def save_prediction(pred: dict, history: list[dict]):
    """Append a new prediction and save to disk."""
    history.append(pred)
    # Keep last 500 predictions
    history = history[-500:]
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)
    return history


def update_history_actuals(history: list[dict], df: pd.DataFrame):
    """Fill in actual prices for past predictions where we now know the outcome."""
    close_map = {}
    for _, row in df.iterrows():
        # key: hour timestamp
        ts = row["close_time"].strftime("%Y-%m-%d %H:00")
        close_map[ts] = float(row["close"])

    for h in history:
        if h.get("actual") is None and h.get("target_hour") in close_map:
            h["actual"] = close_map[h["target_hour"]]
            h["hit"] = h["lower"] <= h["actual"] <= h["upper"]

    return history


# ────────────────────────────────────────────────────────────────
# BACKTEST METRICS
# ────────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600)
def load_backtest_metrics():
    """Load pre-computed backtest metrics from JSONL file."""
    path = Path("backtest_results.jsonl")
    if not path.exists():
        return None

    preds = []
    with open(path) as f:
        for line in f:
            preds.append(json.loads(line.strip()))

    if not preds:
        return None

    alpha = 0.05
    hits = [p["lower"] <= p["actual"] <= p["upper"] for p in preds]
    coverage = float(np.mean(hits))
    widths = [p["upper"] - p["lower"] for p in preds]
    mean_width = float(np.mean(widths))

    winkler_scores = []
    for p in preds:
        w = p["upper"] - p["lower"]
        if p["actual"] < p["lower"]:
            w += (2 / alpha) * (p["lower"] - p["actual"])
        elif p["actual"] > p["upper"]:
            w += (2 / alpha) * (p["actual"] - p["upper"])
        winkler_scores.append(w)
    mean_winkler = float(np.mean(winkler_scores))

    return {
        "coverage_95": coverage,
        "mean_width": mean_width,
        "mean_winkler_95": mean_winkler,
        "n_predictions": len(preds),
    }


# ────────────────────────────────────────────────────────────────
# DATA + PREDICTION
# ────────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def get_live_data_and_prediction():
    """Fetch latest 500 bars, run model, return prediction + data."""
    df = fetch_btc_klines(limit=500)
    prices = pd.Series(df["close"].values, dtype=float)
    pred = predict_range(prices, n_sims=10_000, alpha=0.05)

    # Target hour = close_time of the last bar + 1 hour
    last_close_time = df["close_time"].iloc[-1]
    target_hour = (last_close_time + pd.Timedelta(hours=1)).strftime("%Y-%m-%d %H:00")

    pred["timestamp"] = datetime.now(timezone.utc).isoformat()
    pred["target_hour"] = target_hour

    return df, pred


# ────────────────────────────────────────────────────────────────
# MAIN UI
# ────────────────────────────────────────────────────────────────

def main():
    # Header
    st.markdown("""
        <div style="text-align:center; margin-bottom:1.5rem;">
            <div class="header-title">₿ BTC Next-Hour Predictor</div>
            <div class="header-subtitle">
                Cyber GBM Monte Carlo · FIGARCH Volatility · Student-t Fat Tails
            </div>
        </div>
    """, unsafe_allow_html=True)

    # ── Backtest Metrics Row ──
    metrics = load_backtest_metrics()
    if metrics:
        c1, c2, c3 = st.columns(3)
        with c1:
            cov = metrics["coverage_95"]
            cov_class = "" if abs(cov - 0.95) < 0.03 else " warn"
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Coverage (30-day backtest)</div>
                    <div class="metric-value{cov_class}">{cov:.2%}</div>
                </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Avg Range Width</div>
                    <div class="metric-value">${metrics['mean_width']:,.0f}</div>
                </div>
            """, unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Winkler Score</div>
                    <div class="metric-value">${metrics['mean_winkler_95']:,.0f}</div>
                </div>
            """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

    # ── Live prediction ──
    with st.spinner("⏳ Fetching live data & running 10,000 Monte Carlo simulations ..."):
        df, pred = get_live_data_and_prediction()

    current = pred["current_price"]
    lower = pred["lower"]
    upper = pred["upper"]
    width = upper - lower

    st.markdown(f"""
        <div class="price-display">
            <div class="price-label">Current BTC Price</div>
            <div class="price-value">${current:,.2f}</div>
            <div class="range-text">
                95% Prediction: ${lower:,.2f} — ${upper:,.2f}
            </div>
            <div class="range-width">
                Width: ${width:,.2f} · Target hour: {pred['target_hour']} UTC
            </div>
        </div>
    """, unsafe_allow_html=True)

    # ── Save prediction (Part C) ──
    history = load_history()
    # Only save if we haven't saved this target hour already
    existing_hours = {h["target_hour"] for h in history}
    if pred["target_hour"] not in existing_hours:
        history = save_prediction(
            {
                "timestamp": pred["timestamp"],
                "target_hour": pred["target_hour"],
                "current_price": current,
                "lower": lower,
                "upper": upper,
                "actual": None,
                "hit": None,
            },
            history,
        )

    # Update actuals for past predictions
    history = update_history_actuals(history, df)
    # Re-save with actuals filled in
    if history:
        with open(HISTORY_FILE, "w") as f:
            json.dump(history, f, indent=2)

    # ── Chart ──
    st.markdown("### 📈 Last 50 Bars + Prediction Band")
    last50 = df.tail(50).copy()

    fig = go.Figure()

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=last50["open_time"],
        open=last50["open"],
        high=last50["high"],
        low=last50["low"],
        close=last50["close"],
        name="BTCUSDT 1H",
        increasing_line_color="#26a69a",
        decreasing_line_color="#ef5350",
    ))

    # Prediction band (next hour)
    next_time = last50["open_time"].iloc[-1] + pd.Timedelta(hours=1)

    # Shaded prediction rectangle
    fig.add_shape(
        type="rect",
        x0=last50["open_time"].iloc[-1],
        x1=next_time,
        y0=lower,
        y1=upper,
        fillcolor="rgba(100, 255, 218, 0.15)",
        line=dict(color="rgba(100, 255, 218, 0.5)", width=2, dash="dot"),
    )

    # Prediction midpoint marker
    fig.add_trace(go.Scatter(
        x=[next_time],
        y=[pred["mean"]],
        mode="markers",
        marker=dict(color="#64ffda", size=12, symbol="diamond"),
        name="Predicted Mean",
    ))

    # Annotation
    fig.add_annotation(
        x=next_time,
        y=upper,
        text=f"${upper:,.0f}",
        showarrow=False,
        font=dict(color="#64ffda", size=11),
        yshift=15,
    )
    fig.add_annotation(
        x=next_time,
        y=lower,
        text=f"${lower:,.0f}",
        showarrow=False,
        font=dict(color="#64ffda", size=11),
        yshift=-15,
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0a0a1a",
        plot_bgcolor="#0a0a1a",
        height=500,
        margin=dict(l=50, r=30, t=30, b=50),
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", y=1.05),
        xaxis=dict(gridcolor="rgba(255,255,255,0.04)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.04)", tickformat="$,.0f"),
    )
    st.plotly_chart(fig, width="stretch")

    # ── Prediction History (Part C) ──
    if history:
        st.markdown("### 📜 Prediction History")

        hist_with_actuals = [h for h in history if h.get("actual") is not None]
        hist_pending = [h for h in history if h.get("actual") is None]

        tab1, tab2 = st.tabs(["✅ Resolved", "⏳ Pending"])

        with tab1:
            if hist_with_actuals:
                rows = []
                for h in reversed(hist_with_actuals[-50:]):
                    hit_str = "✅ Hit" if h.get("hit") else "❌ Miss"
                    rows.append({
                        "Target Hour": h["target_hour"],
                        "Price": f"${h['current_price']:,.2f}",
                        "Lower": f"${h['lower']:,.2f}",
                        "Upper": f"${h['upper']:,.2f}",
                        "Actual": f"${h['actual']:,.2f}",
                        "Result": hit_str,
                    })
                st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
            else:
                st.info("No resolved predictions yet. Check back after the next hour closes!")

        with tab2:
            if hist_pending:
                rows = []
                for h in reversed(hist_pending):
                    rows.append({
                        "Target Hour": h["target_hour"],
                        "Current Price": f"${h['current_price']:,.2f}",
                        "Lower": f"${h['lower']:,.2f}",
                        "Upper": f"${h['upper']:,.2f}",
                    })
                st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
            else:
                st.info("No pending predictions.")

    # ── Footer ──
    st.markdown("---")
    st.markdown(
        "<div style='text-align:center; color:#4a5568; font-size:0.8rem;'>"
        "AlphaI × Polaris Challenge · Cyber GBM with FIGARCH volatility & Student-t tails · "
        f"Last updated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
        "</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
