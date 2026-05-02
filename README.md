# ₿ BTC Next-Hour Predictor

Predict Bitcoin's next-hour price range using Monte Carlo simulation.

## Model

- **GBM** (Geometric Brownian Motion) with FIGARCH conditional volatility
- **FIGARCH(1,1)** volatility estimation (with GARCH(1,1) fallback)
- **Student-t** distributed shocks for fat tails
- **Mean-reverted FIGARCH variance** for volatility clustering
- **0.96 volatility calibration scale** selected on a pre-test calibration window
- **10,000** Monte Carlo simulations per prediction

## Project Structure

| File | Description |
|------|-------------|
| `model.py` | Core model: data fetching, volatility estimation, GBM simulation |
| `backtest.py` | 30-day rolling backtest (720 hourly predictions) |
| `app.py` | Live Streamlit dashboard with prediction persistence |
| `backtest_results.jsonl` | Backtest output (one prediction per line) |
| `requirements.txt` | Pinned Python dependencies |

## Quick Start

```bash
pip install -r requirements.txt

# Run the 30-day backtest
python backtest.py

# Launch the dashboard
streamlit run app.py
```

## Backtest Metrics

Run `python backtest.py` to generate `backtest_results.jsonl` and print:
- **Coverage** : fraction of predictions containing the actual price (target: ~0.95)
- **Mean Width** : average prediction range width (narrower = better)
- **Winkler Score** : combined accuracy + tightness metric (lower = better)

Current saved `backtest_results.jsonl` metrics:

| Metric | Value |
|--------|-------|
| `coverage_95` | `0.9513888889` |
| `mean_width` | `$1,188.3203891551` |
| `mean_winkler_95` | `$1,680.9890754666` |
| Predictions | `720` |
| Backtest window | `2026-04-02 17:00 UTC` to `2026-05-02 16:00 UTC` |
| Volatility scale | `0.96` |

## Live Dashboard

**URL**: [btc-predictor-abhiraj.streamlit.app](https://btc-predictor-abhiraj.streamlit.app/)

The dashboard shows:
- Backtest metrics as headline cards
- Current BTC price + 95% prediction range for the next hour
- Candlestick chart of last 50 bars with shaded prediction band
- Prediction history with hit/miss tracking

Persistence uses local JSON storage through isolated `load_history()` / `save_prediction()` functions. On hosted platforms, local files may reset after app sleep or redeploy. The storage code is intentionally small so it can be swapped to an external backend (Supabase, S3, GitHub Gist, etc.).

## Data Source

BTCUSDT 1-hour klines from Binance public API:
```
https://data-api.binance.vision/api/v3/klines?symbol=BTCUSDT&interval=1h
```
No API key required.
