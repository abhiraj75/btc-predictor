# ₿ BTC Next-Hour Predictor

**AlphaI × Polaris Challenge** — Predict Bitcoin's next-hour price range using Monte Carlo simulation.

## Model

- **Cyber GBM** (Geometric Brownian Motion) with adaptive volatility
- **FIGARCH** volatility estimation (with GARCH(1,1) fallback)
- **Student-t** distributed shocks for fat tails
- **Rolling entropy + momentum** features for volatility clustering
- **10,000** Monte Carlo simulations per prediction

## Project Structure

| File | Description |
|------|-------------|
| `model.py` | Core model — data fetching, features, GBM simulation |
| `backtest.py` | Part A — 30-day rolling backtest (720 predictions) |
| `app.py` | Part B+C — Live Streamlit dashboard with persistence |
| `backtest_results.jsonl` | Backtest output (one prediction per line) |
| `requirements.txt` | Python dependencies |

## Quick Start

```bash
pip install -r requirements.txt

# Run the 30-day backtest (Part A)
python backtest.py

# Launch the dashboard (Part B)
streamlit run app.py
```

## Backtest Metrics (Part A)

Run `python backtest.py` to generate `backtest_results.jsonl` and print:
- **Coverage** — fraction of predictions containing the actual price (target: ~0.95)
- **Mean Width** — average prediction range width (narrower = better)
- **Winkler Score** — combined accuracy + tightness metric (lower = better)

Current saved `backtest_results.jsonl` metrics:

| Metric | Value |
|--------|-------|
| `coverage_95` | `0.9513888889` |
| `mean_width` | `1188.3203891551` |
| `mean_winkler_95` | `1680.9890754666` |
| Predictions | `720` |
| Backtest window | `2026-04-02 17:00 UTC` to `2026-05-02 16:00 UTC` |
| Volatility scale | `0.96` |

## Dashboard (Part B)

The Streamlit dashboard shows:
- Backtest metrics as headline cards
- Current BTC price + 95% prediction range for the next hour
- Candlestick chart of last 50 bars with shaded prediction band
- Prediction history with hit/miss tracking (Part C)

Deployment URL: add the public Streamlit/HuggingFace/etc. URL here before submitting the form.

## Data Source

BTCUSDT 1-hour klines from Binance public API:
```
https://data-api.binance.vision/api/v3/klines?symbol=BTCUSDT&interval=1h
```
No API key required.
