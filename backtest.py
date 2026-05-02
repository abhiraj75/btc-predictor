"""
backtest.py — Part A: 30-day rolling backtest

Fetches ~820+ BTCUSDT 1h bars (720 test + 100 warm-up),
runs the Cyber GBM model at each step without peeking at future data,
and writes backtest_results.jsonl.
"""

import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from model import fetch_btc_klines, compute_features, simulate_mc


# ────────────────────────────────────────────────────────────────
# Evaluation (from challenge spec)
# ────────────────────────────────────────────────────────────────

def evaluate(predictions: list[dict], alpha: float = 0.05):
    """
    Compute coverage, mean width, and mean Winkler score.
    predictions: list of dicts with keys 'lower', 'upper', 'actual'.
    """
    hits = [p["lower"] <= p["actual"] <= p["upper"] for p in predictions]
    coverage = float(np.mean(hits))

    widths = [p["upper"] - p["lower"] for p in predictions]
    mean_width = float(np.mean(widths))

    winkler_scores = []
    for p in predictions:
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
    }


# ────────────────────────────────────────────────────────────────
# Backtest loop
# ────────────────────────────────────────────────────────────────

def run_backtest(
    train_window: int = 500,
    test_bars: int = 720,
    n_sims: int = 10_000,
    output_file: str = "backtest_results.jsonl",
):
    """
    Rolling backtest: for each of the last `test_bars` hours,
    train on the preceding `train_window` bars and predict the next close.
    """
    total_needed = train_window + test_bars + 1  # +1 for the actual next bar
    print(f"Fetching {total_needed} hourly BTCUSDT bars from Binance ...")
    df = fetch_btc_klines(limit=total_needed)
    print(f"  Got {len(df)} bars  ({df['open_time'].iloc[0]} → {df['open_time'].iloc[-1]})")

    closes = df["close"].values
    times = df["open_time"].values

    # We need indices [train_window .. train_window + test_bars - 1] as prediction origins.
    # For each origin i, we predict close at i+1 using data up to i.
    start_idx = train_window
    end_idx = len(closes) - 1  # last index where we have an actual next bar

    actual_test_bars = min(test_bars, end_idx - start_idx)
    print(f"  Running backtest over {actual_test_bars} bars ...\n")

    predictions = []
    for step in tqdm(range(actual_test_bars), desc="Backtesting"):
        i = start_idx + step

        # Build training series — ONLY data up to bar i (no peeking)
        train_prices = pd.Series(
            closes[i - train_window : i + 1],
            dtype=float,
        )

        # Compute features & predict
        log_ret = np.log(train_prices / train_prices.shift(1)).dropna()

        try:
            feats = compute_features(train_prices, log_ret)
        except Exception as e:
            # If model fitting fails, use simple fallback
            tqdm.write(f"  ⚠ bar {i}: model fit failed ({e}), using fallback")
            vol = log_ret.std()
            S0 = train_prices.iloc[-1]
            lower = S0 * np.exp(-2 * vol)
            upper = S0 * np.exp(2 * vol)
            actual = float(closes[i + 1])
            predictions.append({
                "timestamp": str(times[i + 1]),
                "lower": float(lower),
                "upper": float(upper),
                "actual": actual,
            })
            continue

        terminals = simulate_mc(
            S0=train_prices.iloc[-1],
            mu=feats["mu"],
            sigma_fig=feats["sigma_fig"],
            H=feats["H_series"],
            M=feats["M_series"],
            bar_sigma2=feats["bar_sigma2"],
            redundancy=feats["redundancy"],
            info_filter=feats["info_filter"],
            nu=feats["nu"],
            base_params=feats["base_params"],
            n_sims=n_sims,
            n_steps=1,
        )

        lower, upper = np.percentile(terminals, [2.5, 97.5])
        actual = float(closes[i + 1])

        predictions.append({
            "timestamp": str(times[i + 1]),
            "lower": float(lower),
            "upper": float(upper),
            "actual": actual,
        })

    # Write JSONL
    with open(output_file, "w") as f:
        for p in predictions:
            f.write(json.dumps(p) + "\n")
    print(f"\n✅ Wrote {len(predictions)} predictions to {output_file}")

    # Evaluate
    metrics = evaluate(predictions)
    print(f"\n{'='*50}")
    print(f"  BACKTEST RESULTS")
    print(f"{'='*50}")
    print(f"  Predictions : {len(predictions)}")
    print(f"  Coverage 95%: {metrics['coverage_95']:.4f}  (target: 0.95)")
    print(f"  Mean width  : ${metrics['mean_width']:.2f}")
    print(f"  Mean Winkler: ${metrics['mean_winkler_95']:.2f}")
    print(f"{'='*50}")

    return predictions, metrics


if __name__ == "__main__":
    predictions, metrics = run_backtest()
