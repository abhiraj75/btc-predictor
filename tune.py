"""Quick tuning sweep — test different sigma2 configs to minimize Winkler."""
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from model import fetch_btc_klines, compute_features

def run_config(closes, times, train_window, test_bars, config_name, sigma2_fn):
    start_idx = train_window
    end_idx = len(closes) - 1
    actual_test = min(test_bars, end_idx - start_idx)
    
    preds = []
    for step in tqdm(range(actual_test), desc=config_name, leave=False):
        i = start_idx + step
        train_prices = pd.Series(closes[i - train_window : i + 1], dtype=float)
        log_ret = np.log(train_prices / train_prices.shift(1)).dropna()
        
        try:
            feats = compute_features(train_prices, log_ret)
        except:
            vol = log_ret.std()
            S0 = train_prices.iloc[-1]
            preds.append({"lower": S0*np.exp(-2*vol), "upper": S0*np.exp(2*vol), "actual": float(closes[i+1])})
            continue
        
        sigma2 = sigma2_fn(feats)
        S0 = train_prices.iloc[-1]
        mu = feats["mu"]
        nu = feats["nu"]
        
        Z = np.random.standard_t(nu, size=10_000) * np.sqrt((nu-2)/nu)
        terminals = S0 * np.exp((mu - 0.5*sigma2) + np.sqrt(sigma2) * Z)
        lo, hi = np.percentile(terminals, [2.5, 97.5])
        
        preds.append({"lower": float(lo), "upper": float(hi), "actual": float(closes[i+1])})
    
    return preds

def evaluate(preds):
    alpha = 0.05
    hits = [p["lower"] <= p["actual"] <= p["upper"] for p in preds]
    cov = np.mean(hits)
    widths = [p["upper"] - p["lower"] for p in preds]
    w_scores = []
    for p in preds:
        w = p["upper"] - p["lower"]
        if p["actual"] < p["lower"]:
            w += (2/alpha)*(p["lower"] - p["actual"])
        elif p["actual"] > p["upper"]:
            w += (2/alpha)*(p["actual"] - p["upper"])
        w_scores.append(w)
    return cov, np.mean(widths), np.mean(w_scores)

# Fetch data once
print("Fetching data...")
df = fetch_btc_klines(limit=1221)
closes = df["close"].values
times = df["open_time"].values

np.random.seed(42)  # reproducible

configs = {
    "A: Pure FIGARCH": lambda f: max(1e-8, f["sigma_fig"].iloc[-1]**2),
    
    "B: FIGARCH + mean-rev": lambda f: max(1e-8, 
        f["sigma_fig"].iloc[-1]**2 + 0.15*(f["bar_sigma2"] - f["sigma_fig"].iloc[-1]**2)),
    
    "C: FIGARCH + light crisis": lambda f: max(1e-8, (lambda sf, bs: 
        sf + 0.1*(bs - sf))(
        f["sigma_fig"].iloc[-1]**2 * (1 + 0.15 * min(f["H_series"].iloc[-1]/max(f["H_series"].max(),1e-12), 1.0) if not np.isnan(f["H_series"].iloc[-1]) else 0),
        f["bar_sigma2"])),

    "D: Current model": lambda f: max(1e-8, (lambda: (
        _d_compute(f)
    ))()),
}

def _d_compute(f):
    """Current model's sigma2 computation."""
    H, M = f["H_series"], f["M_series"]
    params = f["base_params"]
    H_max = max(H.max(), 1e-12)
    M_max = max(M.max(), 1e-12)
    H_val = min(H.iloc[-1]/H_max, 1.0) if not np.isnan(H.iloc[-1]) else 0.0
    M_val = min(M.iloc[-1]/M_max, 1.0) if not np.isnan(M.iloc[-1]) else 0.0
    crisis = (H_val > 0.8) or (M_val > 0.8)
    delta_t = params["delta"] if crisis else 0.0
    s2_base = f["sigma_fig"].iloc[-1]**2
    s2 = s2_base * (1 + params["alpha"]*H_val + delta_t*M_val) + params["gamma"]*(f["bar_sigma2"] - s2_base)
    red = f["redundancy"].iloc[-1] if not np.isnan(f["redundancy"].iloc[-1]) else 1.0
    inf_ = f["info_filter"].iloc[-1] if not np.isnan(f["info_filter"].iloc[-1]) else 0.0
    s2 *= max(1e-12, red)
    s2 *= 1 + 0.25*inf_
    return max(1e-8, min(s2, 0.5))

print(f"\nTesting {len(configs)} configs on {min(720, len(closes)-501)} bars...\n")
print(f"{'Config':<30} {'Coverage':>10} {'Width':>10} {'Winkler':>10}")
print("-" * 65)

best_winkler = float('inf')
best_name = ""
best_preds = None

for name, sigma_fn in configs.items():
    np.random.seed(42)
    preds = run_config(closes, times, 500, 720, name, sigma_fn)
    cov, width, winkler = evaluate(preds)
    marker = ""
    if winkler < best_winkler:
        best_winkler = winkler
        best_name = name
        best_preds = preds
        marker = " ← best"
    print(f"{name:<30} {cov:>9.2%} {width:>9,.0f} {winkler:>9,.0f}{marker}")

print(f"\n✅ Best config: {best_name}")
