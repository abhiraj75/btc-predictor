"""
model.py — Cyber GBM Monte Carlo Simulator for BTCUSDT (1-hour bars)

Adapted from the AlphaI starter Colab notebook.
Uses FIGARCH volatility + Student-t fat tails + entropy/momentum features
to produce 95% prediction intervals for Bitcoin's next-hour close price.
"""

import numpy as np
import pandas as pd
import requests
import scipy.stats as stats
from arch import arch_model
from datetime import datetime, timezone


# ────────────────────────────────────────────────────────────────
# 1. DATA FETCHING
# ────────────────────────────────────────────────────────────────

BASE_URL = "https://data-api.binance.vision/api/v3/klines"


def fetch_btc_klines(limit: int = 500, end_time_ms: int | None = None) -> pd.DataFrame:
    """
    Fetch BTCUSDT 1-hour klines from Binance public API.
    Returns a DataFrame with columns: open_time, open, high, low, close, volume.
    """
    all_data = []
    remaining = limit
    current_end = end_time_ms

    while remaining > 0:
        batch = min(remaining, 1000)
        params = {"symbol": "BTCUSDT", "interval": "1h", "limit": batch}
        if current_end is not None:
            params["endTime"] = current_end

        r = requests.get(BASE_URL, params=params, timeout=15)
        r.raise_for_status()
        data = r.json()

        if not data:
            break

        all_data = data + all_data  # prepend (older data first)
        # Move endTime to just before the oldest candle in this batch
        current_end = data[0][0] - 1
        remaining -= len(data)

    df = pd.DataFrame(all_data, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_vol", "trades", "taker_buy_base",
        "taker_buy_quote", "ignore"
    ])

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)

    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    df = df.drop_duplicates(subset="open_time").sort_values("open_time").reset_index(drop=True)
    return df


# ────────────────────────────────────────────────────────────────
# 2. FEATURE ENGINEERING
# ────────────────────────────────────────────────────────────────

def rolling_entropy(x: pd.Series, window: int = 60, bins: int = 20) -> pd.Series:
    """Shannon entropy of binned values over a rolling window."""
    def _ent(v):
        p, _ = np.histogram(v, bins=bins, density=True)
        p = p[p > 0]
        return -np.sum(p * np.log(p))
    return x.rolling(window, min_periods=window).apply(_ent, raw=True)


def compute_features(prices: pd.Series, log_ret: pd.Series):
    """
    Fit FIGARCH model and compute all features needed for simulation.
    Returns a dict with: mu, sigma_fig, H_series, M_series, bar_sigma2, nu,
                         redundancy, info_filter, base_params.
    """
    mu = log_ret.mean()

    # FIGARCH volatility model
    try:
        am = arch_model(log_ret * 100, vol="FIGARCH", p=1, o=0, q=1, dist="studentst")
        res = am.fit(disp="off", show_warning=False)
    except Exception:
        # Fallback to GARCH(1,1) if FIGARCH fails
        am = arch_model(log_ret * 100, vol="Garch", p=1, q=1, dist="studentst")
        res = am.fit(disp="off", show_warning=False)

    sigma_fig = res.conditional_volatility / 100

    # Standardised residuals → fit Student-t degrees of freedom
    resid = (log_ret * 100 - res.params["mu"]) / res.conditional_volatility
    try:
        nu = max(4, stats.t.fit(resid.dropna(), floc=0, fscale=1)[0])
    except Exception:
        nu = 5.0  # sensible default for BTC

    # Rolling entropy and momentum
    H_series = rolling_entropy(resid)
    M_series = log_ret.abs().rolling(60, min_periods=10).mean()

    bar_sigma2 = (sigma_fig ** 2).mean()

    # Multi-scale variance ratio (redundancy)
    var5 = prices.rolling(5, min_periods=2).var()
    var20 = prices.rolling(20, min_periods=5).var()
    redundancy = 1 + 0.1 * np.log1p(var5 / var20.replace(0, np.nan))
    redundancy = redundancy.fillna(1.0)

    # Information filter
    H_mean = H_series.mean()
    info_filter = (H_series > H_mean).astype(float) if not np.isnan(H_mean) else pd.Series(0.0, index=H_series.index)

    # Adaptive base params (ensure stability) — tuned for tighter ranges
    H_max = H_series.max() if H_series.max() > 0 else 1.0
    M_max = M_series.max() if M_series.max() > 0 else 1.0
    alpha0, delta0 = 0.35, 0.2  # reduced from 0.5/0.3 to tighten ranges
    if alpha0 * H_max + delta0 * M_max >= 1:
        fac = 0.95 / (alpha0 * H_max + delta0 * M_max)
        alpha0 *= fac
        delta0 *= fac

    base_params = {
        "alpha": alpha0, "delta": delta0,
        "gamma": 0.2, "kappa": 0.1, "eta": 1e-3
    }

    return {
        "mu": mu, "sigma_fig": sigma_fig, "H_series": H_series,
        "M_series": M_series, "bar_sigma2": bar_sigma2, "nu": nu,
        "redundancy": redundancy, "info_filter": info_filter,
        "base_params": base_params,
    }


# ────────────────────────────────────────────────────────────────
# 3. SIMULATION ENGINE (vectorised — ~100x faster than loop)
# ────────────────────────────────────────────────────────────────

def _compute_sigma2(sigma_fig: pd.Series, H: pd.Series, M: pd.Series,
                    params: dict, bar_sigma2: float,
                    redundancy: pd.Series, info_filter: pd.Series,
                    eps: float = 1e-6) -> float:
    """Compute the adaptive variance for the next step (deterministic)."""
    H_max = max(H.max(), 1e-12)
    M_max = max(M.max(), 1e-12)

    H_val = min(H.iloc[-1] / H_max, 1.0) if not np.isnan(H.iloc[-1]) else 0.0
    M_val = min(M.iloc[-1] / M_max, 1.0) if not np.isnan(M.iloc[-1]) else 0.0

    crisis = (H_val > 0.8) or (M_val > 0.8)
    delta_t = params["delta"] if crisis else 0.0

    sigma2_base = sigma_fig.iloc[-1] ** 2
    sigma2 = (
        sigma2_base * (1 + params["alpha"] * H_val + delta_t * M_val)
        + params["gamma"] * (bar_sigma2 - sigma2_base)
    )

    red_val = redundancy.iloc[-1] if not np.isnan(redundancy.iloc[-1]) else 1.0
    inf_val = info_filter.iloc[-1] if not np.isnan(info_filter.iloc[-1]) else 0.0

    sigma2 *= max(1e-12, red_val)
    sigma2 *= 1 + 0.25 * inf_val  # reduced from 0.5 to tighten ranges
    sigma2 = max(eps, min(sigma2, 0.5))

    return sigma2


def simulate_mc(
    S0: float, mu: float, sigma_fig: pd.Series,
    H: pd.Series, M: pd.Series,
    bar_sigma2: float, redundancy: pd.Series,
    info_filter: pd.Series, nu: float, base_params: dict,
    n_sims: int = 10_000, n_steps: int = 1
) -> np.ndarray:
    """
    Vectorised Monte Carlo simulation for 1-step ahead prediction.
    Since n_steps=1, sigma2 is deterministic (same for all sims) —
    only the random shock Z differs. This allows full numpy vectorisation.
    """
    sigma2 = _compute_sigma2(
        sigma_fig, H, M, base_params, bar_sigma2,
        redundancy, info_filter
    )

    dt = 1.0
    # Vectorised: draw all shocks at once
    Z = np.random.standard_t(nu, size=n_sims) * np.sqrt((nu - 2) / nu)
    terminals = S0 * np.exp((mu - 0.5 * sigma2) * dt + np.sqrt(sigma2 * dt) * Z)

    return terminals


# ────────────────────────────────────────────────────────────────
# 4. PREDICTION INTERFACE
# ────────────────────────────────────────────────────────────────

def predict_range(
    prices: pd.Series, n_sims: int = 10_000, alpha: float = 0.05
) -> dict:
    """
    Given a price series (closes), predict the 95% range for the next bar.
    Returns dict with: lower, upper, mean, current_price.
    """
    log_ret = np.log(prices / prices.shift(1)).dropna()

    feats = compute_features(prices, log_ret)

    terminals = simulate_mc(
        S0=prices.iloc[-1],
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

    lo = float(alpha / 2 * 100)
    hi = float((1 - alpha / 2) * 100)
    lower, upper = np.percentile(terminals, [lo, hi])

    return {
        "lower": float(lower),
        "upper": float(upper),
        "mean": float(terminals.mean()),
        "current_price": float(prices.iloc[-1]),
    }
