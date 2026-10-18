"""Indicadores causales y escalamiento ajustado exclusivamente en entrenamiento."""
import numpy as np
import pandas as pd

FEATURES = ["return_1", "momentum_5", "momentum_20", "gap_sma20", "volatility_20"]


def load_prices(path):
    df = pd.read_csv(path)
    df.columns = [c.lower().strip() for c in df.columns]
    if not {"timestamp", "close"}.issubset(df.columns):
        raise ValueError("El CSV debe contener timestamp y close.")
    df["timestamp"] = pd.to_datetime(df.timestamp, utc=True, errors="raise")
    if df.timestamp.duplicated().any() or not df.timestamp.is_monotonic_increasing:
        raise ValueError("Las fechas deben ser únicas y estar en orden ascendente.")
    df["close"] = pd.to_numeric(df.close, errors="raise")
    if not np.isfinite(df.close).all() or (df.close <= 0).any():
        raise ValueError("Los precios deben ser positivos y finitos.")
    if len(df) < 150:
        raise ValueError("Se necesitan al menos 150 precios; se recomienda un historial más largo.")
    return df


def features(df):
    out = df.copy()
    p = out.close
    out["return_1"] = p.pct_change(fill_method=None)
    out["momentum_5"] = p.pct_change(5, fill_method=None)
    out["momentum_20"] = p.pct_change(20, fill_method=None)
    out["gap_sma20"] = p / p.rolling(20, min_periods=20).mean() - 1
    out["volatility_20"] = out.return_1.rolling(20, min_periods=20).std(ddof=0)
    return out.dropna(subset=FEATURES).reset_index(drop=True)


def prepare(path, config):
    df = features(load_prices(path))
    a = int(len(df) * config["train_fraction"])
    b = a + int(len(df) * config["validation_fraction"])
    if min(a, b-a, len(df)-b) < 20:
        raise ValueError("Cada partición necesita al menos 20 observaciones útiles.")
    mu = df.loc[:a-1, FEATURES].mean().to_numpy()
    sigma = df.loc[:a-1, FEATURES].std(ddof=0).to_numpy()
    sigma = np.where(sigma < 1e-12, 1, sigma)
    x = np.clip((df[FEATURES].to_numpy()-mu)/sigma, -10, 10).astype(np.float32)
    return df, x, {"train": (0, a-1), "validation": (a, b-1), "test": (b, len(df)-1)}, mu, sigma
