import numpy as np
import pandas as pd


def inject_noise(df: pd.DataFrame, frac: float = 0.01, noise_scale: float = 0.5, random_state: int = 42):
    """
    Inject point anomalies by adding Gaussian noise to randomly selected numeric cells.
    Works safely after train/test split by resetting index and using iloc.

    Returns:
        df_mod (DataFrame): modified copy with injected anomalies
        anomaly_indices (list[int]): row positions (0..n-1) that were modified
    """
    df_mod = df.copy().reset_index(drop=True)

    rng = np.random.default_rng(random_state)
    n = len(df_mod)
    k = max(1, int(n * frac))

    num_cols = df_mod.select_dtypes(include=[np.number]).columns.tolist()
    if not num_cols:
        # No numeric columns to perturb
        return df_mod, []

    anomaly_indices = rng.choice(n, size=k, replace=False)

    for i in anomaly_indices:
        c = rng.choice(num_cols)
        col_std = float(df_mod[c].std()) if df_mod[c].std() == df_mod[c].std() else 0.0  # NaN-safe
        sigma = noise_scale * (col_std if col_std > 0 else 1.0)
        noise = rng.normal(0.0, sigma)

        j = df_mod.columns.get_loc(c)
        df_mod.iloc[i, j] = df_mod.iloc[i, j] + noise

    return df_mod, anomaly_indices.tolist()


def inject_swap(df: pd.DataFrame, frac: float = 0.01, random_state: int = 42):
    """
    Inject contextual anomalies by swapping values between two numeric columns within selected rows.
    Works safely after train/test split by resetting index and using iloc.

    Returns:
        df_mod (DataFrame): modified copy with injected anomalies
        anomaly_indices (list[int]): row positions (0..n-1) that were modified
    """
    df_mod = df.copy().reset_index(drop=True)

    rng = np.random.default_rng(random_state)
    n = len(df_mod)
    k = max(1, int(n * frac))

    num_cols = df_mod.select_dtypes(include=[np.number]).columns.tolist()
    if len(num_cols) < 2:
        return df_mod, []

    anomaly_indices = rng.choice(n, size=k, replace=False)

    for i in anomaly_indices:
        c1, c2 = rng.choice(num_cols, size=2, replace=False)
        j1 = df_mod.columns.get_loc(c1)
        j2 = df_mod.columns.get_loc(c2)

        tmp = df_mod.iloc[i, j1]
        df_mod.iloc[i, j1] = df_mod.iloc[i, j2]
        df_mod.iloc[i, j2] = tmp

    return df_mod, anomaly_indices.tolist()


def inject_anomalies(df: pd.DataFrame, method: str = "noise", frac: float = 0.01, random_state: int = 42, **kwargs):
    """
    Dispatcher for anomaly injection methods.
    Ensures index safety by working on reset-index copies inside each method.
    """
    method = method.lower().strip()

    if method == "noise":
        return inject_noise(df, frac=frac, random_state=random_state, **kwargs)
    elif method == "swap":
        return inject_swap(df, frac=frac, random_state=random_state)
    else:
        raise ValueError(f"Unknown injection method: {method}. Choose from ['noise', 'swap'].")
