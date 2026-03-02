import numpy as np
import pandas as pd
from typing import Tuple, List, Optional


def _continuous_like_columns(df: pd.DataFrame, min_unique: int = 10) -> List[str]:
    """
    Select numeric columns that behave like continuous variables.
    This avoids injecting into one-hot / binary dummy columns.
    """
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cont_cols = [c for c in num_cols if df[c].nunique(dropna=True) > min_unique]
    return cont_cols if len(cont_cols) > 0 else num_cols  # fallback


def inject_noise(
    df: pd.DataFrame,
    frac: float = 0.02,
    scale: float = 5.0,
    random_state: int = 42,
    columns: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, List[int]]:
    """
    Inject point anomalies by adding Gaussian noise to selected rows.

    - Index-safe after train/test split (reset_index + iloc)
    - By default injects into continuous-like numeric columns only
    - 'scale' is in std units. If you inject in scaled space, scale=5 means 5 sigma.
    Returns:
      df_mod, anomaly_indices (0..n-1 row positions)
    """
    df_mod = df.copy().reset_index(drop=True)

    rng = np.random.default_rng(random_state)
    n = len(df_mod)
    k = max(1, int(n * frac))

    if columns is None:
        columns = _continuous_like_columns(df_mod, min_unique=10)

    if len(columns) == 0:
        return df_mod, []

    anomaly_indices = rng.choice(n, size=k, replace=False)

    # Add noise to ALL chosen columns for the selected anomalous rows
    for c in columns:
        j = df_mod.columns.get_loc(c)
        col_std = float(df_mod[c].std())
        if not np.isfinite(col_std) or col_std == 0.0:
            col_std = 1.0

        noise = rng.normal(loc=0.0, scale=scale * col_std, size=k)
        df_mod.iloc[anomaly_indices, j] = df_mod.iloc[anomaly_indices, j].to_numpy() + noise

    return df_mod, anomaly_indices.tolist()


def inject_swap(
    df: pd.DataFrame,
    frac: float = 0.02,
    random_state: int = 42,
    columns: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, List[int]]:
    """
    Inject contextual anomalies by swapping values between two random rows
    for each selected anomalous row, across continuous-like columns.

    Index-safe after split (reset_index + iloc).
    Returns:
      df_mod, anomaly_indices (0..n-1 row positions)
    """
    df_mod = df.copy().reset_index(drop=True)

    rng = np.random.default_rng(random_state)
    n = len(df_mod)
    k = max(1, int(n * frac))

    if columns is None:
        columns = _continuous_like_columns(df_mod, min_unique=10)

    if len(columns) == 0 or n < 2:
        return df_mod, []

    anomaly_indices = rng.choice(n, size=k, replace=False)
    swap_with = rng.choice(n, size=k, replace=False)

    for i, jrow in zip(anomaly_indices, swap_with):
        if i == jrow:
            continue
        for c in columns:
            col_idx = df_mod.columns.get_loc(c)
            tmp = df_mod.iloc[i, col_idx]
            df_mod.iloc[i, col_idx] = df_mod.iloc[jrow, col_idx]
            df_mod.iloc[jrow, col_idx] = tmp

    return df_mod, anomaly_indices.tolist()


def inject_anomalies(
    df: pd.DataFrame,
    method: str = "noise",
    frac: float = 0.02,
    random_state: int = 42,
    **kwargs
) -> Tuple[pd.DataFrame, List[int]]:
    """
    Dispatch anomaly injection.
    Supported methods: 'noise', 'swap'
    """
    method = method.lower().strip()
    if method == "noise":
        return inject_noise(df, frac=frac, random_state=random_state, **kwargs)
    elif method == "swap":
        return inject_swap(df, frac=frac, random_state=random_state, **kwargs)
    else:
        raise ValueError("Unknown injection method. Choose from: 'noise', 'swap'")
