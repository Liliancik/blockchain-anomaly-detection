import numpy as np
import pandas as pd
from typing import Tuple, List


def inject_noise(df: pd.DataFrame, frac: float = 0.01, columns: List[str] = None, scale: float = 3.0) -> Tuple[pd.DataFrame, List[int]]:
    """Inject Gaussian noise into a fraction of rows to create anomalies.

    Args:
        df: Input DataFrame (numeric columns are used).
        frac: Fraction of rows to modify.
        columns: List of columns to modify (defaults to all numeric columns).
        scale: Noise scale multiplier (in std units).

    Returns:
        (modified_df, list_of_indices_changed)
    """
    df_mod = df.copy()
    numeric_cols = df_mod.select_dtypes(include=["int64", "float64"]).columns.tolist()
    if columns is None:
        columns = numeric_cols

    n = len(df_mod)
    k = max(1, int(n * frac))
    idx = np.random.choice(n, size=k, replace=False)

    for c in columns:
        col_std = df_mod[c].std() if df_mod[c].std() != 0 else 1.0
        noise = np.random.normal(loc=0.0, scale=scale * col_std, size=k)
        df_mod.loc[idx, c] = df_mod.loc[idx, c] + noise

    return df_mod, idx.tolist()


def inject_swap(df: pd.DataFrame, frac: float = 0.01, columns: List[str] = None) -> Tuple[pd.DataFrame, List[int]]:
    """Inject anomalies by swapping values between rows.

    Args:
        df: Input DataFrame.
        frac: Fraction of rows to modify.
        columns: Columns to consider (defaults to numeric).

    Returns:
        (modified_df, indices)
    """
    df_mod = df.copy()
    n = len(df_mod)
    k = max(1, int(n * frac))
    idx = np.random.choice(n, size=k, replace=False)
    idx_swap = np.random.choice(n, size=k, replace=False)

    if columns is None:
        columns = df_mod.select_dtypes(include=["int64", "float64"]).columns.tolist()

    for i, j in zip(idx, idx_swap):
        for c in columns:
            df_mod.at[i, c], df_mod.at[j, c] = df_mod.at[j, c], df_mod.at[i, c]

    return df_mod, idx.tolist()


def inject_anomalies(df: pd.DataFrame, method: str = "noise", **kwargs) -> Tuple[pd.DataFrame, List[int]]:
    if method == "noise":
        return inject_noise(df, **kwargs)
    elif method == "swap":
        return inject_swap(df, **kwargs)
    else:
        raise ValueError("Unsupported injection method")
