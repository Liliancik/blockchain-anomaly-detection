"""Synthetic anomaly injection for evaluating detection performance.

Creates controlled test scenarios with known anomalies to measure:
- True positive rate (detected anomalies)
- False positive rate (incorrectly flagged normal data)
- Overall detection accuracy

Two injection methods:
1. Noise: Add Gaussian noise to random samples (point anomalies)
2. Swap: Exchange values between samples (contextual anomalies)
"""
import numpy as np
import pandas as pd
from typing import Tuple, List, Optional


def _continuous_like_columns(df: pd.DataFrame, min_unique: int = 10) -> List[str]:
    """Identify continuous-valued numeric columns.
    
    Purpose: Avoid injecting anomalies into one-hot encoded binary columns.
    Those columns have only 0/1 values and noise injection doesn't make sense.
    
    Strategy: Keep numeric columns with >10 unique values.
    - Continuous features (GPA, Study_hours): Many unique values
    - One-hot binary features (Gender_M): Only 0 or 1
    
    Example:
    Input columns: [GPA, Study_Hours, Gender_M, Grade_A, Grade_B]
    Output: [GPA, Study_Hours] (Gender_M has 2 unique, Grade_A has 2 unique)
    
    Args:
        df: DataFrame to analyze
        min_unique: Threshold for "continuous-like" (>= this many unique values)
    
    Returns:
        List of continuous-like column names (or all numeric if none qualify)
    """
    # Get all numeric columns
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Filter to those with many unique values (continuous-like)
    cont_cols = [c for c in num_cols if df[c].nunique(dropna=True) > min_unique]
    # Fallback: if no continuous columns found, use all numeric (shouldn't happen)
    return cont_cols if len(cont_cols) > 0 else num_cols


def inject_noise(
    df: pd.DataFrame,
    frac: float = 0.02,
    scale: float = 5.0,
    random_state: int = 41,
    columns: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, List[int]]:
    """Inject point anomalies by adding Gaussian noise.
    
    Simulates real anomalies like:
    - Data entry errors (typos, misreadings)
    - Sensor glitches
    - Fraudulent modifications
    
    Process:
    1. Randomly select 'frac' fraction of rows
    2. For each selected row, add random Gaussian noise to continuous columns
    3. Noise magnitude = scale × column_std_dev
       Example: scale=5 means noise has 5× the standard deviation of that column
    4. Return modified data + indices of injected anomalies (ground truth)
    
    Why reset_index? After train/test split, indices might not be 0..n-1.
    This ensures returned anomaly_indices match iloc positions in the result.
    
    Args:
        df: Input DataFrame
        frac: Fraction of rows to inject anomalies into (0.02 = 2%)
        scale: Noise amplitude in units of standard deviation (5.0 = 5-sigma)
        random_state: Seed for reproducibility
        columns: Which columns to inject into (default: auto-detect continuous)
    
    Returns:
        df_mod: DataFrame with injected anomalies
        anomaly_indices: Row indices (0..n-1) that were corrupted
    """
    # Make a copy and reset indices to 0..n-1
    df_mod = df.copy().reset_index(drop=True)

    # Initialize reproducible random generator
    rng = np.random.default_rng(random_state)
    n = len(df_mod)
    # How many rows to corrupt?
    k = max(1, int(n * frac))  # At least 1 anomaly

    # Auto-detect continuous columns if not specified
    if columns is None:
        columns = _continuous_like_columns(df_mod, min_unique=10)

    # If no continuous columns, can't inject anomalies
    if len(columns) == 0:
        return df_mod, []

    # Randomly select which rows to corrupt (without replacement)
    anomaly_indices = rng.choice(n, size=k, replace=False)

    # Add noise to all continuous columns for corrupted rows
    for c in columns:
        # Find column index
        j = df_mod.columns.get_loc(c)
        # Compute this column's standard deviation
        col_std = float(df_mod[c].std())
        # Safety check: if std is 0 or NaN, use 1.0 as fallback
        if not np.isfinite(col_std) or col_std == 0.0:
            col_std = 1.0

        # Generate Gaussian noise with desired scale
        # noise ~ N(0, scale*col_std)
        noise = rng.normal(loc=0.0, scale=scale * col_std, size=k)
        # Add noise to selected rows
        df_mod.iloc[anomaly_indices, j] = df_mod.iloc[anomaly_indices, j].to_numpy() + noise

    return df_mod, anomaly_indices.tolist()


def inject_swap(
    df: pd.DataFrame,
    frac: float = 0.02,
    random_state: int = 41,
    columns: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, List[int]]:
    """Inject contextual anomalies by swapping feature values.
    
    Creates "wrong context" anomalies where values are individually normal
    but combined in impossible ways. Examples:
    - Swap test scores between students: valid scores but wrong person
    - Swap grades between classes: realistic grades but misaligned to student
    
    Process:
    1. Select 'frac' fraction of rows as anomalous
    2. For each anomalous row, pick another random row to swap with
    3. Exchange continuous column values between the two rows
    4. Result: Both rows now contain mixed identity information
    
    This creates more subtle anomalies than noise injection.
    Detection challenge: Individual feature values are normal, but relationships are off.
    
    Args:
        df: Input DataFrame
        frac: Fraction of rows to corrupt (0.02 = 2%)
        random_state: Seed for reproducibility
        columns: Which columns to swap (default: auto-detect continuous)
    
    Returns:
        df_mod: DataFrame with swapped anomalies
        anomaly_indices: Row indices that were modified
    """
    # Copy and reset indices
    df_mod = df.copy().reset_index(drop=True)

    # Random number generator
    rng = np.random.default_rng(random_state)
    n = len(df_mod)
    # Number of rows to corrupt
    k = max(1, int(n * frac))

    # Auto-detect continuous columns
    if columns is None:
        columns = _continuous_like_columns(df_mod, min_unique=10)

    # Check validity: need continuous columns and at least 2 rows
    if len(columns) == 0 or n < 2:
        return df_mod, []

    # Randomly select rows to corrupt
    anomaly_indices = rng.choice(n, size=k, replace=False)
    # For each corrupted row, pick a different row to swap with
    swap_with = rng.choice(n, size=k, replace=False)

    # Perform swaps
    for i, jrow in zip(anomaly_indices, swap_with):
        # Skip if swap would be with itself
        if i == jrow:
            continue
        # Swap values in each continuous column between rows i and jrow
        for c in columns:
            col_idx = df_mod.columns.get_loc(c)
            # Temporary variable for swap
            tmp = df_mod.iloc[i, col_idx]
            df_mod.iloc[i, col_idx] = df_mod.iloc[jrow, col_idx]
            df_mod.iloc[jrow, col_idx] = tmp

    return df_mod, anomaly_indices.tolist()


def inject_anomalies(
    df: pd.DataFrame,
    method: str = "noise",
    frac: float = 0.02,
    random_state: int = 41,
    **kwargs
) -> Tuple[pd.DataFrame, List[int]]:
    """Unified interface for anomaly injection.
    
    Dispatches to appropriate injection method based on 'method' parameter.
    This allows easy switching between different anomaly types without
    changing calling code.
    
    Args:
        df: DataFrame to inject anomalies into
        method: "noise" (Gaussian noise) or "swap" (contextual swap)
        frac: Fraction of rows to corrupt
        random_state: Seed for reproducibility
        **kwargs: Additional arguments passed to specific injection function
    
    Returns:
        df_mod: Modified DataFrame with injected anomalies
        anomaly_indices: Ground-truth labels (which rows were corrupted)
    
    Raises:
        ValueError: If method is not "noise" or "swap"
    """
    # Normalize method name (case-insensitive, strip whitespace)
    method = method.lower().strip()
    
    if method == "noise":
        # Point anomalies: add Gaussian noise
        return inject_noise(df, frac=frac, random_state=random_state, **kwargs)
    elif method == "swap":
        # Contextual anomalies: swap values between rows
        return inject_swap(df, frac=frac, random_state=random_state, **kwargs)
    else:
        # Unknown method
        raise ValueError("Unknown injection method. Choose from: 'noise', 'swap'")
