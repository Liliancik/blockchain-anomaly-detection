"""
Clear visualization of data and injected anomalies.

This script creates:
 - `outputs/clear_injection_visualization.png` : scatter of two features with clear color-coding,
   per-feature histogram overlays, and a count bar.
 - `outputs/clean_vs_injected.csv` : full scaled dataset with an `injected` column (0/1).
 - `outputs/injected_samples.csv` : same as before, injected-only rows.

Usage:
    python scripts/visualize_injection.py --data data/StudentPerformanceFactors.csv --frac 0.02 --scale 5.0
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# make project importable
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_data
from src import preprocessing as prep
from src.anomaly_injection import inject_anomalies


def pick_continuous_columns(df: pd.DataFrame, min_unique: int = 10, max_cols: int = 6):
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cont_cols = [c for c in num_cols if df[c].nunique(dropna=True) > min_unique]
    if len(cont_cols) == 0:
        cont_cols = num_cols
    return cont_cols[:max_cols]


def visualize(args):
    out_dir = Path('outputs')
    out_dir.mkdir(exist_ok=True)

    print('Loading and preprocessing data...')
    df = load_data(args.data)
    df = prep.handle_missing(df)
    df_encoded, _ = prep.encode_categorical(df)
    X_df, _ = prep.select_features(df_encoded)

    # Normalize (match pipeline behaviour where injection happens in scaled space)
    scaler, X_scaled = prep.normalize_features(X_df)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X_df.columns)

    print(f'Data shape: {X_scaled_df.shape}')

    # Inject anomalies
    inj_df, inj_idx = inject_anomalies(X_scaled_df.copy(), frac=args.frac, scale=args.scale, random_state=args.seed)
    inj_idx = list(map(int, inj_idx))

    # Label full dataset
    X_scaled_df['injected'] = 0
    X_scaled_df.loc[inj_idx, 'injected'] = 1

    # Save combined CSV with labels
    combined_csv = out_dir / 'clean_vs_injected.csv'
    X_scaled_df.to_csv(combined_csv, index=False)

    # Save injected-only CSV
    injected_samples = inj_df.iloc[inj_idx].copy()
    injected_samples['injected_row_pos'] = list(range(len(inj_idx)))
    injected_csv = out_dir / 'injected_samples.csv'
    injected_samples.to_csv(injected_csv, index=False)

    # Choose columns for plots
    cont_cols = pick_continuous_columns(X_scaled_df.drop(columns=['injected']), min_unique=10, max_cols=6)
    if len(cont_cols) < 2:
        cont_cols = X_scaled_df.select_dtypes(include=[np.number]).columns.tolist()[:2]

    scatter_x, scatter_y = cont_cols[0], cont_cols[1]

    print('Creating clearer plots...')
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))

    # Scatter: clean vs injected
    ax = axs[0, 0]
    clean = X_scaled_df[X_scaled_df['injected'] == 0]
    injected = X_scaled_df[X_scaled_df['injected'] == 1]
    ax.scatter(clean[scatter_x], clean[scatter_y], c='#7fb3d5', s=12, alpha=0.6, label='clean')
    ax.scatter(injected[scatter_x], injected[scatter_y], c='#e74c3c', s=50, marker='x', label='injected')
    ax.set_xlabel(scatter_x)
    ax.set_ylabel(scatter_y)
    ax.set_title('Clean vs Injected (scatter)')
    ax.legend()

    # Histograms: overlay for scatter_x
    ax = axs[0, 1]
    ax.hist(clean[scatter_x].dropna(), bins=50, density=True, alpha=0.6, color='#7fb3d5', label='clean')
    ax.hist(injected[scatter_x].dropna(), bins=30, density=True, alpha=0.9, color='#e74c3c', label='injected')
    ax.set_title(f'Feature distribution: {scatter_x}')
    ax.legend()

    # Histograms: overlay for scatter_y
    ax = axs[1, 0]
    ax.hist(clean[scatter_y].dropna(), bins=50, density=True, alpha=0.6, color='#7fb3d5', label='clean')
    ax.hist(injected[scatter_y].dropna(), bins=30, density=True, alpha=0.9, color='#e74c3c', label='injected')
    ax.set_title(f'Feature distribution: {scatter_y}')
    ax.legend()

    # Summary counts and small table
    ax = axs[1, 1]
    counts = X_scaled_df['injected'].value_counts().sort_index()
    ax.bar(['clean', 'injected'], [counts.get(0, 0), counts.get(1, 0)], color=['#7fb3d5', '#e74c3c'])
    ax.set_title('Counts: clean vs injected')
    for i, v in enumerate([counts.get(0, 0), counts.get(1, 0)]):
        ax.text(i, v + max(counts.max() * 0.01, 1), str(v), ha='center')

    plt.tight_layout()
    out_png = out_dir / 'clear_injection_visualization.png'
    fig.savefig(out_png, dpi=150)

    print('\nSaved clearer visualization to:', out_png)
    print('Saved combined CSV (clean vs injected) to:', combined_csv)
    print('Saved injected samples CSV to:', injected_csv)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/StudentPerformanceFactors.csv')
    parser.add_argument('--frac', type=float, default=0.02)
    parser.add_argument('--scale', type=float, default=5.0)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    visualize(args)
