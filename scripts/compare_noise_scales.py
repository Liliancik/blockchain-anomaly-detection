"""
Compare visualizations across different noise scales.
Creates side-by-side scatter and histogram plots for scales 5.0, 10.0, and 20.0.
Saves: outputs/noise_scale_comparison.png
"""

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


def compare_scales():
    out_dir = Path('outputs')
    out_dir.mkdir(exist_ok=True)

    print('Loading and preprocessing data...')
    df = load_data('data/StudentPerformanceFactors.csv')
    df = prep.handle_missing(df)
    df_encoded, _ = prep.encode_categorical(df)
    X_df, _ = prep.select_features(df_encoded)

    # Normalize
    scaler, X_scaled = prep.normalize_features(X_df)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X_df.columns)

    print(f'Data shape: {X_scaled_df.shape}')

    scales = [5.0, 10.0, 20.0]
    cont_cols = pick_continuous_columns(X_scaled_df, min_unique=10, max_cols=6)
    scatter_x, scatter_y = cont_cols[0], cont_cols[1]

    # Create figure with 3 columns (one per scale) and 2 rows (scatter + histogram)
    fig, axs = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('Noise Scale Comparison: 5.0 vs 10.0 vs 20.0', fontsize=16, fontweight='bold')

    for col_idx, scale in enumerate(scales):
        print(f'Processing scale {scale}...')
        
        # Inject anomalies with this scale
        inj_df, inj_idx = inject_anomalies(
            X_scaled_df.copy(),
            frac=0.10,
            scale=scale,
            random_state=42
        )
        inj_idx = list(map(int, inj_idx))

        # Label data
        X_labeled = X_scaled_df.copy()
        X_labeled['injected'] = 0
        X_labeled.loc[inj_idx, 'injected'] = 1

        clean = X_labeled[X_labeled['injected'] == 0]
        injected = X_labeled[X_labeled['injected'] == 1]

        # Top row: scatter plot
        ax = axs[0, col_idx]
        ax.scatter(clean[scatter_x], clean[scatter_y], c='#7fb3d5', s=12, alpha=0.6, label='clean')
        ax.scatter(injected[scatter_x], injected[scatter_y], c='#e74c3c', s=50, marker='x', label='injected')
        ax.set_xlabel(scatter_x, fontsize=10)
        ax.set_ylabel(scatter_y, fontsize=10)
        ax.set_title(f'Scale {scale}: Scatter', fontweight='bold')
        ax.legend(fontsize=9)

        # Bottom row: histogram overlay
        ax = axs[1, col_idx]
        ax.hist(clean[scatter_x].dropna(), bins=50, density=True, alpha=0.6, color='#7fb3d5', label='clean')
        ax.hist(injected[scatter_x].dropna(), bins=30, density=True, alpha=0.9, color='#e74c3c', label='injected')
        ax.set_title(f'Scale {scale}: {scatter_x} Distribution', fontweight='bold')
        ax.set_xlabel(scatter_x, fontsize=10)
        ax.set_ylabel('Density', fontsize=10)
        ax.legend(fontsize=9)

    plt.tight_layout()
    out_png = out_dir / 'noise_scale_comparison.png'
    fig.savefig(out_png, dpi=150)
    print(f'\nSaved comparison visualization to: {out_png}')


if __name__ == '__main__':
    compare_scales()
