"""Simple bar chart of encoder input features with importance."""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_data
from src import preprocessing as prep
from src.autoencoder_model import train_detector


def visualize_encoder_data():
    """Create bar charts of encoder input data and feature importance."""
    
    # Load and preprocess
    print("Loading and preprocessing data...")
    df = load_data("data/StudentPerformanceFactors.csv")
    df = prep.handle_missing(df)
    df_encoded, _ = prep.encode_categorical(df)
    X_df, features = prep.select_features(df_encoded)
    
    # Split and normalize
    from sklearn.model_selection import train_test_split
    from src import config
    X_train, X_test = train_test_split(X_df, test_size=config.TEST_SIZE, random_state=config.SEED)
    scaler, X_train_scaled = prep.normalize_features(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train autoencoder
    print("Training autoencoder for importance analysis...")
    model, _ = train_detector(
        X_train_scaled,
        contamination=config.INJECTION_FRAC,
        bottleneck_dim=config.BOTTLENECK_DIM,
        epochs=50,  # Quick training for visualization
        batch_size=config.BATCH_SIZE,
        val_split=0.15,
        random_state=config.SEED,
    )
    
    # Calculate feature importance (reconstruction error reduction)
    print("Calculating feature importance...")
    baseline_error = np.mean((X_test_scaled - model.predict(X_test_scaled, verbose=0)) ** 2)
    importance_scores = []
    
    for i in range(X_test_scaled.shape[1]):
        X_test_permuted = X_test_scaled.copy()
        np.random.shuffle(X_test_permuted[:, i])
        permuted_error = np.mean((X_test_permuted - model.predict(X_test_permuted, verbose=0)) ** 2)
        importance_scores.append(max(0, permuted_error - baseline_error))
    
    importance_df = pd.DataFrame({
        'feature': features,
        'importance': importance_scores
    }).sort_values('importance', ascending=False)
    
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Left: Feature Means
    feature_means = X_df.mean().sort_values(ascending=False)
    colors_means = ['#1f77b4' if i < 10 else '#aec7e8' for i in range(len(feature_means))]
    ax1.bar(range(len(feature_means)), feature_means.values, color=colors_means, edgecolor='black', linewidth=1)
    ax1.set_xticks(range(len(feature_means)))
    ax1.set_xticklabels(feature_means.index, rotation=45, ha='right', fontsize=9)
    ax1.set_ylabel('Mean Value', fontsize=12, fontweight='bold')
    ax1.set_title('Encoder Input Features by Mean Value', fontsize=13, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    # Right: Feature Importance
    top_n = 15
    importance_top = importance_df.head(top_n)
    colors_imp = ['#2ca02c' if i < 5 else '#90ee90' for i in range(len(importance_top))]
    ax2.barh(range(len(importance_top)), importance_top['importance'].values, 
            color=colors_imp, edgecolor='black', linewidth=1)
    ax2.set_yticks(range(len(importance_top)))
    ax2.set_yticklabels(importance_top['feature'].values, fontsize=10)
    ax2.set_xlabel('Reconstruction Error Importance', fontsize=12, fontweight='bold')
    ax2.set_title(f'Top {top_n} Features by Importance', fontsize=13, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)
    
    fig.suptitle(f'Encoder Data Analysis ({X_df.shape[0]} samples × {X_df.shape[1]} features → bottleneck: 4)', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save
    output_path = Path("outputs/encoder_features.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved to {output_path}")
    
    # Print statistics
    print(f"\n{'='*70}")
    print(f"Dataset: {X_df.shape[0]} samples × {X_df.shape[1]} features")
    print(f"{'='*70}")
    print(f"\nTop 10 Important Features:")
    for i, row in importance_df.head(10).iterrows():
        print(f"  {row['feature']}: {row['importance']:.6f}")
    
    plt.show()


if __name__ == "__main__":
    visualize_encoder_data()
