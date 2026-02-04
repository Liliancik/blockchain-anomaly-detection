"""
Hyperparameter tuning for autoencoder anomaly detector.
Tests different architecture and training configurations to find optimal F1 score.
"""

import argparse
import json
from pathlib import Path
from itertools import product
import sys
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score

# Fix encoding for Windows
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_data
from src import preprocessing as prep
from src.anomaly_injection import inject_anomalies
from src.autoencoder_model import train_detector, score_anomalies, predict_anomalies


def make_labels(n: int, anomaly_indices: list) -> np.ndarray:
    """y: 1=anomaly, 0=normal"""
    y = np.zeros(n, dtype=int)
    if len(anomaly_indices) > 0:
        y[np.array(anomaly_indices, dtype=int)] = 1
    return y


def evaluate_config(
    X_train, X_test, y_test,
    bottleneck_dim=16,
    epochs=150,
    batch_size=16,
    val_split=0.15,
    random_state=42
):
    """
    Train autoencoder with given config and evaluate on test set.
    Returns: (precision, recall, f1_score, detected_count, true_positives)
    """
    try:
        # Train detector on clean training data
        model, threshold = train_detector(
            X_train,
            contamination=0.02,
            bottleneck_dim=bottleneck_dim,
            epochs=epochs,
            batch_size=batch_size,
            val_split=val_split,
            random_state=random_state
        )

        # Score and predict on test set
        scores = score_anomalies(model, X_test)
        preds = predict_anomalies(model, X_test, threshold)
        
        # Convert predictions to binary (1=anomaly, 0=normal)
        y_pred = np.where(preds == -1, 1, 0)
        
        # Calculate metrics
        detected = np.sum(y_pred)
        tp = np.sum((y_pred == 1) & (y_test == 1))
        
        if detected == 0:
            precision = 0.0
        else:
            precision = tp / detected
            
        if np.sum(y_test) == 0:
            recall = 0.0
        else:
            recall = tp / np.sum(y_test)
            
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'detected': int(detected),
            'true_positives': int(tp),
            'threshold': float(threshold)
        }
    except Exception as e:
        print(f"  Error: {str(e)[:80]}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Hyperparameter tuning for autoencoder anomaly detector"
    )
    parser.add_argument(
        "--data",
        default="data/StudentPerformanceFactors.csv",
        help="Path to dataset CSV"
    )
    parser.add_argument(
        "--method",
        default="noise",
        choices=["noise", "swap"],
        help="Anomaly injection method"
    )
    parser.add_argument(
        "--frac",
        type=float,
        default=0.02,
        help="Fraction of anomalies to inject"
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.3,
        help="Test set size"
    )
    parser.add_argument(
        "--output",
        default="tuning_results.json",
        help="Output file for results"
    )
    args = parser.parse_args()

    print("=" * 70)
    print("AUTOENCODER HYPERPARAMETER TUNING")
    print("=" * 70)

    # Load and preprocess data
    print("\n[1] Loading and preprocessing data...")
    df = load_data(args.data)
    df = prep.handle_missing(df)
    df_encoded, _ = prep.encode_categorical(df)
    X_df, _ = prep.select_features(df_encoded)
    
    scaler, X_scaled = prep.normalize_features(X_df)
    print(f"  [OK] Data shape: {X_scaled.shape}")

    # Split into train/test
    print(f"\n[2] Splitting data (train/test = {1-args.test_size:.0%}/{args.test_size:.0%})...")
    X_train, X_test = train_test_split(
        X_df,  # Use X_df (DataFrame) instead of X_scaled (ndarray)
        test_size=args.test_size,
        random_state=42
    )
    print(f"  [OK] Train: {X_train.shape}, Test: {X_test.shape}")

    # Inject anomalies in test set only
    print(f"\n[3] Injecting anomalies ({args.frac:.0%}) in test set...")
    X_test_inj, anomaly_indices = inject_anomalies(
        X_test.copy(),  # X_test is already a DataFrame
        method=args.method,
        frac=args.frac
    )
    y_test = make_labels(len(X_test_inj), anomaly_indices)
    print(f"  [OK] Injected {len(anomaly_indices)} anomalies")
    print(f"  [OK] Test set shape: {X_test_inj.shape}")

    # Normalize features AFTER splitting to avoid data leakage
    print(f"\n[4] Normalizing features...")
    scaler_train, X_train_scaled = prep.normalize_features(X_train)
    X_test_inj_scaled = scaler_train.transform(X_test_inj)
    print(f"  [OK] Normalized train: {X_train_scaled.shape}, test: {X_test_inj_scaled.shape}")

    # Hyperparameter grid
    print("\n[5] Starting hyperparameter search...\n")
    param_grid = {
        'bottleneck_dim': [8, 16, 24, 32],
        'epochs': [80, 120, 150],
        'batch_size': [8, 16, 32],
        'val_split': [0.1, 0.15, 0.2],
    }

    results = []
    total_configs = np.prod([len(v) for v in param_grid.values()])
    config_num = 0

    for bottleneck_dim in param_grid['bottleneck_dim']:
        for epochs in param_grid['epochs']:
            for batch_size in param_grid['batch_size']:
                for val_split in param_grid['val_split']:
                    config_num += 1
                    config = {
                        'bottleneck_dim': bottleneck_dim,
                        'epochs': epochs,
                        'batch_size': batch_size,
                        'val_split': val_split,
                    }
                    
                    print(f"[{config_num:2d}/{total_configs}] ", end="")
                    print(f"bn_dim={bottleneck_dim:2d} epochs={epochs:3d} bs={batch_size:2d} vs={val_split:.2f}  ", end="", flush=True)
                    
                    result = evaluate_config(
                        X_train_scaled, X_test_inj_scaled, y_test,
                        **config
                    )
                    
                    if result is not None:
                        result.update(config)
                        results.append(result)
                        print(f"→ F1={result['f1']:.4f} P={result['precision']:.4f} R={result['recall']:.4f}")
                    else:
                        print("FAILED")

    # Find best configs
    print("\n" + "=" * 70)
    print("TOP 10 CONFIGURATIONS BY F1 SCORE")
    print("=" * 70)
    
    sorted_results = sorted(results, key=lambda x: x['f1'], reverse=True)
    
    for rank, result in enumerate(sorted_results[:10], 1):
        print(f"\n[#{rank}] F1={result['f1']:.4f}")
        print(f"  Precision: {result['precision']:.4f}")
        print(f"  Recall:    {result['recall']:.4f}")
        print(f"  Detected:  {result['detected']}/{len(y_test)} ({result['detected']/len(y_test)*100:.1f}%)")
        print(f"  True Positives: {result['true_positives']}/{np.sum(y_test)}")
        print(f"  Config: bottleneck_dim={result['bottleneck_dim']}, epochs={result['epochs']}, "
              f"batch_size={result['batch_size']}, val_split={result['val_split']}")

    # Save results
    output_path = Path(args.output)
    with open(output_path, 'w') as f:
        json.dump({
            'grid': param_grid,
            'best_by_f1': sorted_results[0] if sorted_results else {},
            'top_10': sorted_results[:10],
            'all_results': sorted_results
        }, f, indent=2)
    
    print(f"\n✓ Results saved to {output_path}")
    
    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Configurations tested: {len(results)}")
    print(f"Best F1 score: {sorted_results[0]['f1']:.4f}" if sorted_results else "No valid results")
    if sorted_results:
        best = sorted_results[0]
        print(f"\nBest config:")
        print(f"  bottleneck_dim={best['bottleneck_dim']}")
        print(f"  epochs={best['epochs']}")
        print(f"  batch_size={best['batch_size']}")
        print(f"  val_split={best['val_split']}")


if __name__ == "__main__":
    main()
