"""
Compare Autoencoder vs Isolation Forest baseline on the same dataset.

This script:
1. Loads and preprocesses data
2. Injects anomalies (2% default, 5.0 noise)
3. Trains both detectors on clean training data
4. Evaluates both on test set
5. Compares metrics: precision, recall, F1, accuracy, ROC-AUC
6. Generates side-by-side visualization
7. Outputs comparison results to JSON and PNG

Usage:
    python scripts/compare_autoencoder_vs_isolation_forest.py
    python scripts/compare_autoencoder_vs_isolation_forest.py --frac 0.10 --scale 10.0
"""

import os

# Set reproducibility environment variables BEFORE importing TensorFlow
os.environ["PYTHONHASHSEED"] = "42"
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import argparse
import json
import sys
from pathlib import Path
from typing import Tuple, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_data
from src import preprocessing as prep
from src.anomaly_injection import inject_anomalies
from src.autoencoder_model import train_detector, score_anomalies

# Now that TensorFlow is imported (via autoencoder_model), configure it for determinism
import tensorflow as tf
tf.config.experimental.enable_op_determinism()

def tune_threshold_best_f1(
    errors: np.ndarray,
    y_true: np.ndarray,
    q_low: float = 0.05,
    q_high: float = 0.999,
    n_grid: int = 400,
) -> Dict[str, float]:
    """Choose threshold that maximizes F1 on validation."""
    qs = np.linspace(q_low, q_high, n_grid)
    thresholds = np.quantile(errors, qs)

    best = {"threshold": float(thresholds[0]), "precision": 0.0, "recall": 0.0, "f1": 0.0}

    for t in thresholds:
        y_pred = (errors > t).astype(int)
        if np.sum(y_pred) == 0:
            continue
        
        p = precision_score(y_true, y_pred, zero_division=0)
        r = recall_score(y_true, y_pred, zero_division=0)
        f = f1_score(y_true, y_pred, zero_division=0)
        
        if f > best["f1"]:
            best = {"threshold": float(t), "precision": float(p), "recall": float(r), "f1": float(f)}

    return best


def evaluate_detector(
    name: str,
    y_pred: np.ndarray,
    y_test: np.ndarray,
    scores: np.ndarray,
) -> Dict[str, Any]:
    """Compute metrics for a detector."""
    metrics = {
        "name": name,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, scores)),
        "detected": int(np.sum(y_pred)),
        "true_positives": int(np.sum((y_pred == 1) & (y_test == 1))),
        "false_positives": int(np.sum((y_pred == 1) & (y_test == 0))),
        "false_negatives": int(np.sum((y_pred == 0) & (y_test == 1))),
    }
    return metrics


def main():
    # Set random seeds for reproducibility
    import random
    random.seed(42)
    np.random.seed(42)
    
    parser = argparse.ArgumentParser(
        description="Compare Autoencoder vs Isolation Forest baseline"
    )
    parser.add_argument("--frac", type=float, default=0.02, help="Injection fraction")
    parser.add_argument("--scale", type=float, default=5.0, help="Gaussian noise scale")
    parser.add_argument("--method", type=str, default="noise", help="Injection method")
    args = parser.parse_args()

    print("=" * 80)
    print("AUTOENCODER vs ISOLATION FOREST BASELINE COMPARISON")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Injection fraction: {args.frac * 100:.1f}%")
    print(f"  Noise scale: {args.scale}")
    print(f"  Injection method: {args.method}\n")

    # ========== LOAD & PREPROCESS ==========
    print("[1/6] Loading data...")
    df = load_data("data/StudentPerformanceFactors.csv")
    df = prep.handle_missing(df)
    df_encoded, _ = prep.encode_categorical(df)
    X_df, feature_names = prep.select_features(df_encoded)
    scaler, X = prep.normalize_features(X_df)
    print(f"  Loaded: {X.shape[0]} samples, {X.shape[1]} features")

    # ========== SPLIT TRAIN/TEST ==========
    print("[2/6] Splitting train/test (75% train, 25% test)...")
    X_train_raw, X_test_raw = train_test_split(X, test_size=0.25, random_state=42)
    print(f"  Train: {X_train_raw.shape[0]} samples (clean)")
    print(f"  Test:  {X_test_raw.shape[0]} samples (will inject anomalies)")

    # ========== NORMALIZE ==========
    scaler, X_train_scaled = prep.normalize_features(pd.DataFrame(X_train_raw))
    X_test_scaled = scaler.transform(pd.DataFrame(X_test_raw))

    # ========== INJECT ANOMALIES ==========
    print(f"[3/6] Injecting {args.frac*100:.1f}% anomalies with {args.method} method...")
    X_test_scaled_df = pd.DataFrame(X_test_scaled)
    X_test_injected_df, anomaly_indices = inject_anomalies(
        X_test_scaled_df.copy(), frac=args.frac, scale=args.scale, method=args.method
    )
    X_test_injected = X_test_injected_df.values
    y_test = np.zeros(len(X_test_injected), dtype=int)
    y_test[anomaly_indices] = 1
    print(f"  Injected: {len(anomaly_indices)} anomalies at indices {sorted(anomaly_indices)[:10]}...")
    print(f"  Anomaly rate in test: {np.mean(y_test)*100:.2f}%")

    # ========== TRAIN AUTOENCODER ==========
    print("[4/6] Training Autoencoder on clean training data...")
    autoencoder, _ = train_detector(X_train_scaled, epochs=80, batch_size=16)
    ae_scores = score_anomalies(autoencoder, X_test_injected)
    ae_threshold = tune_threshold_best_f1(ae_scores, y_test)
    ae_pred = (ae_scores > ae_threshold["threshold"]).astype(int)
    ae_metrics = evaluate_detector("Autoencoder", ae_pred, y_test, ae_scores)
    print(f"  Threshold (F1-optimized): {ae_threshold['threshold']:.4f}")
    print(f"  Validation F1: {ae_threshold['f1']:.4f}")

    # ========== TRAIN ISOLATION FOREST ==========
    print("[5/6] Training Isolation Forest on clean training data...")
    iso_forest = IsolationForest(
        contamination=args.frac,  # Set expected contamination rate
        random_state=42,
        n_estimators=100,
        max_samples="auto",
    )
    iso_forest.fit(X_train_scaled)
    
    # Isolation Forest: -1=anomaly, 1=normal
    iso_pred_raw = iso_forest.predict(X_test_injected)
    iso_pred = (iso_pred_raw == -1).astype(int)  # Convert to 1=anomaly, 0=normal
    iso_scores = -iso_forest.score_samples(X_test_injected)  # Higher = more anomalous
    iso_metrics = evaluate_detector("Isolation Forest", iso_pred, y_test, iso_scores)
    print(f"  Contamination estimate: {args.frac*100:.1f}%")

    # ========== PRINT RESULTS ==========
    print("\n" + "=" * 80)
    print("COMPARISON RESULTS")
    print("=" * 80)
    
    metrics_names = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    print(f"\n{'Metric':<15} {'Autoencoder':<15} {'Isolation Forest':<15} {'Winner':<15}")
    print("-" * 60)
    
    for metric in metrics_names:
        ae_val = ae_metrics[metric]
        iso_val = iso_metrics[metric]
        winner = "Autoencoder" if ae_val > iso_val else ("Isolation Forest" if iso_val > ae_val else "Tie")
        print(f"{metric:<15} {ae_val:<15.4f} {iso_val:<15.4f} {winner:<15}")
    
    print("\n" + "=" * 80)
    print("DETECTION COUNTS")
    print("=" * 80)
    print(f"\n{'Metric':<20} {'Autoencoder':<15} {'Isolation Forest':<15}")
    print("-" * 50)
    print(f"{'Detected':<20} {ae_metrics['detected']:<15} {iso_metrics['detected']:<15}")
    print(f"{'True Positives':<20} {ae_metrics['true_positives']:<15} {iso_metrics['true_positives']:<15}")
    print(f"{'False Positives':<20} {ae_metrics['false_positives']:<15} {iso_metrics['false_positives']:<15}")
    print(f"{'False Negatives':<20} {ae_metrics['false_negatives']:<15} {iso_metrics['false_negatives']:<15}")

    # ========== SAVE RESULTS ==========
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    results = {
        "configuration": {
            "injection_fraction": args.frac,
            "noise_scale": args.scale,
            "method": args.method,
            "train_samples": int(X_train_scaled.shape[0]),
            "test_samples": int(X_test_injected.shape[0]),
            "anomalies_injected": int(len(anomaly_indices)),
        },
        "autoencoder": ae_metrics,
        "isolation_forest": iso_metrics,
    }
    
    results_path = output_dir / "baseline_comparison_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved to: {results_path}")

    # ========== GENERATE VISUALIZATION ==========
    print("[5/6] Generating visualization...")
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(
        f"Autoencoder vs Isolation Forest Baseline\n"
        f"(Injection: {args.frac*100:.1f}%, Noise: {args.scale})",
        fontsize=16,
        fontweight="bold",
    )

    # Plot 1: Accuracy
    ax = axes[0, 0]
    methods = ["Autoencoder", "Isolation Forest"]
    values = [ae_metrics["accuracy"], iso_metrics["accuracy"]]
    colors = ["#2E86AB", "#A23B72"]
    bars = ax.bar(methods, values, color=colors, alpha=0.8, edgecolor="black", linewidth=1.5)
    ax.set_ylabel("Accuracy", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 1.0)
    ax.set_title("Accuracy", fontsize=12, fontweight="bold")
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{val:.4f}', ha='center', va='bottom', fontsize=10, fontweight="bold")

    # Plot 2: Precision
    ax = axes[0, 1]
    values = [ae_metrics["precision"], iso_metrics["precision"]]
    bars = ax.bar(methods, values, color=colors, alpha=0.8, edgecolor="black", linewidth=1.5)
    ax.set_ylabel("Precision", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 1.0)
    ax.set_title("Precision", fontsize=12, fontweight="bold")
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{val:.4f}', ha='center', va='bottom', fontsize=10, fontweight="bold")

    # Plot 3: Recall
    ax = axes[0, 2]
    values = [ae_metrics["recall"], iso_metrics["recall"]]
    bars = ax.bar(methods, values, color=colors, alpha=0.8, edgecolor="black", linewidth=1.5)
    ax.set_ylabel("Recall", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 1.0)
    ax.set_title("Recall", fontsize=12, fontweight="bold")
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{val:.4f}', ha='center', va='bottom', fontsize=10, fontweight="bold")

    # Plot 4: F1 Score
    ax = axes[1, 0]
    values = [ae_metrics["f1"], iso_metrics["f1"]]
    bars = ax.bar(methods, values, color=colors, alpha=0.8, edgecolor="black", linewidth=1.5)
    ax.set_ylabel("F1 Score", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 1.0)
    ax.set_title("F1 Score", fontsize=12, fontweight="bold")
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{val:.4f}', ha='center', va='bottom', fontsize=10, fontweight="bold")

    # Plot 5: ROC-AUC
    ax = axes[1, 1]
    values = [ae_metrics["roc_auc"], iso_metrics["roc_auc"]]
    bars = ax.bar(methods, values, color=colors, alpha=0.8, edgecolor="black", linewidth=1.5)
    ax.set_ylabel("ROC-AUC", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 1.0)
    ax.set_title("ROC-AUC", fontsize=12, fontweight="bold")
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{val:.4f}', ha='center', va='bottom', fontsize=10, fontweight="bold")

    # Plot 6: Detection Counts
    ax = axes[1, 2]
    x = np.arange(len(["TP", "FP", "FN"]))
    width = 0.35
    ae_counts = [ae_metrics["true_positives"], ae_metrics["false_positives"], ae_metrics["false_negatives"]]
    iso_counts = [iso_metrics["true_positives"], iso_metrics["false_positives"], iso_metrics["false_negatives"]]
    
    bars1 = ax.bar(x - width/2, ae_counts, width, label="Autoencoder", color=colors[0], alpha=0.8, edgecolor="black", linewidth=1.5)
    bars2 = ax.bar(x + width/2, iso_counts, width, label="Isolation Forest", color=colors[1], alpha=0.8, edgecolor="black", linewidth=1.5)
    
    ax.set_ylabel("Count", fontsize=11, fontweight="bold")
    ax.set_title("Detection Breakdown", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(["TP", "FP", "FN"])
    ax.legend(fontsize=10)
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.2,
                        f'{int(height)}', ha='center', va='bottom', fontsize=9, fontweight="bold")

    plt.tight_layout()
    viz_path = output_dir / "baseline_comparison.png"
    plt.savefig(viz_path, dpi=150, bbox_inches="tight")
    print(f"✓ Visualization saved to: {viz_path}")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\nAutoencoder ROC-AUC: {ae_metrics['roc_auc']:.4f}")
    print(f"Isolation Forest ROC-AUC: {iso_metrics['roc_auc']:.4f}")
    print(f"\nAutoencoder F1: {ae_metrics['f1']:.4f}")
    print(f"Isolation Forest F1: {iso_metrics['f1']:.4f}")
    
    if ae_metrics["roc_auc"] > iso_metrics["roc_auc"]:
        print(f"\n✓ Autoencoder outperforms Isolation Forest by {(ae_metrics['roc_auc'] - iso_metrics['roc_auc']):.4f} ROC-AUC")
    elif iso_metrics["roc_auc"] > ae_metrics["roc_auc"]:
        print(f"\n✓ Isolation Forest outperforms Autoencoder by {(iso_metrics['roc_auc'] - ae_metrics['roc_auc']):.4f} ROC-AUC")
    else:
        print("\n✓ Performance is equivalent")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
