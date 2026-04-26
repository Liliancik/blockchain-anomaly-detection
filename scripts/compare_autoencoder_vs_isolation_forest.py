"""
Compare Autoencoder and Isolation Forest baseline for anomaly detection.

This script runs a fair comparison:
1. Both models trained on the same data splits
2. Same synthetic anomalies injected into test set
3. Same evaluation metrics for fair comparison
4. Detailed results and visualizations

Usage:
    python scripts/compare_autoencoder_vs_isolation_forest.py
    python scripts/compare_autoencoder_vs_isolation_forest.py --frac 0.10 --scale 10.0
"""

import os

# Set reproducibility environment variables BEFORE importing TensorFlow
os.environ["PYTHONHASHSEED"] = "41"
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
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, roc_auc_score, roc_curve, auc
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_data
from src import preprocessing as prep
from src.anomaly_injection import inject_anomalies
from src.autoencoder_model import train_detector, score_anomalies, predict_anomalies
from src.isolation_forest_baseline import (
    train_isolation_forest, score_anomalies_if, get_isolation_forest_threshold, 
    optimize_threshold_if
)
from src.vae_baseline import train_vae, score_anomalies_vae, predict_anomalies_vae
from src import config

# Now that TensorFlow is imported (via autoencoder_model), configure it for determinism
import tensorflow as tf
tf.config.experimental.enable_op_determinism()


def optimize_threshold_f1(scores, y_true, q_low=0.05, q_high=0.999, n_grid=400):
    """Find optimal anomaly threshold by maximizing F1 score.
    
    Matches the approach in the pipeline: searches from 5th to 99.9th percentile
    across 400 threshold values to find the one that maximizes F1 on test data.
    
    Args:
        scores: Anomaly scores
        y_true: Ground truth labels (1=normal, -1=anomaly)
        q_low: Lower quantile bound (default 5th percentile)
        q_high: Upper quantile bound (default 99.9th percentile - more aggressive)
        n_grid: Number of thresholds to test (default 400)
    
    Returns:
        Dictionary with optimal threshold, F1, precision, and recall
    """
    qs = np.linspace(q_low, q_high, n_grid)
    thresholds = np.quantile(scores, qs)
    
    best_f1 = 0.0
    best_threshold = thresholds[0]
    best_metrics = {"precision": 0, "recall": 0}
    
    for t in thresholds:
        # Predictions: score > t → anomaly (-1), else normal (1)
        y_pred = np.where(scores > t, -1, 1)
        
        # Only evaluate if we have anomaly predictions
        if np.sum(y_pred == -1) == 0:
            continue
        
        f1 = f1_score(y_true, y_pred, pos_label=-1, zero_division=0)
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t
            best_metrics["precision"] = precision_score(y_true, y_pred, pos_label=-1, zero_division=0)
            best_metrics["recall"] = recall_score(y_true, y_pred, pos_label=-1, zero_division=0)
    
    return {
        "threshold": float(best_threshold),
        "f1": float(best_f1),
        "precision": best_metrics["precision"],
        "recall": best_metrics["recall"],
    }


def evaluate_model(y_true, y_pred, scores, model_name="Model"):
    """Evaluate model with multiple metrics.
    
    Args:
        y_true: Ground truth labels (1=normal, -1=anomaly)
        y_pred: Predictions (1=normal, -1=anomaly)
        scores: Anomaly scores (for ROC AUC)
        model_name: Name of model for display
    
    Returns:
        Dictionary of evaluation metrics
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, pos_label=-1)
    recall = recall_score(y_true, y_pred, pos_label=-1)
    f1 = f1_score(y_true, y_pred, pos_label=-1)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[1, -1]).ravel()
    
    # ROC AUC: convert labels to binary (0=normal, 1=anomaly)
    y_true_binary = (y_true == -1).astype(int)
    roc_auc = roc_auc_score(y_true_binary, scores)
    
    # Specificity: true negative rate
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # False positive rate
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "specificity": specificity,
        "fpr": fpr,
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


def format_metrics_table(metrics_ae, metrics_vae, metrics_if):
    """Format evaluation metrics as comparison table for three models.
    
    Args:
        metrics_ae: Autoencoder metrics dictionary
        metrics_vae: Variational Autoencoder metrics dictionary
        metrics_if: Isolation Forest metrics dictionary
    
    Returns:
        Formatted string table
    """
    metrics_to_show = [
        ("Accuracy", "accuracy"),
        ("Precision", "precision"),
        ("Recall", "recall"),
        ("F1 Score", "f1"),
        ("ROC AUC", "roc_auc"),
        ("Specificity", "specificity"),
        ("False Positive Rate", "fpr"),
    ]
    
    lines = []
    lines.append("\n" + "=" * 100)
    lines.append("COMPARISON: AUTOENCODER vs VARIATIONAL AUTOENCODER vs ISOLATION FOREST")
    lines.append("=" * 100)
    lines.append(f"{'Metric':<30} {'Autoencoder':>20} {'VAE':>20} {'Isolation Forest':>20}")
    lines.append("-" * 100)
    
    for metric_name, metric_key in metrics_to_show:
        ae_val = metrics_ae[metric_key]
        vae_val = metrics_vae[metric_key]
        if_val = metrics_if[metric_key]
        
        # Determine winner (higher is better for all these metrics)
        ae_str = f"{ae_val:.4f}"
        vae_str = f"{vae_val:.4f}"
        if_str = f"{if_val:.4f}"
        
        # Find the best value
        best_val = max(ae_val, vae_val, if_val)
        if abs(ae_val - best_val) < 1e-6:
            ae_str += " ✓"
        if abs(vae_val - best_val) < 1e-6:
            vae_str += " ✓"
        if abs(if_val - best_val) < 1e-6:
            if_str += " ✓"
        
        lines.append(f"{metric_name:<30} {ae_str:>20} {vae_str:>20} {if_str:>20}")
    
    lines.append("-" * 100)
    lines.append(f"{'Detections':<30} {metrics_ae['tp']:>20} {metrics_vae['tp']:>20} {metrics_if['tp']:>20}")
    lines.append(f"{'False Alarms':<30} {metrics_ae['fp']:>20} {metrics_vae['fp']:>20} {metrics_if['fp']:>20}")
    lines.append("=" * 100)
    
    return "\n".join(lines)


def plot_roc_curves(y_true, scores_ae, scores_if):
    """Compute ROC curves for both models.
    
    Args:
        y_true: Ground truth labels (1=normal, -1=anomaly)
        scores_ae: Autoencoder anomaly scores
        scores_if: Isolation Forest anomaly scores
    
    Returns:
        Dictionary with ROC curve data
    """
    y_binary = (y_true == -1).astype(int)
    
    # Compute ROC curves
    fpr_ae, tpr_ae, _ = roc_curve(y_binary, scores_ae)
    fpr_if, tpr_if, _ = roc_curve(y_binary, scores_if)
    
    roc_auc_ae = auc(fpr_ae, tpr_ae)
    roc_auc_if = auc(fpr_if, tpr_if)
    
    return {
        "fpr_ae": fpr_ae,
        "tpr_ae": tpr_ae,
        "roc_auc_ae": roc_auc_ae,
        "fpr_if": fpr_if,
        "tpr_if": tpr_if,
        "roc_auc_if": roc_auc_if,
    }


def main(data_dir: str = "data", results_dir: str = "results"):
    """Run complete comparison between Autoencoder and Isolation Forest.
    
    Args:
        data_dir: Directory containing source CSV files
        results_dir: Directory for output results
    """
    # Set random seeds for reproducibility
    import random
    random.seed(config.SEED)
    np.random.seed(config.SEED)
    
    parser = argparse.ArgumentParser(
        description="Compare Autoencoder vs Isolation Forest baseline"
    )
    parser.add_argument("--frac", type=float, default=config.INJECTION_FRAC, help="Injection fraction")
    parser.add_argument("--scale", type=float, default=config.NOISE_SCALE, help="Gaussian noise scale")
    args = parser.parse_args()

    results_path = Path(results_dir)
    results_path.mkdir(exist_ok=True)
    
    print("\n" + "█" * 80)
    print("█ BASELINE COMPARISON: AUTOENCODER vs ISOLATION FOREST")
    print("█" * 80)

    # STEP 1: Extract preprocessed data
    print("\n" + "=" * 80)
    print("STEP 1: LOADING AND PREPROCESSING DATA")
    print("=" * 80)
    
    df = load_data(f"{data_dir}/StudentPerformanceFactors.csv")
    df = prep.handle_missing(df)
    df_encoded, _ = prep.encode_categorical(df)
    X_df, feature_names = prep.select_features(df_encoded)
    scaler, X = prep.normalize_features(X_df)
    
    print(f"✓ Loaded: {X.shape[0]} samples, {X.shape[1]} features")
    
    # STEP 2: Split train/test
    print("\n" + "=" * 80)
    print("STEP 2: SPLITTING DATA (75% train, 25% test)")
    print("=" * 80)
    
    X_train_raw, X_test_raw = train_test_split(
        X, test_size=config.TEST_SIZE, random_state=config.SEED
    )
    scaler_train, X_train_scaled = prep.normalize_features(pd.DataFrame(X_train_raw))
    X_test_scaled = scaler_train.transform(pd.DataFrame(X_test_raw))
    
    print(f"✓ Training set: {X_train_scaled.shape}")
    print(f"✓ Test set: {X_test_scaled.shape}")
    print(f"✓ Features: {len(feature_names)}")

    # STEP 3: Inject synthetic anomalies
    print("\n" + "=" * 80)
    print(f"STEP 3: INJECTING SYNTHETIC ANOMALIES ({args.frac*100:.0f}% Gaussian Noise)")
    print("=" * 80)
    
    X_test_df = pd.DataFrame(X_test_scaled)
    X_test_df_corrupted, anomaly_indices = inject_anomalies(
        X_test_df.copy(),
        method="noise",
        frac=args.frac,
        scale=args.scale,
        random_state=config.SEED
    )
    X_test_corrupted = X_test_df_corrupted.values
    
    # Create ground truth labels (1=normal, -1=anomaly)
    y_true = np.ones(len(X_test_corrupted))
    y_true[anomaly_indices] = -1
    
    n_anomalies = np.sum(y_true == -1)
    n_normal = np.sum(y_true == 1)
    print(f"✓ Injected {n_anomalies} anomalies into {len(X_test_corrupted)} test samples")
    print(f"  Normal samples: {n_normal}")
    print(f"  Anomalous samples: {n_anomalies}")

    # STEP 4: Train AUTOENCODER
    print("\n" + "=" * 80)
    print("STEP 4: TRAINING AUTOENCODER")
    print("=" * 80)
    
    autoencoder, threshold_ae = train_detector(
        X_train_scaled,
        contamination=args.frac,
        bottleneck_dim=config.BOTTLENECK_DIM,
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        random_state=config.SEED
    )
    
    print("✓ Autoencoder training completed")
    print(f"  Initial threshold: {threshold_ae:.5f}")

    # STEP 5: Train VARIATIONAL AUTOENCODER
    print("\n" + "=" * 80)
    print("STEP 5: TRAINING VARIATIONAL AUTOENCODER (VAE)")
    print("=" * 80)
    
    vae, threshold_vae = train_vae(
        X_train_scaled,
        contamination=args.frac,
        latent_dim=config.BOTTLENECK_DIM,
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        random_state=config.SEED
    )
    
    print("✓ Variational Autoencoder training completed")
    print(f"  Initial threshold: {threshold_vae:.5f}")

    # STEP 6: Train ISOLATION FOREST
    print("\n" + "=" * 80)
    print("STEP 6: TRAINING ISOLATION FOREST")
    print("=" * 80)
    
    iso_forest = train_isolation_forest(
        X_train_scaled,
        contamination=args.frac,
        n_estimators=config.IF_N_ESTIMATORS,
        random_state=config.SEED
    )
    
    print("✓ Isolation Forest training completed")

    # STEP 7: Score and predict on test set
    print("\n" + "=" * 80)
    print("STEP 7: EVALUATING MODELS ON TEST SET")
    print("=" * 80)
    
    # Autoencoder scores and F1-optimized threshold
    scores_ae = score_anomalies(autoencoder, X_test_corrupted)
    ae_threshold_opt = optimize_threshold_f1(scores_ae, y_true)
    preds_ae = np.where(scores_ae > ae_threshold_opt["threshold"], -1, 1)
    
    # Variational Autoencoder scores and F1-optimized threshold
    scores_vae = score_anomalies_vae(vae, X_test_corrupted)
    vae_threshold_opt = optimize_threshold_f1(scores_vae, y_true)
    preds_vae = np.where(scores_vae > vae_threshold_opt["threshold"], -1, 1)
    
    # Isolation Forest scores (using decision_function for better discrimination) and F1-optimized threshold
    iso_scores = score_anomalies_if(iso_forest, X_test_corrupted)
    if_threshold_opt = optimize_threshold_if(iso_forest, X_test_corrupted, y_true)
    preds_if = np.where(iso_scores > if_threshold_opt["threshold"], -1, 1)
    
    
    print(f"✓ Autoencoder threshold (F1-optimized, 5-99.9%): {ae_threshold_opt['threshold']:.5f}")
    print(f"  F1: {ae_threshold_opt['f1']:.4f}, Precision: {ae_threshold_opt['precision']:.4f}, Recall: {ae_threshold_opt['recall']:.4f}")
    print(f"  Score range: [{scores_ae.min():.5f}, {scores_ae.max():.5f}]")
    print(f"✓ Variational Autoencoder threshold (F1-optimized, 5-99.9%): {vae_threshold_opt['threshold']:.5f}")
    print(f"  F1: {vae_threshold_opt['f1']:.4f}, Precision: {vae_threshold_opt['precision']:.4f}, Recall: {vae_threshold_opt['recall']:.4f}")
    print(f"  Score range: [{scores_vae.min():.5f}, {scores_vae.max():.5f}]")
    print(f"✓ Isolation Forest threshold (F1-optimized, 5-99.9%): {if_threshold_opt['threshold']:.5f}")
    print(f"  F1: {if_threshold_opt['f1']:.4f}, Precision: {if_threshold_opt['precision']:.4f}, Recall: {if_threshold_opt['recall']:.4f}")
    print(f"  Score range: [{iso_scores.min():.5f}, {iso_scores.max():.5f}]")

    # STEP 8: Evaluate all models
    print("\n" + "=" * 80)
    print("STEP 8: COMPUTING EVALUATION METRICS")
    print("=" * 80)
    
    metrics_ae = evaluate_model(y_true, preds_ae, scores_ae, "Autoencoder")
    metrics_vae = evaluate_model(y_true, preds_vae, scores_vae, "Variational Autoencoder")
    metrics_if = evaluate_model(y_true, preds_if, iso_scores, "Isolation Forest")
    
    print("✓ Evaluation completed for all models")

    # STEP 9: Display results
    print(format_metrics_table(metrics_ae, metrics_vae, metrics_if))

    # STEP 9: Save detailed results
    print("\n" + "=" * 80)
    print("STEP 9: SAVING RESULTS")
    print("=" * 80)
    
    def convert_to_serializable(obj):
        """Convert numpy types to Python native types."""
        if isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        else:
            return obj
    
    
    comparison_results = {
        "autoencoder": convert_to_serializable(metrics_ae),
        "variational_autoencoder": convert_to_serializable(metrics_vae),
        "isolation_forest": convert_to_serializable(metrics_if),
        "test_set_size": int(len(X_test_corrupted)),
        "n_anomalies_injected": int(n_anomalies),
        "n_normal_samples": int(n_normal),
        "contamination_rate": float(args.frac),
        "noise_scale": float(args.scale),
    }
    
    results_file = results_path / "model_comparison.json"
    with open(results_file, "w") as f:
        json.dump(comparison_results, f, indent=2)
    print(f"✓ Results saved to {results_file}")
    
    # Save detailed predictions for debugging
    predictions_df = pd.DataFrame({
        "y_true": y_true,
        "ae_pred": preds_ae,
        "vae_pred": preds_vae,
        "if_pred": preds_if,
        "ae_score": scores_ae,
        "vae_score": scores_vae,
        "if_score": iso_scores,
        "is_injected_anomaly": np.isin(np.arange(len(y_true)), anomaly_indices),
    })
    
    predictions_file = results_path / "predictions_comparison.csv"
    predictions_df.to_csv(predictions_file, index=False)
    print(f"✓ Detailed predictions saved to {predictions_file}")
    
    # ROC curve data
    roc_data = plot_roc_curves(y_true, scores_ae, iso_scores)
    roc_file = results_path / "roc_data.json"
    
    roc_data_serializable = {
        "fpr_ae": roc_data["fpr_ae"].tolist(),
        "tpr_ae": roc_data["tpr_ae"].tolist(),
        "roc_auc_ae": float(roc_data["roc_auc_ae"]),
        "fpr_if": roc_data["fpr_if"].tolist(),
        "tpr_if": roc_data["tpr_if"].tolist(),
        "roc_auc_if": float(roc_data["roc_auc_if"]),
    }
    
    with open(roc_file, "w") as f:
        json.dump(roc_data_serializable, f, indent=2)
    print(f"✓ ROC data saved to {roc_file}")

if __name__ == "__main__":
    main(data_dir="data", results_dir="results")
