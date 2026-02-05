"""Run end-to-end pipeline: load data, preprocess, split, inject anomalies, tune threshold, detect, log to ledger."""
import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score

# Ensure project root is importable when running as: python scripts/run_pipeline.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_data
from src import preprocessing as prep
from src.anomaly_injection import inject_anomalies
from src.autoencoder_model import train_detector, score_anomalies, predict_anomalies
from src.blockchain_logger import log_anomaly


def make_labels(n: int, anomaly_indices: List[int]) -> np.ndarray:
    """Binary labels: 1=anomaly, 0=normal (indices are 0..n-1 positions)."""
    y = np.zeros(n, dtype=int)
    if anomaly_indices:
        y[np.array(anomaly_indices, dtype=int)] = 1
    return y


def tune_threshold_best_f1(
    errors: np.ndarray,
    y_true: np.ndarray,
    q_low: float = 0.05,
    q_high: float = 0.999,
    n_grid: int = 400,
) -> Dict[str, float]:
    """
    Choose threshold that maximizes F1 on validation.
    errors: reconstruction errors (higher => more anomalous)
    y_true: 1=anomaly, 0=normal
    """
    qs = np.linspace(q_low, q_high, n_grid)
    thresholds = np.quantile(errors, qs)

    best = {"threshold": float(thresholds[0]), "precision": 0.0, "recall": 0.0, "f1": 0.0}

    for t in thresholds:
        y_pred = (errors > t).astype(int)
        p = precision_score(y_true, y_pred, zero_division=0)
        r = recall_score(y_true, y_pred, zero_division=0)
        f = f1_score(y_true, y_pred, zero_division=0)

        if f > best["f1"]:
            best = {"threshold": float(t), "precision": float(p), "recall": float(r), "f1": float(f)}

    return best


def run(
    path: str,
    injection_method: str = "noise",
    frac: float = 0.01,
    noise_scale: float = 5.0,
    push_outputs: bool = False,
):
    # 1) Load + preprocess
    df = load_data(path)
    df = prep.handle_missing(df)
    df_encoded, _ = prep.encode_categorical(df)
    X_df, _ = prep.select_features(df_encoded)

    # 2) Split FIRST: 70/15/15
    X_train, X_tmp = train_test_split(X_df, test_size=0.30, random_state=42, shuffle=True)
    X_val, X_test = train_test_split(X_tmp, test_size=0.50, random_state=42, shuffle=True)

    # 3) Fit scaler on TRAIN only
    scaler, X_train_scaled = prep.normalize_features(X_train)

    # 4) Train autoencoder on TRAIN only
    # If your AE supports bottleneck_dim, this helps reduce over-reconstruction:
    try:
        model, _ = train_detector(X_train_scaled, contamination=frac, bottleneck_dim=4)
    except TypeError:
        model, _ = train_detector(X_train_scaled, contamination=frac)

    # 5) VALIDATION: inject anomalies in SCALED space for stable strength
    X_val_scaled = scaler.transform(X_val)
    X_val_scaled_df = pd.DataFrame(X_val_scaled)

    X_val_inj_df, val_anom_idx = inject_anomalies(
        X_val_scaled_df.copy(),
        method=injection_method,
        frac=frac,
        scale=noise_scale,     # used by noise injection; ignored by swap
    )
    X_val_inj = X_val_inj_df.values
    y_val = make_labels(len(X_val_inj), val_anom_idx)

    val_errors = score_anomalies(model, X_val_inj)
    best = tune_threshold_best_f1(val_errors, y_val)

    threshold = best["threshold"]
    print("\n--- Threshold tuning on VALIDATION (best F1) ---")
    print(f"Chosen threshold: {threshold:.6f}")
    print(f"Val Precision: {best['precision']:.3f} | Val Recall: {best['recall']:.3f} | Val F1: {best['f1']:.3f}")

    # 6) TEST: inject anomalies in SCALED space and evaluate ONCE
    X_test_scaled = scaler.transform(X_test)
    X_test_scaled_df = pd.DataFrame(X_test_scaled)

    X_test_inj_df, test_anom_idx = inject_anomalies(
        X_test_scaled_df.copy(),
        method=injection_method,
        frac=frac,
        scale=noise_scale,
    )
    X_test_inj = X_test_inj_df.values
    y_test = make_labels(len(X_test_inj), test_anom_idx)

    test_errors = score_anomalies(model, X_test_inj)
    preds = predict_anomalies(model, X_test_inj, threshold)  # -1 anomaly, 1 normal
    y_pred = (preds == -1).astype(int)

    injected = int(np.sum(y_test == 1))
    detected = int(np.sum(y_pred == 1))
    tp = int(np.sum((y_pred == 1) & (y_test == 1)))

    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, test_errors)

    print("\n--- FINAL results on TEST ---")
    print(f"Injected: {injected}")
    print(f"Detected: {detected}")
    print(f"True positives: {tp}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1: {f1:.3f}")
    print(f"ROC-AUC: {roc_auc:.3f}")

    print("\nScores summary (test):")
    print(f" - min: {float(np.min(test_errors))}")
    print(f" - max: {float(np.max(test_errors))}")
    print(f" - mean: {float(np.mean(test_errors))}")

    # 7) Log detected anomalies (TEST indices) to ledger
    anomalies = [int(i) for i, p in enumerate(preds) if p == -1]

    # Record ledger size before logging so we can measure which entries belong to this run
    ledger_path = Path("blockchain_ledger.jsonl")
    if ledger_path.exists():
        with ledger_path.open("r", encoding="utf-8") as f:
            ledger_before = sum(1 for _ in f)
    else:
        ledger_before = 0

    for idx in anomalies:
        log_anomaly(int(idx), {"detected_by": "autoencoder", "score": float(test_errors[idx])})

    # Read newly appended ledger entries for this run
    ledger_after = 0
    new_entries = []
    if ledger_path.exists():
        with ledger_path.open("r", encoding="utf-8") as f:
            lines = f.readlines()
            ledger_after = len(lines)
            if ledger_after > ledger_before:
                new_lines = lines[ledger_before:ledger_after]
                for ln in new_lines:
                    try:
                        new_entries.append(json.loads(ln))
                    except Exception:
                        continue

    blockchain_logged_indices = [e.get("data", {}).get("index") for e in new_entries]
    blockchain_detected = len(blockchain_logged_indices)
    # Build ledger-based prediction vector aligned with test samples
    y_ledger_pred = np.zeros(len(y_test), dtype=int)
    for i in blockchain_logged_indices:
        if i is None:
            continue
        if 0 <= int(i) < len(y_ledger_pred):
            y_ledger_pred[int(i)] = 1

    # Compute accuracies
    autoencoder_accuracy = float(accuracy_score(y_test, y_pred))
    blockchain_accuracy = float(accuracy_score(y_test, y_ledger_pred)) if blockchain_detected >= 0 else 0.0

    # Blockchain-level metrics
    blockchain_tp = int(np.sum((y_ledger_pred == 1) & (y_test == 1)))
    blockchain_fp = int(np.sum((y_ledger_pred == 1) & (y_test == 0)))
    blockchain_fn = int(np.sum((y_ledger_pred == 0) & (y_test == 1)))
    blockchain_precision = (blockchain_tp / (blockchain_tp + blockchain_fp)) if (blockchain_tp + blockchain_fp) > 0 else 0.0
    blockchain_recall = (blockchain_tp / (blockchain_tp + blockchain_fn)) if (blockchain_tp + blockchain_fn) > 0 else 0.0
    blockchain_f1 = (2 * blockchain_precision * blockchain_recall / (blockchain_precision + blockchain_recall)) if (blockchain_precision + blockchain_recall) > 0 else 0.0

    print("\n--- ACCURACY SUMMARY ---")
    print(f"Autoencoder accuracy: {autoencoder_accuracy:.4f}")
    print(f"Blockchain accuracy: {blockchain_accuracy:.4f} (logged {blockchain_detected} entries)")
    print(f"Blockchain precision: {blockchain_precision:.4f} | recall: {blockchain_recall:.4f} | f1: {blockchain_f1:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")

    # 8) Optionally save outputs
    if push_outputs:
        out_path = Path("pipeline_outputs.json")
        out: Dict[str, Any] = {
            "method": injection_method,
            "frac": frac,
            "noise_scale": noise_scale,
            "threshold": float(threshold),
            "validation_metrics": best,
            "test_metrics": {
                "injected": injected,
                "detected": detected,
                "true_positives": tp,
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "roc_auc": float(roc_auc),
                "accuracy": autoencoder_accuracy,
            },
            "blockchain_metrics": {
                "logged": blockchain_detected,
                "true_positives": blockchain_tp,
                "precision": float(blockchain_precision),
                "recall": float(blockchain_recall),
                "f1": float(blockchain_f1),
                "accuracy": float(blockchain_accuracy),
            },
            "scores_summary_test": {
                "min": float(np.min(test_errors)),
                "max": float(np.max(test_errors)),
                "mean": float(np.mean(test_errors)),
            },
        }
        out_path.write_text(json.dumps(out, indent=2))
        print(f"\nSaved outputs to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/StudentPerformanceFactors.csv", help="Path to dataset CSV")
    parser.add_argument("--method", default="noise", choices=["noise", "swap"])
    parser.add_argument("--frac", type=float, default=0.10, help="Fraction of injected anomalies")
    parser.add_argument("--noise-scale", type=float, default=10.0, help="Noise strength in scaled space (sigma units)")
    parser.add_argument("--push-outputs", action="store_true")
    args = parser.parse_args()

    run(args.data, args.method, args.frac, args.noise_scale, args.push_outputs)
