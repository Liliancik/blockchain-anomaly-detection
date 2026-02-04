"""Run end-to-end pipeline: load data, preprocess, split, inject anomalies, tune threshold, detect, log to ledger."""
import argparse
import json
from pathlib import Path
import pandas as pd
from typing import List

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_data
from src import preprocessing as prep
from src.anomaly_injection import inject_anomalies
from src.autoencoder_model import train_detector, score_anomalies, predict_anomalies
from src.blockchain_logger import log_anomaly


def make_labels(n: int, anomaly_indices: List[int]) -> np.ndarray:
    """y: 1=anomaly, 0=normal"""
    y = np.zeros(n, dtype=int)
    if len(anomaly_indices) > 0:
        y[np.array(anomaly_indices, dtype=int)] = 1
    return y


def tune_threshold_precision_first(
    errors: np.ndarray,
    y_true: np.ndarray,
    min_recall: float = 0.85,
    q_low: float = 0.05,
    q_high: float = 0.999,
    n_grid: int = 250,
):
    """
    Choose threshold t that maximizes PRECISION subject to RECALL >= min_recall.
    If no threshold meets the recall constraint, fall back to best F1.

    errors: reconstruction errors (higher => more anomalous)
    y_true: 1=anomaly, 0=normal
    """
    qs = np.linspace(q_low, q_high, n_grid)
    thresholds = np.quantile(errors, qs)

    best = None
    best_fallback = {"threshold": None, "precision": 0.0, "recall": 0.0, "f1": 0.0}

    for t in thresholds:
        y_pred = (errors > t).astype(int)

        p = precision_score(y_true, y_pred, zero_division=0)
        r = recall_score(y_true, y_pred, zero_division=0)
        f = f1_score(y_true, y_pred, zero_division=0)

        # fallback: maximize F1 regardless of recall constraint
        if f > best_fallback["f1"]:
            best_fallback = {"threshold": float(t), "precision": float(p), "recall": float(r), "f1": float(f)}

        # precision-first (with recall constraint)
        if r >= min_recall:
            if best is None or p > best["precision"] or (p == best["precision"] and f > best["f1"]):
                best = {"threshold": float(t), "precision": float(p), "recall": float(r), "f1": float(f)}

    return best if best is not None else best_fallback


def run(
    path: str,
    injection_method: str = "noise",
    frac: float = 0.02,
    min_recall: float = 0.85,
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

    # 4) Train autoencoder on TRAIN only (assumed normal)
    model, _ = train_detector(X_train_scaled, contamination=frac)

    # 5) Inject anomalies into VALIDATION and tune threshold
    # --- Inject anomalies in SCALED space (recommended) ---
    X_val_scaled = scaler.transform(X_val)

    X_val_scaled_df = pd.DataFrame(X_val_scaled)
    X_val_injected_scaled_df, val_anom_idx = inject_anomalies(
        X_val_scaled_df.copy(),
        method=injection_method,
        frac=frac,
        noise_scale=3.0,      # <<< stronger, in z-space
    )

    X_val_injected_scaled = X_val_injected_scaled_df.values
    y_val = make_labels(len(X_val_injected_scaled), val_anom_idx)

    val_errors = score_anomalies(model, X_val_injected_scaled)
    best = tune_threshold_precision_first(val_errors, y_val, min_recall=min_recall)

    threshold = best["threshold"]
    print("\n--- Threshold tuning on VALIDATION ---")
    print(f"Chosen threshold: {threshold:.6f}")
    print(f"Val Precision: {best['precision']:.3f} | Val Recall: {best['recall']:.3f} | Val F1: {best['f1']:.3f}")

    # 6) FINAL evaluation on TEST (inject anomalies into test, evaluate once)
    # --- Inject anomalies in SCALED space (recommended) ---
    X_test_scaled = scaler.transform(X_test)

    X_test_scaled_df = pd.DataFrame(X_test_scaled)
    X_test_injected_scaled_df, test_anom_idx = inject_anomalies(
        X_test_scaled_df.copy(),
        method=injection_method,
        frac=frac,
        noise_scale=5.0,      # <<< stronger, in z-space
    )

    X_test_injected_scaled = X_test_injected_scaled_df.values
    y_test = make_labels(len(X_test_injected_scaled), test_anom_idx)

    test_errors = score_anomalies(model, X_test_injected_scaled)

    preds = predict_anomalies(model, X_test_injected_scaled, threshold)  # -1 anomaly, 1 normal
    y_pred = (preds == -1).astype(int)

    injected = int(np.sum(y_test == 1))
    detected = int(np.sum(y_pred == 1))
    tp = int(np.sum((y_pred == 1) & (y_test == 1)))

    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print("\n--- FINAL results on TEST ---")
    print(f"Injected: {injected}")
    print(f"Detected: {detected}")
    print(f"True positives: {tp}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1: {f1:.3f}")

    print("\nScores summary (test):")
    print(f" - min: {float(np.min(test_errors))}")
    print(f" - max: {float(np.max(test_errors))}")
    print(f" - mean: {float(np.mean(test_errors))}")

    # 7) Log detected anomalies (TEST indices) to ledger
    anomalies = [int(i) for i, p in enumerate(preds) if p == -1]
    for idx in anomalies:
        log_anomaly(int(idx), {"detected_by": "autoencoder", "score": float(test_errors[idx])})

    # 8) Optionally save outputs
    if push_outputs:
        out_path = Path("pipeline_outputs.json")
        out = {
            "method": injection_method,
            "frac": frac,
            "min_recall_constraint": min_recall,
            "threshold": float(threshold),
            "validation_metrics": best,
            "test_metrics": {
                "injected": injected,
                "detected": detected,
                "true_positives": tp,
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
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
    parser.add_argument("--frac", type=float, default=0.02, help="Fraction of injected anomalies")
    parser.add_argument("--min-recall", type=float, default=0.85, help="Recall constraint for precision-first tuning")
    parser.add_argument("--push-outputs", action="store_true")
    args = parser.parse_args()

    run(args.data, args.method, args.frac, args.min_recall, args.push_outputs)
