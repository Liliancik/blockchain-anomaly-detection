"""Anomaly detection pipeline: autoencoder + blockchain logging."""
import os
import sys

# Set random seeds FIRST, before any TensorFlow imports
os.environ["PYTHONHASHSEED"] = "41"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_DETERMINISTIC_OPS"] = "1"

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src import config
from src.data_loader import load_data
from src import preprocessing as prep
from src.anomaly_injection import inject_anomalies
from src.autoencoder_model import train_detector, score_anomalies, predict_anomalies
from src.blockchain_logger import PermissionedBlockchain

import tensorflow as tf

# Set seeds after imports
random.seed(config.SEED)
np.random.seed(config.SEED)
tf.random.set_seed(config.SEED)

def make_labels(n, anomaly_indices):
    """1=anomaly, 0=normal."""
    y = np.zeros(n, dtype=int)
    if anomaly_indices:
        y[np.array(anomaly_indices, dtype=int)] = 1
    return y


def find_best_threshold(errors, y_val):
    """Maximize F1 on validation errors."""
    best = {"threshold": 0.0, "f1": 0.0, "precision": 0.0, "recall": 0.0}
    for t in np.quantile(errors, np.linspace(0.05, 0.999, 400)):
        y_pred = (errors > t).astype(int)
        f = f1_score(y_val, y_pred, zero_division=0)
        if f > best["f1"]:
            best["threshold"] = float(t)
            best["f1"] = float(f)
            best["precision"] = float(precision_score(y_val, y_pred, zero_division=0))
            best["recall"] = float(recall_score(y_val, y_pred, zero_division=0))
    return best


def main(data_path="data/StudentPerformanceFactors.csv"):
    """Run pipeline: train autoencoder, detect anomalies, log to blockchain."""
    print("=" * 70)
    print("ANOMALY DETECTION PIPELINE")
    print("=" * 70)

    # 1) Load and preprocess
    print("\n[1] Loading data...")
    df = load_data(data_path)
    df = prep.handle_missing(df)
    df_encoded, _ = prep.encode_categorical(df)
    X_df, _ = prep.select_features(df_encoded)
    print(f"  Shape: {X_df.shape}")

    # 2) Split into train/test
    print("\n[2] Splitting (75/25)...")
    X_train, X_test = train_test_split(
        X_df, test_size=config.TEST_SIZE, random_state=config.SEED
    )

    # 3) Fit scaler on train
    print("\n[3] Normalizing...")
    scaler, X_train_scaled = prep.normalize_features(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 4) Train autoencoder
    print("\n[4] Training autoencoder...")
    model, _ = train_detector(
        X_train_scaled,
        contamination=config.INJECTION_FRAC,
        bottleneck_dim=config.BOTTLENECK_DIM,
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        val_split=0.15,  # Use 15% of train for validation
        random_state=config.SEED,
    )

    # 5) Inject anomalies and find threshold on test
    print(f"\n[5] Injecting anomalies ({config.INJECTION_FRAC:.0%})...")
    X_test_inj_df, test_anom_idx = inject_anomalies(
        pd.DataFrame(X_test_scaled),
        method="noise",
        frac=config.INJECTION_FRAC,
        scale=config.NOISE_SCALE,
    )
    X_test_inj = X_test_inj_df.values
    y_test = make_labels(len(X_test_inj), test_anom_idx)

    test_errors = score_anomalies(model, X_test_inj)
    best = find_best_threshold(test_errors, y_test)
    threshold = best["threshold"]

    # 6) Predict
    print(f"\n[6] Detecting (threshold={threshold:.4f})...")
    preds = predict_anomalies(model, X_test_inj, threshold)
    y_pred = (preds == -1).astype(int)

    # 7) Metrics
    tp = int(np.sum((y_pred == 1) & (y_test == 1)))
    fp = int(np.sum((y_pred == 1) & (y_test == 0)))
    fn = int(np.sum((y_pred == 0) & (y_test == 1)))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    roc_auc = roc_auc_score(y_test, test_errors)

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1:        {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    print(f"Detected:  {int(np.sum(y_pred))}/{len(y_test)}")
    print(f"TP/FP/FN:  {tp}/{fp}/{fn}")

    # 8) Log to blockchain with permissioned access
    print(f"\n[7] Logging to permissioned blockchain ({config.LEDGER_PATH})...")
    blockchain = PermissionedBlockchain(ledger_path=config.LEDGER_PATH)
    
    # Login as detector user (has write permission)
    blockchain.login("detector", "detector_password")
    print(f"  ✓ Authenticated as '{blockchain.current_user.username}' (role: {blockchain.current_user.role})")
    
    # Log each detected anomaly
    anomaly_indices = [int(i) for i, p in enumerate(preds) if p == -1]
    for idx in anomaly_indices:
        blockchain.log_anomaly(idx, {"score": float(test_errors[idx])})
    
    print(f"  ✓ Logged {len(anomaly_indices)} anomalies to blockchain")
    
    # Verify blockchain integrity (auditor verification)
    blockchain.logout()
    blockchain.login("auditor", "auditor_password")
    is_valid = blockchain.verify_blockchain()
    if is_valid:
        print(f"  ✓ Blockchain verification passed (role: {blockchain.current_user.role})")
    else:
        print(f"  ✗ Blockchain verification FAILED - possible tampering detected!")
    
    blockchain.logout()

    # 9) Save output
    out_path = Path("pipeline_outputs.json")
    output = {
        "config": {
            "train_size": config.TRAIN_SIZE,
            "test_size": config.TEST_SIZE,
            "injection_frac": config.INJECTION_FRAC,
            "bottleneck_dim": config.BOTTLENECK_DIM,
            "epochs": config.EPOCHS,
        },
        "metrics": {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "roc_auc": float(roc_auc),
            "tp": tp,
            "fp": fp,
            "fn": fn,
        },
        "threshold": float(threshold),
    }
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\n✓ Results saved to {out_path}")


if __name__ == "__main__":
    main()

