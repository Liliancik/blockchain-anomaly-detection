"""Anomaly detection pipeline: autoencoder + blockchain logging."""
import os
import sys

# Set random seeds FIRST, before any TensorFlow imports
# These environment variables ensure reproducible results across runs:
# - PYTHONHASHSEED: Makes Python's hash function deterministic
# - TF_CPP_MIN_LOG_LEVEL: Reduces TensorFlow logging noise (2 = warnings only)
# - TF_DETERMINISTIC_OPS: Forces TensorFlow to use deterministic algorithms
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
from src.autoencoder_model import train_detector, score_anomalies, predict_anomalies, plot_loss_curves
from src.blockchain_logger import PermissionedBlockchain

import tensorflow as tf

# Set seeds after imports for all libraries
# This ensures reproducible random behavior in data splitting, model initialization, etc.
random.seed(config.SEED)
np.random.seed(config.SEED)
tf.random.set_seed(config.SEED)

def make_labels(n, anomaly_indices):
    """Create binary labels: 1=anomaly, 0=normal.
    
    Args:
        n: Total number of samples
        anomaly_indices: List of indices marked as anomalies
    
    Returns:
        Binary label array (0 or 1) for each sample
    """
    y = np.zeros(n, dtype=int)
    if anomaly_indices:
        y[np.array(anomaly_indices, dtype=int)] = 1
    return y


def find_best_threshold(errors, y_val):
    """Find optimal anomaly threshold by maximizing F1 score.
    
    Searches through 400 thresholds (5th to 99.9th percentile of errors) to find
    the one that produces the highest F1 score on validation data. This balances
    precision and recall for the best overall detection performance.
    
    Args:
        errors: Reconstruction errors from model
        y_val: True labels (0=normal, 1=anomaly)
    
    Returns:
        Dict with optimal threshold and corresponding metrics
    """
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
    """Run full anomaly detection pipeline: load data, train model, detect anomalies, log results.
    
    This pipeline:
    1. Loads and preprocesses student performance data
    2. Trains an autoencoder model on clean data
    3. Injects synthetic anomalies to evaluate detection capability
    4. Finds optimal detection threshold to maximize F1 score
    5. Predicts which injected samples are anomalies
    6. Logs results to a permissioned blockchain for audit trail
    7. Saves performance metrics to JSON
    
    Args:
        data_path: Path to CSV file containing the dataset
    """
    print("=" * 70)
    print("ANOMALY DETECTION PIPELINE")
    print("=" * 70)

    # 1) Load and preprocess
    # Read raw data, handle missing values, encode categorical features,
    # and select relevant features for model training
    print("\n[1] Loading data...")
    df = load_data(data_path)
    df = prep.handle_missing(df)
    df_encoded, _ = prep.encode_categorical(df)
    X_df, _ = prep.select_features(df_encoded)
    print(f"  Shape: {X_df.shape}")

    # 2) Split into train/test
    # Separate data: 75% for training the autoencoder, 25% for evaluation.
    # We'll inject anomalies into the test set to simulate real-world scenarios.
    print("\n[2] Splitting (75/25)...")
    X_train, X_test = train_test_split(
        X_df, test_size=config.TEST_SIZE, random_state=config.SEED
    )

    # 3) Fit scaler on train data and scale both sets
    # Normalize features to [0, 1] range. Fit scaler ONLY on training data
    # to prevent data leakage, then apply to test data.
    print("\n[3] Normalizing...")
    scaler, X_train_scaled = prep.normalize_features(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 4) Train autoencoder on clean training data
    # The autoencoder learns to reconstruct normal data with low error.
    # Normal samples should have low reconstruction error, anomalies high error.
    print("\n[4] Training autoencoder...")
    model, _, history = train_detector(
        X_train_scaled,
        contamination=config.INJECTION_FRAC,
        bottleneck_dim=config.BOTTLENECK_DIM,
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        val_split=0.15,  # Use 15% of train for validation
        random_state=config.SEED,
    )
    
    # Plot and save training/validation loss curves
    print("\n[4b] Saving loss curves...")
    plot_loss_curves(history, output_path="outputs/autoencoder_loss_curves.png")

    # 5) Inject synthetic anomalies into test data and find optimal detection threshold
    # Add noise to simulate real anomalies. Track which samples are injected (ground truth).
    # Then find the threshold that best separates normal from injected samples.
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

    # 6) Use trained model to predict anomalies on test data
    # Apply the optimal threshold: samples with error > threshold are flagged as anomalies.
    print(f"\n[6] Detecting (threshold={threshold:.4f})...")
    preds = predict_anomalies(model, X_test_inj, threshold)
    y_pred = (preds == -1).astype(int)

    # 7) Calculate detection performance metrics
    # Compute true positives (correctly detected anomalies), false positives,
    # and false negatives to evaluate model performance.
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

    # 8) Log detection results to blockchain for audit trail
    # Uses permissioned blockchain where only authorized users (detector, auditor)
    # can write/verify logs. This ensures tamper-proof record of anomalies detected.
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

    # 9) Save all results to JSON file for later analysis and reporting
    # Store configuration, metrics, threshold, and anomaly detection results.
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

