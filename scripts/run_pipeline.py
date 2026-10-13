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
from src.poa_blockchain import PoABlockchain
from scripts.tamper_detection_test import test_tamper_detection

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

    # 2) Split into train/validation/test
    # Keep the final test set completely separate from model and threshold selection.
    print("\n[2] Splitting (60/15/25)...")

    # First reserve 25% as the untouched final test set.
    X_train_val, X_test = train_test_split(
        X_df,
        test_size=config.TEST_SIZE,
        random_state=config.SEED,
    )

    # Split the remaining 75% into 60% training and 15% validation.
    # 15 / 75 = 0.20, so validation is 20% of X_train_val.
    val_fraction_of_train_val = config.VAL_SIZE / (
        config.TRAIN_SIZE + config.VAL_SIZE
    )

    X_train, X_val = train_test_split(
        X_train_val,
        test_size=val_fraction_of_train_val,
        random_state=config.SEED,
    )

    print(f"  Train:      {len(X_train)}")
    print(f"  Validation: {len(X_val)}")
    print(f"  Test:       {len(X_test)}")

    # 3) Fit scaler ONLY on training data.
    # Apply the same fitted scaler to validation and test data.
    print("\n[3] Normalizing...")
    scaler, X_train_scaled = prep.normalize_features(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # 4) Train autoencoder on clean training data
    # The autoencoder learns to reconstruct normal data with low error.
    # Normal samples should have low reconstruction error, anomalies high error.
    print("\n[4] Training autoencoder...")
    model, _, history = train_detector(
        X_train_scaled,
        X_val=X_val_scaled,
        contamination=config.INJECTION_FRAC,
        bottleneck_dim=config.BOTTLENECK_DIM,
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        random_state=config.SEED,
    )
    
    # Plot and save training/validation loss curves
    print("\n[4b] Saving loss curves...")
    plot_loss_curves(history, output_path="outputs/autoencoder_loss_curves.png")

    # 5) Inject synthetic anomalies independently into validation and test sets.
    # Validation anomalies are used ONLY for threshold selection.
    # Test anomalies are used ONLY for final evaluation.
    print(f"\n[5] Injecting anomalies ({config.INJECTION_FRAC:.0%})...")

    # ----- Validation set -----
    X_val_inj_df, val_anom_idx = inject_anomalies(
        pd.DataFrame(X_val_scaled),
        method="noise",
        frac=config.INJECTION_FRAC,
        scale=config.NOISE_SCALE,
        random_state=config.SEED,
    )

    X_val_inj = X_val_inj_df.values
    y_val = make_labels(len(X_val_inj), val_anom_idx)

    # Select threshold using validation labels only.
    val_errors = score_anomalies(model, X_val_inj)
    best = find_best_threshold(val_errors, y_val)
    threshold = best["threshold"]

    print(f"  Validation anomalies: {len(val_anom_idx)}")
    print(f"  Selected threshold:   {threshold:.6f}")
    print(f"  Validation F1:         {best['f1']:.4f}")

    # ----- Final test set -----
    # Use a different random seed so the test corruption is generated
    # independently from the validation corruption.
    X_test_inj_df, test_anom_idx = inject_anomalies(
        pd.DataFrame(X_test_scaled),
        method="noise",
        frac=config.INJECTION_FRAC,
        scale=config.NOISE_SCALE,
        random_state=config.SEED + 1,
    )

    X_test_inj = X_test_inj_df.values
    y_test = make_labels(len(X_test_inj), test_anom_idx)

    print(f"  Test anomalies:       {len(test_anom_idx)}")

    # 6) Use trained model to predict anomalies on test data
    # Apply the optimal threshold: samples with error > threshold are flagged as anomalies.
    print(f"\n[6] Detecting (threshold={threshold:.4f})...")

    # Compute reconstruction errors on the untouched final test set.
    # These scores are used only for final evaluation (e.g. ROC-AUC),
    # NOT for threshold selection.
    test_errors = score_anomalies(model, X_test_inj)

    # Apply the threshold that was already selected on validation data.
    preds = np.where(test_errors > threshold, -1, 1)
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

    # 8) Log detection results to permissioned PoA blockchain
    # Uses three authorized validator nodes with round-robin Proof-of-Authority.
    # Each validated block is replicated across all validator ledgers.
    print("\n[7] Logging to permissioned PoA blockchain...")

    poa_blockchain = PoABlockchain(
        ledger_dir="results/poa_ledgers"
    )

    # Log each detected anomaly
    anomaly_indices = [int(i) for i, p in enumerate(preds) if p == -1]

    for idx in anomaly_indices:

        # Select the next authorized validator using round-robin PoA
        validator = poa_blockchain.get_current_validator()

        anomaly_data = {
            "sample_index": idx,
            "score": float(test_errors[idx]),
            "detector": "autoencoder",
        }

        poa_blockchain.create_block(
            validator=validator,
            anomaly_data=anomaly_data,
        )

    print(
        f"  ✓ Logged {len(anomaly_indices)} anomalies "
        f"to permissioned PoA blockchain"
    )

    # Verify all validator ledgers and replication consistency
    is_valid = poa_blockchain.verify_all_validators()

    if is_valid:
        print("  ✓ PoA blockchain verification passed")
        print("  ✓ 3/3 validator ledgers valid and identical")
    else:
        print(
            "  ✗ PoA blockchain verification FAILED - "
            "possible tampering or replica inconsistency detected!"
        )

    # Run controlled blockchain tampering experiment
    print("\n[8] Running controlled blockchain tampering experiment...")

    tamper_results = test_tamper_detection()

    if tamper_results["tampering_detected"]:
        print("  ✓ Controlled tampering successfully detected")
    else:
        print("  ✗ Controlled tampering was not detected")

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

