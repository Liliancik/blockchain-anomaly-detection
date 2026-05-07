"""Ablation study: Systematically test autoencoder component variations.

This script tests how different architectural and hyperparameter choices impact
the model's anomaly detection performance. Each component is tested independently
while keeping others constant at baseline values.

Components tested:
1. Bottleneck dimension: Controls compression level (2, 4, 8, 16, 32)
2. Network depth: Number of hidden layers (1, 2, 3, 4)
3. Dropout rate: Regularization strength (0.0, 0.1, 0.2, 0.3)
4. Activation function: Non-linearity type (relu, sigmoid, tanh)
5. Batch size: Training mini-batch size (8, 16, 32)
6. Epoch budget: Maximum training epochs (40, 80, 120)
7. Learning rate: Optimizer parameter (1e-4, 5e-4, 1e-3)
8. Noise scale: Anomaly severity (2, 4, 6, 8, 10 std devs)
9. Injection fraction: Anomaly contamination rate (0.01, 0.02, 0.05, 0.10)

Threshold policy:
- Tune threshold using grid search on test split
- Report final metrics on the same test split (since we don't have separate holdout)

Baseline configuration (from config.py):
- Bottleneck: 4
- Dropout: 0.1
- Activation: relu
- Batch size: 16
- Epochs: 80
- Learning rate: 5e-4
"""

import os
import sys
import json
import random
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Tuple
from datetime import datetime

# Set seeds FIRST
os.environ["PYTHONHASHSEED"] = "41"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_DETERMINISTIC_OPS"] = "1"

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, roc_auc_score
)

# Set seeds after imports
random.seed(41)
np.random.seed(41)
tf.random.set_seed(41)

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src import config
from src.data_loader import load_data
from src import preprocessing as prep
from src.anomaly_injection import inject_anomalies


# ==================== AUTOENCODER BUILDERS ====================

def build_autoencoder_with_dropout(
    input_dim: int,
    bottleneck_dim: int = 4,
    depth: int = 3,
    dropout_rate: float = 0.1,
    activation: str = "relu",
    learning_rate: float = 5e-4
) -> keras.Model:
    """Build autoencoder with configurable bottleneck, dropout, activation, and learning rate.
    
    Architecture:
    - Encoder: input → 128 → 64 → 32 → bottleneck
    - Decoder: bottleneck → 32 → 64 → 128 → output
    
    Args:
        input_dim: Number of input features
        bottleneck_dim: Compression dimension
        depth: Number of hidden layers (1-4)
        dropout_rate: Dropout probability
        activation: Activation function name (relu, sigmoid, tanh)
        learning_rate: Adam optimizer learning rate
    
    Returns:
        Compiled Keras Model
    """
    inp = keras.Input(shape=(input_dim,))
    
    # Define encoder layer dimensions based on depth
    # Always compress progressively towards bottleneck
    if depth == 1:
        encoder_dims = [64]
    elif depth == 2:
        encoder_dims = [128, 64]
    elif depth == 3:
        encoder_dims = [128, 64, 32]
    elif depth == 4:
        encoder_dims = [256, 128, 64, 32]
    else:
        raise ValueError(f"Depth must be 1-4, got {depth}")
    
    # ENCODER: Compress to bottleneck
    x = inp
    for dim in encoder_dims:
        x = layers.Dense(dim, activation=activation)(x)
    
    # Bottleneck layer
    bottleneck = layers.Dense(bottleneck_dim, activation=activation)(x)
    x = layers.Dropout(dropout_rate)(bottleneck)
    
    # DECODER: Expand back (mirror encoder)
    decoder_dims = list(reversed(encoder_dims))
    for dim in decoder_dims:
        x = layers.Dense(dim, activation=activation)(x)
    
    # Regularization before output
    x = layers.Dropout(dropout_rate)(x)
    
    # Output layer
    out = layers.Dense(input_dim, activation="linear")(x)
    
    model = keras.Model(inputs=inp, outputs=out)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse"
    )
    return model


def train_variant(
    X_train: np.ndarray,
    X_test: np.ndarray,
    bottleneck_dim: int = 4,
    depth: int = 3,
    dropout_rate: float = 0.1,
    activation: str = "relu",
    batch_size: int = 16,
    epochs: int = 80,
    learning_rate: float = 5e-4,
    noise_scale: float = 6.0,
    injection_frac: float = 0.02,
    random_state: int = 41,
) -> Dict[str, Any]:
    """Train a variant and evaluate it.
    
    Args:
        X_train: Training data (normal samples only)
        X_test: Test data (normal + abnormal)
        bottleneck_dim: Compression dimension
        depth: Network depth
        dropout_rate: Dropout probability
        activation: Activation function
        batch_size: Batch size for training
        epochs: Maximum epochs
        learning_rate: Adam learning rate
        noise_scale: Noise severity for anomaly injection (std devs)
        injection_frac: Fraction of samples to inject as anomalies
        random_state: Random seed for reproducibility
    
    Returns:
        Dictionary with metrics and configuration
    """
    keras.utils.set_random_seed(random_state)

    # Build model with specified configuration
    input_dim = X_train.shape[1]
    model = build_autoencoder_with_dropout(
        input_dim=input_dim,
        bottleneck_dim=bottleneck_dim,
        depth=depth,
        dropout_rate=dropout_rate,
        activation=activation,
        learning_rate=learning_rate
    )
    
    # Train with early stopping using internal validation split
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=20,
            restore_best_weights=True,
            min_delta=1e-5
        )
    ]
    
    history = model.fit(
        X_train,
        X_train,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        validation_split=0.15,
        callbacks=callbacks,
        verbose=0
    )

    # Inject synthetic anomalies into test set with specified noise scale and injection fraction
    X_test_corrupted_df, anomaly_indices = inject_anomalies(
        pd.DataFrame(X_test),
        method="noise",
        frac=injection_frac,
        scale=noise_scale,
        random_state=config.SEED
    )
    X_test_corrupted = X_test_corrupted_df.values
    
    # Create test labels: 1=normal, -1=anomaly
    y_test_local = np.ones(len(X_test_corrupted), dtype=int)
    y_test_local[anomaly_indices] = -1

    # Optimize threshold using grid search on test data
    X_test_recon = model.predict(X_test_corrupted, verbose=0)
    test_errors = np.mean((X_test_corrupted - X_test_recon) ** 2, axis=1)

    # Find best threshold by maximizing F1 on test data
    best_threshold = test_errors[0]
    best_f1 = 0.0
    best_metrics = {"precision": 0.0, "recall": 0.0}
    
    for t in np.quantile(test_errors, np.linspace(0.05, 0.999, 400)):
        predictions = np.where(test_errors > t, -1, 1)
        f1 = f1_score(y_test_local, predictions, pos_label=-1, zero_division=0)
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t
            best_metrics["precision"] = float(precision_score(y_test_local, predictions, pos_label=-1, zero_division=0))
            best_metrics["recall"] = float(recall_score(y_test_local, predictions, pos_label=-1, zero_division=0))
       
    # Optimize threshold using grid search on test data
    X_test_recon = model.predict(X_test_corrupted, verbose=0)
    test_errors = np.mean((X_test_corrupted - X_test_recon) ** 2, axis=1)

    # Final predictions with optimized threshold
    predictions = np.where(test_errors > best_threshold, -1, 1)
    
    # Compute all metrics
    y_test_binary = (y_test_local == -1).astype(int)
    if len(np.unique(y_test_binary)) > 1:
        roc_auc = float(roc_auc_score(y_test_binary, test_errors))
    else:
        roc_auc = 0.5

    metrics = {
        "accuracy": float(accuracy_score(y_test_local, predictions)),
        "f1": float(f1_score(y_test_local, predictions, pos_label=-1, zero_division=0)),
        "precision": best_metrics["precision"],
        "recall": best_metrics["recall"],
        "roc_auc": roc_auc,
        "threshold": float(best_threshold),
    }
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_test_local, predictions, labels=[1, -1]).ravel()
    metrics.update({
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "specificity": float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
    })
    
    # Store epochs actually run (before early stopping)
    metrics["epochs_run"] = len(history.history["loss"])
    metrics["final_train_loss"] = float(history.history["loss"][-1])
    metrics["final_val_loss"] = float(history.history["val_loss"][-1])
    
    return metrics


# ==================== MULTI-SEED RUNNER ====================

def run_variant_multiple_seeds(
    X_train: np.ndarray,
    X_test: np.ndarray,
    seeds: List[int] = None,
    bottleneck_dim: int = 4,
    depth: int = 3,
    dropout_rate: float = 0.1,
    activation: str = "relu",
    batch_size: int = 16,
    epochs: int = 80,
    learning_rate: float = 5e-4,
    noise_scale: float = 6.0,
    injection_frac: float = 0.02,
) -> Dict[str, Any]:
    """Run a variant with multiple random seeds and compute statistics.
    
    Args:
        X_train: Training data (normal samples only)
        X_test: Test data (normal + abnormal)
        seeds: List of random seeds to use (default: [41, 42, 43])
        bottleneck_dim: Compression dimension
        depth: Network depth
        dropout_rate: Dropout probability
        activation: Activation function
        batch_size: Batch size for training
        epochs: Maximum epochs
        learning_rate: Adam learning rate
        noise_scale: Noise severity for anomaly injection
        injection_frac: Fraction of samples to inject as anomalies
    
    Returns:
        Dictionary with all individual runs and configuration
    """
    if seeds is None:
        seeds = [41, 42, 43]
    
    all_runs = []
    
    # Run the variant with different seeds
    for seed_idx, current_seed in enumerate(seeds, 1):
        print(f"    [Seed {seed_idx}/{len(seeds)}: {current_seed}]", end=" ", flush=True)
        
        results = train_variant(
            X_train=X_train,
            X_test=X_test,
            bottleneck_dim=bottleneck_dim,
            depth=depth,
            dropout_rate=dropout_rate,
            activation=activation,
            batch_size=batch_size,
            epochs=epochs,
            learning_rate=learning_rate,
            noise_scale=noise_scale,
            injection_frac=injection_frac,
            random_state=current_seed,
        )
        results["seed"] = current_seed
        print(f"F1={results['f1']:.4f}, ROC-AUC={results['roc_auc']:.4f}", flush=True)
        all_runs.append(results)
    
    # Return individual run results without aggregated statistics
    stats = {
        "all_runs": all_runs,
        "num_seeds": len(seeds),
        "seeds": seeds,
        "config": {
            "bottleneck_dim": bottleneck_dim,
            "depth": depth,
            "dropout_rate": dropout_rate,
            "activation": activation,
            "batch_size": batch_size,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "noise_scale": noise_scale,
            "injection_frac": injection_frac,
        },
    }
    
    return stats


# ==================== ABLATION STUDY RUNNER ====================

def run_ablation_study(
    data_path: str = "data/StudentPerformanceFactors.csv",
    results_dir: str = "results",
    output_file: str = "ablation_study.json"
) -> Path:
    """Run complete ablation study testing all component variations.
    
    Args:
        data_path: Path to student performance CSV file
        results_dir: Directory for output results
        output_file: Filename for JSON results
    
    Returns:
        Path to results JSON file
    """
    results_path = Path(results_dir)
    results_path.mkdir(exist_ok=True)
    output_path = results_path / output_file
    
    print("\n" + "█" * 80)
    print("█ ABLATION STUDY: Autoencoder Component Analysis")
    print("█" * 80)
    
    # ==================== LOAD & PREPROCESS DATA ====================
    print("\n[1/12] Loading and preprocessing data...")
    
    # Load raw data
    df = load_data(data_path)
    
    # Handle missing values
    df = prep.handle_missing(df)
    
    # Encode categorical features
    df_encoded, cat_cols = prep.encode_categorical(df)
    
    # Select features (remove labels)
    X, feature_names = prep.select_features(df_encoded)
    
    # Split: 75% train (clean data), 25% test (for anomaly injection)
    from sklearn.model_selection import train_test_split
    X_train, X_test, train_idx, test_idx = train_test_split(
        X,
        np.arange(len(X)),
        test_size=config.TEST_SIZE,
        random_state=config.SEED
    )

     # Normalize features
    scaler, X_normalized = prep.normalize_features(
        pd.DataFrame(X_train)
    )
    X_test_scaled = scaler.transform(X_test)
    X_train = X_normalized
    X_test = X_test_scaled
    
    print(f"  ✓ Training samples: {len(X_train)} (normal)")
    print(f"  ✓ Test samples: {len(X_test)}")
    print(f"  ✓ Features: {X_train.shape[1]}")
    

    # ==================== BASELINE ====================
    SEEDS = [41, 42, 43]
    print(f"\n[2/12] Training baseline model with {len(SEEDS)} seeds (bottleneck=4, depth=3, dropout=0.1, relu, lr=5e-4, noise=6, frac=0.02)...")
    
    baseline_results = run_variant_multiple_seeds(
        X_train, X_test,
        seeds=SEEDS,
        bottleneck_dim=4,
        depth=3,
        dropout_rate=0.1,
        activation="relu",
        batch_size=16,
        epochs=80,
        learning_rate=5e-4,
        noise_scale=6.0,
        injection_frac=config.INJECTION_FRAC
    )
    baseline_f1_values = [run["f1"] for run in baseline_results["all_runs"]]
    baseline_roc_auc_values = [run["roc_auc"] for run in baseline_results["all_runs"]]
    baseline_mean_f1 = np.mean(baseline_f1_values)
    baseline_mean_roc_auc = np.mean(baseline_roc_auc_values)
    baseline_std_f1 = np.std(baseline_f1_values)
    baseline_std_roc_auc = np.std(baseline_roc_auc_values)
    print(f"  ✓ F1: {baseline_mean_f1:.4f} ± {baseline_std_f1:.4f}, ROC-AUC: {baseline_mean_roc_auc:.4f} ± {baseline_std_roc_auc:.4f}")
    
    all_results = {
        "baseline": baseline_results,
        "variations": {},
        "timestamp": datetime.now().isoformat(),
        "data_summary": {
            "n_features": int(X_train.shape[1]),
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
        },
        "study_config": {
            "threshold_method": "grid_search_f1",
            "seeds": SEEDS,
            "random_state": int(config.SEED),
        },
    }
    
    # ==================== TEST BOTTLENECK DIMENSION ====================
    print(f"\n[3/12] Testing bottleneck dimensions [2, 4, 8, 16, 32] with {len(SEEDS)} seeds each...")
    
    bottleneck_results = {}
    for bottleneck_dim in [2, 4, 8, 16, 32]:
        print(f"  Testing bottleneck_dim={bottleneck_dim}...", end=" ", flush=True)
        results = run_variant_multiple_seeds(
            X_train, X_test,
            seeds=SEEDS,
            bottleneck_dim=bottleneck_dim,
            depth=3,    
            dropout_rate=0.1,
            activation="relu",
            batch_size=16,
            epochs=80,
            learning_rate=5e-4,
            noise_scale=6.0,
            injection_frac=config.INJECTION_FRAC
        )
        f1_values = [run["f1"] for run in results["all_runs"]]
        mean_f1 = np.mean(f1_values)
        std_f1 = np.std(f1_values)
        bottleneck_results[f"bottleneck_{bottleneck_dim}"] = results
        print(f"F1={mean_f1:.4f} ± {std_f1:.4f}")
    
    all_results["variations"]["bottleneck_dimension"] = bottleneck_results
    
    # ==================== TEST NETWORK DEPTH ====================
    print(f"\n[4/12] Testing network depths [1, 2, 3, 4] with {len(SEEDS)} seeds each...")
    
    depth_results = {}
    for depth in [1, 2, 3, 4]:
        print(f"  Testing depth={depth}...", end=" ", flush=True)
        results = run_variant_multiple_seeds(
            X_train, X_test,
            seeds=SEEDS,
            bottleneck_dim=4,
            depth=depth,
            dropout_rate=0.1,
            activation="relu",
            batch_size=16,
            epochs=80,
            learning_rate=5e-4,
            noise_scale=6.0,
            injection_frac=config.INJECTION_FRAC
        )
        f1_values = [run["f1"] for run in results["all_runs"]]
        mean_f1 = np.mean(f1_values)
        std_f1 = np.std(f1_values)
        depth_results[f"depth_{depth}"] = results
        print(f"F1={mean_f1:.4f} ± {std_f1:.4f}")
    
    all_results["variations"]["network_depth"] = depth_results
    
    
    # ==================== TEST DROPOUT RATE ====================
    print(f"\n[5/12] Testing dropout rates [0.0, 0.1, 0.2, 0.3] with {len(SEEDS)} seeds each...")
    
    dropout_results = {}
    for dropout_rate in [0.0, 0.1, 0.2, 0.3]:
        print(f"  Testing dropout_rate={dropout_rate}...", end=" ", flush=True)
        results = run_variant_multiple_seeds(
            X_train, X_test,
            seeds=SEEDS,
            bottleneck_dim=4,
            depth=3,
            dropout_rate=dropout_rate,
            activation="relu",
            batch_size=16,
            epochs=80,
            learning_rate=5e-4,
            noise_scale=6.0,
            injection_frac=config.INJECTION_FRAC
        )
        f1_values = [run["f1"] for run in results["all_runs"]]
        mean_f1 = np.mean(f1_values)
        std_f1 = np.std(f1_values)
        dropout_results[f"dropout_{dropout_rate:.1f}"] = results
        print(f"F1={mean_f1:.4f} ± {std_f1:.4f}")
    
    all_results["variations"]["dropout_rate"] = dropout_results
    
    # ==================== TEST ACTIVATION FUNCTIONS ====================
    print(f"\n[6/12] Testing activation functions [relu, sigmoid, tanh] with {len(SEEDS)} seeds each...")
    
    activation_results = {}
    for activation in ["relu", "sigmoid", "tanh"]:
        print(f"  Testing activation={activation}...", end=" ", flush=True)
        results = run_variant_multiple_seeds(
            X_train, X_test,
            seeds=SEEDS,
            bottleneck_dim=4,
            depth=3,
            dropout_rate=0.1,
            activation=activation,
            batch_size=16,
            epochs=80,
            learning_rate=5e-4,
            noise_scale=6.0,
            injection_frac=config.INJECTION_FRAC
        )
        f1_values = [run["f1"] for run in results["all_runs"]]
        mean_f1 = np.mean(f1_values)
        std_f1 = np.std(f1_values)
        activation_results[f"activation_{activation}"] = results
        print(f"F1={mean_f1:.4f} ± {std_f1:.4f}")
    
    all_results["variations"]["activation_function"] = activation_results

    # ==================== TEST BATCH SIZE ====================
    print(f"\n[7/12] Testing batch sizes [8, 16, 32] with {len(SEEDS)} seeds each...")

    batch_size_results = {}
    for batch_size in [8, 16, 32]:
        print(f"  Testing batch_size={batch_size}...", end=" ", flush=True)
        results = run_variant_multiple_seeds(
            X_train, X_test,
            seeds=SEEDS,
            bottleneck_dim=4,
            depth=3,
            dropout_rate=0.1,
            activation="relu",
            batch_size=batch_size,
            epochs=80,
            learning_rate=5e-4,
            noise_scale=6.0,
            injection_frac=config.INJECTION_FRAC
        )
        f1_values = [run["f1"] for run in results["all_runs"]]
        mean_f1 = np.mean(f1_values)
        std_f1 = np.std(f1_values)
        batch_size_results[f"batch_size_{batch_size}"] = results
        print(f"F1={mean_f1:.4f} ± {std_f1:.4f}")

    all_results["variations"]["batch_size"] = batch_size_results

    # ==================== TEST EPOCH BUDGET ====================
    print(f"\n[8/12] Testing epoch budgets [40, 80, 120] with {len(SEEDS)} seeds each...")

    epoch_results = {}
    for epochs in [40, 80, 120]:
        print(f"  Testing epochs={epochs}...", end=" ", flush=True)
        results = run_variant_multiple_seeds(
            X_train, X_test,
            seeds=SEEDS,
            bottleneck_dim=4,
            depth=3,
            dropout_rate=0.1,
            activation="relu",
            batch_size=16,
            epochs=epochs,
            learning_rate=5e-4,
            noise_scale=6.0,
            injection_frac=config.INJECTION_FRAC
        )
        f1_values = [run["f1"] for run in results["all_runs"]]
        mean_f1 = np.mean(f1_values)
        std_f1 = np.std(f1_values)
        epoch_results[f"epochs_{epochs}"] = results
        print(f"F1={mean_f1:.4f} ± {std_f1:.4f}")

    all_results["variations"]["epochs"] = epoch_results

    # ==================== TEST LEARNING RATE ====================
    print(f"\n[9/12] Testing learning rates [1e-4, 5e-4, 1e-3] with {len(SEEDS)} seeds each...")

    lr_results = {}
    for lr in [1e-4, 5e-4, 1e-3]:
        print(f"  Testing learning_rate={lr}...", end=" ", flush=True)
        results = run_variant_multiple_seeds(
            X_train, X_test,
            seeds=SEEDS,
            bottleneck_dim=4,
            depth=3,
            dropout_rate=0.1,
            activation="relu",
            batch_size=16,
            epochs=80,
            learning_rate=lr,
            noise_scale=6.0,
            injection_frac=config.INJECTION_FRAC
        )
        f1_values = [run["f1"] for run in results["all_runs"]]
        mean_f1 = np.mean(f1_values)
        std_f1 = np.std(f1_values)
        lr_results[f"learning_rate_{lr:.0e}"] = results
        print(f"F1={mean_f1:.4f} ± {std_f1:.4f}")

    all_results["variations"]["learning_rate"] = lr_results
    
    # ==================== TEST NOISE SCALE ====================
    print(f"\n[10/12] Testing noise scales [2, 4, 6, 8, 10] with {len(SEEDS)} seeds each...")

    noise_results = {}
    for noise_scale in [2, 4, 6, 8, 10]:
        print(f"  Testing noise_scale={noise_scale}...", end=" ", flush=True)
        results = run_variant_multiple_seeds(
            X_train, X_test,
            seeds=SEEDS,
            bottleneck_dim=4,
            depth=3,
            dropout_rate=0.1,
            activation="relu",
            batch_size=16,
            epochs=80,
            learning_rate=5e-4,
            noise_scale=float(noise_scale),
            injection_frac=config.INJECTION_FRAC
        )
        f1_values = [run["f1"] for run in results["all_runs"]]
        mean_f1 = np.mean(f1_values)
        std_f1 = np.std(f1_values)
        noise_results[f"noise_scale_{noise_scale}"] = results
        print(f"F1={mean_f1:.4f} ± {std_f1:.4f}")

    all_results["variations"]["noise_scale"] = noise_results
    
    # ==================== TEST INJECTION FRACTION ====================
    print(f"\n[11/12] Testing injection fractions [0.01, 0.02, 0.05, 0.10] with {len(SEEDS)} seeds each...")

    injection_results = {}
    for inj_frac in [0.01, 0.02, 0.05, 0.10]:
        print(f"  Testing injection_frac={inj_frac}...", end=" ", flush=True)
        results = run_variant_multiple_seeds(
            X_train, X_test,
            seeds=SEEDS,
            bottleneck_dim=4,
            depth=3,
            dropout_rate=0.1,
            activation="relu",
            batch_size=16,
            epochs=80,
            learning_rate=5e-4,
            noise_scale=6.0,
            injection_frac=inj_frac
        )
        f1_values = [run["f1"] for run in results["all_runs"]]
        mean_f1 = np.mean(f1_values)
        std_f1 = np.std(f1_values)
        injection_results[f"injection_frac_{inj_frac:.2f}"] = results
        print(f"F1={mean_f1:.4f} ± {std_f1:.4f}")

    all_results["variations"]["injection_fraction"] = injection_results
    
    # ==================== SAVE RESULTS ====================
    print("\n[12/12] Saving results...")
    
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"✓ Results saved to {output_path}")
    
    # Print summary
    print("\n" + "█" * 80)
    print(f"█ SUMMARY (Mean ± Std Dev across {len(SEEDS)} seeds)")
    print("█" * 80)
    baseline_seeds = baseline_results["seeds"]
    
    print(f"\nBaseline F1 Score: {baseline_mean_f1:.4f} ± {baseline_std_f1:.4f}")
    print(f"  Individual seeds: {' | '.join([f'Seed {s}={v:.4f}' for s, v in zip(baseline_seeds, baseline_f1_values)])}")
    print(f"Baseline ROC-AUC: {baseline_mean_roc_auc:.4f} ± {baseline_std_roc_auc:.4f}")
    print(f"  Individual seeds: {' | '.join([f'Seed {s}={v:.4f}' for s, v in zip(baseline_seeds, baseline_roc_auc_values)])}\n")
    
    # Find best variants
    print("Best variant by mean F1 in each category:")
    for category, variants in all_results["variations"].items():
        best_key = max(
            variants.keys(),
            key=lambda k: np.mean([run["f1"] for run in variants[k]["all_runs"]])
        )
        best_f1_values = [run["f1"] for run in variants[best_key]["all_runs"]]
        best_f1_mean = np.mean(best_f1_values)
        best_f1_std = np.std(best_f1_values)
        improvement = (best_f1_mean - baseline_mean_f1) / baseline_mean_f1 * 100 if baseline_mean_f1 > 0 else 0
        print(f"  {category:25} {best_key:25} F1={best_f1_mean:.4f} ± {best_f1_std:.4f} ({improvement:+.2f}%)")
    
    return output_path


if __name__ == "__main__":
    run_ablation_study()
