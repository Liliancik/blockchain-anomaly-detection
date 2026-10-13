"""Ablation study for the StudentPerformanceFactors Autoencoder.

This script evaluates architectural and training choices while using a fixed
train/validation/test procedure.

Experimental procedure:
- Fixed 60/15/25 train/validation/test split
- Training data are used for model fitting
- Synthetic anomalies are injected independently into validation and test sets
- Threshold is selected using validation data only
- Test data are evaluated only after the threshold has been fixed
- Experiments are repeated across five random seeds

Components tested:
1. Bottleneck dimension: 2, 4, 8, 16, 32
2. Network depth: 1, 2, 3, 4
3. Dropout rate: 0.0, 0.1, 0.2, 0.3
4. Activation function: relu, sigmoid, tanh
5. Batch size: 8, 16, 32
6. Epoch budget: 40, 80, 120
7. Learning rate: 1e-4, 5e-4, 1e-3
8. Noise scale: 2, 4, 6, 8, 10 standard deviations

The anomaly injection fraction is fixed at 2% and is not optimized.
"""

import os
import sys
import json
import random
import numpy as np
import pandas as pd

from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime


# ---------------------------------------------------------------------------
# REPRODUCIBILITY
# ---------------------------------------------------------------------------

os.environ["PYTHONHASHSEED"] = "41"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_DETERMINISTIC_OPS"] = "1"

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


random.seed(41)
np.random.seed(41)
tf.random.set_seed(41)


# ---------------------------------------------------------------------------
# PROJECT IMPORTS
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src import config
from src.data_loader import load_data
from src import preprocessing as prep
from src.anomaly_injection import inject_anomalies


# ===========================================================================
# AUTOENCODER
# ===========================================================================

def build_autoencoder_with_dropout(
    input_dim: int,
    bottleneck_dim: int = 4,
    depth: int = 3,
    dropout_rate: float = 0.1,
    activation: str = "relu",
    learning_rate: float = 5e-4,
) -> keras.Model:
    """Build configurable Autoencoder."""

    inp = keras.Input(shape=(input_dim,))

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

    # Encoder
    x = inp

    for dim in encoder_dims:
        x = layers.Dense(dim, activation=activation)(x)

    # Bottleneck
    bottleneck = layers.Dense(
        bottleneck_dim,
        activation=activation
    )(x)

    x = layers.Dropout(dropout_rate)(bottleneck)

    # Decoder
    decoder_dims = list(reversed(encoder_dims))

    for dim in decoder_dims:
        x = layers.Dense(dim, activation=activation)(x)

    x = layers.Dropout(dropout_rate)(x)

    # Output
    out = layers.Dense(
        input_dim,
        activation="linear"
    )(x)

    model = keras.Model(
        inputs=inp,
        outputs=out
    )

    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=learning_rate
        ),
        loss="mse"
    )

    return model


# ===========================================================================
# TRAIN AND EVALUATE ONE VARIANT
# ===========================================================================

def train_variant(
    X_train: np.ndarray,
    X_val: np.ndarray,
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
    """Train one Autoencoder configuration and evaluate on held-out test data."""

    keras.backend.clear_session()
    keras.utils.set_random_seed(random_state)

    input_dim = X_train.shape[1]

    model = build_autoencoder_with_dropout(
        input_dim=input_dim,
        bottleneck_dim=bottleneck_dim,
        depth=depth,
        dropout_rate=dropout_rate,
        activation=activation,
        learning_rate=learning_rate,
    )

    # -----------------------------------------------------------------------
    # MODEL TRAINING
    # -----------------------------------------------------------------------

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=20,
            restore_best_weights=True,
            min_delta=1e-5,
        )
    ]

    history = model.fit(
        X_train,
        X_train,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,

        # X_val is kept for threshold selection.
        # Internal validation is used only for early stopping.
        validation_split=0.15,

        callbacks=callbacks,
        verbose=0,
    )

    # -----------------------------------------------------------------------
    # INDEPENDENT VALIDATION ANOMALIES
    # -----------------------------------------------------------------------

    X_val_corrupted_df, val_anomaly_indices = inject_anomalies(
        pd.DataFrame(X_val),
        method="noise",
        frac=injection_frac,
        scale=noise_scale,
        random_state=random_state,
    )

    X_val_corrupted = X_val_corrupted_df.values

    y_val_local = np.ones(
        len(X_val_corrupted),
        dtype=int
    )

    y_val_local[val_anomaly_indices] = -1

    # -----------------------------------------------------------------------
    # INDEPENDENT TEST ANOMALIES
    # -----------------------------------------------------------------------

    X_test_corrupted_df, test_anomaly_indices = inject_anomalies(
        pd.DataFrame(X_test),
        method="noise",
        frac=injection_frac,
        scale=noise_scale,

        # Different seed prevents the validation and test perturbations
        # from being identical.
        random_state=random_state + 1000,
    )

    X_test_corrupted = X_test_corrupted_df.values

    y_test_local = np.ones(
        len(X_test_corrupted),
        dtype=int
    )

    y_test_local[test_anomaly_indices] = -1

    # -----------------------------------------------------------------------
    # THRESHOLD SELECTION — VALIDATION DATA ONLY
    # -----------------------------------------------------------------------

    X_val_recon = model.predict(
        X_val_corrupted,
        verbose=0
    )

    val_errors = np.mean(
        (X_val_corrupted - X_val_recon) ** 2,
        axis=1
    )

    best_threshold = float(val_errors[0])
    best_val_f1 = -1.0

    candidate_thresholds = np.quantile(
        val_errors,
        np.linspace(0.05, 0.999, 400)
    )

    for threshold in candidate_thresholds:

        val_predictions = np.where(
            val_errors > threshold,
            -1,
            1
        )

        current_f1 = f1_score(
            y_val_local,
            val_predictions,
            pos_label=-1,
            zero_division=0,
        )

        if current_f1 > best_val_f1:
            best_val_f1 = current_f1
            best_threshold = float(threshold)

    # -----------------------------------------------------------------------
    # HELD-OUT TEST EVALUATION
    # -----------------------------------------------------------------------

    X_test_recon = model.predict(
        X_test_corrupted,
        verbose=0
    )

    test_errors = np.mean(
        (X_test_corrupted - X_test_recon) ** 2,
        axis=1
    )

    # Apply the already-fixed validation threshold.
    predictions = np.where(
        test_errors > best_threshold,
        -1,
        1
    )

    # -----------------------------------------------------------------------
    # TEST METRICS
    # -----------------------------------------------------------------------

    y_test_binary = (
        y_test_local == -1
    ).astype(int)

    if len(np.unique(y_test_binary)) > 1:
        roc_auc = float(
            roc_auc_score(
                y_test_binary,
                test_errors
            )
        )
    else:
        roc_auc = 0.5

    precision = float(
        precision_score(
            y_test_local,
            predictions,
            pos_label=-1,
            zero_division=0,
        )
    )

    recall = float(
        recall_score(
            y_test_local,
            predictions,
            pos_label=-1,
            zero_division=0,
        )
    )

    test_f1 = float(
        f1_score(
            y_test_local,
            predictions,
            pos_label=-1,
            zero_division=0,
        )
    )

    accuracy = float(
        accuracy_score(
            y_test_local,
            predictions
        )
    )

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(
        y_test_local,
        predictions,
        labels=[1, -1]
    ).ravel()

    specificity = (
        float(tn / (tn + fp))
        if (tn + fp) > 0
        else 0.0
    )

    metrics = {
        "accuracy": accuracy,
        "f1": test_f1,
        "precision": precision,
        "recall": recall,
        "roc_auc": roc_auc,
        "specificity": specificity,

        "threshold": float(best_threshold),
        "validation_f1": float(best_val_f1),

        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),

        "n_validation_anomalies": int(
            len(val_anomaly_indices)
        ),

        "n_test_anomalies": int(
            len(test_anomaly_indices)
        ),

        "epochs_run": len(
            history.history["loss"]
        ),

        "final_train_loss": float(
            history.history["loss"][-1]
        ),

        "final_val_loss": float(
            history.history["val_loss"][-1]
        ),
    }

    return metrics


# ===========================================================================
# MULTI-SEED RUNNER
# ===========================================================================

def run_variant_multiple_seeds(
    X_train: np.ndarray,
    X_val: np.ndarray,
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
    """Run one configuration across multiple random seeds."""

    if seeds is None:
        seeds = [41, 42, 43, 44, 45]

    all_runs = []

    for seed_idx, current_seed in enumerate(
        seeds,
        1
    ):

        print(
            f"    [Seed {seed_idx}/{len(seeds)}: "
            f"{current_seed}]",
            end=" ",
            flush=True
        )

        results = train_variant(
            X_train=X_train,
            X_val=X_val,
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

        print(
            f"F1={results['f1']:.4f}, "
            f"ROC-AUC={results['roc_auc']:.4f}",
            flush=True
        )

        all_runs.append(results)

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


# ===========================================================================
# ABLATION STUDY
# ===========================================================================

def run_ablation_study(
    data_path: str = "data/StudentPerformanceFactors.csv",
    results_dir: str = "results",
    output_file: str = "ablation_study_revised.json",
) -> Path:
    """Run the complete revised ablation study."""

    results_path = Path(results_dir)
    results_path.mkdir(exist_ok=True)

    output_path = results_path / output_file

    print("\n" + "=" * 80)
    print("REVISED ABLATION STUDY")
    print("Fixed 60/15/25 split + validation-based threshold")
    print("=" * 80)

    # -----------------------------------------------------------------------
    # LOAD DATA
    # -----------------------------------------------------------------------

    print("\n[1/11] Loading and preprocessing data...")

    df = load_data(data_path)

    df = prep.handle_missing(df)

    df_encoded, cat_cols = prep.encode_categorical(
        df
    )

    X, feature_names = prep.select_features(
        df_encoded
    )

    indices = np.arange(len(X))

    # -----------------------------------------------------------------------
    # FIXED 60/15/25 SPLIT
    # -----------------------------------------------------------------------

    # First reserve 25% for the final test set.
    X_trainval, X_test, idx_trainval, test_idx = train_test_split(
        X,
        indices,
        test_size=config.TEST_SIZE,
        random_state=config.SEED,
    )

    # trainval now contains 75% of the complete dataset.
    # We need 15% of the complete dataset for validation:
    #
    # 15 / 75 = 0.20
    #
    # Therefore 20% of trainval becomes validation.

    val_relative = (
        config.VAL_SIZE /
        (config.TRAIN_SIZE + config.VAL_SIZE)
    )

    X_train, X_val, train_idx, val_idx = train_test_split(
        X_trainval,
        idx_trainval,
        test_size=val_relative,
        random_state=config.SEED,
    )

    # -----------------------------------------------------------------------
    # NORMALIZATION
    # -----------------------------------------------------------------------

    # Fit scaler ONLY on training data.
    scaler, X_train_scaled = prep.normalize_features(
        pd.DataFrame(X_train)
    )

    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    X_train = np.asarray(X_train_scaled)
    X_val = np.asarray(X_val_scaled)
    X_test = np.asarray(X_test_scaled)

    print(
        f"  ✓ Training samples: {len(X_train)}"
    )

    print(
        f"  ✓ Validation samples: {len(X_val)}"
    )

    print(
        f"  ✓ Test samples: {len(X_test)}"
    )

    print(
        f"  ✓ Features: {X_train.shape[1]}"
    )

    # -----------------------------------------------------------------------
    # FIVE RANDOM SEEDS
    # -----------------------------------------------------------------------

    SEEDS = [
        41,
        42,
        43,
        44,
        45
    ]

    # -----------------------------------------------------------------------
    # BASELINE
    # -----------------------------------------------------------------------

    print(
        "\n[2/11] Training baseline model "
        f"with {len(SEEDS)} seeds..."
    )

    baseline_results = run_variant_multiple_seeds(
        X_train,
        X_val,
        X_test,

        seeds=SEEDS,

        bottleneck_dim=4,
        depth=3,
        dropout_rate=0.1,
        activation="relu",
        batch_size=16,
        epochs=80,
        learning_rate=5e-4,

        noise_scale=config.NOISE_SCALE,
        injection_frac=config.INJECTION_FRAC,
    )

    baseline_f1_values = [
        run["f1"]
        for run in baseline_results["all_runs"]
    ]

    baseline_roc_auc_values = [
        run["roc_auc"]
        for run in baseline_results["all_runs"]
    ]

    baseline_mean_f1 = np.mean(
        baseline_f1_values
    )

    baseline_mean_roc_auc = np.mean(
        baseline_roc_auc_values
    )

    baseline_std_f1 = np.std(
        baseline_f1_values
    )

    baseline_std_roc_auc = np.std(
        baseline_roc_auc_values
    )

    print(
        f"  ✓ F1: {baseline_mean_f1:.4f} "
        f"± {baseline_std_f1:.4f}"
    )

    print(
        f"  ✓ ROC-AUC: "
        f"{baseline_mean_roc_auc:.4f} "
        f"± {baseline_std_roc_auc:.4f}"
    )

    all_results = {
        "baseline": baseline_results,

        "variations": {},

        "timestamp": datetime.now().isoformat(),

        "data_summary": {
            "n_features": int(
                X_train.shape[1]
            ),

            "n_train": int(
                len(X_train)
            ),

            "n_validation": int(
                len(X_val)
            ),

            "n_test": int(
                len(X_test)
            ),
        },

        "study_config": {
            "split": "60/15/25",

            "threshold_method":
                "validation_grid_search_f1",

            "seeds": SEEDS,

            "random_state":
                int(config.SEED),

            "injection_fraction":
                float(config.INJECTION_FRAC),

            "baseline_noise_scale":
                float(config.NOISE_SCALE),
        },
    }

    # =======================================================================
    # BOTTLENECK
    # =======================================================================

    print(
        "\n[3/11] Testing bottleneck dimensions "
        "[2, 4, 8, 16, 32]..."
    )

    bottleneck_results = {}

    for bottleneck_dim in [
        2,
        4,
        8,
        16,
        32
    ]:

        print(
            f"  Testing bottleneck_dim="
            f"{bottleneck_dim}..."
        )

        results = run_variant_multiple_seeds(
            X_train,
            X_val,
            X_test,

            seeds=SEEDS,

            bottleneck_dim=bottleneck_dim,
            depth=3,
            dropout_rate=0.1,
            activation="relu",
            batch_size=16,
            epochs=80,
            learning_rate=5e-4,

            noise_scale=config.NOISE_SCALE,
            injection_frac=config.INJECTION_FRAC,
        )

        bottleneck_results[
            f"bottleneck_{bottleneck_dim}"
        ] = results

    all_results["variations"][
        "bottleneck_dimension"
    ] = bottleneck_results

    # =======================================================================
    # NETWORK DEPTH
    # =======================================================================

    print(
        "\n[4/11] Testing network depths "
        "[1, 2, 3, 4]..."
    )

    depth_results = {}

    for depth in [
        1,
        2,
        3,
        4
    ]:

        print(
            f"  Testing depth={depth}..."
        )

        results = run_variant_multiple_seeds(
            X_train,
            X_val,
            X_test,

            seeds=SEEDS,

            bottleneck_dim=4,
            depth=depth,
            dropout_rate=0.1,
            activation="relu",
            batch_size=16,
            epochs=80,
            learning_rate=5e-4,

            noise_scale=config.NOISE_SCALE,
            injection_frac=config.INJECTION_FRAC,
        )

        depth_results[
            f"depth_{depth}"
        ] = results

    all_results["variations"][
        "network_depth"
    ] = depth_results

    # =======================================================================
    # DROPOUT
    # =======================================================================

    print(
        "\n[5/11] Testing dropout rates "
        "[0.0, 0.1, 0.2, 0.3]..."
    )

    dropout_results = {}

    for dropout_rate in [
        0.0,
        0.1,
        0.2,
        0.3
    ]:

        print(
            f"  Testing dropout_rate="
            f"{dropout_rate}..."
        )

        results = run_variant_multiple_seeds(
            X_train,
            X_val,
            X_test,

            seeds=SEEDS,

            bottleneck_dim=4,
            depth=3,
            dropout_rate=dropout_rate,
            activation="relu",
            batch_size=16,
            epochs=80,
            learning_rate=5e-4,

            noise_scale=config.NOISE_SCALE,
            injection_frac=config.INJECTION_FRAC,
        )

        dropout_results[
            f"dropout_{dropout_rate:.1f}"
        ] = results

    all_results["variations"][
        "dropout_rate"
    ] = dropout_results

    # =======================================================================
    # ACTIVATION FUNCTION
    # =======================================================================

    print(
        "\n[6/11] Testing activation functions "
        "[relu, sigmoid, tanh]..."
    )

    activation_results = {}

    for activation in [
        "relu",
        "sigmoid",
        "tanh"
    ]:

        print(
            f"  Testing activation="
            f"{activation}..."
        )

        results = run_variant_multiple_seeds(
            X_train,
            X_val,
            X_test,

            seeds=SEEDS,

            bottleneck_dim=4,
            depth=3,
            dropout_rate=0.1,
            activation=activation,
            batch_size=16,
            epochs=80,
            learning_rate=5e-4,

            noise_scale=config.NOISE_SCALE,
            injection_frac=config.INJECTION_FRAC,
        )

        activation_results[
            f"activation_{activation}"
        ] = results

    all_results["variations"][
        "activation_function"
    ] = activation_results

    # =======================================================================
    # BATCH SIZE
    # =======================================================================

    print(
        "\n[7/11] Testing batch sizes "
        "[8, 16, 32]..."
    )

    batch_size_results = {}

    for batch_size in [
        8,
        16,
        32
    ]:

        print(
            f"  Testing batch_size="
            f"{batch_size}..."
        )

        results = run_variant_multiple_seeds(
            X_train,
            X_val,
            X_test,

            seeds=SEEDS,

            bottleneck_dim=4,
            depth=3,
            dropout_rate=0.1,
            activation="relu",
            batch_size=batch_size,
            epochs=80,
            learning_rate=5e-4,

            noise_scale=config.NOISE_SCALE,
            injection_frac=config.INJECTION_FRAC,
        )

        batch_size_results[
            f"batch_size_{batch_size}"
        ] = results

    all_results["variations"][
        "batch_size"
    ] = batch_size_results

    # =======================================================================
    # EPOCHS
    # =======================================================================

    print(
        "\n[8/11] Testing epoch budgets "
        "[40, 80, 120]..."
    )

    epoch_results = {}

    for epochs in [
        40,
        80,
        120
    ]:

        print(
            f"  Testing epochs={epochs}..."
        )

        results = run_variant_multiple_seeds(
            X_train,
            X_val,
            X_test,

            seeds=SEEDS,

            bottleneck_dim=4,
            depth=3,
            dropout_rate=0.1,
            activation="relu",
            batch_size=16,
            epochs=epochs,
            learning_rate=5e-4,

            noise_scale=config.NOISE_SCALE,
            injection_frac=config.INJECTION_FRAC,
        )

        epoch_results[
            f"epochs_{epochs}"
        ] = results

    all_results["variations"][
        "epochs"
    ] = epoch_results

    # =======================================================================
    # LEARNING RATE
    # =======================================================================

    print(
        "\n[9/11] Testing learning rates "
        "[1e-4, 5e-4, 1e-3]..."
    )

    lr_results = {}

    for learning_rate in [
        1e-4,
        5e-4,
        1e-3
    ]:

        print(
            f"  Testing learning_rate="
            f"{learning_rate}..."
        )

        results = run_variant_multiple_seeds(
            X_train,
            X_val,
            X_test,

            seeds=SEEDS,

            bottleneck_dim=4,
            depth=3,
            dropout_rate=0.1,
            activation="relu",
            batch_size=16,
            epochs=80,
            learning_rate=learning_rate,

            noise_scale=config.NOISE_SCALE,
            injection_frac=config.INJECTION_FRAC,
        )

        lr_results[
            f"learning_rate_"
            f"{learning_rate:.0e}"
        ] = results

    all_results["variations"][
        "learning_rate"
    ] = lr_results

    # =======================================================================
    # NOISE-SEVERITY SENSITIVITY ANALYSIS
    # =======================================================================

    print(
        "\n[10/11] Testing anomaly severity "
        "noise scales [2, 4, 6, 8, 10]..."
    )

    noise_results = {}

    for noise_scale in [
        2,
        4,
        6,
        8,
        10
    ]:

        print(
            f"  Testing noise_scale="
            f"{noise_scale}..."
        )

        results = run_variant_multiple_seeds(
            X_train,
            X_val,
            X_test,

            seeds=SEEDS,

            bottleneck_dim=4,
            depth=3,
            dropout_rate=0.1,
            activation="relu",
            batch_size=16,
            epochs=80,
            learning_rate=5e-4,

            noise_scale=float(
                noise_scale
            ),

            # Fixed prevalence.
            injection_frac=config.INJECTION_FRAC,
        )

        noise_results[
            f"noise_scale_{noise_scale}"
        ] = results

    all_results["variations"][
        "noise_scale_sensitivity"
    ] = noise_results

    # =======================================================================
    # SAVE RESULTS
    # =======================================================================

    print(
        "\n[11/11] Saving results..."
    )

    with open(
        output_path,
        "w"
    ) as f:

        json.dump(
            all_results,
            f,
            indent=2
        )

    print(
        f"\n✓ Results saved to "
        f"{output_path}"
    )

    # -----------------------------------------------------------------------
    # SUMMARY
    # -----------------------------------------------------------------------

    print(
        "\n" + "=" * 80
    )

    print(
        f"SUMMARY — Mean ± Std Dev "
        f"across {len(SEEDS)} seeds"
    )

    print(
        "=" * 80
    )

    print(
        f"\nBaseline F1: "
        f"{baseline_mean_f1:.4f} "
        f"± {baseline_std_f1:.4f}"
    )

    print(
        f"Baseline ROC-AUC: "
        f"{baseline_mean_roc_auc:.4f} "
        f"± {baseline_std_roc_auc:.4f}"
    )

    print(
        "\nBest variant by mean F1 "
        "in each architectural/training category:"
    )

    # Do NOT treat noise severity as an optimized model parameter.
    for category, variants in all_results[
        "variations"
    ].items():

        if category == "noise_scale_sensitivity":
            continue

        best_key = max(
            variants.keys(),

            key=lambda key: np.mean([
                run["f1"]
                for run
                in variants[key]["all_runs"]
            ])
        )

        best_f1_values = [
            run["f1"]
            for run
            in variants[best_key]["all_runs"]
        ]

        best_f1_mean = np.mean(
            best_f1_values
        )

        best_f1_std = np.std(
            best_f1_values
        )

        print(
            f"  {category:25} "
            f"{best_key:25} "
            f"F1={best_f1_mean:.4f} "
            f"± {best_f1_std:.4f}"
        )

    # Separate severity results
    print(
        "\nNoise-scale sensitivity "
        "(anomaly severity; NOT model optimization):"
    )

    for key, results in noise_results.items():

        f1_values = [
            run["f1"]
            for run
            in results["all_runs"]
        ]

        roc_values = [
            run["roc_auc"]
            for run
            in results["all_runs"]
        ]

        print(
            f"  {key:20} "
            f"F1={np.mean(f1_values):.4f} "
            f"± {np.std(f1_values):.4f}, "
            f"ROC-AUC={np.mean(roc_values):.4f} "
            f"± {np.std(roc_values):.4f}"
        )

    print(
        "\n✓ Ablation study complete."
    )

    return output_path


# ===========================================================================
# MAIN
# ===========================================================================

if __name__ == "__main__":
    run_ablation_study()