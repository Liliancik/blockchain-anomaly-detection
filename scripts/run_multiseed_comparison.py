"""Run the model comparison using different random seeds.

The models are compared using the same train, validation and test
splits. PCA is also included as a simple reconstruction baseline.
"""

import os
import sys
import json
import numpy as np
import pandas as pd

from scipy import stats

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    confusion_matrix,
)


# Allow imports from src
sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            ".."
        )
    )
)


from src import config
from src.data_loader import load_data
from src import preprocessing as prep
from src.anomaly_injection import inject_anomalies

from src.autoencoder_model import (
    train_detector,
    score_anomalies,
)

from src.vae_baseline import (
    train_vae,
    score_anomalies_vae,
)

from src.isolation_forest_baseline import (
    train_isolation_forest,
    score_anomalies_if,
    optimize_threshold_if,
)

from src.pca_baseline import (
    train_pca,
    score_anomalies_pca,
)


# Start with one seed to test PCA
SEEDS = [41, 42, 43, 44, 45]


def find_best_threshold(
    scores,
    y_true,
    q_low=0.05,
    q_high=0.999,
    n_grid=400
):
    """Find the threshold with the best F1 score on validation data."""

    qs = np.linspace(
        q_low,
        q_high,
        n_grid
    )

    thresholds = np.quantile(
        scores,
        qs
    )

    best_f1 = 0.0
    best_threshold = thresholds[0]

    for threshold in thresholds:

        predictions = np.where(
            scores > threshold,
            -1,
            1
        )

        # Skip if no anomalies are predicted
        if np.sum(predictions == -1) == 0:
            continue

        f1 = f1_score(
            y_true,
            predictions,
            pos_label=-1,
            zero_division=0
        )

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    return float(best_threshold)


def calculate_metrics(
    y_true,
    predictions,
    scores
):
    """Calculate the evaluation metrics."""

    accuracy = accuracy_score(
        y_true,
        predictions
    )

    precision = precision_score(
        y_true,
        predictions,
        pos_label=-1,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        predictions,
        pos_label=-1,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        predictions,
        pos_label=-1,
        zero_division=0
    )

    # Convert anomaly labels to 1 for ROC-AUC and PR-AUC
    y_binary = (
        np.array(y_true) == -1
    ).astype(int)

    roc_auc = roc_auc_score(
        y_binary,
        scores
    )

    pr_auc = average_precision_score(
        y_binary,
        scores
    )

    balanced_accuracy = balanced_accuracy_score(
        y_true,
        predictions
    )

    mcc = matthews_corrcoef(
        y_true,
        predictions
    )

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[1, -1]
    ).ravel()

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0
    )

    fpr = (
        fp / (fp + tn)
        if (fp + tn) > 0
        else 0
    )

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "balanced_accuracy": float(balanced_accuracy),
        "mcc": float(mcc),
        "specificity": float(specificity),
        "fpr": float(fpr),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def run_experiment(
    X_df,
    seed
):
    """Run one experiment using one random seed."""

    print("\n" + "=" * 70)
    print(f"RUNNING SEED {seed}")
    print("=" * 70)

    # ---------------------------------------------------------
    # Split data into train, validation and test
    # ---------------------------------------------------------

    X_train_val, X_test = train_test_split(
        X_df,
        test_size=config.TEST_SIZE,
        random_state=seed
    )

    val_size = (
        config.VAL_SIZE
        / (
            config.TRAIN_SIZE
            + config.VAL_SIZE
        )
    )

    X_train, X_val = train_test_split(
        X_train_val,
        test_size=val_size,
        random_state=seed
    )

    print(f"Train:      {len(X_train)}")
    print(f"Validation: {len(X_val)}")
    print(f"Test:       {len(X_test)}")

    # ---------------------------------------------------------
    # Normalize data
    # Scaler is fitted only on training data
    # ---------------------------------------------------------

    scaler, X_train_scaled = (
        prep.normalize_features(
            X_train
        )
    )

    X_val_scaled = scaler.transform(
        X_val
    )

    X_test_scaled = scaler.transform(
        X_test
    )

    # ---------------------------------------------------------
    # Add anomalies to validation data
    # ---------------------------------------------------------

    X_val_df = pd.DataFrame(
        X_val_scaled
    )

    X_val_corrupted_df, val_anomaly_indices = (
        inject_anomalies(
            X_val_df.copy(),
            method="noise",
            frac=config.INJECTION_FRAC,
            scale=config.NOISE_SCALE,
            random_state=seed
        )
    )

    X_val_corrupted = (
        X_val_corrupted_df.values
    )

    y_val = np.ones(
        len(X_val_corrupted)
    )

    y_val[
        val_anomaly_indices
    ] = -1

    # ---------------------------------------------------------
    # Add anomalies to test data
    # Different seed keeps test corruption independent
    # ---------------------------------------------------------

    X_test_df = pd.DataFrame(
        X_test_scaled
    )

    X_test_corrupted_df, test_anomaly_indices = (
        inject_anomalies(
            X_test_df.copy(),
            method="noise",
            frac=config.INJECTION_FRAC,
            scale=config.NOISE_SCALE,
            random_state=seed + 1
        )
    )

    X_test_corrupted = (
        X_test_corrupted_df.values
    )

    y_test = np.ones(
        len(X_test_corrupted)
    )

    y_test[
        test_anomaly_indices
    ] = -1

    print(
        f"Validation anomalies: "
        f"{len(val_anomaly_indices)}"
    )

    print(
        f"Test anomalies:       "
        f"{len(test_anomaly_indices)}"
    )

    # =========================================================
    # AUTOENCODER
    # =========================================================

    print("\nTraining Autoencoder...")

    autoencoder, _, _ = train_detector(
        X_train_scaled,
        X_val=X_val_scaled,
        contamination=config.INJECTION_FRAC,
        bottleneck_dim=config.BOTTLENECK_DIM,
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        random_state=seed
    )

    ae_val_scores = score_anomalies(
        autoencoder,
        X_val_corrupted
    )

    ae_threshold = find_best_threshold(
        ae_val_scores,
        y_val
    )

    print(
        f"AE validation threshold: "
        f"{ae_threshold:.5f}"
    )

    ae_test_scores = score_anomalies(
        autoencoder,
        X_test_corrupted
    )

    ae_predictions = np.where(
        ae_test_scores > ae_threshold,
        -1,
        1
    )

    ae_results = calculate_metrics(
        y_test,
        ae_predictions,
        ae_test_scores
    )

    # =========================================================
    # VAE
    # =========================================================

    print("Training VAE...")

    vae, _ = train_vae(
        X_train_scaled,
        X_val=X_val_scaled,
        contamination=config.INJECTION_FRAC,
        latent_dim=config.BOTTLENECK_DIM,
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        random_state=seed
    )

    vae_val_scores = score_anomalies_vae(
        vae,
        X_val_corrupted
    )

    vae_threshold = find_best_threshold(
        vae_val_scores,
        y_val
    )

    print(
        f"VAE validation threshold: "
        f"{vae_threshold:.5f}"
    )

    vae_test_scores = score_anomalies_vae(
        vae,
        X_test_corrupted
    )

    vae_predictions = np.where(
        vae_test_scores > vae_threshold,
        -1,
        1
    )

    vae_results = calculate_metrics(
        y_test,
        vae_predictions,
        vae_test_scores
    )

    # =========================================================
    # ISOLATION FOREST
    # =========================================================

    print("Training Isolation Forest...")

    isolation_forest = train_isolation_forest(
        X_train_scaled,
        contamination=config.INJECTION_FRAC,
        n_estimators=200,
        random_state=seed
    )

    if_threshold_result = optimize_threshold_if(
        isolation_forest,
        X_val_corrupted,
        y_val
    )

    if_threshold = (
        if_threshold_result["threshold"]
    )

    if_test_scores = score_anomalies_if(
        isolation_forest,
        X_test_corrupted
    )

    if_predictions = np.where(
        if_test_scores > if_threshold,
        -1,
        1
    )

    if_results = calculate_metrics(
        y_test,
        if_predictions,
        if_test_scores
    )

    # =========================================================
    # PCA
    # =========================================================

    print("Training PCA...")

    pca = train_pca(
        X_train_scaled,
        n_components=config.BOTTLENECK_DIM
    )

    # PCA threshold is also selected using validation data
    pca_val_scores = score_anomalies_pca(
        pca,
        X_val_corrupted
    )

    pca_threshold = find_best_threshold(
        pca_val_scores,
        y_val
    )

    print(
        f"PCA validation threshold: "
        f"{pca_threshold:.5f}"
    )

    pca_test_scores = score_anomalies_pca(
        pca,
        X_test_corrupted
    )

    pca_predictions = np.where(
        pca_test_scores > pca_threshold,
        -1,
        1
    )

    pca_results = calculate_metrics(
        y_test,
        pca_predictions,
        pca_test_scores
    )

    # =========================================================
    # PRINT RESULTS
    # =========================================================

    print("\nResults:")

    print(
        f"AE  - F1: {ae_results['f1']:.4f}, "
        f"ROC-AUC: {ae_results['roc_auc']:.4f}"
    )

    print(
        f"VAE - F1: {vae_results['f1']:.4f}, "
        f"ROC-AUC: {vae_results['roc_auc']:.4f}"
    )

    print(
        f"IF  - F1: {if_results['f1']:.4f}, "
        f"ROC-AUC: {if_results['roc_auc']:.4f}"
    )

    print(
        f"PCA - F1: {pca_results['f1']:.4f}, "
        f"ROC-AUC: {pca_results['roc_auc']:.4f}"
    )

    return {
        "seed": seed,
        "Autoencoder": ae_results,
        "VAE": vae_results,
        "Isolation Forest": if_results,
        "PCA": pca_results,
    }


def calculate_summary(results):
    """Calculate mean, standard deviation and confidence intervals."""

    models = [
        "Autoencoder",
        "VAE",
        "Isolation Forest",
        "PCA",
    ]

    metrics = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "pr_auc",
        "balanced_accuracy",
        "mcc",
        "specificity",
        "fpr",
    ]

    summary = []

    for model in models:

        for metric in metrics:

            values = [
                result[model][metric]
                for result in results
            ]

            values = np.array(
                values
            )

            mean = np.mean(
                values
            )

            # One seed is used while testing PCA
            if len(values) > 1:

                sd = np.std(
                    values,
                    ddof=1
                )

                sem = stats.sem(
                    values
                )

                ci = stats.t.interval(
                    0.95,
                    len(values) - 1,
                    loc=mean,
                    scale=sem
                )

                # Metrics are between 0 and 1
                ci_lower = max(
                    0.0,
                    ci[0]
                )

                ci_upper = min(
                    1.0,
                    ci[1]
                )

            else:

                sd = 0.0
                ci_lower = mean
                ci_upper = mean

            summary.append({
                "model": model,
                "metric": metric,
                "mean": float(mean),
                "sd": float(sd),
                "ci_lower": float(ci_lower),
                "ci_upper": float(ci_upper),
            })

    return summary


def main():

    print("=" * 70)
    print("MULTI-SEED MODEL COMPARISON")
    print("=" * 70)

    print(
        f"Seeds: {SEEDS}"
    )

    # ---------------------------------------------------------
    # Load dataset
    # ---------------------------------------------------------

    df = load_data(
        "data/StudentPerformanceFactors.csv"
    )

    df = prep.handle_missing(
        df
    )

    df_encoded, _ = (
        prep.encode_categorical(
            df
        )
    )

    X_df, feature_names = (
        prep.select_features(
            df_encoded
        )
    )

    print(
        f"Dataset: {X_df.shape[0]} samples, "
        f"{X_df.shape[1]} features"
    )

    # ---------------------------------------------------------
    # Run experiments
    # ---------------------------------------------------------

    all_results = []

    for seed in SEEDS:

        result = run_experiment(
            X_df,
            seed
        )

        all_results.append(
            result
        )

    # ---------------------------------------------------------
    # Calculate summary
    # ---------------------------------------------------------

    summary = calculate_summary(
        all_results
    )

    # ---------------------------------------------------------
    # Save results
    # ---------------------------------------------------------

    os.makedirs(
        "results",
        exist_ok=True
    )

    models = [
        "Autoencoder",
        "VAE",
        "Isolation Forest",
        "PCA",
    ]

    rows = []

    for result in all_results:

        for model in models:

            row = {
                "seed": result["seed"],
                "model": model,
            }

            row.update(
                result[model]
            )

            rows.append(
                row
            )

    individual_df = pd.DataFrame(
        rows
    )

    individual_df.to_csv(
        "results/multiseed_individual_results.csv",
        index=False
    )

    summary_df = pd.DataFrame(
        summary
    )

    summary_df.to_csv(
        "results/multiseed_summary.csv",
        index=False
    )

    with open(
        "results/multiseed_results.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            {
                "seeds": SEEDS,
                "individual_results": all_results,
                "summary": summary,
            },
            file,
            indent=2
        )

    # ---------------------------------------------------------
    # Print final results
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("FINAL MULTI-SEED RESULTS")
    print("=" * 70)

    for model in models:

        print(f"\n{model}")
        print("-" * 50)

        model_summary = [
            row
            for row in summary
            if row["model"] == model
        ]

        for row in model_summary:

            print(
                f"{row['metric']:12s}: "
                f"{row['mean']:.4f} "
                f"± {row['sd']:.4f} "
                f"(95% CI: "
                f"{row['ci_lower']:.4f} - "
                f"{row['ci_upper']:.4f})"
            )

    print(
        "\nResults saved in the results folder."
    )


if __name__ == "__main__":
    main()