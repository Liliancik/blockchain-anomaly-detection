"""Run end-to-end pipeline: load data, preprocess, inject anomalies, detect, log to ledger."""
import argparse
import json
from pathlib import Path

from src.data_loader import load_data
from src import preprocessing as prep
from src.anomaly_injection import inject_anomalies
from src.autoencoder_model import train_detector, score_anomalies, predict_anomalies
from src.blockchain_logger import log_anomaly

import numpy as np


def run(path: str, injection_method: str = "noise", frac: float = 0.01, push_outputs: bool = False):
    df = load_data(path)
    df = prep.handle_missing(df)
    df_encoded, _ = prep.encode_categorical(df)
    X_df, _ = prep.select_features(df_encoded)
    scaler, X_scaled = prep.normalize_features(X_df)

    # Inject anomalies
    df_injected, anomaly_indices = inject_anomalies(X_df, method=injection_method, frac=frac)

    # Normalize injected
    _, X_injected_scaled = prep.normalize_features(df_injected)

    # Train detector on injected data (for demonstration we train on full set)
    clf = train_detector(X_injected_scaled)
    scores = score_anomalies(clf, X_injected_scaled)
    preds = predict_anomalies(clf, X_injected_scaled)

    anomalies = [int(i) for i, p in enumerate(preds) if p == -1]

    print(f"Injected anomaly indices (ground truth): {anomaly_indices}")
    print(f"Detected anomalies by model: {anomalies}")

    # Log detected anomalies to the ledger
    for idx in anomalies:
        entry = log_anomaly(int(idx), {"detected_by": "isolation_forest", "score": float(scores[idx])})
        print(f"Logged anomaly {idx} -> hash: {entry['hash']}")

    # Optionally save outputs
    if push_outputs:
        out_path = Path("pipeline_outputs.json")
        out = {
            "injected": anomaly_indices,
            "detected": anomalies,
            "scores_summary": {
                "min": float(np.min(scores)),
                "max": float(np.max(scores)),
                "mean": float(np.mean(scores)),
            }
        }
        out_path.write_text(json.dumps(out, indent=2))
        print(f"Saved outputs to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/StudentPerformanceFactors.csv", help="Path to dataset CSV")
    parser.add_argument("--method", default="noise", choices=["noise", "swap"])
    parser.add_argument("--frac", type=float, default=0.01)
    parser.add_argument("--push-outputs", action="store_true")
    args = parser.parse_args()
    run(args.data, args.method, args.frac, args.push_outputs)
