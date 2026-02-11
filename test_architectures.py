"""
Test different autoencoder architecture layer configurations.

Compares multiple encoder-decoder depth/width combinations:
- shallow_2: 64→32→bottleneck→32→64
- baseline_3: 128→64→32→bottleneck→32→64→128 (current best)
- deep_4: 256→128→64→32→bottleneck→32→64→128→256
- moderate_4: 128→64→32→16→bottleneck→16→32→64→128
- narrow_3: 64→32→16→bottleneck→16→32→64
- wide_3: 256→192→128→bottleneck→128→192→256

Tests 6 architectures × 6 bottleneck dimensions (2, 4, 6, 8, 10, 12) = 36 total configurations
Output: architecture_tuning_results.json with detailed comparison
"""
import os
os.environ["PYTHONHASHSEED"] = "42"
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import subprocess
import json
import re
from pathlib import Path
import sys

# Define architecture configurations to test
architectures = {
    "shallow_2": {
        "name": "64-32-bottleneck-32-64",
        "encoder_sizes": [64, 32],
        "decoder_sizes": [32, 64],
    },
    "baseline_3": {
        "name": "128-64-32-bottleneck-32-64-128",
        "encoder_sizes": [128, 64, 32],
        "decoder_sizes": [32, 64, 128],
    },
    "deep_4": {
        "name": "256-128-64-32-bottleneck-32-64-128-256",
        "encoder_sizes": [256, 128, 64, 32],
        "decoder_sizes": [32, 64, 128, 256],
    },
    "moderate_4": {
        "name": "128-64-32-16-bottleneck-16-32-64-128",
        "encoder_sizes": [128, 64, 32, 16],
        "decoder_sizes": [16, 32, 64, 128],
    },
    "narrow_3": {
        "name": "64-32-16-bottleneck-16-32-64",
        "encoder_sizes": [64, 32, 16],
        "decoder_sizes": [16, 32, 64],
    },
    "wide_3": {
        "name": "256-192-128-bottleneck-128-192-256",
        "encoder_sizes": [256, 192, 128],
        "decoder_sizes": [128, 192, 256],
    },
}

# Bottleneck dimensions to test
bottleneck_dims = [2, 4, 6, 8, 10, 12]

# Store results
results = {}

print("=" * 80)
print("AUTOENCODER ARCHITECTURE & BOTTLENECK TUNING")
print("=" * 80)
print(f"\nTesting {len(architectures)} architectures × {len(bottleneck_dims)} bottleneck dimensions")
print(f"Total configurations: {len(architectures) * len(bottleneck_dims)}")
print("=" * 80 + "\n")

# Get original model code and config
model_path = Path("src/autoencoder_model.py")
config_path = Path("src/config.py")
original_model_content = model_path.read_text()
original_config_content = config_path.read_text()

try:
    for arch_key, arch_config in architectures.items():
        results[arch_key] = {}
        
        for bottleneck_dim in bottleneck_dims:
            print(f"{arch_config['name']} (bottleneck={bottleneck_dim}):", end="")
            sys.stdout.flush()
            
            # Update config with current bottleneck dimension
            config_content = original_config_content.replace(
                "BOTTLENECK_DIM = 4",
                f"BOTTLENECK_DIM = {bottleneck_dim}"
            )
            config_path.write_text(config_content)
            
            # Generate model code for this configuration
            encoder_sizes_str = str(arch_config['encoder_sizes'])
            decoder_sizes_str = str(arch_config['decoder_sizes'])
            
            model_code = f'''import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

def build_autoencoder(input_dim: int, bottleneck_dim: int = {bottleneck_dim}, num_dropout_layers: int = 2) -> keras.Model:
    """
    Autoencoder with architecture: {arch_config['name']}
    Bottleneck dimension: {bottleneck_dim}
    """
    inp = keras.Input(shape=(input_dim,))
    
    # Encoder
    x = inp
    encoder_sizes = {encoder_sizes_str}
    for i, size in enumerate(encoder_sizes):
        x = layers.Dense(size, activation="relu")(x)
    
    bottleneck = layers.Dense(bottleneck_dim, activation="relu")(x)
    
    # Dropout after bottleneck
    if num_dropout_layers >= 1:
        x = layers.Dropout(0.1)(bottleneck)
    else:
        x = bottleneck
    
    # Decoder
    decoder_sizes = {decoder_sizes_str}
    for i, size in enumerate(decoder_sizes):
        x = layers.Dense(size, activation="relu")(x)
        if num_dropout_layers >= 2 and i == 0:  # One dropout in decoder
            x = layers.Dropout(0.1)(x)
    
    out = layers.Dense(input_dim, activation="linear")(x)
    
    model = keras.Model(inputs=inp, outputs=out)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=5e-4), loss="mse")
    return model

def train_detector(
    X_train: np.ndarray,
    contamination: float = 0.02,
    bottleneck_dim: int = {bottleneck_dim},
    num_dropout_layers: int = 2,
    epochs: int = 80,
    batch_size: int = 16,
    val_split: float = 0.2,
    random_state: int = 42,
):
    np.random.seed(random_state)
    keras.utils.set_random_seed(random_state)

    input_dim = X_train.shape[1]
    model = build_autoencoder(input_dim=input_dim, bottleneck_dim=bottleneck_dim, num_dropout_layers=num_dropout_layers)

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=20,
            restore_best_weights=True,
            min_delta=1e-5
        )
    ]

    model.fit(
        X_train,
        X_train,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        validation_split=val_split,
        callbacks=callbacks,
        verbose=0,
    )

    X_hat = model.predict(X_train, verbose=0)
    train_errors = np.mean((X_train - X_hat) ** 2, axis=1)

    q = max(0.0, min(1.0, 1.0 - contamination))
    threshold = float(np.quantile(train_errors, q))

    return model, threshold

def score_anomalies(model: keras.Model, X: np.ndarray) -> np.ndarray:
    X_hat = model.predict(X, verbose=0)
    errors = np.mean((X - X_hat) ** 2, axis=1)
    return errors

def predict_anomalies(model: keras.Model, X: np.ndarray, threshold: float) -> np.ndarray:
    errors = score_anomalies(model, X)
    preds = np.where(errors > threshold, -1, 1)
    return preds
'''
            
            # Write temporary model file
            model_path.write_text(model_code)
            
            # Run pipeline and capture output
            try:
                result = subprocess.run(
                    [sys.executable, "scripts/run_pipeline.py"],
                    capture_output=True,
                    text=True,
                    timeout=300
                )
                
                output = result.stdout + result.stderr
                
                # Parse metrics
                precision_match = re.search(r'Precision:\s*([\d.]+)', output)
                recall_match = re.search(r'Recall:\s*([\d.]+)', output)
                f1_match = re.search(r'F1:\s*([\d.]+)', output)
                roc_match = re.search(r'ROC-AUC:\s*([\d.]+)', output)
                tp_match = re.search(r'TP/FP/FN:\s*(\d+)/(\d+)/(\d+)', output)
                
                if precision_match and recall_match and f1_match:
                    metrics = {
                        'precision': float(precision_match.group(1)),
                        'recall': float(recall_match.group(1)),
                        'f1': float(f1_match.group(1)),
                        'fp': int(tp_match.group(2)) if tp_match else 0,
                    }
                    if roc_match:
                        metrics['roc_auc'] = float(roc_match.group(1))
                    
                    results[arch_key][bottleneck_dim] = metrics
                    print(f" F1={metrics['f1']:.4f} | Prec={metrics['precision']:.4f} | Rec={metrics['recall']:.4f} | FP={metrics['fp']}")
                else:
                    print(f" [ERROR] Failed to parse metrics")
                    print(f"  Output: {output[:150]}")
                    results[arch_key][bottleneck_dim] = {"error": "parse_failed"}
                    
            except subprocess.TimeoutExpired:
                print(f" [TIMEOUT]")
                results[arch_key][bottleneck_dim] = {"error": "timeout"}
            except Exception as e:
                print(f" [ERROR] {str(e)[:50]}")
                results[arch_key][bottleneck_dim] = {"error": str(e)}

finally:
    # Restore original files
    model_path.write_text(original_model_content)
    config_path.write_text(original_config_content)

# Save results
output_file = Path("architecture_tuning_results.json")
with open(output_file, "w") as f:
    json.dump(results, f, indent=2)

print("\n" + "=" * 80)
print("RESULTS SUMMARY")
print("=" * 80)

# Print summary statistics
for arch_key in results:
    f1_scores = [results[arch_key][d].get('f1', 0) for d in bottleneck_dims if isinstance(results[arch_key][d], dict) and 'f1' in results[arch_key][d]]
    if f1_scores:
        avg_f1 = sum(f1_scores) / len(f1_scores)
        max_f1 = max(f1_scores)
        best_dim = [d for d in bottleneck_dims if isinstance(results[arch_key][d], dict) and results[arch_key][d].get('f1', 0) == max_f1][0]
        print(f"{arch_key:15} | Avg F1={avg_f1:.4f} | Best F1={max_f1:.4f} (dim={best_dim})")

print(f"\nResults saved to {output_file}")
