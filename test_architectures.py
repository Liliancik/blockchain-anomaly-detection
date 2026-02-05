"""
Test different autoencoder architecture layer sizes to find optimal configuration
Current: 128→64→32→bottleneck→32→64→128
Test alternatives like:
- 64→32→16→bottleneck→16→32→64
- 256→128→64→bottleneck→64→128→256
- 128→64→bottleneck→64→128
etc.
"""
import subprocess
import json
import re
from pathlib import Path
import tempfile
import shutil

# Define architecture configurations to test
architectures = {
    "shallow_2": {
        "name": "64-32-bottleneck-32-64",
        "encoder_sizes": [64, 32],
        "decoder_sizes": [32, 64],
        "description": "Shallow (2 hidden layers each side)"
    },
    "baseline_3": {
        "name": "128-64-32-bottleneck-32-64-128",
        "encoder_sizes": [128, 64, 32],
        "decoder_sizes": [32, 64, 128],
        "description": "Baseline (3 hidden layers each side)"
    },
    "deep_4": {
        "name": "256-128-64-32-bottleneck-32-64-128-256",
        "encoder_sizes": [256, 128, 64, 32],
        "decoder_sizes": [32, 64, 128, 256],
        "description": "Deep (4 hidden layers each side)"
    },
    "moderate_4": {
        "name": "128-64-32-16-bottleneck-16-32-64-128",
        "encoder_sizes": [128, 64, 32, 16],
        "decoder_sizes": [16, 32, 64, 128],
        "description": "Moderate (4 hidden layers, smaller sizes)"
    },
    "narrow_3": {
        "name": "64-32-16-bottleneck-16-32-64",
        "encoder_sizes": [64, 32, 16],
        "decoder_sizes": [16, 32, 64],
        "description": "Narrow (3 layers, smaller sizes)"
    },
    "wide_3": {
        "name": "256-192-128-bottleneck-128-192-256",
        "encoder_sizes": [256, 192, 128],
        "decoder_sizes": [128, 192, 256],
        "description": "Wide (3 layers, larger sizes)"
    },
}

results = {}

print("="*80)
print("AUTOENCODER ARCHITECTURE TUNING (with bottleneck_dim=4)")
print("="*80)

for arch_key, arch_config in architectures.items():
    print(f"\n{'='*80}")
    print(f"Testing: {arch_config['name']}")
    print(f"Description: {arch_config['description']}")
    print('='*80)
    
    model_code = f'''import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

def build_autoencoder(input_dim: int, bottleneck_dim: int = 4, num_dropout_layers: int = 2) -> keras.Model:
    """
    Autoencoder with architecture: {arch_config['name']}
    """
    inp = keras.Input(shape=(input_dim,))
    
    # Encoder
    x = inp
    encoder_sizes = {arch_config['encoder_sizes']}
    for i, size in enumerate(encoder_sizes):
        x = layers.Dense(size, activation="relu")(x)
    
    bottleneck = layers.Dense(bottleneck_dim, activation="relu")(x)
    
    # Dropout after bottleneck
    if num_dropout_layers >= 1:
        x = layers.Dropout(0.1)(bottleneck)
    else:
        x = bottleneck
    
    # Decoder
    decoder_sizes = {arch_config['decoder_sizes']}
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
    bottleneck_dim: int = 4,
    num_dropout_layers: int = 2,
    epochs: int = 150,
    batch_size: int = 42,
    val_split: float = 0.15,
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
    model_path = Path("src/autoencoder_model.py")
    original_content = model_path.read_text()
    
    try:
        model_path.write_text(model_code)
        
        # Run pipeline
        result = subprocess.run(
            [".venv/Scripts/python.exe", "scripts/run_pipeline.py"],
            capture_output=True,
            text=True,
            timeout=300
        )
        
        output = result.stdout + result.stderr
        
        # Extract metrics
        metrics = {}
        try:
            if "Precision:" in output:
                metrics['test_precision'] = float(re.search(r'Precision: ([\d.]+)', output).group(1))
                metrics['test_recall'] = float(re.search(r'Recall: ([\d.]+)', output).group(1))
                metrics['test_f1'] = float(re.search(r'F1: ([\d.]+)', output).group(1))
                metrics['test_accuracy'] = float(re.search(r'Autoencoder accuracy: ([\d.]+)', output).group(1))
                
                detected = re.search(r'Detected: (\d+)', output)
                injected = re.search(r'Injected: (\d+)', output)
                true_pos = re.search(r'True positives: (\d+)', output)
                if detected and injected:
                    metrics['detected'] = int(detected.group(1))
                    metrics['injected'] = int(injected.group(1))
                    metrics['false_positives'] = metrics['detected'] - int(true_pos.group(1)) if true_pos else 0
            
            results[arch_key] = {
                "architecture": arch_config['name'],
                "metrics": metrics
            }
            print(f"✓ Results: {metrics}")
        except Exception as e:
            print(f"✗ Error parsing output: {e}")
            results[arch_key] = {
                "architecture": arch_config['name'],
                "metrics": {"error": str(e)}
            }
    
    finally:
        # Restore original model file
        model_path.write_text(original_content)

print(f"\n\n{'='*80}")
print("ARCHITECTURE COMPARISON")
print('='*80)

# Print comparison table
print(f"\n{'Architecture':<40} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Accuracy':<12} {'False+':<10}")
print('-' * 98)
for arch_key in sorted(architectures.keys()):
    if arch_key in results:
        arch_name = architectures[arch_key]['name']
        m = results[arch_key]['metrics']
        if 'error' not in m:
            fp = m.get('false_positives', 0)
            print(f"{arch_name:<40} {m.get('test_precision', 0):<12.4f} {m.get('test_recall', 0):<12.4f} {m.get('test_f1', 0):<12.4f} {m.get('test_accuracy', 0):<12.4f} {fp:<10}")

# Save results
with open("architecture_tuning_results.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\n\nDetailed results saved to architecture_tuning_results.json")

# Determine optimal
print("\n" + "="*80)
print("ANALYSIS & RECOMMENDATION")
print("="*80)
valid_results = {k: v for k, v in results.items() if 'error' not in v['metrics']}

if valid_results:
    best_f1_arch = max(valid_results.items(), key=lambda x: x[1]['metrics'].get('test_f1', 0))
    best_acc_arch = max(valid_results.items(), key=lambda x: x[1]['metrics'].get('test_accuracy', 0))
    best_fp_arch = min(valid_results.items(), key=lambda x: x[1]['metrics'].get('false_positives', 999))
    
    print(f"\nBest F1: {best_f1_arch[1]['architecture']}")
    print(f"  F1={best_f1_arch[1]['metrics'].get('test_f1', 0):.4f}, Accuracy={best_f1_arch[1]['metrics'].get('test_accuracy', 0):.4f}")
    
    print(f"\nBest Accuracy: {best_acc_arch[1]['architecture']}")
    print(f"  Accuracy={best_acc_arch[1]['metrics'].get('test_accuracy', 0):.4f}, F1={best_acc_arch[1]['metrics'].get('test_f1', 0):.4f}")
    
    print(f"\nFewest False Positives: {best_fp_arch[1]['architecture']}")
    print(f"  FP={best_fp_arch[1]['metrics'].get('false_positives', 0)}, Accuracy={best_fp_arch[1]['metrics'].get('test_accuracy', 0):.4f}")
