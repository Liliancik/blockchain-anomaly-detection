"""Configuration constants for anomaly detection pipeline."""

SEED = 41  # Random seed for reproducibility

TRAIN_SIZE = 0.75
TEST_SIZE = 0.25

INJECTION_FRAC = 0.02  # Synthetic anomalies for evaluation
NOISE_SCALE = 6  # Noise scale in std deviations (lower = more realistic anomalies)

BOTTLENECK_DIM = 4  # Compression dimension
EPOCHS = 80
BATCH_SIZE = 16

# Isolation Forest parameters
IF_N_ESTIMATORS = 35  # Number of isolation trees
IF_MAX_SAMPLES = "auto"  # Number of samples to draw for training each tree
IF_CONTAMINATION = 0.02  # Match injection rate for IF to learn the pattern

LEDGER_PATH = "blockchain_ledger.jsonl"  # Blockchain storage

# Threshold Optimization
OPTIMIZE_THRESHOLD = True
THRESHOLD_METRIC = "f1"  # "f1" (best balance), "precision" (fewer false positives), "recall" (fewer false negatives)
