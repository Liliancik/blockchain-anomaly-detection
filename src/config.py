# src/config.py
# Core constants for anomaly detection pipeline

SEED = 42

# Data split
TRAIN_SIZE = 0.75
TEST_SIZE = 0.25

# Anomaly injection (constant)
INJECTION_FRAC = 0.02
NOISE_SCALE = 5.0

# Autoencoder
BOTTLENECK_DIM = 4
EPOCHS = 80
BATCH_SIZE = 16

# Blockchain
LEDGER_PATH = "blockchain_ledger.jsonl"
