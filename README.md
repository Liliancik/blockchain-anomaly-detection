# blockchain-anomaly-detection

## Overview

A clean, production-ready **anomaly detection pipeline** combining a deep autoencoder with blockchain logging for auditable anomaly detection on tabular data.

**Key Results** (75/25 split, Bottleneck Dim=4):
- Precision: 54.4% | Recall: 93.9% | F1: 0.689 | ROC-AUC: 1.0

## What It Does

1. **Trains a deep autoencoder** on clean training data
2. **Injects synthetic anomalies** (Gaussian noise, 2% of test set)
3. **Detects anomalies** via reconstruction error + threshold
4. **Logs detections** to an immutable, auditable blockchain ledger

## Quick Start

### 1. Setup
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements/requirements.txt
```

### 2. Run Pipeline
```powershell
python scripts/run_pipeline.py
# Output: pipeline_outputs.json + blockchain_ledger.jsonl
```

### 3. View Results
```powershell
# Metrics: cat pipeline_outputs.json
# Ledger: cat blockchain_ledger.jsonl
```

## Configuration

All hyperparameters in `src/config.py`:

```python
SEED = 42
TRAIN_SIZE = 0.75
TEST_SIZE = 0.25
INJECTION_FRAC = 0.02        # 2% anomalies in test
NOISE_SCALE = 5.0             # Gaussian noise strength
BOTTLENECK_DIM = 4            # Autoencoder bottleneck
EPOCHS = 150                  # Training epochs
BATCH_SIZE = 16               # Training batch size
LEDGER_PATH = "blockchain_ledger.jsonl"
```

## Architecture

- **Autoencoder**: 128→64→32→bottleneck→32→64→128
- **Data Split**: 75% train (clean), 25% test
- **Blockchain**: SHA256 hash-chained, HMAC-SHA256 signed
- **Threshold Selection**: Grid search to maximize F1 score

## Dataset

**StudentPerformanceFactors.csv** (6,607 × 20):
- 17 numeric features (after categorical encoding)
- 3 label columns (excluded)
- Normalized with StandardScaler (μ=0, σ=1)

## File Structure

```
blockchain-anomaly-detection/
├── src/                      # Core modules
│   ├── config.py             # Hyperparameters
│   ├── data_loader.py        # Data loading
│   ├── preprocessing.py      # Feature engineering
│   ├── anomaly_injection.py  # Synthetic anomalies
│   ├── autoencoder_model.py  # Detector
│   └── blockchain_logger.py  # Immutable ledger
├── scripts/
│   ├── run_pipeline.py       # Main pipeline
│   ├── tune_hyperparameters.py
│   ├── compare_autoencoder_vs_isolation_forest.py
│   └── visualize_*.py        # Visualization scripts
├── data/                     # StudentPerformanceFactors.csv
├── outputs/                  # Generated artifacts
├── blockchain_ledger.jsonl   # Detection log
└── pipeline_outputs.json     # Latest results
```

## Results Format

**pipeline_outputs.json**:
```json
{
  "config": {"train_size": 0.75, "test_size": 0.25, ...},
  "metrics": {
    "precision": 0.544,
    "recall": 0.939,
    "f1": 0.689,
    "roc_auc": 1.0
  },
  "threshold": 1.135
}
```

**blockchain_ledger.jsonl** (append-only log):
```json
{"ts": 1707470400.123, "data": {"index": 42, "score": 1.234}, "prev_hash": "...", "record_hash": "..."}
```

## Key Features

✅ **Hash-chained blockchain** — Tamper-evident ledger (SHA256)  
✅ **HMAC signing** — Authentic entries (HMAC-SHA256)  
✅ **Append-only** — Immutable history  
✅ **Deep autoencoder** — Learns normal data manifold  
✅ **Configurable thresholds** — Adapt to use cases  

## Baseline Comparison

Autoencoder vs Isolation Forest (2% injection, noise 5.0):

| Metric | Autoencoder | Isolation Forest |
|--------|-------------|------------------|
| F1 | 0.833 | 0.087 |
| ROC-AUC | 0.989 | 0.737 |
| Precision | 77.8% | 7.55% |

**Winner**: Autoencoder (+0.252 ROC-AUC) — Better at capturing complex patterns

## Troubleshooting

**Import errors**:
```powershell
$env:PYTHONPATH=(Resolve-Path .).Path
python scripts/run_pipeline.py
```

**Different results on each run**:  
Normal! Neural networks have inherent randomness. Variation: ±0.01 ROC-AUC, ±5% precision/recall.

**No output files**:
```powershell
New-Item -ItemType Directory -Path outputs -ErrorAction SilentlyContinue
```

## For Details

See **`CHANGELOG.md`** for:
- Full development history
- All experiment results
- Architectural decisions
- Findings & observations
- Future enhancements
- Testing approaches

---

**Status**: Production-ready | **Best Config**: 75/25 split, bottleneck=4, F1=0.689
