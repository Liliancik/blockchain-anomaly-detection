
# blockchain-anomaly-detection

## Overview

A complete **anomaly detection pipeline with blockchain logging** for tabular student performance data. The system trains a deep autoencoder to detect injected anomalies and records all detections in an immutable, auditable blockchain ledger. Achieves **99.5% accuracy with perfect recall** on test data.

### Key Features
- **Deep Autoencoder Detector**: 128→64→32→bottleneck→32→64→128 architecture with dropout regularization
- **Flexible Anomaly Injection**: Gaussian noise or value-swapping with configurable fractions (1-20%)
- **Blockchain Ledger**: SHA256 hash-chained append-only log for audit trails
- **High Performance**: Precision 0.79, Recall 1.0, F1 0.88 on test data
- **Rich Visualizations**: Noise scale comparison, injection effects, distribution plots
- **Modular Design**: Easy to swap detectors, injection methods, or backends

## Quick Start

### 1. Setup Environment

```powershell
# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements/requirements.txt
```

If you encounter dependency issues, install core packages individually:
```powershell
pip install pandas numpy scikit-learn tensorflow keras matplotlib
```

### 2. Add Data

Place your dataset at `data/StudentPerformanceFactors.csv`. The `data/` folder is ignored by git.

### 3. Run the Pipeline

```powershell
# Default: 2% injection, noise scale 20.0
python scripts/run_pipeline.py

# Custom parameters
python scripts/run_pipeline.py --frac 0.05 --noise-scale 15.0 --method noise

# With output saving
python scripts/run_pipeline.py --frac 0.02 --noise-scale 20.0 --push-outputs
```

### 4. Generate Visualizations

```powershell
# Compare different noise scales (5.0, 10.0, 20.0)
python scripts/compare_noise_scales.py

# Visualize injection effects
python scripts/visualize_injection.py
```

## Current Performance (Last Run)

With **2% injection fraction, noise scale 20.0**:

| Metric | Validation | Test |
|--------|-----------|------|
| Precision | 0.950 | 0.792 |
| Recall | 1.000 | 1.000 |
| F1 Score | 0.974 | 0.884 |
| Accuracy | — | 0.9950 |

**Blockchain Ledger**: 24 entries logged, 19 true positives detected (perfect recall)

## Architecture

### Core Modules

| Module | Purpose |
|--------|---------|
| `src/data_loader.py` | Load and parse CSV datasets |
| `src/preprocessing.py` | Handle missing values, categorical encoding, feature selection, normalization |
| `src/anomaly_injection.py` | Inject anomalies (Gaussian noise or value swapping) |
| `src/autoencoder_model.py` | Train and evaluate deep autoencoder detector |
| `src/blockchain_logger.py` | Append-only SHA256 hash-chained ledger |

### Scripts

| Script | Purpose |
|--------|---------|
| `scripts/run_pipeline.py` | Full pipeline: load → preprocess → train → inject → detect → log |
| `scripts/compute_metrics.py` | Calculate precision, recall, F1 from outputs |
| `scripts/compare_noise_scales.py` | Generate side-by-side visualizations for noise scales 5.0, 10.0, 20.0 |
| `scripts/visualize_injection.py` | Visualize clean vs injected data distributions |
| `scripts/tune_hyperparameters.py` | Grid search for optimal bottleneck dimensions |

### Data Flow

```
Input CSV
    ↓
[Data Loader] → Load & validate
    ↓
[Preprocessing] → Impute, encode, select features, normalize
    ↓
[Train-Test Split] → 75% train (clean), 25% test
    ↓
[Autoencoder Training] → Learn reconstruction on clean training data
    ↓
[Anomaly Injection] → Add Gaussian noise to test set
    ↓
[Detection] → Score reconstruction errors, threshold-based classification
    ↓
[Blockchain Logging] → Record all detections with SHA256 hashes
    ↓
[Metrics & Outputs] → Precision, recall, F1, accuracy, visualizations
```

## Outputs & Artifacts

### JSON Outputs
- **`pipeline_outputs.json`** — Complete metrics and results from latest run:
  - Ground-truth injected indices
  - Detected indices and scores
  - Precision, recall, F1, accuracy
  - Reconstruction error statistics

- **`blockchain_ledger.jsonl`** — Append-only detection log with:
  - Detection timestamp
  - Anomaly score
  - SHA256 hash (links to previous entry)
  - Optional user/role information

### Visualizations
- **`outputs/noise_scale_comparison.png`** — Side-by-side scatter and histogram plots showing effect of noise scales 5.0, 10.0, 20.0
- **`outputs/clear_injection_visualization.png`** — Clean (blue) vs injected (red) data distributions
- **`outputs/injection_fraction_comparison.png`** — Impact of different injection fractions (1%, 5%, 10%, 20%)

### Data Exports
- **`outputs/clean_vs_injected.csv`** — All samples with injection labels
- **`outputs/injected_samples.csv`** — Only injected anomaly samples

### Parameters

| Parameter | Default | Options | Description |
|-----------|---------|---------|-------------|
| `--frac` | 0.02 | 0.01–0.20 | Fraction of samples to inject |
| `--noise-scale` | 5.0 | 5.0–20.0 | Gaussian noise strength (σ units) |
| `--method` | noise | noise, swap | Injection method |
| `--push-outputs` | False | True/False | Save outputs and ledger entries |

### Ledger Format

Each line in `blockchain_ledger.jsonl`:
```json
{
  "timestamp": "2026-02-05 17:34:29.123456",
  "anomaly_score": 45.32,
  "sample_index": 1250,
  "sha256_hash": "a1b2c3d4...",
  "previous_hash": "z9y8x7w6...",
  "user": "system",
  "role": "detector"
}
```

## Git Workflow

### Committing Results

```powershell
# After a successful run, commit the ledger
git add blockchain_ledger.jsonl
git commit -m "Append ledger entries: evaluation run with frac=0.02"
git push

# (Optional) Publish specific run outputs
git add pipeline_outputs.json
git commit -m "Results: F1=0.884, Precision=0.792, Recall=1.0"
git push
```

### Evaluation Methodology

The pipeline implements **Option A evaluation** (train on clean, test on injected):
1. Train autoencoder on 75% clean training data only
2. Inject anomalies into 25% test set (configurable fraction)
3. Evaluate detection against ground-truth injected labels
4. This ensures **unbiased generalization** assessment

## Customization

### Swap Detector with Your Own

Replace in `src/autoencoder_model.py`:
```python
# Your detector class
class CustomDetector:
    def fit(self, X):
        # Train on clean data
        pass
    
    def predict(self, X):
        # Return anomaly scores
        return scores
```

### Modify Injection Method

Edit `src/anomaly_injection.py` to add new injection strategies:
```python
def inject_custom(df, frac=0.02, **kwargs):
    # Your injection logic
    return modified_df, anomaly_indices
```

### Change Ledger Backend

Extend `src/blockchain_logger.py` to use database, cloud storage, etc.:
```python
class CustomBlockchain:
    def append(self, entry):
        # Custom storage logic
        pass
```

## Troubleshooting

### Import Errors

If imports fail, ensure the project root is in `PYTHONPATH`:
```powershell
$env:PYTHONPATH=(Resolve-Path .).Path
python scripts/run_pipeline.py
```

Or install in development mode:
```powershell
pip install -e .
```

### TensorFlow/Keras Issues

Ensure TensorFlow is properly installed:
```powershell
pip install --upgrade tensorflow keras
```

Disable oneDNN warnings (optional):
```powershell
$env:TF_CPP_MIN_LOG_LEVEL='2'
python scripts/run_pipeline.py
```

### Memory Issues

For large datasets, reduce batch size:
```python
# In src/autoencoder_model.py
model.fit(X_train, epochs=50, batch_size=8)  # Reduce from 16
```

### No Output Files

Ensure `outputs/` directory exists:
```powershell
New-Item -ItemType Directory -Path outputs -ErrorAction SilentlyContinue
python scripts/run_pipeline.py --push-outputs
```

## Project Structure

```
blockchain-anomaly-detection/
├── data/                           # Datasets (gitignored, ~100+ MB)
│   └── StudentPerformanceFactors.csv
├── outputs/                        # Visualizations and exports
│   ├── noise_scale_comparison.png
│   ├── clear_injection_visualization.png
│   ├── injection_fraction_comparison.png
│   ├── clean_vs_injected.csv
│   └── injected_samples.csv
├── requirements/
│   └── requirements.txt             # Python dependencies
├── scripts/
│   ├── run_pipeline.py             # Main pipeline runner
│   ├── compute_metrics.py          # Metric calculation
│   ├── compare_noise_scales.py     # Noise comparison visualization
│   ├── visualize_injection.py      # Data distribution plots
│   ├── tune_hyperparameters.py     # Hyperparameter search
│   └── __init__.py
├── src/
│   ├── data_loader.py              # CSV loading
│   ├── preprocessing.py            # Data preprocessing
│   ├── anomaly_injection.py        # Anomaly injection methods
│   ├── autoencoder_model.py        # Detector implementation
│   ├── blockchain_logger.py        # Ledger storage
│   └── __init__.py
├── blockchain_ledger.jsonl         # Immutable detection log
├── pipeline_outputs.json           # Latest run results
├── tuning_results.json             # Hyperparameter tuning history
├── dataset_explore.ipynb           # Exploratory analysis notebook
├── .gitignore
├── README.md                        # This file
└── .git/                           # Version control
```

## Performance Benchmarks

### Effect of Noise Scale

| Noise Scale | Precision | Recall | F1 Score | Accuracy |
|-------------|-----------|--------|----------|----------|
| 5.0 | 0.733 | 0.579 | 0.647 | 0.993 |
| 10.0 | 0.769 | 0.842 | 0.804 | 0.997 |
| 20.0 | 0.792 | 1.000 | 0.884 | 0.995 |

**Insight**: Higher noise scales make anomalies more detectable (higher recall), though slightly more false positives occur.

### Effect of Injection Fraction

Testing with 2% baseline injection (noise scale 20.0):
- **1%**: Higher precision, lower recall (fewer anomalies to detect)
- **2%**: Balanced performance (current default)
- **5-20%**: Good recall, manageable false positives

**Recommendation**: Use 2-5% for realistic anomaly detection scenarios.

## Related Work

- **Autoencoder for Anomaly Detection**: [Kingma & Welling, 2013](https://arxiv.org/abs/1312.6114)
- **Blockchain Audit Logs**: [Swan, 2015](https://link.springer.com/book/10.1007/978-1-4842-2143-3)
- **Student Performance Analysis**: Educational data mining (EDM) community

## Contact & Support

For questions or contributions:
- Check existing issues on GitHub
- Review the README section on Troubleshooting
- Examine logs in `blockchain_ledger.jsonl` for debugging
