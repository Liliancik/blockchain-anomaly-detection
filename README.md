
# blockchain-anomaly-detection

## Overview

This repository demonstrates a lightweight pipeline for injecting and detecting anomalies in tabular student-performance data with an append-only, blockchain-style ledger for auditability. The code is modular so you can swap detectors, injection strategies, or the ledger backend.

## Quick Start

1. Place your dataset at `data/StudentPerformanceFactors.csv`. The `data/` folder is ignored by git by default.

2. (Recommended) Create and activate a virtual environment, then install dependencies:

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements/requirements.txt
```

If `pip install -r` hits a dependency resolution issue, install the minimal runtime packages directly:

```powershell
pip install pandas numpy scikit-learn jupyter
```

3. Run the pipeline (default uses `data/StudentPerformanceFactors.csv`). The pipeline has options for injection method and fraction; it now implements the evaluation variant that trains the detector on the clean data and tests on an injected copy (Option A):

```powershell
# run end-to-end (inject noise, detect, log)
python scripts/run_pipeline.py --data data/StudentPerformanceFactors.csv --method noise --frac 0.02 --push-outputs

# compute precision/recall/F1 from latest outputs
python scripts/compute_metrics.py
```

Notes:
- `--method` accepts `noise` and `swap` (see `src/anomaly_injection.py`).
- `--frac` sets the fraction of rows to inject as anomalous.
- `--push-outputs` will (optionally) cause the script to save `pipeline_outputs.json` locally and the pipeline appends detection logs to `blockchain_ledger.jsonl`.

## Files and purpose

- `src/data_loader.py` — CSV loader
- `src/preprocessing.py` — imputation, encoding, normalization
- `src/anomaly_injection.py` — inject anomalies (noise, swap)
- `src/autoencoder_model.py` — IsolationForest wrapper (baseline detector)
- `src/blockchain_logger.py` — append-only JSONL ledger with SHA256 hashes
- `scripts/run_pipeline.py` — orchestrates load → preprocess → inject → detect → log → save outputs
- `scripts/compute_metrics.py` — compute precision/recall/F1 from `pipeline_outputs.json`

## Ledger and outputs

- `blockchain_ledger.jsonl` is an append-only JSONL file that records logged detections with timestamps and SHA256 hashes. This file is tracked in the repo by default and is suitable for audit trails.
- `pipeline_outputs.json` contains the ground-truth injected indices, detected indices, and score summaries for each run; this file is currently ignored by `.gitignore` to avoid committing large runs. If you want to publish specific runs, you can add and commit `pipeline_outputs.json` manually.

To commit the ledger and/or outputs (PowerShell):

```powershell
# commit ledger changes
git add blockchain_ledger.jsonl
git commit -m "Append ledger entries: evaluation run"
git push

# (optional) publish a run's outputs
git add pipeline_outputs.json
git commit -m "Add pipeline_outputs.json (evaluation outputs)"
git push
```

## Reproduce evaluation variants

- Default behavior (current): train detector on clean data, test on injected copy (Option A). This yields a stricter, unbiased evaluation for detector generalization.
- To modify behavior or experiment with other detectors, update `src/autoencoder_model.py` or replace it with your own detector implementing `fit(X)` and `predict(X)`.

## Troubleshooting

- If imports fail, ensure `PYTHONPATH` includes the project root or run commands from the repository root. In PowerShell you can set it for a session:

```powershell
$env:PYTHONPATH=(Resolve-Path .).Path
python scripts/run_pipeline.py ...
```

- If `pip install -r requirements/requirements.txt` fails due to unrelated dependency resolution issues, install core packages individually (see above).

## Project structure

```
.
├── data/                 # Large datasets (gitignored)
├── requirements/         # Requirements files
├── scripts/              # Runner scripts
├── src/                  # Core modules (loader, preprocessing, injection, detection, logging)
└── README.md             # This file
```

## License

TBD
