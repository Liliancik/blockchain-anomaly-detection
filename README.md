# blockchain-anomaly-detection

## Overview

This project explores anomaly detection in student performance data with an emphasis on traceability and tamper-resistance using a blockchain-inspired ledger. It provides a lightweight scaffold to inject anomalies into datasets, detect them with a model, and record detection events in an append-only ledger for auditability.

## Quick Start

1. Place your dataset at `data/StudentPerformanceFactors.csv`. The `data/` folder is ignored by git by default.

2. (Optional) Create and activate a virtual environment, then install dependencies:

```powershell
python -m venv .venv; .\\.venv\\Scripts\\Activate.ps1
pip install -r requirements/requirements.txt
```

3. Run the pipeline (default uses `data/StudentPerformanceFactors.csv`):

```powershell
python scripts/run_pipeline.py --data data/StudentPerformanceFactors.csv --method noise --frac 0.01 --push-outputs
```

## What I added (scaffold)

- `src/data_loader.py` — CSV loader
- `src/preprocessing.py` — missing value handling, encoding, normalization
- `src/anomaly_injection.py` — anomaly injection helpers (noise, swap)
- `src/autoencoder_model.py` — Isolation Forest wrapper for anomaly detection
- `src/blockchain_logger.py` — simple append-only JSONL ledger with hashing
- `scripts/run_pipeline.py` — end-to-end runner script

## Notes

- `blockchain_ledger.jsonl` is a local, append-only file used to simulate a ledger. For production replace the logging with a real blockchain or secure audit service.
- Keep large or sensitive datasets in `data/` and do not commit them to git.

## Project Structure

```
.
├── data/                 # Large datasets (gitignored)
├── requirements/         # Requirements files
├── scripts/              # Runner scripts
├── src/                  # Core modules (loader, preprocessing, injection, detection, logging)
└── README.md             # This file
```

## Next steps

- Upload `StudentPerformanceFactors.csv` into the `data/` folder and tell me to run the pipeline; I will execute the notebook or script here, save outputs, and push the notebook with outputs so GitHub renders results.

## License

TBD
