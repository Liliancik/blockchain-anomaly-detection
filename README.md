# Blockchain-Enabled Anomaly Detection in Educational Data with Autoencoders

## Overview
This project explores how autoencoders can be used to detect anomalies in student learning data (e.g., assessment scores, engagement logs, or progress metrics) while leveraging blockchain-inspired mechanisms to ensure traceable, immutable datasets. The goal is to validate how secure data provenance can help prevent tampering in machine learning inputs and strengthen trust in anomaly detection outcomes for education.

## Objectives
- **Implement an autoencoder-based anomaly detector** for educational performance logs.
- **Validate the system on synthetic data**, demonstrating reconstruction error as an anomaly signal.
- **Simulate blockchain integration** (e.g., Hyperledger Fabric style ledger) to ensure data traceability.
- **Discuss ethics and data ownership**, including student privacy and consent.

## Methodology (Scoped)
1. **Focused literature review** on ML + blockchain in education.
2. **Autoencoder training (Keras)** on synthetic or anonymized datasets.
3. **Blockchain logging simulation** to record immutable dataset entries and hashes.
4. **Evaluation** using reconstruction error metrics and anomaly thresholds.
5. **Ethical analysis** covering governance, ownership, and risks of misuse.

## Expected Outcomes
- A **secure anomaly detection prototype** for educational data.
- Demonstration of **fraud/tamper resistance** in ML inputs via blockchain logging.
- A thesis/report with **use cases for personalized learning systems** and integrity guarantees.

## Project Structure (Planned)
```
.
├── data/                 # Synthetic datasets and logs
├── notebooks/            # Experiments and evaluation
├── src/                  # Autoencoder + blockchain simulation
└── README.md             # Project overview
```

## Ethics & Data Ownership
Educational data is highly sensitive. This project emphasizes:
- **Privacy**: anonymization and minimal data exposure.
- **Consent**: clear ownership and usage rights for student records.
- **Transparency**: blockchain logging to ensure trust in dataset provenance.

## Status
This is an early-stage research prototype focused on core implementation and evaluation.

## License
TBD (add appropriate license once repository requirements are finalized).
