# Blockchain-Enabled Anomaly Detection in Educational Data with Autoencoders

A machine learning system combining **autoencoder-based anomaly detection** with a **permissioned Proof-of-Authority (PoA) blockchain** for educational data provenance and integrity.

The project uses the StudentPerformanceFactors dataset to detect unusual student profiles and records detected anomalies using replicated, cryptographically verified blockchain ledgers.

---

## Overview

The framework combines:

- **Autoencoder-based anomaly detection**
- **Variational Autoencoder (VAE) baseline**
- **Isolation Forest baseline**
- **PCA reconstruction baseline**
- **Validation-based anomaly threshold selection**
- **Repeated-seed robustness evaluation**
- **Statistical model comparison**
- **Ablation and sensitivity experiments**
- **Permissioned Proof-of-Authority blockchain**
- **Controlled blockchain tamper-detection experiment**

The framework separates two responsibilities:

1. **Detection** — machine learning models identify unusual student profiles.
2. **Integrity and provenance** — the blockchain records detected anomalies and provides tamper-evident verification.

---

## Dataset

The project uses the **StudentPerformanceFactors** dataset.

The dataset contains **6,607 student records** with educational and behavioural factors related to student performance.

The raw dataset is stored as:

```text
data/StudentPerformanceFactors.csv
```

After preprocessing and categorical encoding, the anomaly-detection models operate on **28 input features**.

---

## Project Structure

```text
blockchain-anomaly-detection/
│
├── data/
│   └── StudentPerformanceFactors.csv
│
├── outputs/                           # Generated figures and pipeline outputs
│
├── requirements/
│   └── requirements.txt              # Fixed Python dependency versions
│
├── results/
│   ├── poa_ledgers/                  # Replicated validator ledgers
│   ├── poa_tamper_test/              # Controlled tampering experiment outputs
│   ├── poa_test_ledgers/             # PoA verification test outputs
│   ├── ablation_analysis_report.txt
│   ├── ablation_comparison.png
│   ├── ablation_heatmap.png
│   ├── ablation_study_revised.json
│   ├── model_comparison.json
│   ├── multiseed_individual_results.csv
│   ├── multiseed_results.json
│   ├── multiseed_summary.csv
│   ├── predictions_comparison.csv
│   └── roc_data.json
│
├── scripts/
│   ├── __init__.py
│   ├── ablation_study.py
│   ├── analyze_ablation.py
│   ├── compare_autoencoder_vs_isolation_forest.py
│   ├── run_multiseed_comparison.py
│   ├── run_pipeline.py
│   ├── statistical_comparison.py
│   ├── tamper_detection_test.py
│   ├── test_poa_blockchain.py
│   ├── tune_hyperparameters.py
│   └── visualize_encoder_data.py
│
├── src/
│   ├── config.py
│   ├── data_loader.py
│   ├── isolation_forest_baseline.py
│   ├── pca_baseline.py
│   ├── poa_blockchain.py
│   ├── preprocessing.py
│   └── vae_baseline.py
│
├── architecture_tuning_results.json
├── blockchain_ledger.jsonl
├── CHANGELOG.md
├── pipeline_outputs.json
├── README.md
├── SETUP.md
├── test_architectures.py
└── tuning_results.json
```

Generated experiment files may change when the corresponding scripts are rerun.

Local runtime folders such as `__pycache__` and virtual environments are not part of the experimental implementation.

---

## Installation and Setup

### 1. Create a Virtual Environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install Dependencies

```powershell
pip install -r requirements/requirements.txt
```

The requirements file contains fixed dependency versions to improve reproducibility.

### 3. Check the Dataset

Ensure that the following file exists:

```text
data/StudentPerformanceFactors.csv
```

### 4. Run the Main Pipeline

```powershell
python .\scripts\run_pipeline.py
```

---

## Experimental Protocol

The revised experimental protocol uses separate training, validation, and test sets:

```text
Training:   60%
Validation: 15%
Test:       25%
```

For the 6,607 StudentPerformanceFactors records, this corresponds to:

```text
Training:   3,964 samples
Validation:   991 samples
Test:       1,652 samples
```

Each partition has a separate purpose:

- **Training set** — model training
- **Validation set** — anomaly-threshold selection
- **Test set** — final held-out evaluation

The test labels are not used to select the anomaly threshold.

---

## Data Preprocessing

The preprocessing pipeline performs:

1. Load the StudentPerformanceFactors dataset
2. Handle missing values
3. Encode categorical variables
4. Exclude label-like variables from model inputs
5. Split data into training, validation, and test sets
6. Fit feature normalization using the training data
7. Apply the fitted transformation to validation and test data

Numerical missing values are handled using median imputation, while categorical missing values are handled using the mode.

Categorical variables are converted into numerical features using one-hot encoding.

The resulting model input contains **28 features**.

---

## Feature Normalization

Features are standardized using `StandardScaler`.

The transformation is:

```text
x_scaled = (x - mean) / standard deviation
```

The scaler is fitted on the **training data only**.

```text
Training data   → fit + transform
Validation data → transform using training scaler
Test data       → transform using training scaler
```

This prevents information from the validation or test distributions from influencing preprocessing.

---

## Autoencoder Method

The primary anomaly detector is an Autoencoder.

The model learns to reconstruct student profiles through a compressed latent representation.

The basic process is:

```text
Student Features
       ↓
     Encoder
       ↓
Latent Representation
       ↓
     Decoder
       ↓
Reconstructed Features
       ↓
Reconstruction Error
       ↓
  Anomaly Score
```

Samples that follow patterns learned during training are expected to have relatively low reconstruction error.

Samples that differ substantially from learned patterns are expected to have higher reconstruction error.

---

## Main Autoencoder Architecture

The main experiment uses a compressed bottleneck representation.

Alternative bottleneck dimensions and architectural settings are evaluated separately in the ablation and hyperparameter experiments.

The ablation configurations should therefore not be interpreted as replacing the baseline architecture used for the main experiment.

---

## Synthetic Anomaly Injection

Synthetic anomalies are used to provide controlled ground-truth labels.

The main StudentPerformanceFactors experiment injects anomalies into approximately **2%** of the validation and test samples.

This produces approximately:

```text
Validation anomalies: 19
Test anomalies:       33
```

Synthetic anomalies are introduced after the train/validation/test split.

The training data remain clean for Autoencoder training.

---

## Threshold Selection

The anomaly threshold is selected using the **validation set only**.

The procedure is:

1. Train the Autoencoder using the training data
2. Calculate reconstruction errors on the validation set
3. Evaluate candidate thresholds
4. Select the threshold that maximizes validation F1-score
5. Fix the selected threshold
6. Apply the fixed threshold to the held-out test set

The test set is therefore used only for final evaluation.

This prevents test-label leakage during threshold optimization.

---

## Main Pipeline

Run:

```powershell
python .\scripts\run_pipeline.py
```

The pipeline performs:

1. Load the dataset
2. Split the data using 60/15/25
3. Normalize the features
4. Train the Autoencoder
5. Save training-loss curves
6. Inject synthetic anomalies into validation and test sets
7. Select the threshold using validation data
8. Evaluate the fixed threshold on the held-out test set
9. Calculate anomaly-detection metrics
10. Log detected anomalies to the permissioned PoA blockchain
11. Verify all validator ledgers
12. Run the controlled blockchain tampering experiment
13. Save pipeline outputs

---

## Evaluation Metrics

The anomaly-detection models are evaluated using:

- **Precision**
- **Recall**
- **F1-score**
- **ROC-AUC**
- **PR-AUC**, where applicable
- **Specificity**
- **Balanced Accuracy**, where applicable
- **Matthews Correlation Coefficient (MCC)**, where applicable

The evaluation also reports confusion-matrix values:

- True Positives
- False Positives
- False Negatives
- True Negatives

---

## Baseline Models

The Autoencoder is compared with multiple anomaly-detection approaches.

### Variational Autoencoder

The VAE implementation is located at:

```text
src/vae_baseline.py
```

The VAE uses a probabilistic latent representation and provides a deep-learning comparison with the standard Autoencoder.

### Isolation Forest

The Isolation Forest implementation is located at:

```text
src/isolation_forest_baseline.py
```

Isolation Forest provides a classical tree-based anomaly-detection baseline.

### PCA

The PCA baseline is located at:

```text
src/pca_baseline.py
```

PCA provides a simpler reconstruction-based comparison.

Reconstruction error from a lower-dimensional PCA representation is used as the anomaly score.

---

## Repeated-Seed Evaluation

To examine robustness to random initialization and stochastic variation, the main model comparison is repeated using five random seeds:

```text
41
42
43
44
45
```

Run:

```powershell
python .\scripts\run_multiseed_comparison.py
```

The experiment compares:

- Autoencoder
- Variational Autoencoder
- Isolation Forest
- PCA

Results are summarized using:

```text
Mean ± Standard Deviation
```

Generated results include:

```text
results/multiseed_individual_results.csv
results/multiseed_results.json
results/multiseed_summary.csv
```

This provides an indication of variability across repeated runs rather than relying only on a single experimental run.

---

## Statistical Comparison

Statistical comparison of the repeated model results is implemented in:

```text
scripts/statistical_comparison.py
```

This analysis complements the five-seed experiment by examining differences between model performance across repeated runs.

---

## Ablation and Sensitivity Analysis

The repository includes experiments examining the sensitivity of the Autoencoder to different experimental configurations.

Relevant scripts include:

```text
scripts/ablation_study.py
scripts/analyze_ablation.py
scripts/tune_hyperparameters.py
```

Generated results include:

```text
results/ablation_study_revised.json
results/ablation_analysis_report.txt
results/ablation_comparison.png
results/ablation_heatmap.png
```

The ablation experiments are treated as sensitivity analyses and are separate from the primary experimental configuration.

---

## Permissioned Proof-of-Authority Blockchain

The revised provenance component uses a small permissioned **Proof-of-Authority (PoA)** blockchain.

The implementation is located at:

```text
src/poa_blockchain.py
```

The network contains three authorized validator identities:

```text
validator_1
validator_2
validator_3
```

Only authorized validators can create blocks.

Authority rotates between validators using a simple round-robin mechanism.

---

## Blockchain Block Structure

Each block contains information including:

- Block index
- Timestamp
- Validator identity
- Anomaly information
- Previous block hash
- SHA-256 block hash
- HMAC-SHA256 authentication tag

Each newly created block is replicated across the three validator ledgers.

---

## Blockchain Verification

Blockchain verification checks:

1. The validator is authorized
2. The block is linked to the correct previous block
3. The stored SHA-256 hash matches the block contents
4. The HMAC-SHA256 authentication tag is valid
5. Each validator ledger is internally valid
6. All validator replicas contain the same blockchain

This provides tamper-evident provenance for stored anomaly records.

---

## Controlled Tamper-Detection Experiment

A controlled tampering experiment is included in:

```text
scripts/tamper_detection_test.py
```

The experiment demonstrates the integrity function of the blockchain separately from the anomaly-detection function.

The procedure is:

1. Create a legitimate anomaly record
2. Replicate it across the authorized validator ledgers
3. Modify the anomaly score stored by one validator
4. Verify the blockchain again
5. Compare the validator replicas

The modified validator ledger fails verification because the stored data no longer correspond to the original cryptographic values.

The untouched validator replicas remain valid.

Replica comparison also detects that the validator ledgers are no longer identical.

---

## PoA Blockchain Testing

Additional tests of the permissioned PoA implementation are available in:

```text
scripts/test_poa_blockchain.py
```

These tests support verification of:

- Authorized validator behaviour
- Unauthorized validator rejection
- Block creation
- Hash-chain integrity
- Authentication-tag verification
- Validator-ledger replication
- Replica consistency

---

## Detection vs Blockchain Integrity

The framework separates anomaly detection from blockchain integrity.

### 1. Detection — Autoencoder's Role

The machine-learning component:

- Learns patterns in student data
- Calculates reconstruction errors
- Produces anomaly scores
- Identifies unusual student profiles

### 2. Integrity — Blockchain's Role

The blockchain component:

- Records detected anomalies
- Maintains cryptographic hash linkage
- Authenticates stored blocks
- Replicates records across authorized validators
- Detects modification of stored records
- Detects inconsistencies between validator copies

The blockchain does **not** improve the Autoencoder's anomaly-detection accuracy.

Its role is to protect the provenance and integrity of anomaly records after detection.

---

## Security and Trust Assumptions

The PoA blockchain is a proof-of-concept permissioned provenance mechanism.

The validators operate within an authorized environment.

HMAC-SHA256 provides **symmetric authentication**.

It should not be interpreted as an asymmetric digital-signature mechanism.

A party with access to the shared secret key could potentially generate a valid authentication tag.

The implementation therefore demonstrates tamper-evident provenance under the stated trust assumptions rather than absolute immutability against every possible attacker.

---

## Generated Outputs

The project generates experiment outputs in the `outputs/` and `results/` directories.

Examples include:

```text
outputs/
    autoencoder_loss_curves.png
    baseline comparison outputs
    pipeline outputs

results/
    model_comparison.json
    predictions_comparison.csv
    roc_data.json
    multiseed_individual_results.csv
    multiseed_results.json
    multiseed_summary.csv
    ablation_study_revised.json
    ablation_analysis_report.txt
    ablation_comparison.png
    ablation_heatmap.png
    poa_ledgers/
    poa_tamper_test/
    poa_test_ledgers/
```

These files are generated by the corresponding experimental scripts and may be recreated by rerunning the experiments.

---

## Reproducibility

The fixed Python dependencies are provided in:

```text
requirements/requirements.txt
```

Install them using:

```powershell
pip install -r requirements/requirements.txt
```

The repository contains code for:

- Data preprocessing
- Autoencoder anomaly detection
- VAE comparison
- Isolation Forest comparison
- PCA comparison
- Validation-based threshold selection
- Synthetic anomaly injection
- Ablation analysis
- Five-seed robustness evaluation
- Statistical comparison
- Permissioned PoA blockchain logging
- Blockchain integrity verification
- Controlled tamper detection

Random seeds are controlled where applicable.

Because machine-learning libraries, hardware, operating systems, and numerical operations can introduce small differences, exact bit-for-bit equality across all systems is not assumed.

The repeated-seed evaluation provides an additional measure of experimental variability.

---

## Current Results

The README does not use historical hard-coded performance tables as the authoritative source of the current experimental results.

Current results should be obtained from:

- The revised manuscript
- `pipeline_outputs.json`
- Files in `results/`
- Files in `outputs/`
- Repeated-seed experiment outputs

This prevents results from earlier development configurations from being confused with the revised experimental protocol.

---

## Important Experimental Notes

### Validation-Based Threshold Selection

Threshold selection is performed using validation data only.

The held-out test set is not used to optimize the anomaly threshold.

### Synthetic Anomalies

The main StudentPerformanceFactors experiment uses approximately 2% injected synthetic anomalies.

These provide controlled ground-truth labels for evaluation.

### Repeated Experiments

The main model comparison is repeated across five random seeds and summarized using mean ± standard deviation.

### Ablation Experiments

Ablation experiments are sensitivity analyses and should not be confused with the main experimental configuration.

### Blockchain Provenance

The PoA blockchain protects the integrity and provenance of anomaly records but does not alter the machine-learning predictions.

---

## Common Issues and Troubleshooting

### Import Errors

From the project root, run:

```powershell
$env:PYTHONPATH=(Resolve-Path .).Path
python .\scripts\run_pipeline.py
```

### Missing Dataset

Check:

```powershell
Test-Path .\data\StudentPerformanceFactors.csv
```

The result should be:

```text
True
```

### TensorFlow Warnings

TensorFlow may display informational CPU or optimization messages.

To reduce logging:

```powershell
$env:TF_CPP_MIN_LOG_LEVEL='2'
python .\scripts\run_pipeline.py
```

### Recreate PoA Ledgers

If a previous blockchain run needs to be cleared before rerunning the complete experiment:

```powershell
Remove-Item -Recurse -Force .\results\poa_ledgers -ErrorAction SilentlyContinue
python .\scripts\run_pipeline.py
```

---

## Code Availability

This repository contains the StudentPerformanceFactors implementation used for the experiments described in the associated manuscript.

The current implementation includes:

- Autoencoder anomaly detection
- Variational Autoencoder baseline
- Isolation Forest baseline
- PCA baseline
- 60/15/25 train/validation/test evaluation
- Validation-only threshold selection
- Synthetic anomaly evaluation
- Ablation experiments
- Five-seed robustness evaluation
- Statistical model comparison
- Permissioned PoA blockchain
- Three replicated validator ledgers
- Blockchain integrity verification
- Controlled blockchain tamper-detection experiment

---

## Associated Manuscript

**Autoencoder-Based Anomaly Detection with Blockchain-Assisted Provenance for Educational Data**

This repository corresponds to the revised StudentPerformanceFactors experimental implementation described in the manuscript.