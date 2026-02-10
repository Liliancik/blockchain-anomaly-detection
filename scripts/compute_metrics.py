"""
Enhanced metrics computation with threshold analysis.
Shows percentile-based thresholds and validation-optimized thresholds.
"""
import json
from pathlib import Path
import numpy as np


def load_pipeline_outputs():
    """Load pipeline results."""
    p = 'pipeline_outputs.json'
    with open(p) as f:
        return json.load(f)


def print_basic_metrics(data):
    """Print basic detection metrics."""
    test_metrics = data.get('test_metrics', {})
    
    injected = test_metrics.get('injected', 0)
    detected = test_metrics.get('detected', 0)
    tp = test_metrics.get('true_positives', 0)
    
    precision = test_metrics.get('precision', 0)
    recall = test_metrics.get('recall', 0)
    f1 = test_metrics.get('f1', 0)
    
    print("\n" + "="*80)
    print("BASIC DETECTION METRICS")
    print("="*80)
    print(f"Injected anomalies:  {injected}")
    print(f"Detected anomalies:  {detected}")
    print(f"True positives:      {tp}")
    print(f"False positives:     {detected - tp}")
    print(f"\nPrecision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    roc_auc = test_metrics.get('roc_auc', None)
    if roc_auc:
        print(f"ROC-AUC:   {roc_auc:.4f}")


def print_threshold_analysis(data):
    """Analyze and display threshold decision criteria."""
    print("\n" + "="*80)
    print("THRESHOLD DECISION ANALYSIS")
    print("="*80)
    
    # Current threshold (from validation optimization)
    current_threshold = data.get('threshold', 'Unknown')
    print(f"\nUsed Threshold: {current_threshold}")
    print(f"  Method: Validation-set F1 optimization (greedy search)")
    print(f"  Optimization: Maximizes F1 on validation set")
    print(f"  Strategy: Balanced precision/recall compromise")
    
    # Show validation metrics
    val_metrics = data.get('validation_metrics', {})
    if val_metrics:
        print(f"\n  Validation Performance at this threshold:")
        print(f"    Precision: {val_metrics.get('precision', 'N/A'):.4f}")
        print(f"    Recall:    {val_metrics.get('recall', 'N/A'):.4f}")
        print(f"    F1:        {val_metrics.get('f1', 'N/A'):.4f}")
    
    # Percentile-based alternatives
    scores_summary = data.get('scores_summary_test', {})
    if scores_summary:
        print(f"\n" + "-"*80)
        print("ALTERNATIVE PERCENTILE-BASED THRESHOLDS")
        print("-"*80)
        
        min_score = scores_summary.get('min', 0)
        max_score = scores_summary.get('max', 1)
        mean_score = scores_summary.get('mean', 0.5)
        
        print(f"\nReconstruction Error Statistics (Test Set):")
        print(f"  Min:  {min_score:.6f}  (cleanest normal sample)")
        print(f"  Max:  {max_score:.6f}  (most anomalous sample)")
        print(f"  Mean: {mean_score:.6f}  (average error)")
        print(f"  Range: {max_score - min_score:.6f}")
        
        # Common percentile thresholds
        percentiles = [80, 85, 90, 95, 99]
        print(f"\nSimple percentile-based thresholds:")
        print(f"  (Marks reconstruction errors above Nth percentile as anomalies)")
        print()
        
        for p in percentiles:
            # Estimate percentile threshold
            threshold_est = min_score + (max_score - min_score) * (p / 100)
            pct_range = 100 - p
            print(f"  {p}th percentile: ~{threshold_est:.6f}  (flags top {pct_range}%)")
        
        print(f"\n  How to use percentile thresholds:")
        print(f"    1. Choose percentile based on use case")
        print(f"    2. Compute threshold: min_score + (max_score - min_score) * (p / 100)")
        print(f"    3. Flag samples with error > threshold as anomalies")
        
        print(f"\n  Trade-offs:")
        print(f"    - 80th: More detections, higher false positive rate")
        print(f"    - 95th: Fewer detections, higher confidence")
        print(f"    - 99th: Very few detections, only confident anomalies")


def print_configuration(data):
    """Print experiment configuration."""
    print("\n" + "="*80)
    print("EXPERIMENT CONFIGURATION")
    print("="*80)
    
    print(f"Injection method:   {data.get('method', 'Unknown')}")
    frac = data.get('frac', 0)
    print(f"Injection fraction: {frac} ({frac*100:.1f}% of test set)")
    print(f"Noise scale:        {data.get('noise_scale', 'Unknown')}")
    test_metrics = data.get('test_metrics', {})
    print(f"Test samples:       {test_metrics.get('injected', 'Unknown')} injected anomalies")


def print_scores_summary(data):
    """Print reconstruction error statistics."""
    scores = data.get('scores_summary_test', {})
    if not scores:
        return
    
    print("\n" + "="*80)
    print("RECONSTRUCTION ERROR STATISTICS")
    print("="*80)
    
    min_err = scores.get('min', 0)
    max_err = scores.get('max', 0)
    mean_err = scores.get('mean', 0)
    
    print(f"Min error:  {min_err:.6f}  (best reconstruction)")
    print(f"Max error:  {max_err:.6f}  (worst reconstruction)")
    print(f"Mean error: {mean_err:.6f}  (average)")
    print(f"Range:      {max_err - min_err:.6f}")
    
    if max_err > min_err:
        print(f"\nInterpretation:")
        print(f"  - Normal samples cluster near {min_err:.6f}")
        print(f"  - Anomalies spread toward {max_err:.6f}")
        print(f"  - Threshold decision separates these distributions")
        
        # Show where current threshold sits
        current_threshold = data.get('threshold', None)
        if current_threshold:
            pct_position = (current_threshold - min_err) / (max_err - min_err) * 100
            print(f"\nCurrent threshold position: {pct_position:.1f}% from minimum")


def main():
    """Compute and display comprehensive metrics."""
    data = load_pipeline_outputs()
    
    print("\n" + "="*80)
    print("BLOCKCHAIN ANOMALY DETECTION: COMPREHENSIVE METRICS")
    print("="*80)
    
    print_configuration(data)
    print_basic_metrics(data)
    print_threshold_analysis(data)
    print_scores_summary(data)
    
    print("\n" + "="*80)
    print("DECISION GUIDANCE: CHOOSING THRESHOLDS")
    print("="*80)
    print("""
For different use cases, choose thresholds strategically:

[HIGH RECALL] Catch all anomalies:
  -> Use lower percentile (80-85th)
  -> Maximize true positives, accept false positives
  -> Good for: Fraud detection, safety-critical systems
  -> Trade-off: More alerts, more investigation needed

[BALANCED] F1-optimized (CURRENT APPROACH):
  -> Use validation-optimized threshold
  -> Balances precision and recall
  -> Good for: General anomaly detection
  -> Trade-off: Neither extreme, reasonable both ways

[HIGH PRECISION] Minimize false alarms:
  -> Use higher percentile (95-99th)
  -> Only flag very confident anomalies
  -> Good for: Regulatory compliance, audit trails
  -> Trade-off: May miss subtle anomalies

Current configuration uses validation-optimized F1 approach,
which is recommended for general-purpose anomaly detection.
    """)
    
    print("="*80 + "\n")


if __name__ == "__main__":
    main()