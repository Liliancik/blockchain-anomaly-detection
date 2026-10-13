"""Analyze and visualize ablation study results.

This script loads the ablation study results and generates:
1. Comparison charts for each component
2. Heatmap showing metric relationships
3. Detailed analysis report
4. Recommendations based on findings
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

# Set seeds for reproducibility (matching ablation_study.py)
SEEDS = [41, 42, 43, 44, 45]
SEEDS_STR = ", ".join(map(str, SEEDS))

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (14, 10)
plt.rcParams["font.size"] = 10


def load_results(results_path: Path) -> Dict[str, Any]:
    """Load ablation study results from JSON file."""
    with open(results_path, "r") as f:
        return json.load(f)


def extract_comparison_data(results: Dict[str, Any]) -> pd.DataFrame:
    """Extract all variants into a single DataFrame for analysis."""
    rows = []
    
    # Helper function to compute stats from all_runs
    def compute_stats_from_runs(all_runs: list) -> dict:
        """Compute mean statistics across multiple runs."""
        # Get all metric names from first run
        metric_names = [k for k in all_runs[0].keys() if k != "seed"]
        
        # Fields to skip (these are metadata, not metrics to average)
        skip_fields = {"seed", "threshold_metric", "threshold_metric_score"}
        
        stats = {}
        for metric in metric_names:
            if metric in skip_fields:
                continue
            
            values = [run[metric] for run in all_runs if metric in run]
            if values:
                try:
                    # Try to convert to float in case they're numeric strings
                    numeric_values = [float(v) for v in values if isinstance(v, (int, float))]
                    if numeric_values:
                        stats[metric] = np.mean(numeric_values)
                except (ValueError, TypeError):
                    # Skip non-numeric fields
                    pass
        return stats
    
    # Add baseline
    baseline = results["baseline"]
    baseline_config = baseline["config"]
    baseline_stats = compute_stats_from_runs(baseline["all_runs"])
    row = {
        "variant": "baseline",
        "category": "baseline",
        **baseline_config,
        **baseline_stats
    }
    rows.append(row)
    
    # Add all variations
    for category, variants in results["variations"].items():
        for variant_name, variant_data in variants.items():
            config = variant_data["config"]
            stats = compute_stats_from_runs(variant_data["all_runs"])
            row = {
                "variant": variant_name,
                "category": category,
                **config,
                **stats
            }
            
            rows.append(row)
    
    return pd.DataFrame(rows)


def plot_component_comparison(df: pd.DataFrame, results_dir: Path):
    """Create comparison charts for each ablated component."""
    
    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    fig.suptitle(f"Ablation Study: Component Impact on Performance - Mean Across Seeds: {SEEDS_STR}", 
                 fontsize=16, fontweight="bold", y=0.995)

    baseline_row = df[df["variant"] == "baseline"].iloc[0]
    
    # 1. Bottleneck Dimension
    ax = axes[0, 0]
    bottleneck_df = df[df["category"] == "bottleneck_dimension"].copy()
    bottleneck_df = bottleneck_df.sort_values("bottleneck_dim")
    ax.plot(bottleneck_df["bottleneck_dim"], bottleneck_df["f1"], 
            marker="o", linewidth=2, markersize=8, label="F1 Score", color="#1f77b4")
    ax.plot(bottleneck_df["bottleneck_dim"], bottleneck_df["roc_auc"], 
            marker="s", linewidth=2, markersize=8, label="ROC-AUC", color="#ff7f0e")
    ax.set_xlabel("Bottleneck Dimension")
    ax.set_ylabel("Score")
    ax.set_title("Bottleneck Dimension Impact (Mean Across Seeds)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(bottleneck_df["bottleneck_dim"].unique())
    
    # 2. Network Depth
    ax = axes[0, 1]
    depth_df = df[df["category"] == "network_depth"].copy()
    depth_df = depth_df.sort_values("depth")
    ax.plot(depth_df["depth"], depth_df["f1"], 
            marker="o", linewidth=2, markersize=8, label="F1 Score", color="#1f77b4")
    ax.plot(depth_df["depth"], depth_df["roc_auc"], 
            marker="s", linewidth=2, markersize=8, label="ROC-AUC", color="#ff7f0e")
    ax.set_xlabel("Network Depth (# Hidden Layers)")
    ax.set_ylabel("Score")
    ax.set_title("Network Depth Impact (Mean Across Seeds)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(depth_df["depth"].unique())
    
    # 3. Dropout Rate
    ax = axes[0, 2]
    dropout_df = df[df["category"] == "dropout_rate"].copy()
    dropout_df = dropout_df.sort_values("dropout_rate")
    ax.plot(dropout_df["dropout_rate"], dropout_df["f1"], 
            marker="o", linewidth=2, markersize=8, label="F1 Score", color="#1f77b4")
    ax.plot(dropout_df["dropout_rate"], dropout_df["roc_auc"], 
            marker="s", linewidth=2, markersize=8, label="ROC-AUC", color="#ff7f0e")
    ax.set_xlabel("Dropout Rate")
    ax.set_ylabel("Score")
    ax.set_title("Dropout Rate Impact (Mean Across Seeds)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Activation Function
    ax = axes[1, 0]
    activation_df = df[df["category"] == "activation_function"].copy()
    activations = activation_df["activation"].values
    x_pos = np.arange(len(activations))
    width = 0.35
    ax.bar(x_pos - width/2, activation_df["f1"], width, label="F1 Score", alpha=0.8)
    ax.bar(x_pos + width/2, activation_df["roc_auc"], width, label="ROC-AUC", alpha=0.8)
    ax.set_xlabel("Activation Function")
    ax.set_ylabel("Score")
    ax.set_title("Activation Function Impact (Mean Across Seeds)")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(activations)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim([0.8, 1.0])

    # 5. Epochs Comparison
    ax = axes[1, 1]
    epochs_df = df[df["category"] == "epochs"].copy() if "epochs" in df["category"].values else None
    if epochs_df is not None and len(epochs_df) > 0:
        epochs_df = epochs_df.sort_values("epochs")
        ax.plot(epochs_df["epochs"], epochs_df["f1"], 
                marker="o", linewidth=2, markersize=8, label="F1 Score", color="#1f77b4")
        ax.plot(epochs_df["epochs"], epochs_df["roc_auc"], 
                marker="s", linewidth=2, markersize=8, label="ROC-AUC", color="#ff7f0e")
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Score")
        ax.set_title("Epochs Impact (Mean Across Seeds)")
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        # Show baseline epochs in a text format
        baseline_epochs = baseline_row.get("epochs", "N/A")
        ax.text(0.5, 0.5, f"Baseline Epochs: {baseline_epochs}", 
                ha="center", va="center", fontsize=14, fontweight="bold",
                transform=ax.transAxes)
        ax.set_title("Epoch Information")
        ax.axis("off")
    
    # 6. Batch Size Comparison
    ax = axes[1, 2]
    batch_df = df[df["category"] == "batch_size"].copy() if "batch_size" in df["category"].values else None
    if batch_df is not None and len(batch_df) > 0:
        batch_df = batch_df.sort_values("batch_size")
        ax.plot(batch_df["batch_size"], batch_df["f1"], 
                marker="o", linewidth=2, markersize=8, label="F1 Score", color="#1f77b4")
        ax.plot(batch_df["batch_size"], batch_df["roc_auc"], 
                marker="s", linewidth=2, markersize=8, label="ROC-AUC", color="#ff7f0e")
        ax.set_xlabel("Batch Size")
        ax.set_ylabel("Score")
        ax.set_title("Batch Size Impact (Mean Across Seeds)")
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        # Show baseline batch size in a text format
        baseline_batch = baseline_row.get("batch_size", "N/A")
        ax.text(0.5, 0.5, f"Baseline Batch Size: {baseline_batch}", 
                ha="center", va="center", fontsize=14, fontweight="bold",
                transform=ax.transAxes)
        ax.set_title("Batch Size Information")
        ax.axis("off")
    
    # 7. Key Metrics Comparison
    ax = axes[2, 0]
    metrics = ["f1", "precision", "recall", "roc_auc"]
    values = [baseline_row[m] for m in metrics]
    colors = plt.cm.Set3(np.linspace(0, 1, len(metrics)))
    bars = ax.barh(metrics, values, color=colors)
    ax.set_xlabel("Score")
    ax.set_title("Baseline Performance (All Metrics)")
    ax.set_xlim([0.8, 1.0])
    for i, (bar, val) in enumerate(zip(bars, values)):
        ax.text(val - 0.02, i, f"{val:.4f}", ha="right", va="center", 
                fontweight="bold", color="white")
    ax.grid(True, alpha=0.3, axis="x")
    
    # 8. Training Configuration Summary
    ax = axes[2, 1]
    config_text = f"""Configuration Summary:
    
Baseline Epochs: {baseline_row.get('epochs', 'N/A')}
Baseline Batch Size: {baseline_row.get('batch_size', 'N/A')}
Bottleneck Dim: {baseline_row.get('bottleneck_dim', 'N/A')}
Network Depth: {baseline_row.get('depth', 'N/A')}
Dropout Rate: {baseline_row.get('dropout_rate', 'N/A')}
Activation: {baseline_row.get('activation', 'N/A')}"""
    ax.text(0.1, 0.5, config_text, fontsize=11, verticalalignment="center",
            family="monospace", transform=ax.transAxes)
    ax.set_title("Baseline Configuration")
    ax.axis("off")
    
    plt.tight_layout()
    output_path = results_dir / "ablation_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"✓ Comparison chart saved: {output_path}")
    plt.close()


def plot_metric_heatmap(df: pd.DataFrame, results_dir: Path):
    """Create heatmaps showing relationships between components and metrics."""
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle(f"Ablation Study: Metric Heatmaps - Mean Across Seeds: {SEEDS_STR}", 
                 fontsize=14, fontweight="bold")
    
    # Prepare data for heatmaps
    # We'll create a matrix of F1 vs ROC-AUC for different components
    
    # 1. Bottleneck dimension heatmap
    ax = axes[0]
    bottleneck_df = df[df["category"] == "bottleneck_dimension"].copy()
    bottleneck_df = bottleneck_df.sort_values("bottleneck_dim")
    
    heatmap_data = bottleneck_df[["bottleneck_dim", "f1", "precision", "recall", "roc_auc"]].set_index("bottleneck_dim")
    sns.heatmap(heatmap_data.T, annot=True, fmt=".4f", cmap="RdYlGn", 
                vmin=0.8, vmax=1.0, ax=ax, cbar_kws={"label": "Score"})
    ax.set_title("Metrics by Bottleneck Dimension (Mean Across Seeds)")
    ax.set_xlabel("Bottleneck Dimension")
    ax.set_ylabel("Metric")
    
    # 2. Depth heatmap
    ax = axes[1]
    depth_df = df[df["category"] == "network_depth"].copy()
    depth_df = depth_df.sort_values("depth")
    
    heatmap_data = depth_df[["depth", "f1", "precision", "recall", "roc_auc"]].set_index("depth")
    sns.heatmap(heatmap_data.T, annot=True, fmt=".4f", cmap="RdYlGn", 
                vmin=0.8, vmax=1.0, ax=ax, cbar_kws={"label": "Score"})
    ax.set_title("Metrics by Network Depth (Mean Across Seeds)")
    ax.set_xlabel("Network Depth (# Hidden Layers)")
    ax.set_ylabel("Metric")
    
    plt.tight_layout()
    output_path = results_dir / "ablation_heatmap.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"✓ Heatmap saved: {output_path}")
    plt.close()


def generate_analysis_report(results: Dict[str, Any], df: pd.DataFrame, 
                            results_dir: Path):
    """Generate a detailed text analysis report."""
    
    report = []
    report.append("=" * 80)
    report.append("ABLATION STUDY ANALYSIS REPORT")
    report.append("=" * 80)
    report.append(f"Timestamp: {results.get('timestamp', 'Unknown')}")
    report.append(f"Random Seeds: {SEEDS_STR}\n")
    
    # Experimental Setup
    report.append("EXPERIMENTAL SETUP")
    report.append("-" * 80)
    report.append(f"  Random Seeds: {SEEDS_STR}")
    report.append(f"  Purpose: Systematic ablation study of autoencoder components")
    report.append(f"  Components tested: bottleneck dimension, network depth, dropout rate,")
    report.append(f"                     activation function, batch size, epoch budget, learning rate")
    report.append(f"                     and noise scale sensitivity\n")
    
    # Data Summary
    report.append("DATA SUMMARY")
    report.append("-" * 80)
    data_summary = results.get("data_summary", {})
    report.append(f"  Total features: {data_summary.get('n_features', 'N/A')}")
    report.append(f"  Training samples: {data_summary.get('n_train', 'N/A')} (normal only)")
    report.append(f"  Test samples: {data_summary.get('n_test', 'N/A')}")
    report.append(f"  Normal test samples: {data_summary.get('n_normal', 'N/A')}")
    report.append(f"  Abnormal test samples: {data_summary.get('n_abnormal', 'N/A')}\n")
    
    # Baseline Performance
    report.append("BASELINE PERFORMANCE")
    report.append("-" * 80)
    baseline = results["baseline"]

    # Get mean results across the five seeds
    baseline_runs = baseline["all_runs"]
    baseline_f1_values = [run["f1"] for run in baseline_runs]
    baseline_roc_auc_values = [run["roc_auc"] for run in baseline_runs]
    baseline_accuracy_values = [run["accuracy"] for run in baseline_runs]
    baseline_precision_values = [run["precision"] for run in baseline_runs]
    baseline_recall_values = [run["recall"] for run in baseline_runs]
    baseline_specificity_values = [run["specificity"] for run in baseline_runs]

    baseline_f1 = np.mean(baseline_f1_values)
    baseline_roc_auc = np.mean(baseline_roc_auc_values)
    baseline_accuracy = np.mean(baseline_accuracy_values)
    baseline_precision = np.mean(baseline_precision_values)
    baseline_recall = np.mean(baseline_recall_values)
    baseline_specificity = np.mean(baseline_specificity_values)
    
    report.append(f"  Configuration:")
    for key, val in baseline["config"].items():
        report.append(f"    {key}: {val}")
    report.append(f"\n  Metrics (Mean ± SD across 5 seeds):")
    report.append(f"    F1 Score:  {baseline_f1:.4f} ± {np.std(baseline_f1_values, ddof=1):.4f}")
    report.append(f"    ROC-AUC:   {baseline_roc_auc:.4f} ± {np.std(baseline_roc_auc_values, ddof=1):.4f}")
    report.append(f"    Accuracy:  {baseline_accuracy:.4f} ± {np.std(baseline_accuracy_values, ddof=1):.4f}")
    report.append(f"    Precision: {baseline_precision:.4f} ± {np.std(baseline_precision_values, ddof=1):.4f}")
    report.append(f"    Recall:    {baseline_recall:.4f} ± {np.std(baseline_recall_values, ddof=1):.4f}")
    report.append(f"    Specificity: {baseline_specificity:.4f} ± {np.std(baseline_specificity_values, ddof=1):.4f}\n")
    
    # Component Analysis
    report.append("COMPONENT IMPACT ANALYSIS")
    report.append("-" * 80)
    
    for category, variants in results["variations"].items():
        report.append(f"\n{category.upper().replace('_', ' ')}")
        report.append("  " + "-" * 76)
        
        # Create comparison table with results for each seed
        for variant_name, variant_info in variants.items():
            if "all_runs" in variant_info and variant_info["all_runs"]:
                # Group results by seed
                report.append(f"  {variant_name}")
                for run in variant_info["all_runs"]:
                    seed = run.get("seed", "?")
                    f1 = run.get("f1", 0)
                    roc_auc = run.get("roc_auc", 0)
                    report.append(f"    Seed {seed}: F1={f1:.4f} ROC-AUC={roc_auc:.4f}")

                f1_values = [run.get("f1", 0) for run in variant_info["all_runs"]]
                roc_values = [run.get("roc_auc", 0) for run in variant_info["all_runs"]]
                report.append(f"    Mean: F1={np.mean(f1_values):.4f} ± {np.std(f1_values, ddof=1):.4f} "
                              f"ROC-AUC={np.mean(roc_values):.4f} ± {np.std(roc_values, ddof=1):.4f}")
        
        # Find and highlight best variant overall (using mean F1 to match KEY FINDINGS)
        best_mean_f1 = -1
        best_variant_name = None
        for variant_name, variant_info in variants.items():
            if "all_runs" in variant_info:
                # Calculate mean F1 across all runs
                f1_values = [run.get("f1", 0) for run in variant_info["all_runs"]]
                if f1_values:
                    mean_f1 = np.mean(f1_values)
                    if mean_f1 > best_mean_f1:
                        best_mean_f1 = mean_f1
                        best_variant_name = variant_name
        
        if best_variant_name:
            improvement = (best_mean_f1 - baseline_f1) / baseline_f1 * 100
            if category == "noise_scale":
                report.append(f"  ✓ Highest: {best_variant_name:27} Mean F1={best_mean_f1:.4f}")
            else:
                report.append(f"  ✓ Best: {best_variant_name:30} Mean F1={best_mean_f1:.4f} ({improvement:+.2f}%)")
    
    # Key Findings
    report.append("\n\n" + "=" * 80)
    report.append("KEY FINDINGS & RECOMMENDATIONS")
    report.append("=" * 80)
    
    # Best overall variant
    non_baseline = df[(df["variant"] != "baseline") & (df["category"] != "noise_scale")].copy()
    best_variant_idx = non_baseline["f1"].idxmax()
    best_variant = non_baseline.loc[best_variant_idx]
    baseline_row = df[df["variant"] == "baseline"].iloc[0]
    
    report.append(f"\nBest Overall Configuration:")
    report.append(f"  Variant: {best_variant['variant']}")
    report.append(f"  Category: {best_variant['category']}")
    
    for key in best_variant.index:
        if key not in ["variant", "category", "n_features_used", "feature_subset_pct"] and not key.startswith(("f1", "roc_auc", "accuracy", "precision", "recall", "specificity", "tp", "fp", "fn", "tn", "epochs_run", "final_")):
            value = best_variant[key]
            # Use baseline value if NaN
            if pd.isna(value):
                value = baseline_row[key]
            report.append(f"  {key}: {value}")

    report.append(f"  Mean F1 Score: {best_variant['f1']:.4f}")
    report.append(f"  Mean ROC-AUC: {best_variant['roc_auc']:.4f}")
    improvement = (best_variant['f1'] - baseline_f1) / baseline_f1 * 100
    report.append(f"  Improvement over baseline: {improvement:+.2f}%\n")
    
    # Baseline seed analysis
    report.append("Baseline Performance by Seed:")
    if "all_runs" in results["baseline"]:
        for run in results["baseline"]["all_runs"]:
            report.append(f"  Seed {run['seed']}: F1={run['f1']:.4f}, ROC-AUC={run['roc_auc']:.4f}")
    
    # Component rankings
    report.append("\nComponent Sensitivity Ranking (by mean F1 improvement):")
    for category in results["variations"].keys():
        if category == "noise_scale":
            continue
        cat_df = df[df["category"] == category]
        max_f1 = cat_df["f1"].max()
        improvement = (max_f1 - baseline_f1) / baseline_f1 * 100
        report.append(f"  {category:25} {improvement:+.2f}%")

    # Noise scale sensitivity
    if "noise_scale" in results["variations"]:
        report.append("\nNoise Scale Sensitivity:")
        report.append("  Higher noise scales represent stronger synthetic anomalies and are")
        report.append("  treated as a sensitivity analysis rather than model selection.")
    
    # Recommendations
    report.append("\n\nRecommendations:")
    report.append("  1. Compare configurations using mean performance across all seeds")
    report.append("  2. Balance F1, Precision, and Recall based on use case requirements")
    report.append("  3. Treat noise scale as anomaly severity rather than model tuning")
    report.append("  4. Avoid selecting configurations based on a single random seed")
    
    # Write report with UTF-8 encoding to handle special characters (✓, etc.)
    report_text = "\n".join(report)
    output_path = results_dir / "ablation_analysis_report.txt"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    
    print(f"✓ Analysis report saved: {output_path}")
    
    # Also print to console
    print("\n" + report_text)


def analyze_ablation_study(
    results_path: Path = None,
    results_dir: Path = None
):
    """Main analysis function."""
    
    if results_dir is None:
        results_dir = Path("results")
    
    if results_path is None:
        results_path = results_dir / "ablation_study_revised.json"
    
    print("\n" + "█" * 80)
    print("█ ABLATION STUDY ANALYSIS")
    print("█" * 80)
    
    if not results_path.exists():
        print(f"\n✗ Results file not found: {results_path}")
        print("  Run ablation_study.py first to generate results.")
        return
    
    print(f"\n[1/4] Loading results from {results_path}...")
    results = load_results(results_path)
    
    print("[2/4] Extracting data...")
    df = extract_comparison_data(results)
    print(f"  ✓ Loaded {len(df)} variants (including baseline)")
    
    print("[3/4] Generating visualizations...")
    plot_component_comparison(df, results_dir)
    plot_metric_heatmap(df, results_dir)
    
    print("[4/4] Generating analysis report...")
    generate_analysis_report(results, df, results_dir)
    
    print("\n" + "█" * 80)
    print("█ Analysis complete!")
    print("█" * 80)
    
    return df, results


if __name__ == "__main__":
    analyze_ablation_study()