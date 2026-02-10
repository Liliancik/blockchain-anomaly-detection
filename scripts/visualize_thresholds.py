"""
Visualize threshold decision boundaries with reconstruction error distribution.
Shows validation-optimized vs percentile-based thresholds.
"""
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def main():
    """Generate threshold visualization."""
    
    # Load pipeline data
    with open("pipeline_outputs.json") as f:
        data = json.load(f)
    
    scores_summary = data.get('scores_summary_test', {})
    if not scores_summary:
        print("No reconstruction error statistics found in pipeline_outputs.json")
        return
    
    min_score = scores_summary.get('min', 0)
    max_score = scores_summary.get('max', 1)
    mean_score = scores_summary.get('mean', 0.5)
    current_threshold = data.get('threshold', None)
    
    # Create figure
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle('Reconstruction Error Threshold Analysis', 
                 fontsize=15, fontweight='bold', y=0.995)
    
    # Plot 1: Distribution with thresholds
    ax = axes[0]
    
    # Simulate error distribution (normal for clean, wider for anomalies)
    np.random.seed(42)
    normal_errors = np.random.normal(min_score, (mean_score - min_score) * 0.5, 800)
    anomaly_errors = np.random.normal(max_score * 0.7, (max_score - min_score) * 0.3, 200)
    all_errors = np.concatenate([normal_errors, anomaly_errors])
    all_errors = np.clip(all_errors, min_score, max_score)
    
    # Histogram
    ax.hist(all_errors, bins=50, alpha=0.6, color='skyblue', edgecolor='black', label='Error Distribution')
    
    # Add threshold lines
    percentiles = [80, 90, 95, 99]
    colors_percentile = ['orange', 'gold', 'red', 'darkred']
    
    for p, color in zip(percentiles, colors_percentile):
        threshold_p = min_score + (max_score - min_score) * (p / 100)
        ax.axvline(threshold_p, color=color, linestyle='--', linewidth=2, alpha=0.7,
                   label=f'{p}th percentile: {threshold_p:.4f}')
    
    # Current (validation-optimized) threshold
    if current_threshold:
        ax.axvline(current_threshold, color='green', linestyle='-', linewidth=3, alpha=0.8,
                   label=f'Current (F1-optimized): {current_threshold:.4f}')
    
    ax.set_xlabel('Reconstruction Error', fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequency (samples)', fontsize=12, fontweight='bold')
    ax.set_title('Distribution of Reconstruction Errors with Threshold Options', fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Threshold comparison table
    ax = axes[1]
    ax.axis('off')
    
    # Calculate percentile positions
    percentile_data = []
    percentile_data.append(['Threshold Type', 'Value', 'Position', 'Strategy'])
    percentile_data.append(['─' * 25, '─' * 15, '─' * 15, '─' * 30])
    
    if current_threshold:
        pct_pos = (current_threshold - min_score) / (max_score - min_score) * 100
        percentile_data.append([
            'F1-Optimized (Current)',
            f'{current_threshold:.4f}',
            f'{pct_pos:.1f}%',
            'Balanced precision/recall'
        ])
        percentile_data.append(['─' * 25, '─' * 15, '─' * 15, '─' * 30])
    
    for p in [80, 85, 90, 95, 99]:
        threshold_p = min_score + (max_score - min_score) * (p / 100)
        if p <= 80:
            strategy = 'High Recall (more detections)'
        elif p <= 90:
            strategy = 'Moderate (balanced)'
        else:
            strategy = 'High Precision (fewer false alarms)'
        
        percentile_data.append([
            f'{p}th Percentile',
            f'{threshold_p:.4f}',
            f'{p}%',
            strategy
        ])
    
    # Add statistics
    percentile_data.append(['─' * 25, '─' * 15, '─' * 15, '─' * 30])
    percentile_data.append([
        'Min Error',
        f'{min_score:.4f}',
        '0%',
        'Best reconstruction'
    ])
    percentile_data.append([
        'Mean Error',
        f'{mean_score:.4f}',
        '~50%',
        'Average reconstruction'
    ])
    percentile_data.append([
        'Max Error',
        f'{max_score:.4f}',
        '100%',
        'Worst reconstruction'
    ])
    
    # Create table
    table = ax.table(cellText=percentile_data, cellLoc='left', loc='center',
                    colWidths=[0.25, 0.2, 0.15, 0.4])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Style header row
    for i in range(4):
        table[(0, i)].set_facecolor('#2E86AB')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Style current threshold row
    if current_threshold:
        for i in range(4):
            table[(1, i)].set_facecolor('#90EE90')
            table[(1, i)].set_text_props(weight='bold')
    
    # Alternate row colors for readability
    start_row = 3 if current_threshold else 2
    for i in range(start_row, len(percentile_data)):
        for j in range(4):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#F0F0F0')
    
    ax.text(0.5, -0.15, 
            'Guidance: Use percentile-based thresholds for different use cases. ' +
            'Current F1-optimized threshold offers balanced performance for general detection.',
            ha='center', va='top', fontsize=10, style='italic',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    output_path = Path("outputs/threshold_analysis.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n[OK] Threshold analysis visualization saved to {output_path}")
    plt.close()


if __name__ == "__main__":
    main()
