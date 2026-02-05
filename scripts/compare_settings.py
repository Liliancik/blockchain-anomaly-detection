"""
Create side-by-side comparison visualization of aggressive vs moderate detection settings.
"""
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Load the moderate results (already in pipeline_outputs.json)
moderate_results = json.load(open("pipeline_outputs.json"))

# Define aggressive results (from earlier test)
aggressive_results = {
    "frac": 0.1,
    "noise_scale": 10.0,
    "test_metrics": {
        "injected": 99,
        "detected": 101,
        "true_positives": 99,
        "precision": 0.9801980198019802,
        "recall": 1.0,
        "f1": 0.99,
        "roc_auc": 0.999966066035495,
        "accuracy": 0.9979838709677419
    }
}

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Anomaly Detection Performance: Aggressive vs Moderate Settings', 
             fontsize=15, fontweight='bold', y=0.995)

# Configuration info
ax = axes[0, 0]
ax.axis('off')
config_text = """AGGRESSIVE SETTING
━━━━━━━━━━━━━━━
Injection: 10%
Noise Scale: 10.0

Anomalies: Large/Obvious
Threshold: Unambiguous
Use Case: Best-case scenario
Real-world: Optimistic"""
ax.text(0.05, 0.95, config_text, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', family='monospace',
        bbox=dict(boxstyle='round', facecolor='#FFE5E5', alpha=0.8))

ax = axes[0, 1]
ax.axis('off')
config_text = """MODERATE SETTING (Recommended)
━━━━━━━━━━━━━━━━━━━━━━━━━━━
Injection: 2%
Noise Scale: 5.0

Anomalies: Subtle/Realistic
Threshold: Ambiguous cases
Use Case: Real-world scenario
Real-world: Realistic"""
ax.text(0.05, 0.95, config_text, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', family='monospace',
        bbox=dict(boxstyle='round', facecolor='#E5F5FF', alpha=0.8))

# Comparison table
ax = axes[0, 2]
ax.axis('off')

metrics = ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC']
aggressive_vals = [
    aggressive_results['test_metrics']['accuracy'],
    aggressive_results['test_metrics']['precision'],
    aggressive_results['test_metrics']['recall'],
    aggressive_results['test_metrics']['f1'],
    aggressive_results['test_metrics']['roc_auc']
]
moderate_vals = [
    moderate_results['test_metrics']['accuracy'],
    moderate_results['test_metrics']['precision'],
    moderate_results['test_metrics']['recall'],
    moderate_results['test_metrics']['f1'],
    moderate_results['test_metrics']['roc_auc']
]

table_data = [['Metric', 'Aggressive', 'Moderate']]
for metric, agg, mod in zip(metrics, aggressive_vals, moderate_vals):
    table_data.append([metric, f'{agg:.4f}', f'{mod:.4f}'])

table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                colWidths=[0.3, 0.35, 0.35])
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 2.5)

for i in range(3):
    table[(0, i)].set_facecolor('#2E86AB')
    table[(0, i)].set_text_props(weight='bold', color='white')

for i in range(1, len(table_data)):
    for j in range(3):
        if i % 2 == 0:
            table[(i, j)].set_facecolor('#F0F0F0')

ax.set_title('Performance Metrics Comparison', fontweight='bold', pad=10)

# Accuracy comparison
ax = axes[1, 0]
settings = ['Aggressive\n(10%, 10.0)', 'Moderate\n(2%, 5.0)']
accuracies = [aggressive_results['test_metrics']['accuracy'], 
              moderate_results['test_metrics']['accuracy']]
colors = ['#FF6B6B', '#4ECDC4']
bars = ax.bar(settings, accuracies, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
ax.set_ylabel('Accuracy', fontweight='bold', fontsize=11)
ax.set_ylim([0.95, 1.0])
ax.set_title('Accuracy Comparison', fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
for bar, acc in zip(bars, accuracies):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{acc:.4f}', ha='center', va='bottom', fontweight='bold')

# Precision vs Recall
ax = axes[1, 1]
x = np.arange(2)
width = 0.35
aggressive_prec = aggressive_results['test_metrics']['precision']
aggressive_rec = aggressive_results['test_metrics']['recall']
moderate_prec = moderate_results['test_metrics']['precision']
moderate_rec = moderate_results['test_metrics']['recall']

bars1 = ax.bar(x - width/2, [aggressive_prec, moderate_prec], width, label='Precision', 
               color='#2E86AB', alpha=0.8, edgecolor='black')
bars2 = ax.bar(x + width/2, [aggressive_rec, moderate_rec], width, label='Recall', 
               color='#A23B72', alpha=0.8, edgecolor='black')

ax.set_ylabel('Score', fontweight='bold', fontsize=11)
ax.set_title('Precision vs Recall', fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(settings)
ax.legend(fontsize=10)
ax.set_ylim([0, 1.1])
ax.grid(True, alpha=0.3, axis='y')

# ROC-AUC comparison
ax = axes[1, 2]
roc_aucs = [aggressive_results['test_metrics']['roc_auc'], 
            moderate_results['test_metrics']['roc_auc']]
bars = ax.bar(settings, roc_aucs, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
ax.set_ylabel('ROC-AUC Score', fontweight='bold', fontsize=11)
ax.set_ylim([0.9, 1.001])
ax.set_title('ROC-AUC Comparison', fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
for bar, auc in zip(bars, roc_aucs):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{auc:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=9)

plt.tight_layout()
output_path = Path("outputs/settings_comparison.png")
output_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✓ Comparison visualization saved to {output_path}")
plt.close()

# Print summary
print("\n" + "="*80)
print("SETTINGS COMPARISON SUMMARY")
print("="*80)
print(f"\nAGGRESSIVE (10% injection, 10.0 noise scale):")
print(f"  - Best for: Demonstrating upper-bound performance")
print(f"  - Accuracy: {aggressive_results['test_metrics']['accuracy']:.4f}")
print(f"  - Precision: {aggressive_results['test_metrics']['precision']:.4f}")
print(f"  - Recall: {aggressive_results['test_metrics']['recall']:.4f}")
print(f"  - ROC-AUC: {aggressive_results['test_metrics']['roc_auc']:.4f}")

print(f"\nMODERATE (2% injection, 5.0 noise scale) ⭐ RECOMMENDED")
print(f"  - Best for: Real-world applicability")
print(f"  - Accuracy: {moderate_results['test_metrics']['accuracy']:.4f}")
print(f"  - Precision: {moderate_results['test_metrics']['precision']:.4f}")
print(f"  - Recall: {moderate_results['test_metrics']['recall']:.4f}")
print(f"  - ROC-AUC: {moderate_results['test_metrics']['roc_auc']:.4f}")

print(f"\nKEY FINDING:")
print(f"  The moderate setting shows realistic performance on subtle anomalies")
print(f"  while maintaining strong ROC-AUC ({moderate_results['test_metrics']['roc_auc']:.4f})")
print(f"  and good accuracy ({moderate_results['test_metrics']['accuracy']:.4f}).")
print("="*80)
