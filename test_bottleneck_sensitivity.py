"""
Test bottleneck dimension sensitivity with simplified regularization (2 dropout layers, patience=20)
"""
import subprocess
import json
import re
from pathlib import Path

results = {}
bottleneck_dims = [2, 4, 8]

for bottleneck in bottleneck_dims:
    print(f"\n{'='*60}")
    print(f"Testing bottleneck_dim = {bottleneck}")
    print('='*60)
    
    # Update run_pipeline.py with new bottleneck
    script_path = Path("scripts/run_pipeline.py")
    content = script_path.read_text()
    
    # Replace bottleneck_dim
    updated = re.sub(
        r'bottleneck_dim=\d+',
        f'bottleneck_dim={bottleneck}',
        content
    )
    script_path.write_text(updated)
    
    # Run pipeline
    result = subprocess.run(
        [".venv/Scripts/python.exe", "scripts/run_pipeline.py"],
        capture_output=True,
        text=True,
        timeout=300
    )
    
    output = result.stdout + result.stderr
    
    # Extract metrics
    metrics = {}
    try:
        if "Precision:" in output:
            metrics['test_precision'] = float(re.search(r'Precision: ([\d.]+)', output).group(1))
            metrics['test_recall'] = float(re.search(r'Recall: ([\d.]+)', output).group(1))
            metrics['test_f1'] = float(re.search(r'F1: ([\d.]+)', output).group(1))
            metrics['test_accuracy'] = float(re.search(r'Autoencoder accuracy: ([\d.]+)', output).group(1))
            
            # Get detected count
            detected = re.search(r'Detected: (\d+)', output)
            injected = re.search(r'Injected: (\d+)', output)
            if detected and injected:
                metrics['detected'] = int(detected.group(1))
                metrics['injected'] = int(injected.group(1))
        
        results[bottleneck] = metrics
        print(f"Results: {metrics}")
    except Exception as e:
        print(f"Error parsing output: {e}")
        results[bottleneck] = {"error": str(e)}

print(f"\n\n{'='*60}")
print("BOTTLENECK SENSITIVITY COMPARISON")
print('='*60)

# Print comparison table
print(f"\n{'Bottleneck':<15} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Accuracy':<12} {'Detected':<12}")
print('-' * 75)
for bn in bottleneck_dims:
    if bn in results and 'error' not in results[bn]:
        m = results[bn]
        print(f"{bn:<15} {m.get('test_precision', 0):<12.4f} {m.get('test_recall', 0):<12.4f} {m.get('test_f1', 0):<12.4f} {m.get('test_accuracy', 0):<12.4f} {m.get('detected', 0):<12}")

# Save results
with open("bottleneck_sensitivity_results.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\n\nResults saved to bottleneck_sensitivity_results.json")
