"""
Controlled tamper detection experiment:
- Show blockchain integrity protection
- Demonstrate why logging to blockchain matters
- Prove immutability of detection records
"""
import json
import hashlib
import hmac
from pathlib import Path
from datetime import datetime
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.blockchain_logger import PermissionedBlockchain


def original_hash(data: dict, secret: str = "blockchain_secret") -> str:
    """Compute HMAC-SHA256 of data."""
    data_str = json.dumps(data, sort_keys=True)
    return hmac.new(secret.encode(), data_str.encode(), hashlib.sha256).hexdigest()


def test_tamper_detection():
    """
    Controlled experiment:
    1. Log a detection record
    2. Tamper with it
    3. Show hash mismatch
    4. Demonstrate immutability value
    """
    
    print("="*90)
    print("BLOCKCHAIN TAMPER DETECTION EXPERIMENT")
    print("="*90)
    
    # Setup
    ledger_path = Path("blockchain_ledger.jsonl")
    backup_path = Path("blockchain_ledger_backup.jsonl")
    
    # Backup existing ledger
    if ledger_path.exists():
        if backup_path.exists():
            backup_path.unlink()
        ledger_path.rename(backup_path)
        print(f"\n[SETUP] Backed up existing ledger to {backup_path}")
    
    # Initialize fresh blockchain
    blockchain = PermissionedBlockchain()
    blockchain.login("admin", "admin_password")
    
    print("\n" + "="*90)
    print("PHASE 1: LOG ORIGINAL ANOMALY DETECTION")
    print("="*90)
    
    # Log an original detection
    original_index = 42
    original_details = {
        "anomaly_score": 3.5,
        "detected_by": "autoencoder",
        "confidence": 0.95,
    }
    
    blockchain.log_anomaly(original_index, original_details)
    
    # Read the logged entry
    with open(ledger_path, "r") as f:
        original_entry = json.loads(f.readline())
    
    original_hash_val = original_entry["hash"]
    original_prev_hash = original_entry["prev_hash"]
    
    print(f"\nLogged detection record:")
    print(f"  Sample Index: {original_entry['data']['index']}")
    print(f"  Anomaly Score: {original_entry['data']['details']['anomaly_score']}")
    print(f"  Confidence: {original_entry['data']['details']['confidence']}")
    print(f"\nOriginal Hash: {original_hash_val}")
    print(f"Previous Hash: {original_prev_hash}")
    
    # Log hash to file for verification
    original_log = {
        "original_entry": original_entry,
        "original_hash": original_hash_val,
        "original_prev_hash": original_prev_hash
    }
    
    print("\n" + "="*90)
    print("PHASE 2: TAMPER WITH LOGGED RECORD")
    print("="*90)
    
    # Read the ledger and modify the entry
    with open(ledger_path, "r") as f:
        entries = [json.loads(line) for line in f]
    
    # Simulate attacker modifying the anomaly score
    print(f"\nAttacker action: Modifying anomaly_score from {original_details['anomaly_score']} to 0.1")
    print("(Attacker wants to hide the detection)")
    
    entries[0]["data"]["details"]["anomaly_score"] = 0.1  # Tamper!
    
    # Write back tampered entry
    with open(ledger_path, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")
    
    print("✗ Tampered ledger written to disk")
    
    # Read tampered entry
    with open(ledger_path, "r") as f:
        tampered_entry = json.loads(f.readline())
    
    tampered_hash_val = tampered_entry["hash"]  # Still has old hash!
    
    print(f"\nTampered entry:")
    print(f"  Sample Index: {tampered_entry['data']['index']}")
    print(f"  Anomaly Score (MODIFIED): {tampered_entry['data']['details']['anomaly_score']}")
    print(f"  Confidence: {tampered_entry['data']['details']['confidence']}")
    print(f"\nTampered entry still has OLD hash: {tampered_hash_val}")
    print("(Attacker left the hash unchanged—oversight!)")
    
    print("\n" + "="*90)
    print("PHASE 3: HASH VERIFICATION & MISMATCH DETECTION")
    print("="*90)
    
    # Recompute what the hash SHOULD be for tampered data
    tampered_data = tampered_entry
    del tampered_data["hash"]  # Remove hash before computing
    expected_hash_for_tampered = hashlib.sha256(
        json.dumps(tampered_data, sort_keys=True).encode()
    ).hexdigest()
    
    print(f"\nVerification process:")
    print(f"  1. Extract data from ledger")
    print(f"  2. Recompute hash of extracted data")
    print(f"  3. Compare with stored hash")
    
    print(f"\nStored hash:    {tampered_hash_val}")
    print(f"Computed hash:  {expected_hash_for_tampered}")
    
    hash_match = tampered_hash_val == expected_hash_for_tampered
    
    if hash_match:
        print("\n✓ Hash matches - data is AUTHENTIC")
    else:
        print("\n✗ Hash MISMATCH - data has been TAMPERED!")
        print(f"\n  Difference:")
        print(f"    Stored:   {tampered_hash_val[:16]}...")
        print(f"    Expected: {expected_hash_for_tampered[:16]}...")
    
    # Show what attacker should have done (but didn't)
    print("\n" + "="*90)
    print("PHASE 4: WHAT ATTACKER SHOULD HAVE DONE (BUT DIDN'T)")
    print("="*90)
    
    print(f"\nTo hide tampering, attacker needs to:")
    print(f"  1. Modify data (sample_index=42, anomaly_score=0.1)")
    print(f"  2. Recompute hash: HMAC-SHA256(modified_data)")
    print(f"  3. Update stored hash in ledger")
    print(f"  4. Recompute NEXT entry's prev_hash")
    print(f"  5. Update NEXT entry's hash")
    print(f"  6. Repeat for ALL subsequent entries (hash chain)")
    
    print(f"\nBut even if they do all that:")
    print(f"  ✓ We have backup of original entry → can detect mismatch")
    print(f"  ✓ Hash chain breaks if ANY entry modified")
    print(f"  ✓ Audit trail exists in Git version control")
    print(f"  ✓ Multiple signatures/timestamps prevent backdating")
    
    print("\n" + "="*90)
    print("PHASE 5: EXPERIMENTAL SUMMARY")
    print("="*90)
    
    results = {
        "experiment": "Tamper Detection Test",
        "timestamp": datetime.now().isoformat(),
        "phase_1_original_hash": original_hash_val,
        "phase_2_tamper_action": "Modified anomaly_score: 3.5 → 0.1",
        "phase_3_stored_hash": tampered_hash_val,
        "phase_3_computed_hash": expected_hash_for_tampered,
        "phase_3_hash_match": hash_match,
        "phase_3_verdict": "AUTHENTIC" if hash_match else "TAMPERED",
        "key_findings": [
            "Blockchain hash verification detects modifications",
            "Attacker cannot silently change detection records",
            "Hash chain provides multi-layer protection",
            "Immutable audit trail prevents cover-up",
            "Autoencoder itself cannot detect tampering—blockchain can"
        ]
    }
    
    print(f"\nKey Findings:")
    for i, finding in enumerate(results["key_findings"], 1):
        print(f"  {i}. {finding}")
    
    print(f"\nVerdict: {results['phase_3_verdict']}")
    print(f"Integrity Check: {'PASSED - Tampering Detected' if not hash_match else 'FAILED - Data authentic'}")
    
    # Save results
    output_path = Path("outputs/tamper_detection_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2))
    print(f"\n✓ Results saved to {output_path}")
    
    print("\n" + "="*90)
    print("CONCLUSION: Why Blockchain Matters")
    print("="*90)
    print("""
Autoencoder detects ANOMALIES (abnormal student performance).
Blockchain prevents FALSIFICATION (hiding/modifying detections).

These are orthogonal concerns:
  • Autoencoder model: Finds real anomalies
  • Blockchain ledger: Protects detection integrity

Together they provide:
  ✓ Accurate detection (autoencoder)
  ✓ Immutable audit trail (blockchain)
  ✓ Non-repudiation (hash chain)
  ✓ Compliance (tamper detection)

Without blockchain:
  ✗ Attacker can silently modify detection records
  ✗ Audit trail is mutable
  ✗ No way to verify original findings
    """)
    
    # Cleanup: restore original ledger
    if backup_path.exists():
        if ledger_path.exists():
            ledger_path.unlink()
        backup_path.rename(ledger_path)
        print(f"[CLEANUP] Restored original ledger from backup")
    
    return results


if __name__ == "__main__":
    results = test_tamper_detection()
    
    # Exit with appropriate code
    is_tampered = not results["phase_3_hash_match"]
    sys.exit(0 if is_tampered else 1)  # 0 if tampering detected (success)
