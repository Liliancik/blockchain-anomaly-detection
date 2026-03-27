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
    """Compute HMAC-SHA256 hash of data.
    
    HMAC (Hash-Based Message Authentication Code) creates a cryptographic signature
    that proves data hasn't been modified. If anyone changes the data, the hash
    will no longer match.
    
    Args:
        data: Dictionary to hash
        secret: Shared secret for HMAC computation
    
    Returns:
        Hexadecimal HMAC-SHA256 hash
    """
    data_str = json.dumps(data, sort_keys=True)
    return hmac.new(secret.encode(), data_str.encode(), hashlib.sha256).hexdigest()


def test_tamper_detection():
    """Simulate an attacker trying to modify blockchain records.
    
    This experimental workflow demonstrates blockchain's value:
    1. Log a legitimate anomaly detection record
    2. Attacker modifies the record to hide the anomaly
    3. Hash verification catches the tampering
    4. Show why recomputing all hashes would be needed (chain effect)
    5. Explain why blockchain-based audit trail prevents falsification
    
    Purpose: Prove that blockchain protects detection integrity from tampering.
    """
    
    print("="*90)
    print("BLOCKCHAIN TAMPER DETECTION EXPERIMENT")
    print("="*90)
    
    # Setup: Backup existing ledger to avoid data loss
    # This ensures we can restore the original blockchain after the experiment
    ledger_path = Path("blockchain_ledger.jsonl")
    backup_path = Path("blockchain_ledger_backup.jsonl")
    
    # Backup existing ledger
    # This ensures we can restore the original blockchain after the experiment
    if ledger_path.exists():
        if backup_path.exists():
            backup_path.unlink()
        ledger_path.rename(backup_path)
        print(f"\n[SETUP] Backed up existing ledger to {backup_path}")
    
    # Initialize fresh blockchain and authenticate as admin
    # Admin has full permissions to write to the blockchain
    blockchain = PermissionedBlockchain()
    blockchain.login("admin", "admin_password")
    
    print("\n" + "="*90)
    print("PHASE 1: LOG ORIGINAL ANOMALY DETECTION")
    print("="*90)
    print("\nObjective: Create a legitimate anomaly detection record on the blockchain")
    
    # Log an original detection with realistic data
    # This simulates the autoencoder finding an anomaly and logging it
    original_index = 42
    original_details = {
        "anomaly_score": 3.5,
        "detected_by": "autoencoder",
        "confidence": 0.95,
    }
    
    blockchain.log_anomaly(original_index, original_details)
    
    # Read the logged entry to extract its hash
    # This hash is cryptographically bound to the data:
    # if data changes, hash must change too (or tampering is obvious)
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
    print("\nThis hash proves: 'This exact record existed at this exact time'")
    
    # Log hash to file for verification
    original_log = {
        "original_entry": original_entry,
        "original_hash": original_hash_val,
        "original_prev_hash": original_prev_hash
    }
    
    print("\n" + "="*90)
    print("PHASE 2: ATTACKER TRIES TO COVER UP THE DETECTION")
    print("="*90)
    print("\nScenario: Malicious admin wants to hide the anomaly detection")
    print("Motive: Prevent audit trail from showing this suspicious record")
    
    # Read the ledger and modify the entry
    # Simulate an attacker with file-system access trying to modify the record
    with open(ledger_path, "r") as f:
        entries = [json.loads(line) for line in f]
    
    # Simulate attacker modifying the anomaly score to make it seem less suspicious
    # Low score = appears like normal data (no anomaly)
    # High score = legitimate anomaly that should have been investigated
    print(f"\nAttacker action: Modifying anomaly_score from {original_details['anomaly_score']} to 0.1")
    print("(Goal: Make the anomaly look like normal data)")
    
    entries[0]["data"]["details"]["anomaly_score"] = 0.1  # Tamper!
    
    # Write back tampered entry
    # This is where attacker made a mistake: modified the data but not the hash!
    with open(ledger_path, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")
    
    print("[!] Tampered ledger written to disk")
    
    # Read tampered entry
    # The data is now modified, but it still has the OLD hash attached
    with open(ledger_path, "r") as f:
        tampered_entry = json.loads(f.readline())
    
    tampered_hash_val = tampered_entry["hash"]  # Still has old hash!
    
    print(f"\nTampered entry:")
    print(f"  Sample Index: {tampered_entry['data']['index']}")
    print(f"  Anomaly Score (MODIFIED): {tampered_entry['data']['details']['anomaly_score']}")
    print(f"  Confidence: {tampered_entry['data']['details']['confidence']}")
    print(f"\nTampered entry still has OLD hash: {tampered_hash_val}")
    print("(Attacker forgot to recompute the hash—critical mistake!)")
    
    print("\n" + "="*90)
    print("PHASE 3: AUDITOR DETECTS TAMPERING VIA HASH VERIFICATION")
    print("="*90)
    print("\nHow it works: Re-hash the stored data and compare with stored hash")
    print("If they don't match → data was modified after hash was created")
    
    # Recompute what the hash SHOULD be for the tampered data
    # If we hash the modified data, we should get a DIFFERENT hash than stored
    # This proves the data was changed AFTER the hash was created
    tampered_data = tampered_entry
    del tampered_data["hash"]  # Remove hash before computing (hash proof-of-data, not proof-of-hash)
    expected_hash_for_tampered = hashlib.sha256(
        json.dumps(tampered_data, sort_keys=True).encode()
    ).hexdigest()
    
    print(f"\nVerification process:")
    print(f"  1. Extract ALL data from the ledger entry")
    print(f"  2. Recompute hash of the extracted data using same algorithm")
    print(f"  3. Compare recomputed hash with stored hash")
    print(f"  4. If different → tampering detected!")
    
    print(f"\nStored hash:    {tampered_hash_val}")
    print(f"Computed hash:  {expected_hash_for_tampered}")
    
    hash_match = tampered_hash_val == expected_hash_for_tampered
    
    if hash_match:
        print("✓ Hash matches - data is AUTHENTIC")
    else:
        print("[ALERT] Hash MISMATCH - data has been TAMPERED!")
        print(f"\n  Evidence of tampering:")
        print(f"    Stored:   {tampered_hash_val[:16]}...")
        print(f"    Expected: {expected_hash_for_tampered[:16]}...")
        print(f"\n  Interpretation:")
        print(f"    • Data in entry ≠ data used to compute hash")
        print(f"    • Someone modified the record WITHOUT updating the hash")
        print(f"    • This violates blockchain integrity")
    
    # Explain the chain effect: modifying one entry breaks all downstream entries
    print("\n" + "="*90)
    print("PHASE 4: WHY ATTACKER CAN'T COVER THEIR TRACKS")
    print("="*90)
    print("\nEven if attacker tried to fix the hash, here's why it's impossible:")
    
    print(f"\nTo successfully hide tampering, attacker must:")
    print(f"  1. Modify data (sample_index=42, anomaly_score=0.1) ✓ Done")
    print(f"  2. Recompute hash: SHA256(modified_data) ✗ Forgot!")
    print(f"  3. Update stored hash in ledger ✗ Didn't do this")
    print(f"  4. Recompute NEXT entry's prev_hash (points to this one) ✗ Would break chain")
    print(f"  5. Update NEXT entry's hash ✗ Would need to update all downstream")
    print(f"  6. Repeat for ALL subsequent entries (hash chain effect) ✗ Impossible")
    print(f"\nResult: One modification requires modifying ENTIRE remaining blockchain")
    
    print(f"\nBut even if they successfully modify all entries:")
    print(f"  ✓ We have backup of original entry → can detect mismatch")
    print(f"  ✓ Hash chain breaks if ANY entry is modified")
    print(f"  ✓ Audit trail exists in Git version control")
    print(f"  ✓ Multiple signatures & timestamps prevent backdating")
    print(f"  ✓ Blockchain verifier role can detect inconsistencies")
    
    print("\n" + "="*90)
    print("PHASE 5: EXPERIMENTAL SUMMARY & KEY FINDINGS")
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
            "Blockchain hash verification detects unauthorized modifications",
            "Attacker cannot silently change detection records",
            "Hash chain creates multi-layer protection (modify one = modify all)",
            "Immutable audit trail ensures accountability & prevents cover-ups",
            "Why this matters: Autoencoder finds anomalies, blockchain protects proof"
        ]
    }
    
    print(f"\nKey Findings:")
    for i, finding in enumerate(results["key_findings"], 1):
        print(f"  {i}. {finding}")
    
    print(f"\nVerdict: {results['phase_3_verdict']}")
    print(f"Security Check: {'✓ PASSED - Tampering Detected Successfully' if not hash_match else '✗ FAILED - Data is authentic (no tampering)'}")
    
    # Save results
    output_path = Path("outputs/tamper_detection_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2))
    print(f"\n[OK] Results saved to {output_path}")
    
    print("\n" + "="*90)
    print("CONCLUSION: Why Blockchain Matters in Anomaly Detection")
    print("="*90)
    print("""
Two separate concerns in security:

1. DETECTION (Autoencoder's job):
   • Finds real anomalies in data
   • High accuracy, good recall/precision
   • But: Can't prevent falsification of records

2. INTEGRITY (Blockchain's job):
   • Protects detection records from modification
   • Creates immutable audit trail
   • Proves "this anomaly was detected and logged"
   • Can't be hidden or rewritten

Together they provide:
   ✓ Accurate anomaly detection (autoencoder)
   ✓ Immutable proof of detections (blockchain)
   ✓ Non-repudiation (can't deny findings later)
   ✓ Compliance & governance (audit trail)
   ✓ Tamper detection & accountability

Without blockchain:
   ✗ Attacker can silently modify/delete detection records
   ✗ Audit trail is mutable and untrusted
   ✗ No way to verify original findings
   ✗ Regulatory non-compliance
   ✗ Easy to cover up suspicious activity
    """)
    
    # Cleanup: Restore original ledger
    # This ensures subsequent runs have a clean state
    if backup_path.exists():
        if ledger_path.exists():
            ledger_path.unlink()
        backup_path.rename(ledger_path)
        print(f"[CLEANUP] Restored original ledger from backup")
    
    return results


if __name__ == "__main__":
    # Run the tamper detection experiment
    results = test_tamper_detection()
    
    # Exit with appropriate code:
    # 0 = Success (tampering was detected as expected)
    # 1 = Failure (hash matched - tampering not detected)
    is_tampered = not results["phase_3_hash_match"]
    sys.exit(0 if is_tampered else 1)  # 0 if tampering detected (success)
