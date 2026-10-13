"""Controlled tamper detection experiment:

- Show blockchain integrity protection
- Demonstrate why logging to blockchain matters
- Test tamper detection across replicated validator ledgers
"""

import json
import shutil
from pathlib import Path
from datetime import datetime
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


from src.poa_blockchain import PoABlockchain


def test_tamper_detection():
    """Simulate an attacker trying to modify blockchain records.

    This experimental workflow demonstrates blockchain's value:
    1. Log a legitimate anomaly detection record
    2. Replicate the record across authorized validators
    3. Attacker modifies one validator's record
    4. Blockchain verification detects the tampering
    5. Validator comparison detects the inconsistent replica

    Purpose: Test whether the permissioned blockchain detects record tampering.
    """

    print("="*90)
    print("BLOCKCHAIN TAMPER DETECTION EXPERIMENT")
    print("="*90)


    # Setup: Use a separate ledger directory for the experiment
    # This ensures the pipeline blockchain is not modified
    ledger_dir = Path("results/poa_tamper_test")

    if ledger_dir.exists():
        shutil.rmtree(ledger_dir)


    # Initialize fresh permissioned PoA blockchain
    blockchain = PoABlockchain(
        ledger_dir=str(ledger_dir)
    )


    print("\n" + "="*90)
    print("PHASE 1: LOG ORIGINAL ANOMALY DETECTION")
    print("="*90)
    print("\nObjective: Create a legitimate anomaly detection record on the blockchain")


    # Log an original detection with realistic data
    # This simulates the autoencoder finding an anomaly and logging it
    original_index = 42

    original_details = {
        "sample_index": original_index,
        "anomaly_score": 3.5,
        "detected_by": "autoencoder",
        "confidence": 0.95,
    }


    # Get the authorized validator for the next block
    validator = blockchain.get_current_validator()

    original_block = blockchain.create_block(
        validator=validator,
        anomaly_data=original_details
    )


    print(f"\nLogged detection record:")
    print(f"  Sample Index: {original_block['data']['sample_index']}")
    print(f"  Anomaly Score: {original_block['data']['anomaly_score']}")
    print(f"  Confidence: {original_block['data']['confidence']}")
    print(f"  Validator: {original_block['validator']}")

    print(f"\nOriginal Hash: {original_block['hash']}")
    print(f"Previous Hash: {original_block['previous_hash']}")


    # Verify all validator copies before tampering
    initial_verification = blockchain.verify_all_validators()

    print(
        f"\nInitial blockchain verification: "
        f"{'VALID' if initial_verification else 'INVALID'}"
    )


    print("\n" + "="*90)
    print("PHASE 2: ATTACKER TRIES TO COVER UP THE DETECTION")
    print("="*90)

    print("\nScenario: Attacker tries to modify the stored anomaly detection")
    print("Motive: Make the suspicious record appear less anomalous")


    # Select one validator ledger to tamper with
    tampered_validator = "validator_2"

    ledger_path = (
        ledger_dir /
        f"{tampered_validator}_ledger.jsonl"
    )


    # Read the validator ledger and modify the entry
    # Simulate an attacker with file-system access trying to modify the record
    with open(ledger_path, "r") as f:
        entries = [json.loads(line) for line in f]


    # Simulate attacker modifying the anomaly score to make it seem less suspicious
    print(
        f"\nAttacker action: Modifying anomaly_score from "
        f"{original_details['anomaly_score']} to 0.1"
    )
    print("(Goal: Make the anomaly look like normal data)")


    entries[0]["data"]["anomaly_score"] = 0.1  # Tamper!


    # Write back tampered entry
    # The original hash and authentication tag are intentionally left unchanged
    with open(ledger_path, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")


    print("[!] Tampered validator ledger written to disk")


    # Read tampered entry
    # The data is modified, but the original cryptographic values remain
    with open(ledger_path, "r") as f:
        tampered_entry = json.loads(f.readline())


    print(f"\nTampered entry:")
    print(f"  Sample Index: {tampered_entry['data']['sample_index']}")
    print(
        f"  Anomaly Score (MODIFIED): "
        f"{tampered_entry['data']['anomaly_score']}"
    )
    print(f"  Confidence: {tampered_entry['data']['confidence']}")
    print(f"  Validator Ledger: {tampered_validator}")


    print("\n" + "="*90)
    print("PHASE 3: AUDITOR DETECTS TAMPERING VIA BLOCKCHAIN VERIFICATION")
    print("="*90)

    print(
        "\nHow it works: Verify the stored block hash, "
        "authentication tag, and hash-chain linkage"
    )
    print("If verification fails → the stored blockchain record was modified")


    # Verify each validator independently
    # The tampered validator should fail while untouched validators remain valid
    validator_results = {}

    for validator in blockchain.validators:

        valid = blockchain.verify_chain(validator)
        validator_results[validator] = valid

        print(
            f"\n{validator}: "
            f"{'VALID' if valid else 'TAMPERED'}"
        )


    tamper_detected = not validator_results[tampered_validator]


    if tamper_detected:

        print("\n[ALERT] Blockchain verification FAILED for modified validator!")
        print("\n  Evidence of tampering:")
        print(f"    Modified validator: {tampered_validator}")
        print(f"    Original anomaly score: 3.5")
        print(f"    Modified anomaly score: 0.1")

        print("\n  Interpretation:")
        print("    • Stored data was modified after the block was created")
        print("    • Original cryptographic values no longer match the record")
        print("    • Blockchain integrity verification detects the modification")

    else:

        print("\n✗ Tampering was not detected")


    print("\n" + "="*90)
    print("PHASE 4: VALIDATOR REPLICA CONSISTENCY")
    print("="*90)

    print(
        "\nThe permissioned blockchain stores replicated copies "
        "across three authorized validators."
    )

    print(
        "After modifying one validator ledger, the replicas "
        "should no longer agree."
    )


    # Verify both blockchain integrity and replica consistency
    replicas_valid = blockchain.verify_all_validators()


    if not replicas_valid:

        print("\n[ALERT] Validator replica inconsistency detected!")
        print(f"  Modified replica: {tampered_validator}")
        print("  Untouched replicas remain unchanged")
        print("  Validator ledgers are no longer identical")

    else:

        print("\n✗ Validator inconsistency was not detected")


    print("\n" + "="*90)
    print("PHASE 5: EXPERIMENTAL SUMMARY & KEY FINDINGS")
    print("="*90)


    results = {
        "experiment": "Permissioned PoA Blockchain Tamper Detection Test",
        "timestamp": datetime.now().isoformat(),
        "validators": blockchain.validators,
        "tampered_validator": tampered_validator,
        "original_anomaly_score": 3.5,
        "tampered_anomaly_score": 0.1,
        "initial_verification": initial_verification,
        "validator_verification": validator_results,
        "replica_consistency_after_tampering": replicas_valid,
        "tampering_detected": tamper_detected,
        "verdict": "TAMPERED" if tamper_detected else "NOT DETECTED",
        "key_findings": [
            "Three authorized validators maintain replicated blockchain ledgers",
            "Blockchain verification detects unauthorized record modification",
            "Untampered validator copies remain valid",
            "Replica comparison detects inconsistency between validator ledgers",
            "Autoencoder detects anomalies while blockchain protects their provenance",
        ]
    }


    print(f"\nKey Findings:")

    for i, finding in enumerate(results["key_findings"], 1):
        print(f"  {i}. {finding}")


    print(f"\nVerdict: {results['verdict']}")

    print(
        f"Security Check: "
        f"{'✓ PASSED - Tampering Detected Successfully' if tamper_detected else '✗ FAILED - Tampering Not Detected'}"
    )

    print(
        f"Replica Check: "
        f"{'✓ PASSED - Inconsistency Detected Successfully' if not replicas_valid else '✗ FAILED - Replicas Still Match'}"
    )


    # Save results
    output_path = Path("outputs/poa_tamper_detection_results.json")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_path.write_text(
        json.dumps(results, indent=2)
    )

    print(f"\n[OK] Results saved to {output_path}")


    print("\n" + "="*90)
    print("CONCLUSION: Blockchain Integrity in Anomaly Detection")
    print("="*90)

    print("""

Two separate concerns in the framework:

1. DETECTION (Autoencoder's job):
   • Finds anomalous patterns in student data
   • Produces anomaly scores and detection records

2. INTEGRITY (Blockchain's job):
   • Records detected anomalies
   • Replicates records across authorized validators
   • Provides cryptographic integrity verification
   • Detects inconsistency between validator copies

Controlled tampering test:

   Original anomaly score: 3.5
   Modified anomaly score: 0.1
   Modified validator: validator_2

The experiment tests whether unauthorized modification of one
stored anomaly record can be detected by blockchain verification
and comparison of the replicated validator ledgers.

""")


    return results


if __name__ == "__main__":

    # Run the tamper detection experiment
    results = test_tamper_detection()


    # Exit with appropriate code:
    # 0 = Success (tampering was detected as expected)
    # 1 = Failure (tampering was not detected)
    is_tampered = results["tampering_detected"]

    sys.exit(0 if is_tampered else 1)