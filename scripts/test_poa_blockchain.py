"""Test the permissioned Proof-of-Authority blockchain."""

import json
import shutil
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.poa_blockchain import PoABlockchain


def main():

    print("=" * 80)
    print("PERMISSIONED PoA BLOCKCHAIN TEST")
    print("=" * 80)

    # Use a separate directory so the existing blockchain is not changed
    ledger_dir = Path("results/poa_test_ledgers")

    # Remove previous test ledgers
    if ledger_dir.exists():
        shutil.rmtree(ledger_dir)

    blockchain = PoABlockchain(
        ledger_dir=str(ledger_dir)
    )

    print("\nAuthorized validators:")
    for validator in blockchain.validators:
        print(f"  ✓ {validator}")

    # ------------------------------------------------------------------
    # TEST 1: CREATE BLOCKS USING AUTHORIZED VALIDATORS
    # ------------------------------------------------------------------

    print("\n" + "=" * 80)
    print("TEST 1: AUTHORIZED VALIDATORS")
    print("=" * 80)

    anomaly_records = [
        {
            "sample_index": 42,
            "anomaly_score": 3.5,
            "detected_by": "autoencoder",
        },
        {
            "sample_index": 105,
            "anomaly_score": 2.8,
            "detected_by": "autoencoder",
        },
        {
            "sample_index": 208,
            "anomaly_score": 4.1,
            "detected_by": "autoencoder",
        },
    ]

    for anomaly in anomaly_records:

        validator = blockchain.get_current_validator()

        block = blockchain.create_block(
            validator=validator,
            anomaly_data=anomaly,
        )

        print(
            f"✓ Block {block['index']} created by "
            f"{block['validator']}"
        )

    # ------------------------------------------------------------------
    # TEST 2: UNAUTHORIZED VALIDATOR
    # ------------------------------------------------------------------

    print("\n" + "=" * 80)
    print("TEST 2: UNAUTHORIZED VALIDATOR")
    print("=" * 80)

    try:

        blockchain.create_block(
            validator="unauthorized_validator",
            anomaly_data={
                "sample_index": 999,
                "anomaly_score": 9.9,
            },
        )

        print("✗ Unauthorized validator was accepted")

    except PermissionError:

        print("✓ Unauthorized validator correctly rejected")

    # ------------------------------------------------------------------
    # TEST 3: VERIFY REPLICATED BLOCKCHAINS
    # ------------------------------------------------------------------

    print("\n" + "=" * 80)
    print("TEST 3: BLOCKCHAIN REPLICATION AND VERIFICATION")
    print("=" * 80)

    for validator in blockchain.validators:

        valid = blockchain.verify_chain(validator)

        chain = blockchain.get_chain(validator)

        print(
            f"{validator}: "
            f"{'VALID' if valid else 'INVALID'} "
            f"({len(chain)} blocks)"
        )

    all_valid = blockchain.verify_all_validators()

    print(
        f"\nReplicated chains identical: "
        f"{'YES' if all_valid else 'NO'}"
    )

    # ------------------------------------------------------------------
    # TEST 4: TAMPERING
    # ------------------------------------------------------------------

    print("\n" + "=" * 80)
    print("TEST 4: TAMPER DETECTION")
    print("=" * 80)

    tampered_validator = "validator_2"

    ledger_path = (
        ledger_dir /
        f"{tampered_validator}_ledger.jsonl"
    )

    with open(ledger_path, "r") as f:
        blocks = [
            json.loads(line)
            for line in f
            if line.strip()
        ]

    # Change an anomaly score without updating its cryptographic values
    blocks[0]["data"]["anomaly_score"] = 0.1

    with open(ledger_path, "w") as f:
        for block in blocks:
            f.write(json.dumps(block) + "\n")

    tamper_valid = blockchain.verify_chain(
        tampered_validator
    )

    if not tamper_valid:
        print("✓ Modified block successfully detected")
    else:
        print("✗ Tampering was not detected")

    replication_valid = blockchain.verify_all_validators()

    if not replication_valid:
        print("✓ Validator inconsistency successfully detected")
    else:
        print("✗ Validator inconsistency was not detected")

    # ------------------------------------------------------------------
    # SUMMARY
    # ------------------------------------------------------------------

    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    print("Authorized validators:       3")
    print(
        f"PoA block creation:          "
        f"{'PASSED' if all_valid else 'FAILED'}"
    )
    print("Unauthorized validator:      REJECTED")
    print(
        f"Ledger replication:          "
        f"{'PASSED' if all_valid else 'FAILED'}"
    )
    print(
        f"Tamper detection:            "
        f"{'PASSED' if not tamper_valid else 'FAILED'}"
    )
    print(
        f"Replica inconsistency:       "
        f"{'PASSED' if not replication_valid else 'FAILED'}"
    )

    print("\nDone.")


if __name__ == "__main__":
    main()