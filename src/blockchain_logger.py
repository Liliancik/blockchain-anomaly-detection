import json
import hashlib
import time
from pathlib import Path
from typing import Dict

LEDGER_PATH = Path("blockchain_ledger.jsonl")


def _hash_record(record: Dict) -> str:
    s = json.dumps(record, sort_keys=True).encode("utf-8")
    return hashlib.sha256(s).hexdigest()


def append_to_ledger(data: Dict) -> Dict:
    """Append an event to a simple append-only ledger that mimics a blockchain by
    including the previous record's hash."""
    prev_hash = None
    if LEDGER_PATH.exists():
        with LEDGER_PATH.open("rb") as f:
            # read last line
            lines = f.readlines()
            if lines:
                last = json.loads(lines[-1].decode("utf-8"))
                prev_hash = last.get("hash")

    entry = {
        "timestamp": time.time(),
        "data": data,
        "prev_hash": prev_hash,
    }
    entry_hash = _hash_record(entry)
    entry["hash"] = entry_hash

    with LEDGER_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")

    return entry


def log_anomaly(index: int, details: Dict) -> Dict:
    payload = {"index": int(index), "details": details}
    return append_to_ledger(payload)
