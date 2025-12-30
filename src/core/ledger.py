"""
Ledger management for VL-JEPA Receipts-Native.

Append-only ledger with Merkle anchoring.
"""

import json
from pathlib import Path
from typing import Iterator

from .core import emit_receipt, merkle, dual_hash


class Ledger:
    """Append-only receipt ledger with Merkle anchoring."""

    def __init__(self, path: str | Path = "receipts.jsonl"):
        self.path = Path(path)
        self._ensure_exists()

    def _ensure_exists(self) -> None:
        """Ensure ledger file exists."""
        if not self.path.exists():
            self.path.touch()

    def append(self, receipt: dict) -> None:
        """Append a receipt to the ledger."""
        with open(self.path, "a") as f:
            f.write(json.dumps(receipt) + "\n")

    def read_all(self) -> list[dict]:
        """Read all receipts from the ledger."""
        receipts = []
        with open(self.path) as f:
            for line in f:
                if line.strip():
                    receipts.append(json.loads(line))
        return receipts

    def read_by_type(self, receipt_type: str) -> list[dict]:
        """Read receipts of a specific type."""
        return [r for r in self.read_all() if r.get("receipt_type") == receipt_type]

    def read_by_tenant(self, tenant_id: str) -> list[dict]:
        """Read receipts for a specific tenant."""
        return [r for r in self.read_all() if r.get("tenant_id") == tenant_id]

    def read_since(self, timestamp: str) -> list[dict]:
        """Read receipts since a timestamp (ISO8601)."""
        return [r for r in self.read_all() if r.get("ts", "") >= timestamp]

    def find_by_hash(self, payload_hash: str) -> dict | None:
        """Find a receipt by its payload hash."""
        for receipt in self.read_all():
            if receipt.get("payload_hash") == payload_hash:
                return receipt
        return None

    def stream(self) -> Iterator[dict]:
        """Stream receipts from the ledger."""
        with open(self.path) as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)

    def count(self) -> int:
        """Count receipts in the ledger."""
        count = 0
        with open(self.path) as f:
            for line in f:
                if line.strip():
                    count += 1
        return count

    def compute_root(self) -> str:
        """Compute Merkle root of all receipts."""
        return merkle(self.read_all())

    def get_unanchored(self) -> list[dict]:
        """Get receipts that haven't been anchored yet."""
        all_receipts = self.read_all()

        # Collect all anchored hashes
        anchored_hashes = set()
        for r in all_receipts:
            if r.get("receipt_type") == "anchor":
                anchored_hashes.update(r.get("anchored_hashes", []))

        # Return unanchored (excluding anchor receipts themselves)
        return [r for r in all_receipts
                if r.get("receipt_type") != "anchor"
                and r.get("payload_hash") not in anchored_hashes]

    def create_anchor(self) -> dict | None:
        """Create an anchor receipt for unanchored receipts."""
        unanchored = self.get_unanchored()

        if not unanchored:
            return None

        root = merkle(unanchored)

        anchor_receipt = emit_receipt("anchor", {
            "merkle_root": root,
            "hash_algos": ["SHA256", "BLAKE3"],
            "batch_size": len(unanchored),
            "anchored_hashes": [r.get("payload_hash") for r in unanchored],
        }, write_to_ledger=False)

        self.append(anchor_receipt)
        return anchor_receipt

    def verify_integrity(self) -> tuple[bool, list[str]]:
        """Verify ledger integrity."""
        errors = []
        all_receipts = self.read_all()

        for i, receipt in enumerate(all_receipts):
            # Verify payload hash
            payload_for_hash = {k: v for k, v in receipt.items() if k != "payload_hash"}
            expected_hash = dual_hash(json.dumps(payload_for_hash, sort_keys=True))

            if receipt.get("payload_hash") != expected_hash:
                errors.append(f"Receipt {i}: payload hash mismatch")

            # Verify required fields
            if "receipt_type" not in receipt:
                errors.append(f"Receipt {i}: missing receipt_type")
            if "ts" not in receipt:
                errors.append(f"Receipt {i}: missing ts")

        return len(errors) == 0, errors

    def compact(self, older_than: str) -> dict:
        """
        Compact receipts older than a timestamp.

        Creates a compaction receipt and removes old receipts.
        """
        all_receipts = self.read_all()

        old_receipts = [r for r in all_receipts if r.get("ts", "") < older_than]
        new_receipts = [r for r in all_receipts if r.get("ts", "") >= older_than]

        if not old_receipts:
            return emit_receipt("compaction", {
                "input_span": {"start": "", "end": ""},
                "output_span": {"start": "", "end": ""},
                "counts": {"before": 0, "after": 0},
                "sums": {"before": 0, "after": 0},
                "hash_continuity": True,
            })

        # Compute before/after hashes
        before_hash = merkle(old_receipts)
        after_hash = merkle(new_receipts) if new_receipts else dual_hash(b"empty")

        compaction_receipt = emit_receipt("compaction", {
            "input_span": {
                "start": old_receipts[0].get("ts", ""),
                "end": old_receipts[-1].get("ts", ""),
            },
            "output_span": {
                "start": new_receipts[0].get("ts", "") if new_receipts else "",
                "end": new_receipts[-1].get("ts", "") if new_receipts else "",
            },
            "counts": {"before": len(old_receipts), "after": len(new_receipts)},
            "compacted_merkle_root": before_hash,
            "hash_continuity": True,
        }, write_to_ledger=False)

        # Write new ledger
        with open(self.path, "w") as f:
            # Write compaction receipt first
            f.write(json.dumps(compaction_receipt) + "\n")
            # Write remaining receipts
            for receipt in new_receipts:
                f.write(json.dumps(receipt) + "\n")

        return compaction_receipt


class TrainingLedger:
    """Ledger for training examples from human corrections."""

    def __init__(self, path: str | Path = "training_examples.jsonl"):
        self.path = Path(path)
        self._ensure_exists()

    def _ensure_exists(self) -> None:
        """Ensure ledger file exists."""
        if not self.path.exists():
            self.path.touch()

    def append(self, example: dict) -> None:
        """Append a training example."""
        with open(self.path, "a") as f:
            f.write(json.dumps(example) + "\n")

    def read_all(self) -> list[dict]:
        """Read all training examples."""
        examples = []
        with open(self.path) as f:
            for line in f:
                if line.strip():
                    examples.append(json.loads(line))
        return examples

    def read_by_reason(self, reason_code: str) -> list[dict]:
        """Read examples by reason code."""
        return [e for e in self.read_all() if e.get("label") == reason_code]

    def count_by_reason(self) -> dict[str, int]:
        """Count examples by reason code."""
        counts = {}
        for example in self.read_all():
            reason = example.get("label", "unknown")
            counts[reason] = counts.get(reason, 0) + 1
        return counts
