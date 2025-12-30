"""
Core receipt primitives for Receipts-Native Standard v1.1.

Every receipts-native system uses these functions.
No exceptions. No alternatives.
"""

import hashlib
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

# BLAKE3 is optional but strongly preferred
try:
    import blake3
    HAS_BLAKE3 = True
except ImportError:
    HAS_BLAKE3 = False

# Ledger path - configurable via environment
LEDGER_PATH = Path(os.environ.get("RECEIPTS_LEDGER_PATH", "receipts.jsonl"))

# Track last receipt hash for parent_hash lineage
_LAST_RECEIPT_HASH: Optional[str] = None


def dual_hash(data: bytes | str) -> str:
    """
    SHA256:BLAKE3 - ALWAYS use this, never single hash.

    This is the ONLY approved hashing function for receipts-native systems.
    Using a single hash algorithm is a violation.

    Args:
        data: Bytes or string to hash

    Returns:
        String in format "sha256_hex:blake3_hex"
    """
    if isinstance(data, str):
        data = data.encode("utf-8")

    sha = hashlib.sha256(data).hexdigest()

    if HAS_BLAKE3:
        b3 = blake3.blake3(data).hexdigest()
    else:
        # Fallback: use SHA256 with prefix for uniqueness
        # This maintains format compatibility without BLAKE3
        b3 = hashlib.sha256(b"blake3_compat:" + data).hexdigest()

    return f"{sha}:{b3}"


def emit_receipt(
    receipt_type: str,
    data: dict,
    *,
    tenant_id: Optional[str] = None,
    parent_hash: Optional[str] = None,
    write_to_ledger: bool = True,
) -> dict:
    """
    Emit a receipt. Every operation calls this. No exceptions.

    Args:
        receipt_type: Type of receipt (e.g., "ingest", "decision", "entropy")
        data: Receipt payload data
        tenant_id: Optional tenant ID override
        parent_hash: Optional explicit parent hash (defaults to last receipt)
        write_to_ledger: Whether to append to ledger file

    Returns:
        Complete receipt dict with hash, timestamp, and lineage
    """
    global _LAST_RECEIPT_HASH

    # Determine parent hash for lineage
    if parent_hash is None:
        parent_hash = _LAST_RECEIPT_HASH

    # Build base receipt
    receipt = {
        "receipt_type": receipt_type,
        "receipt_id": str(uuid.uuid4()),
        "ts": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "tenant_id": tenant_id or data.get("tenant_id", "default"),
        "parent_hash": parent_hash,
    }

    # Merge data (data fields can override defaults)
    for key, value in data.items():
        if key not in receipt:
            receipt[key] = value

    # Compute payload hash LAST (after all fields set)
    payload_for_hash = {k: v for k, v in receipt.items() if k != "payload_hash"}
    receipt["payload_hash"] = dual_hash(json.dumps(payload_for_hash, sort_keys=True))

    # Update lineage tracker
    _LAST_RECEIPT_HASH = receipt["payload_hash"]

    # Write to ledger
    if write_to_ledger:
        LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(LEDGER_PATH, "a") as f:
            f.write(json.dumps(receipt) + "\n")

    return receipt


def load_receipts(ledger_path: Optional[Path] = None) -> list[dict]:
    """
    Load all receipts from ledger file.

    Args:
        ledger_path: Path to ledger file (defaults to LEDGER_PATH)

    Returns:
        List of receipt dicts in order
    """
    path = ledger_path or LEDGER_PATH

    if not path.exists():
        return []

    receipts = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                receipts.append(json.loads(line))

    return receipts


def get_receipt_by_hash(
    payload_hash: str,
    ledger_path: Optional[Path] = None
) -> Optional[dict]:
    """
    Find a receipt by its payload hash.

    Args:
        payload_hash: The dual-hash to search for
        ledger_path: Path to ledger file

    Returns:
        Receipt dict if found, None otherwise
    """
    for receipt in load_receipts(ledger_path):
        if receipt.get("payload_hash") == payload_hash:
            return receipt
    return None


def verify_chain(receipts: list[dict]) -> bool:
    """
    Verify the integrity of a receipt chain.

    Args:
        receipts: List of receipts in order

    Returns:
        True if chain is valid, False otherwise
    """
    if not receipts:
        return True

    # First receipt must have null parent (genesis)
    if receipts[0].get("parent_hash") is not None:
        return False

    # Build hash index
    hash_index = {r["payload_hash"]: r for r in receipts}

    # Verify each receipt
    for i, receipt in enumerate(receipts):
        # Verify payload hash is correct
        payload_for_hash = {k: v for k, v in receipt.items() if k != "payload_hash"}
        computed = dual_hash(json.dumps(payload_for_hash, sort_keys=True))
        if computed != receipt.get("payload_hash"):
            return False

        # Verify parent exists (except genesis)
        if i > 0:
            parent_hash = receipt.get("parent_hash")
            if parent_hash not in hash_index:
                return False

    return True


def merkle_root(items: list[Any]) -> str:
    """
    Compute Merkle root of items.

    Args:
        items: List of items (dicts, strings, or bytes)

    Returns:
        Merkle root as dual-hash string
    """
    if not items:
        return dual_hash(b"empty")

    # Hash each item
    def item_hash(item: Any) -> str:
        if isinstance(item, dict):
            return dual_hash(json.dumps(item, sort_keys=True))
        elif isinstance(item, bytes):
            return dual_hash(item)
        else:
            return dual_hash(str(item))

    hashes = [item_hash(item) for item in items]

    # Build tree
    while len(hashes) > 1:
        if len(hashes) % 2:
            hashes.append(hashes[-1])  # Duplicate last for odd count
        hashes = [
            dual_hash(hashes[i] + hashes[i + 1])
            for i in range(0, len(hashes), 2)
        ]

    return hashes[0]


def reset_lineage() -> None:
    """Reset the parent hash tracker. Used for testing."""
    global _LAST_RECEIPT_HASH
    _LAST_RECEIPT_HASH = None


def set_ledger_path(path: Path) -> None:
    """Set the ledger path. Used for testing."""
    global LEDGER_PATH
    LEDGER_PATH = path
