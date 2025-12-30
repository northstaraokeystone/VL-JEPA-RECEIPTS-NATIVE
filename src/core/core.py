"""
Core functions for VL-JEPA Receipts-Native v3.0.

Every function in this codebase uses these primitives.
No exceptions. No alternatives.
"""

import hashlib
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# BLAKE3 is optional but preferred
try:
    import blake3
    HAS_BLAKE3 = True
except ImportError:
    HAS_BLAKE3 = False

# Ledger paths
LEDGER_PATH = Path(os.environ.get("VLJEPA_LEDGER_PATH", "receipts.jsonl"))
TRAINING_PATH = Path(os.environ.get("VLJEPA_TRAINING_PATH", "training_examples.jsonl"))

# RACI matrix cache
_RACI_MATRIX: dict | None = None


def dual_hash(data: bytes | str) -> str:
    """
    SHA256:BLAKE3 - ALWAYS use this, never single hash.

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
        # Fallback: use SHA256 twice with different prefixes for uniqueness
        b3 = hashlib.sha256(b"blake3_compat:" + data).hexdigest()

    return f"{sha}:{b3}"


def load_raci_for_event(event_type: str, domain: str = "default") -> dict:
    """
    Load RACI assignment for an event type.

    Args:
        event_type: The type of event (e.g., "authenticity", "fsd_frame")
        domain: The domain context (e.g., "x_twitter", "tesla_fsd")

    Returns:
        RACI dict with responsible, accountable, consulted, informed, escalation_path
    """
    global _RACI_MATRIX

    if _RACI_MATRIX is None:
        raci_path = Path(__file__).parent.parent.parent / "config" / "raci_matrix.json"
        if raci_path.exists():
            with open(raci_path) as f:
                _RACI_MATRIX = json.load(f)
        else:
            _RACI_MATRIX = {}

    # Look up domain-specific RACI, fall back to default
    domain_raci = _RACI_MATRIX.get(domain, {})
    event_raci = domain_raci.get(event_type, {})

    # Default RACI if not found
    if not event_raci:
        event_raci = {
            "responsible": f"{domain}_system",
            "accountable": f"{domain}_team",
            "consulted": [],
            "informed": ["audit_log"],
            "escalation_path": [f"{domain}_lead", "platform_safety"]
        }

    return event_raci


def emit_receipt(
    receipt_type: str,
    data: dict,
    *,
    tenant_id: str | None = None,
    include_raci: bool = True,
    domain: str = "default",
    write_to_ledger: bool = True,
) -> dict:
    """
    Emit a receipt. Every function calls this. No exceptions.

    Args:
        receipt_type: Type of receipt (e.g., "ingest", "anchor", "authenticity")
        data: Receipt data payload
        tenant_id: Optional tenant ID override
        include_raci: Whether to include RACI accountability (default True)
        domain: Domain for RACI lookup
        write_to_ledger: Whether to write to ledger file (default True)

    Returns:
        Complete receipt dict with hash, timestamp, and optional RACI
    """
    # Build base receipt
    receipt = {
        "receipt_type": receipt_type,
        "ts": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "tenant_id": tenant_id or data.get("tenant_id", "default"),
        "receipt_id": str(uuid.uuid4()),
    }

    # Add RACI if requested
    if include_raci:
        raci = load_raci_for_event(receipt_type, domain)
        receipt["raci"] = raci

    # Merge data
    receipt.update(data)

    # Compute payload hash (exclude the hash field itself)
    payload_for_hash = {k: v for k, v in receipt.items() if k != "payload_hash"}
    receipt["payload_hash"] = dual_hash(json.dumps(payload_for_hash, sort_keys=True))

    # Write to ledger
    if write_to_ledger:
        with open(LEDGER_PATH, "a") as f:
            f.write(json.dumps(receipt) + "\n")

    return receipt


class StopRule(Exception):
    """
    Raised when a stoprule triggers. Never catch silently.

    Stoprules are the enforcement mechanism for SLOs.
    When violated, they emit an anomaly receipt and halt processing.
    """

    def __init__(self, message: str, metric: str = "unknown", delta: float = 0.0, action: str = "halt"):
        super().__init__(message)
        self.metric = metric
        self.delta = delta
        self.action = action

        # Emit anomaly receipt on creation
        emit_receipt("anomaly", {
            "metric": metric,
            "baseline": 0.0,
            "delta": delta,
            "classification": "violation",
            "action": action,
            "message": message,
        }, include_raci=True, domain="system")


def merkle(items: list) -> str:
    """
    Compute Merkle root of items using BLAKE3.

    Args:
        items: List of items to hash into tree

    Returns:
        Merkle root as dual-hash string
    """
    if not items:
        return dual_hash(b"empty")

    # Hash each item
    hashes = [dual_hash(json.dumps(item, sort_keys=True) if isinstance(item, dict) else str(item))
              for item in items]

    # Build tree
    while len(hashes) > 1:
        if len(hashes) % 2:
            hashes.append(hashes[-1])  # Duplicate last for odd count
        hashes = [dual_hash(hashes[i] + hashes[i+1])
                  for i in range(0, len(hashes), 2)]

    return hashes[0]


def verify_merkle_proof(item: Any, proof: list[tuple[str, str]], root: str) -> bool:
    """
    Verify a Merkle inclusion proof.

    Args:
        item: The item to verify inclusion for
        proof: List of (sibling_hash, position) tuples
        root: Expected Merkle root

    Returns:
        True if proof is valid
    """
    current = dual_hash(json.dumps(item, sort_keys=True) if isinstance(item, dict) else str(item))

    for sibling_hash, position in proof:
        if position == "left":
            current = dual_hash(sibling_hash + current)
        else:
            current = dual_hash(current + sibling_hash)

    return current == root


def generate_merkle_proof(items: list, index: int) -> tuple[str, list[tuple[str, str]]]:
    """
    Generate a Merkle inclusion proof for an item.

    Args:
        items: List of all items
        index: Index of item to prove

    Returns:
        Tuple of (root, proof) where proof is list of (sibling_hash, position)
    """
    if not items:
        return dual_hash(b"empty"), []

    hashes = [dual_hash(json.dumps(item, sort_keys=True) if isinstance(item, dict) else str(item))
              for item in items]

    proof = []
    current_index = index

    while len(hashes) > 1:
        if len(hashes) % 2:
            hashes.append(hashes[-1])

        # Find sibling
        if current_index % 2 == 0:
            sibling_index = current_index + 1
            position = "right"
        else:
            sibling_index = current_index - 1
            position = "left"

        if sibling_index < len(hashes):
            proof.append((hashes[sibling_index], position))

        # Move to next level
        current_index = current_index // 2
        hashes = [dual_hash(hashes[i] + hashes[i+1])
                  for i in range(0, len(hashes), 2)]

    return hashes[0], proof


def load_thresholds() -> dict:
    """Load current thresholds from config."""
    thresholds_path = Path(__file__).parent.parent.parent / "config" / "thresholds.json"
    if thresholds_path.exists():
        with open(thresholds_path) as f:
            return json.load(f)

    # Default thresholds
    return {
        "compression_threshold": 0.85,
        "confidence_threshold": 0.85,
        "temporal_consistency_threshold": 2.0,  # sigma
        "adversarial_detection_threshold": 0.30,
        "coherence_threshold": 0.80,
        "merkle_tree_depth": 8,
    }


def save_thresholds(thresholds: dict) -> None:
    """Save thresholds to config."""
    thresholds_path = Path(__file__).parent.parent.parent / "config" / "thresholds.json"
    thresholds_path.parent.mkdir(parents=True, exist_ok=True)
    with open(thresholds_path, "w") as f:
        json.dump(thresholds, f, indent=2)
