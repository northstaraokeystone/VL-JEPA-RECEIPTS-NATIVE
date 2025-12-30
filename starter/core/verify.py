"""
Verification functions for Receipts-Native Standard v1.1.

These functions verify Merkle proofs, entropy bounds, and chain integrity.
"""

import json
from typing import Any, Optional

from .receipt import dual_hash, get_receipt_by_hash


def verify_merkle_proof(
    item: Any,
    proof: list[tuple[str, str]],
    root: str
) -> bool:
    """
    Verify a Merkle inclusion proof.

    Args:
        item: The item to verify inclusion for
        proof: List of (sibling_hash, position) tuples where position is "left" or "right"
        root: Expected Merkle root

    Returns:
        True if proof is valid, False otherwise
    """
    if isinstance(item, dict):
        current = dual_hash(json.dumps(item, sort_keys=True))
    elif isinstance(item, bytes):
        current = dual_hash(item)
    else:
        current = dual_hash(str(item))

    for sibling_hash, position in proof:
        if position == "left":
            current = dual_hash(sibling_hash + current)
        else:
            current = dual_hash(current + sibling_hash)

    return current == root


def generate_merkle_proof(
    items: list[Any],
    index: int
) -> tuple[str, list[tuple[str, str]]]:
    """
    Generate a Merkle inclusion proof for an item.

    Args:
        items: List of all items
        index: Index of item to prove (0-based)

    Returns:
        Tuple of (root, proof) where proof is list of (sibling_hash, position)
    """
    if not items:
        return dual_hash(b"empty"), []

    if index < 0 or index >= len(items):
        raise ValueError(f"Index {index} out of range for {len(items)} items")

    def item_hash(item: Any) -> str:
        if isinstance(item, dict):
            return dual_hash(json.dumps(item, sort_keys=True))
        elif isinstance(item, bytes):
            return dual_hash(item)
        else:
            return dual_hash(str(item))

    hashes = [item_hash(item) for item in items]
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
        hashes = [
            dual_hash(hashes[i] + hashes[i + 1])
            for i in range(0, len(hashes), 2)
        ]

    return hashes[0], proof


def check_entropy_bounds(
    s_before: float,
    s_after: float,
    work: float,
    threshold: float = 0.01
) -> tuple[bool, float]:
    """
    Check if entropy change is within bounds.

    P5: |ΔS_total - work_done| < threshold

    Args:
        s_before: Entropy before operation
        s_after: Entropy after operation
        work: Work performed (expected entropy change)
        threshold: Maximum allowed delta (default 0.01)

    Returns:
        Tuple of (is_within_bounds, actual_delta)
    """
    delta_s = s_after - s_before
    actual_delta = abs(delta_s - work)
    return actual_delta < threshold, actual_delta


def trace_to_genesis(
    receipt: dict,
    receipts: list[dict]
) -> list[dict]:
    """
    Trace a receipt back to genesis via parent_hash chain.

    Args:
        receipt: The receipt to trace from
        receipts: All receipts in the ledger

    Returns:
        List of receipts from target back to genesis (inclusive)

    Raises:
        ValueError: If chain is broken (missing parent)
    """
    # Build hash index for O(1) lookup
    hash_index = {r["payload_hash"]: r for r in receipts}

    chain = [receipt]
    current = receipt

    while current.get("parent_hash") is not None:
        parent_hash = current["parent_hash"]

        if parent_hash not in hash_index:
            raise ValueError(
                f"Broken chain: parent {parent_hash[:16]}... not found"
            )

        parent = hash_index[parent_hash]
        chain.append(parent)
        current = parent

    # Return in reverse order (genesis first)
    return list(reversed(chain))


def verify_decision_determinism(
    decision_receipt: dict,
    receipts: list[dict],
    decision_function: Optional[callable] = None
) -> bool:
    """
    Verify that a decision is deterministically derivable from inputs.

    P3: Every decision is auditable without source code.

    Args:
        decision_receipt: The decision receipt to verify
        receipts: All receipts for input lookup
        decision_function: Optional function to recompute decision

    Returns:
        True if decision is verifiable, False otherwise
    """
    # Must have input hashes
    input_hashes = decision_receipt.get("input_hashes", [])
    if not input_hashes:
        return False

    # All inputs must exist
    hash_index = {r["payload_hash"]: r for r in receipts}
    for input_hash in input_hashes:
        if input_hash not in hash_index:
            return False

    # If decision function provided, verify determinism
    if decision_function is not None:
        inputs = [hash_index[h] for h in input_hashes]
        recomputed = decision_function(inputs)
        return recomputed == decision_receipt.get("output")

    # Without function, we can only verify inputs exist
    return True


def verify_no_precomputed_results(
    system_state: dict
) -> bool:
    """
    Verify that a system has no pre-computed results tables.

    P4: Proofs derived, not stored.

    Args:
        system_state: Dictionary of system state/tables

    Returns:
        True if no pre-computed results found
    """
    forbidden_patterns = [
        "fraud_alerts",
        "precomputed",
        "cached_results",
        "stored_proofs",
        "detection_results",
        "alert_table",
    ]

    for key in system_state.keys():
        key_lower = key.lower()
        for pattern in forbidden_patterns:
            if pattern in key_lower:
                return False

    return True
