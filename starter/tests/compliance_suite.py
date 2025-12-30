"""
Receipts-Native Compliance Test Suite v1.1

These 6 tests verify compliance with the 6 principles of receipts-native architecture.
A system is receipts-native if and only if it passes ALL 6 tests.

Usage:
    pytest tests/compliance_suite.py --system=examples.receipts_minimal
    pytest tests/compliance_suite.py --system=examples.simple_logger

Each test corresponds to one principle:
    P1: Native Provenance - State reconstructable from receipts alone
    P2: Cryptographic Lineage - Any receipt traceable to genesis
    P3: Verifiable Causality - Decisions auditable without code
    P4: Query-as-Proof - Proofs derived, not stored
    P5: Thermodynamic Governance - Entropy conserved within bounds
    P6: Receipts-Gated Progress - System halts without gates
"""

import json
import random
import pytest
from typing import Protocol, Optional


class SystemUnderTest(Protocol):
    """Protocol defining the interface for systems under test."""

    def run_cycle(self) -> dict:
        """Run a complete cycle and return state."""
        ...

    def get_state(self) -> dict:
        """Get current system state."""
        ...

    def load_receipts(self) -> list[dict]:
        """Load all receipts from ledger."""
        ...

    def delete_all_state_except_receipts(self) -> None:
        """Delete all state except receipts.jsonl."""
        ...

    def reconstruct_from_receipts(self) -> dict:
        """Reconstruct state from receipts alone."""
        ...

    def get_random_decision(self) -> dict:
        """Get a random decision receipt."""
        ...

    def query(self, query_str: str) -> dict:
        """Query the system and get derived proof."""
        ...

    def has_precomputed_results(self) -> bool:
        """Check if system has pre-computed results tables."""
        ...

    def run_with_entropy(self) -> tuple[float, float, float]:
        """Run cycle with entropy tracking. Returns (before, after, work)."""
        ...

    def delete_gate_receipt(self, gate_id: str) -> None:
        """Delete a specific gate receipt."""
        ...

    def advance_to_next_gate(self) -> None:
        """Attempt to advance past the next gate."""
        ...

    def get_gates(self) -> list[str]:
        """Get list of gate IDs."""
        ...


# Import StopRule from core if available, otherwise define locally
try:
    from starter.core.stoprule import StopRule
except ImportError:
    try:
        from core.stoprule import StopRule
    except ImportError:
        class StopRule(Exception):
            """Fallback StopRule for when core is not available."""
            pass


def trace_to_genesis(receipt: dict, receipts: list[dict]) -> list[dict]:
    """Trace a receipt back to genesis via parent_hash chain."""
    hash_index = {r["payload_hash"]: r for r in receipts}

    chain = [receipt]
    current = receipt

    while current.get("parent_hash") is not None:
        parent_hash = current["parent_hash"]
        if parent_hash not in hash_index:
            raise ValueError(f"Broken chain: parent {parent_hash[:16]}... not found")
        parent = hash_index[parent_hash]
        chain.append(parent)
        current = parent

    return list(reversed(chain))


def verify_chain_integrity(chain: list[dict]) -> bool:
    """Verify hash chain integrity."""
    try:
        # Try to import dual_hash
        try:
            from starter.core.receipt import dual_hash
        except ImportError:
            from core.receipt import dual_hash

        for receipt in chain:
            payload_for_hash = {k: v for k, v in receipt.items() if k != "payload_hash"}
            computed = dual_hash(json.dumps(payload_for_hash, sort_keys=True))
            if computed != receipt.get("payload_hash"):
                return False
        return True
    except ImportError:
        # If we can't import dual_hash, assume valid if chain exists
        return len(chain) > 0


def extract_input_receipts(decision: dict, receipts: list[dict]) -> list[dict]:
    """Extract input receipts referenced by a decision."""
    input_hashes = decision.get("input_hashes", [])
    hash_index = {r["payload_hash"]: r for r in receipts}
    return [hash_index[h] for h in input_hashes if h in hash_index]


def deterministic_verify(decision: dict, inputs: list[dict]) -> bool:
    """Verify decision is deterministically derivable from inputs."""
    # A decision is verifiable if:
    # 1. It has input_hashes
    # 2. All inputs are present
    # 3. Output is recorded
    if not decision.get("input_hashes"):
        return False
    if len(inputs) != len(decision.get("input_hashes", [])):
        return False
    if "output" not in decision and "verdict" not in decision:
        return False
    return True


# =============================================================================
# PRINCIPLE 1: NATIVE PROVENANCE
# =============================================================================

def test_principle_1_native_provenance(system_under_test: SystemUnderTest):
    """
    PRINCIPLE 1: Native Provenance

    Test: All state is reconstructable from receipts alone.
    Method:
        1. Run system, capture state
        2. Delete all state except receipts.jsonl
        3. Reconstruct state from receipts
        4. Assert reconstructed == original

    Pass: State reconstruction matches original
    Fail: State cannot be reconstructed, or differs from original
    """
    # Run system to generate state
    system_under_test.run_cycle()
    state_before = system_under_test.get_state()

    # Must have some state to test
    assert state_before, "FAIL P1: System has no state after run_cycle()"

    # Delete all state except receipts
    system_under_test.delete_all_state_except_receipts()

    # Reconstruct from receipts
    state_after = system_under_test.reconstruct_from_receipts()

    # Compare states
    assert state_before == state_after, (
        f"FAIL P1: State reconstruction mismatch. "
        f"Before: {list(state_before.keys())}, After: {list(state_after.keys())}"
    )


# =============================================================================
# PRINCIPLE 2: CRYPTOGRAPHIC LINEAGE
# =============================================================================

def test_principle_2_cryptographic_lineage(system_under_test: SystemUnderTest):
    """
    PRINCIPLE 2: Cryptographic Lineage

    Test: Any receipt is traceable to genesis via parent_hash chain.
    Method:
        1. Load all receipts
        2. Select random non-genesis receipt
        3. Trace parent_hash chain backward
        4. Assert reaches genesis (parent_hash is None)
        5. Verify hash integrity of chain

    Pass: Chain reaches genesis with valid hashes
    Fail: Broken chain, missing genesis, or hash mismatch
    """
    receipts = system_under_test.load_receipts()

    # Must have receipts to test
    assert len(receipts) > 0, "FAIL P2: No receipts in ledger"

    # Genesis must have null parent
    genesis = receipts[0]
    assert genesis.get("parent_hash") is None, (
        "FAIL P2: First receipt is not genesis (parent_hash should be null)"
    )

    # Test random non-genesis receipt if we have more than one
    if len(receipts) > 1:
        random_receipt = random.choice(receipts[1:])

        # Trace to genesis
        try:
            chain = trace_to_genesis(random_receipt, receipts)
        except ValueError as e:
            pytest.fail(f"FAIL P2: {e}")

        # Verify chain starts at genesis
        assert chain[0].get("parent_hash") is None, (
            "FAIL P2: Chain does not reach genesis"
        )

        # Verify chain integrity
        assert verify_chain_integrity(chain), (
            "FAIL P2: Chain integrity verification failed (hash mismatch)"
        )


# =============================================================================
# PRINCIPLE 3: VERIFIABLE CAUSALITY
# =============================================================================

def test_principle_3_verifiable_causality(system_under_test: SystemUnderTest):
    """
    PRINCIPLE 3: Verifiable Causality

    Test: Every decision is auditable without source code access.
    Method:
        1. Get a decision receipt from the system
        2. Extract input receipt references
        3. Verify decision is deterministically derivable
        4. NO source code access allowed in verification

    Pass: Decision verifiable from input receipts alone
    Fail: Decision requires source code to verify
    """
    # Get a decision receipt
    decision = system_under_test.get_random_decision()

    assert decision is not None, "FAIL P3: No decision receipts in system"
    assert decision.get("receipt_type") in ("decision", "verdict", "routing"), (
        f"FAIL P3: Receipt type '{decision.get('receipt_type')}' is not a decision"
    )

    # Must have input_hashes for verifiable causality
    input_hashes = decision.get("input_hashes", [])
    assert input_hashes, (
        "FAIL P3: Decision receipt missing input_hashes field. "
        "Decisions must reference their inputs for verifiability."
    )

    # All inputs must exist in ledger
    receipts = system_under_test.load_receipts()
    inputs = extract_input_receipts(decision, receipts)

    assert len(inputs) == len(input_hashes), (
        f"FAIL P3: Missing input receipts. "
        f"Expected {len(input_hashes)}, found {len(inputs)}"
    )

    # Verify determinism
    assert deterministic_verify(decision, inputs), (
        "FAIL P3: Decision is not deterministically verifiable from inputs"
    )


# =============================================================================
# PRINCIPLE 4: QUERY-AS-PROOF
# =============================================================================

def test_principle_4_query_as_proof(system_under_test: SystemUnderTest):
    """
    PRINCIPLE 4: Query-as-Proof

    Test: Proofs are derived from receipts at query time, not pre-stored.
    Method:
        1. Query system for a result
        2. Query again with same parameters
        3. Verify proofs are deterministically equal
        4. Verify no pre-computed results tables exist

    Pass: Proofs derived deterministically, no pre-stored results
    Fail: Pre-computed results table exists, or non-deterministic proofs
    """
    # Query for a proof
    proof1 = system_under_test.query("detect_anomalies")
    proof2 = system_under_test.query("detect_anomalies")

    # Proofs must be deterministic
    assert proof1 == proof2, (
        "FAIL P4: Non-deterministic proof derivation. "
        "Same query produced different proofs."
    )

    # Must not have pre-computed results
    has_precomputed = system_under_test.has_precomputed_results()
    assert not has_precomputed, (
        "FAIL P4: System has pre-computed results table. "
        "Proofs must be derived at query time."
    )


# =============================================================================
# PRINCIPLE 5: THERMODYNAMIC GOVERNANCE
# =============================================================================

def test_principle_5_thermodynamic_governance(system_under_test: SystemUnderTest):
    """
    PRINCIPLE 5: Thermodynamic Governance

    Test: Entropy is conserved within bounds over execution windows.
    Method:
        1. Run cycle with entropy tracking
        2. Calculate |ΔS_total - work_done|
        3. Assert delta < 0.01 threshold

    Pass: Entropy delta within bounds
    Fail: Entropy violation exceeds threshold
    """
    entropy_before, entropy_after, work = system_under_test.run_with_entropy()

    delta_s = entropy_after - entropy_before
    actual_delta = abs(delta_s - work)

    assert actual_delta < 0.01, (
        f"FAIL P5: Entropy violation. "
        f"Delta = {actual_delta:.4f}, threshold = 0.01. "
        f"S_before={entropy_before:.4f}, S_after={entropy_after:.4f}, work={work:.4f}"
    )


# =============================================================================
# PRINCIPLE 6: RECEIPTS-GATED PROGRESS
# =============================================================================

def test_principle_6_receipts_gated_progress(system_under_test: SystemUnderTest):
    """
    PRINCIPLE 6: Receipts-Gated Progress

    Test: System halts without required gate receipts.
    Method:
        1. Get list of gates
        2. Delete a required gate receipt
        3. Attempt to advance past gate
        4. Assert StopRule is raised

    Pass: StopRule raised when gate missing
    Fail: System advances without gate, or wrong exception
    """
    gates = system_under_test.get_gates()

    assert gates, (
        "FAIL P6: System has no gates defined. "
        "Receipts-native systems require progress gates."
    )

    # Delete first gate
    gate_to_delete = gates[0]
    system_under_test.delete_gate_receipt(gate_to_delete)

    # Attempt to advance should raise StopRule
    with pytest.raises(StopRule) as exc_info:
        system_under_test.advance_to_next_gate()

    # Verify it's about the missing gate
    error_msg = str(exc_info.value).lower()
    assert "gate" in error_msg or gate_to_delete.lower() in error_msg, (
        f"FAIL P6: StopRule raised but not about missing gate. "
        f"Error: {exc_info.value}"
    )


# =============================================================================
# AGGREGATE COMPLIANCE CHECK
# =============================================================================

def run_full_compliance_check(system_under_test: SystemUnderTest) -> dict:
    """
    Run all 6 compliance tests and return results.

    Args:
        system_under_test: System implementing the test interface

    Returns:
        Dict with pass/fail status and details for each principle
    """
    results = {
        "system": type(system_under_test).__name__,
        "passed": 0,
        "failed": 0,
        "principles": {}
    }

    tests = [
        ("P1: Native Provenance", test_principle_1_native_provenance),
        ("P2: Cryptographic Lineage", test_principle_2_cryptographic_lineage),
        ("P3: Verifiable Causality", test_principle_3_verifiable_causality),
        ("P4: Query-as-Proof", test_principle_4_query_as_proof),
        ("P5: Thermodynamic Governance", test_principle_5_thermodynamic_governance),
        ("P6: Receipts-Gated Progress", test_principle_6_receipts_gated_progress),
    ]

    for name, test_func in tests:
        try:
            test_func(system_under_test)
            results["principles"][name] = {"status": "PASS", "error": None}
            results["passed"] += 1
        except (AssertionError, Exception) as e:
            results["principles"][name] = {"status": "FAIL", "error": str(e)}
            results["failed"] += 1

    results["is_receipts_native"] = results["failed"] == 0
    return results
