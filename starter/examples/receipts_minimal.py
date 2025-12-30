"""
Receipts-Native Minimal Implementation

This is a MINIMAL receipts-native system that PASSES all 6 compliance tests.
Use this as a reference implementation when building receipts-native systems.

Every operation emits a receipt. State is reconstructable from receipts alone.
"""

import json
import math
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# Try importing from starter package, fall back to local
try:
    from starter.core.receipt import (
        dual_hash,
        emit_receipt,
        load_receipts,
        merkle_root,
        reset_lineage,
        set_ledger_path,
        LEDGER_PATH,
    )
    from starter.core.stoprule import StopRule
except ImportError:
    from core.receipt import (
        dual_hash,
        emit_receipt,
        load_receipts,
        merkle_root,
        reset_lineage,
        set_ledger_path,
        LEDGER_PATH,
    )
    from core.stoprule import StopRule


class ReceiptsMinimal:
    """
    Minimal receipts-native system that passes all 6 compliance tests.

    This demonstrates:
    - P1: State reconstructable from receipts
    - P2: Receipt chain traceable to genesis
    - P3: Decisions auditable without code
    - P4: Proofs derived at query time
    - P5: Entropy conserved within bounds
    - P6: Progress gated by receipts
    """

    def __init__(self, ledger_path: Optional[Path] = None):
        """
        Initialize the minimal receipts-native system.

        Args:
            ledger_path: Optional custom ledger path
        """
        if ledger_path:
            set_ledger_path(ledger_path)

        self.ledger_path = ledger_path or LEDGER_PATH

        # Initialize state (will be populated from receipts)
        self._state = {}
        self._gates = {}
        self._entropy = 0.0

        # Initialize by emitting genesis
        reset_lineage()
        self._emit_genesis()

    def _emit_genesis(self) -> dict:
        """Emit genesis receipt (first receipt in chain)."""
        receipt = emit_receipt("genesis", {
            "system": "receipts_minimal",
            "version": "1.1",
            "initialized_at": datetime.now(timezone.utc).isoformat(),
        })
        self._state["genesis"] = receipt["payload_hash"]
        return receipt

    def run_cycle(self) -> dict:
        """
        Run a complete cycle with receipt emission.

        Returns:
            Current state after cycle
        """
        # Ingest some data
        ingest_receipt = emit_receipt("ingest", {
            "source_type": "cycle_data",
            "data_size": random.randint(100, 1000),
        })

        # Make a decision based on ingested data
        decision_receipt = emit_receipt("decision", {
            "input_hashes": [ingest_receipt["payload_hash"]],
            "output": "processed",
            "confidence": 0.95,
            "verdict": "accept",
        })

        # Track entropy for P5
        entropy_before = self._entropy
        work = 0.005  # Simulated work
        self._entropy += work
        entropy_receipt = emit_receipt("entropy", {
            "s_before": entropy_before,
            "s_after": self._entropy,
            "work": work,
            "delta": abs((self._entropy - entropy_before) - work),
        })

        # Update state from receipts
        self._state["last_ingest"] = ingest_receipt["payload_hash"]
        self._state["last_decision"] = decision_receipt["payload_hash"]
        self._state["last_entropy"] = entropy_receipt["payload_hash"]
        self._state["cycle_count"] = self._state.get("cycle_count", 0) + 1

        return self._state.copy()

    def get_state(self) -> dict:
        """
        Get current system state.

        Returns:
            Copy of current state
        """
        return self._state.copy()

    def load_receipts(self) -> list[dict]:
        """
        Load all receipts from ledger.

        Returns:
            List of receipts in order
        """
        return load_receipts(self.ledger_path)

    def delete_all_state_except_receipts(self) -> None:
        """
        Delete all state except receipts.jsonl.

        This simulates a complete state loss - only receipts survive.
        """
        self._state = {}
        self._gates = {}
        self._entropy = 0.0

    def reconstruct_from_receipts(self) -> dict:
        """
        Reconstruct state from receipts alone.

        This is the key P1 test - state must be fully reconstructable.

        Returns:
            Reconstructed state
        """
        receipts = self.load_receipts()

        # Rebuild state from receipt chain
        for receipt in receipts:
            rtype = receipt.get("receipt_type")

            if rtype == "genesis":
                self._state["genesis"] = receipt["payload_hash"]

            elif rtype == "ingest":
                self._state["last_ingest"] = receipt["payload_hash"]

            elif rtype == "decision":
                self._state["last_decision"] = receipt["payload_hash"]

            elif rtype == "entropy":
                self._state["last_entropy"] = receipt["payload_hash"]
                self._entropy = receipt.get("s_after", 0.0)

            elif rtype == "gate":
                self._gates[receipt.get("gate_id")] = receipt["payload_hash"]

            # Count cycles
            if rtype == "ingest":
                self._state["cycle_count"] = self._state.get("cycle_count", 0) + 1

        return self._state.copy()

    def get_random_decision(self) -> Optional[dict]:
        """
        Get a random decision receipt for P3 testing.

        Returns:
            A decision receipt, or None if no decisions exist
        """
        receipts = self.load_receipts()
        decisions = [r for r in receipts if r.get("receipt_type") == "decision"]

        if not decisions:
            # Generate a decision if none exist
            self.run_cycle()
            receipts = self.load_receipts()
            decisions = [r for r in receipts if r.get("receipt_type") == "decision"]

        return random.choice(decisions) if decisions else None

    def query(self, query_str: str) -> dict:
        """
        Query the system and derive proof from receipts.

        P4: Proofs are derived at query time, NOT pre-stored.

        Args:
            query_str: The query (e.g., "detect_anomalies")

        Returns:
            Proof derived from receipt chain
        """
        receipts = self.load_receipts()

        if query_str == "detect_anomalies":
            # Derive anomaly detection from entropy receipts
            entropy_receipts = [r for r in receipts if r.get("receipt_type") == "entropy"]

            anomalies = []
            for r in entropy_receipts:
                if r.get("delta", 0) >= 0.01:
                    anomalies.append(r["payload_hash"])

            return {
                "query": query_str,
                "derived_from": [r["payload_hash"] for r in entropy_receipts],
                "anomalies": anomalies,
                "count": len(anomalies),
                "merkle_root": merkle_root(entropy_receipts) if entropy_receipts else None,
            }

        elif query_str == "get_decisions":
            decisions = [r for r in receipts if r.get("receipt_type") == "decision"]
            return {
                "query": query_str,
                "decisions": [r["payload_hash"] for r in decisions],
                "count": len(decisions),
            }

        else:
            # Generic query - return receipt count
            return {
                "query": query_str,
                "receipt_count": len(receipts),
                "receipt_types": list(set(r.get("receipt_type") for r in receipts)),
            }

    def has_precomputed_results(self) -> bool:
        """
        Check if system has pre-computed results.

        P4: Receipts-native systems derive proofs at query time.

        Returns:
            False (this system has no pre-computed results)
        """
        # This system derives all proofs from receipts
        # No stored fraud_alerts, cached_results, etc.
        return False

    def run_with_entropy(self) -> tuple[float, float, float]:
        """
        Run cycle with entropy tracking for P5 testing.

        Returns:
            Tuple of (entropy_before, entropy_after, work_done)
        """
        entropy_before = self._entropy

        # Simulate work that changes entropy
        work = 0.005  # Small, bounded work

        # Update entropy (conserved within bounds)
        self._entropy = entropy_before + work

        # Emit entropy receipt
        emit_receipt("entropy", {
            "s_before": entropy_before,
            "s_after": self._entropy,
            "work": work,
            "delta": abs((self._entropy - entropy_before) - work),
        })

        return entropy_before, self._entropy, work

    def get_gates(self) -> list[str]:
        """
        Get list of gate IDs.

        Returns:
            List of gate identifiers
        """
        return ["t2h", "t24h", "t48h"]

    def delete_gate_receipt(self, gate_id: str) -> None:
        """
        Delete a gate receipt (for P6 testing).

        Args:
            gate_id: The gate to delete
        """
        if gate_id in self._gates:
            del self._gates[gate_id]

    def advance_to_next_gate(self) -> None:
        """
        Attempt to advance past the next gate.

        Raises:
            StopRule: If required gate receipt is missing
        """
        required_gate = "t2h"  # First gate required

        # Check if gate receipt exists
        if required_gate not in self._gates:
            raise StopRule(
                f"Missing required gate receipt: {required_gate}",
                metric="gate",
                gate_id=required_gate,
                action="halt",
            )

        # Gate passed - emit progress receipt
        emit_receipt("progress", {
            "gate_passed": required_gate,
            "next_gate": "t24h",
        })

    def emit_gate(self, gate_id: str) -> dict:
        """
        Emit a gate receipt.

        Args:
            gate_id: The gate identifier

        Returns:
            The gate receipt
        """
        receipt = emit_receipt("gate", {
            "gate_id": gate_id,
            "gate_type": "timeline",
            "passed": True,
        })

        self._gates[gate_id] = receipt["payload_hash"]
        return receipt


# Alias for test discovery
SystemUnderTest = ReceiptsMinimal


if __name__ == "__main__":
    # Demo: Show that this system passes compliance
    print("Receipts-Native Minimal System Demo")
    print("=" * 40)

    system = ReceiptsMinimal()

    # Run some cycles
    for i in range(3):
        state = system.run_cycle()
        print(f"Cycle {i+1}: {state.get('cycle_count', 0)} cycles completed")

    # Show receipts
    receipts = system.load_receipts()
    print(f"\nTotal receipts: {len(receipts)}")

    for r in receipts:
        print(f"  - {r['receipt_type']}: {r['payload_hash'][:16]}...")

    # Query for proof
    proof = system.query("detect_anomalies")
    print(f"\nAnomaly detection proof: {proof['count']} anomalies found")
