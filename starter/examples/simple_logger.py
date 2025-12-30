"""
Simple Logger - Traditional Logging System (NOT Receipts-Native)

This system demonstrates what receipts-native is NOT.
It uses traditional logging and FAILS all 6 compliance tests.

Use this as a counter-example to understand the difference between
logging and receipts-native architecture.

VIOLATIONS:
- P1: State stored in variables, not reconstructable from logs
- P2: No cryptographic lineage, no parent hashes
- P3: Decisions not auditable without source code
- P4: Pre-computed fraud alerts stored in table
- P5: No entropy tracking
- P6: No gates, progress not receipts-controlled
"""

import logging
import random
from typing import Optional

# Configure traditional logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("simple_logger")


class SimpleLogger:
    """
    Traditional logging system that FAILS all receipts-native compliance tests.

    This demonstrates:
    - State in variables (not reconstructable)
    - Logs without cryptographic integrity
    - Pre-computed results tables
    - No entropy tracking
    - No progress gates
    """

    def __init__(self, ledger_path=None):
        """Initialize simple logger system."""
        # State stored in variables (NOT receipts)
        self._state = {
            "processed_count": 0,
            "decisions": [],
        }

        # PRE-COMPUTED results table - VIOLATES P4
        self._fraud_alerts = {}  # Pre-stored, not derived

        # No gates
        self._gates = {}

        logger.info("SimpleLogger initialized")

    def run_cycle(self) -> dict:
        """Run a cycle using traditional logging."""
        # State changes without receipts
        data = {"value": random.randint(1, 100)}
        self._state["processed_count"] += 1
        self._state["last_data"] = data

        # Traditional logging (NOT receipt emission)
        logger.info(f"Processed data: {data}")

        # Make decision and store in state
        decision = "accept" if data["value"] > 50 else "reject"
        self._state["decisions"].append(decision)
        logger.info(f"Decision: {decision}")

        # Pre-compute fraud alert - VIOLATES P4
        if data["value"] > 90:
            self._fraud_alerts[self._state["processed_count"]] = {
                "type": "high_value",
                "value": data["value"],
            }
            logger.warning(f"Fraud alert stored: {data['value']}")

        return self._state.copy()

    def get_state(self) -> dict:
        """Get current state."""
        return self._state.copy()

    def load_receipts(self) -> list[dict]:
        """
        Load receipts - but this system has none.

        Returns empty or fake receipts without proper structure.
        """
        # No proper receipts - just fake entries
        return [
            {
                "receipt_type": "log",
                "message": "SimpleLogger has no receipts",
                # Missing: payload_hash, parent_hash, ts, tenant_id
            }
        ]

    def delete_all_state_except_receipts(self) -> None:
        """Delete state - but we can't reconstruct it."""
        self._state = {}
        self._fraud_alerts = {}
        logger.info("State deleted")

    def reconstruct_from_receipts(self) -> dict:
        """
        Attempt to reconstruct state - FAILS because no receipts.

        Returns empty or different state.
        """
        # Cannot reconstruct - logs don't contain enough info
        return {
            "error": "Cannot reconstruct - no receipts",
        }

    def get_random_decision(self) -> Optional[dict]:
        """
        Get a decision - but not in receipt format.

        Returns a decision without input_hashes (violates P3).
        """
        return {
            "receipt_type": "decision",
            "output": "accept",
            # Missing: input_hashes - violates P3
        }

    def query(self, query_str: str) -> dict:
        """
        Query using pre-computed results - VIOLATES P4.

        Returns from stored table, not derived from receipts.
        """
        if query_str == "detect_anomalies":
            # Return from pre-computed table - VIOLATES P4
            return {
                "query": query_str,
                "from_stored_table": True,  # This is the violation
                "alerts": list(self._fraud_alerts.values()),
            }

        return {"query": query_str, "result": "unknown"}

    def has_precomputed_results(self) -> bool:
        """
        Check for pre-computed results - YES, we have them.

        Returns True because we store fraud_alerts.
        """
        return True  # We have _fraud_alerts table

    def run_with_entropy(self) -> tuple[float, float, float]:
        """
        Run with entropy - but we don't track it.

        Returns values that VIOLATE P5 threshold.
        """
        # No entropy tracking - return violation
        return 0.0, 0.5, 0.0  # delta = 0.5 >> 0.01 threshold

    def get_gates(self) -> list[str]:
        """
        Get gates - but we don't have any.

        Returns empty list - violates P6.
        """
        return []  # No gates defined

    def delete_gate_receipt(self, gate_id: str) -> None:
        """Delete gate - but we have none."""
        pass

    def advance_to_next_gate(self) -> None:
        """
        Advance without checking gates - VIOLATES P6.

        Does NOT raise StopRule.
        """
        # No gate checking - just proceed
        logger.info("Advancing without gate check")
        # Should raise StopRule but doesn't - VIOLATION


# Alias for test discovery
SystemUnderTest = SimpleLogger


if __name__ == "__main__":
    # Demo: Show that this system FAILS compliance
    print("Simple Logger Demo (NOT Receipts-Native)")
    print("=" * 40)

    system = SimpleLogger()

    # Run some cycles
    for i in range(3):
        state = system.run_cycle()
        print(f"Cycle {i+1}: {state.get('processed_count', 0)} processed")

    # Show why it fails each principle
    print("\nCompliance Violations:")
    print("  P1: State not reconstructable from logs")
    print("  P2: No cryptographic lineage")
    print("  P3: Decisions missing input_hashes")
    print("  P4: Has pre-computed _fraud_alerts table")
    print("  P5: No entropy tracking")
    print("  P6: No gates defined, no StopRule on missing gate")

    print("\nThis system would FAIL all 6 compliance tests.")
