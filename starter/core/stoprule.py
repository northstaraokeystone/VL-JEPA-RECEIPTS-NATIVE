"""
StopRule exception for Receipts-Native Standard v1.1.

Stoprules are the enforcement mechanism for SLOs and gates.
When violated, they emit an anomaly receipt and halt processing.
NEVER catch a StopRule silently.
"""

from typing import Optional


class StopRule(Exception):
    """
    Raised when a stoprule triggers.

    Stoprules enforce:
    - SLO violations (entropy, bias, latency)
    - Missing gate receipts
    - Chain integrity failures
    - Threshold breaches

    NEVER catch a StopRule with `except: pass`.
    Always handle explicitly or let it propagate.
    """

    def __init__(
        self,
        message: str,
        *,
        metric: str = "unknown",
        delta: float = 0.0,
        action: str = "halt",
        gate_id: Optional[str] = None,
    ):
        """
        Initialize StopRule.

        Args:
            message: Human-readable error message
            metric: The metric that was violated
            delta: How far over the threshold (positive = over)
            action: What action triggered (halt, escalate, reject)
            gate_id: If gate-related, which gate was missing
        """
        super().__init__(message)
        self.metric = metric
        self.delta = delta
        self.action = action
        self.gate_id = gate_id

    def to_dict(self) -> dict:
        """Convert to dict for receipt emission."""
        return {
            "message": str(self),
            "metric": self.metric,
            "delta": self.delta,
            "action": self.action,
            "gate_id": self.gate_id,
        }

    def __repr__(self) -> str:
        return (
            f"StopRule('{self}', metric='{self.metric}', "
            f"delta={self.delta}, action='{self.action}')"
        )
