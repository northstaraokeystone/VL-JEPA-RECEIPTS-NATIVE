"""
Intervention Capture - Singularity 2 (Part 1)

Captures human corrections and generates training examples.
Every correction becomes a labeled example for threshold tuning.

Human Correction Flow:
1. VL-JEPA makes prediction
2. Human overrides
3. System captures correction with reason code
4. System generates training example
5. System triggers threshold update if needed
"""

from enum import Enum
from typing import Any
import uuid
import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core import emit_receipt, dual_hash
from src.core.ledger import TrainingLedger


class ReasonCode(Enum):
    """
    Reason codes for human corrections.

    Each code maps to a specific threshold for auto-tuning.
    """
    COMPRESSION_FAILURE = "RC001"      # Deepfake compressed unexpectedly well
    TEMPORAL_INCONSISTENCY = "RC002"   # Merkle tree missed semantic jitter
    ADVERSARIAL_NOVEL = "RC003"        # New attack not caught
    CROSS_MODAL_MISMATCH = "RC004"     # High coherence but wrong alignment
    CONFIDENCE_OVERCONFIDENT = "RC005" # High confidence, wrong prediction
    MERKLE_CHAIN_BREAK = "RC006"       # Temporal chain integrity failure

    @property
    def severity(self) -> str:
        """Get severity level for this reason code."""
        severities = {
            "RC001": "HIGH",
            "RC002": "HIGH",
            "RC003": "CRITICAL",
            "RC004": "MEDIUM",
            "RC005": "MEDIUM",
            "RC006": "CRITICAL",
        }
        return severities.get(self.value, "MEDIUM")

    @property
    def auto_tune_target(self) -> str:
        """Get the threshold to auto-tune for this reason code."""
        targets = {
            "RC001": "compression_threshold",
            "RC002": "temporal_consistency_threshold",
            "RC003": "adversarial_detection_threshold",
            "RC004": "coherence_threshold",
            "RC005": "confidence_calibration",
            "RC006": "merkle_tree_depth",
        }
        return targets.get(self.value, "unknown")


def capture_intervention(
    original_receipt: dict,
    corrected_verdict: str,
    reason_code: ReasonCode,
    justification: str,
    corrector_id: str,
    domain: str = "default",
) -> tuple[dict, dict]:
    """
    Capture a human correction and generate training example.

    Args:
        original_receipt: The VL-JEPA prediction receipt
        corrected_verdict: Human-corrected verdict
        reason_code: Reason for correction
        justification: Free-text explanation
        corrector_id: Human operator ID
        domain: Domain for RACI

    Returns:
        Tuple of (intervention_receipt, training_example)
    """
    intervention_id = str(uuid.uuid4())

    # Extract original prediction details
    original_verdict = original_receipt.get("verdict", original_receipt.get("is_adversarial", "unknown"))

    # Compute deltas for learning
    confidence_delta = 0.0
    if "confidence" in original_receipt:
        # If we corrected, confidence was misleading
        confidence_delta = original_receipt["confidence"] if original_verdict != corrected_verdict else 0.0

    compression_delta = 0.0
    if "compression_ratio" in original_receipt:
        compression_delta = original_receipt["compression_ratio"]

    # Emit intervention receipt
    intervention_receipt = emit_receipt("human_intervention", {
        "intervention_id": intervention_id,
        "original_receipt_hash": original_receipt.get("payload_hash", dual_hash(str(original_receipt))),
        "original_verdict": str(original_verdict),
        "corrected_verdict": corrected_verdict,
        "reason_code": reason_code.value,
        "reason_name": reason_code.name,
        "severity": reason_code.severity,
        "justification": justification,
        "corrector_id": corrector_id,
        "training_example_generated": True,
        "auto_tune_target": reason_code.auto_tune_target,
    }, domain=domain)

    # Generate training example
    training_example = {
        "example_id": str(uuid.uuid4()),
        "intervention_id": intervention_id,
        "original_receipt_type": original_receipt.get("receipt_type", "unknown"),
        "input_hash": original_receipt.get("payload_hash", dual_hash(str(original_receipt))),
        "bad_prediction": {
            "verdict": str(original_verdict),
            "confidence": original_receipt.get("confidence", 0),
            "compression_ratio": original_receipt.get("compression_ratio", 0),
            "detection_score": original_receipt.get("detection_score", 0),
        },
        "good_output": {
            "verdict": corrected_verdict,
        },
        "label": reason_code.value,
        "label_name": reason_code.name,
        "confidence_delta": confidence_delta,
        "compression_delta": compression_delta,
        "severity": reason_code.severity,
        "ts": intervention_receipt["ts"],
    }

    # Persist training example
    training_ledger = TrainingLedger()
    training_ledger.append(training_example)

    return intervention_receipt, training_example


class InterventionCapture:
    """
    Intervention capture manager.

    Tracks corrections, generates training examples, and triggers auto-tuning.
    """

    def __init__(
        self,
        auto_tune_threshold: int = 10,
        domain: str = "default",
    ):
        """
        Args:
            auto_tune_threshold: Number of corrections before triggering auto-tune
            domain: Domain for RACI
        """
        self.auto_tune_threshold = auto_tune_threshold
        self.domain = domain

        self.training_ledger = TrainingLedger()
        self.intervention_count = 0
        self.reason_counts: dict[str, int] = {}

    def capture(
        self,
        original_receipt: dict,
        corrected_verdict: str,
        reason_code: ReasonCode,
        justification: str,
        corrector_id: str,
    ) -> tuple[dict, dict, bool]:
        """
        Capture an intervention.

        Args:
            original_receipt: Original prediction receipt
            corrected_verdict: Human correction
            reason_code: Reason for correction
            justification: Explanation
            corrector_id: Operator ID

        Returns:
            Tuple of (intervention_receipt, training_example, should_auto_tune)
        """
        intervention_receipt, training_example = capture_intervention(
            original_receipt,
            corrected_verdict,
            reason_code,
            justification,
            corrector_id,
            self.domain,
        )

        self.intervention_count += 1

        # Track reason code counts
        code = reason_code.value
        self.reason_counts[code] = self.reason_counts.get(code, 0) + 1

        # Check if we should trigger auto-tune
        should_auto_tune = self.reason_counts[code] >= self.auto_tune_threshold

        # Emit critical escalation if needed
        if reason_code.severity == "CRITICAL":
            emit_receipt("critical_intervention", {
                "intervention_id": training_example["intervention_id"],
                "reason_code": reason_code.value,
                "severity": "CRITICAL",
                "action": "escalate_to_security",
            }, domain=self.domain)

        return intervention_receipt, training_example, should_auto_tune

    def get_training_examples_for_reason(self, reason_code: ReasonCode) -> list[dict]:
        """Get all training examples for a specific reason code."""
        return self.training_ledger.read_by_reason(reason_code.value)

    def get_correction_stats(self) -> dict:
        """Get correction statistics."""
        return {
            "total_interventions": self.intervention_count,
            "reason_counts": self.reason_counts,
            "training_examples_total": len(self.training_ledger.read_all()),
        }

    def should_trigger_tuning(self, reason_code: ReasonCode) -> bool:
        """Check if auto-tuning should be triggered for a reason code."""
        return self.reason_counts.get(reason_code.value, 0) >= self.auto_tune_threshold

    def get_tuning_candidates(self) -> list[ReasonCode]:
        """Get reason codes that have enough examples for tuning."""
        candidates = []
        for code_value, count in self.reason_counts.items():
            if count >= self.auto_tune_threshold:
                try:
                    candidates.append(ReasonCode(code_value))
                except ValueError:
                    pass
        return candidates

    def emit_learning_summary(self) -> dict:
        """Emit a learning summary receipt."""
        examples = self.training_ledger.read_all()

        # Compute improvement metrics
        recent_examples = examples[-100:] if len(examples) > 100 else examples

        receipt = emit_receipt("learning_summary", {
            "total_interventions": self.intervention_count,
            "total_training_examples": len(examples),
            "recent_examples": len(recent_examples),
            "reason_distribution": self.reason_counts,
            "tuning_candidates": [c.value for c in self.get_tuning_candidates()],
            "auto_tune_threshold": self.auto_tune_threshold,
        }, domain=self.domain)

        return receipt
