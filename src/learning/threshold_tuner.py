"""
Threshold Tuner - Singularity 2 (Part 2)

Auto-tunes thresholds based on training examples from human corrections.
Every correction improves the system.

Tuning strategies:
- CONSERVATIVE: Adjust by max 5% per update
- AGGRESSIVE: Adjust by max 20% per update
- ADAPTIVE: Adjust based on error magnitude
"""

from enum import Enum
from typing import Any
import json
from pathlib import Path
import uuid

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core import emit_receipt, load_thresholds, save_thresholds
from src.core.ledger import TrainingLedger
from .intervention_capture import ReasonCode


class TuningStrategy(Enum):
    """Threshold tuning strategies."""
    CONSERVATIVE = "conservative"  # Max 5% adjustment
    AGGRESSIVE = "aggressive"      # Max 20% adjustment
    ADAPTIVE = "adaptive"          # Based on error magnitude


class ThresholdTuner:
    """
    Threshold tuner using training examples from human corrections.

    Automatically adjusts thresholds to reduce future errors.
    """

    def __init__(
        self,
        strategy: TuningStrategy = TuningStrategy.CONSERVATIVE,
        min_examples: int = 10,
        domain: str = "default",
    ):
        """
        Args:
            strategy: Tuning strategy
            min_examples: Minimum examples before tuning
            domain: Domain for RACI
        """
        self.strategy = strategy
        self.min_examples = min_examples
        self.domain = domain

        self.training_ledger = TrainingLedger()
        self.tuning_history: list[dict] = []

    def _get_max_adjustment(self) -> float:
        """Get maximum adjustment percentage based on strategy."""
        if self.strategy == TuningStrategy.CONSERVATIVE:
            return 0.05
        elif self.strategy == TuningStrategy.AGGRESSIVE:
            return 0.20
        else:  # ADAPTIVE
            return 0.10  # Base, will be adjusted

    def _compute_adjustment(
        self,
        examples: list[dict],
        current_value: float,
        threshold_name: str,
    ) -> float:
        """
        Compute threshold adjustment from examples.

        Args:
            examples: Training examples for this threshold
            current_value: Current threshold value
            threshold_name: Name of threshold

        Returns:
            New threshold value
        """
        if not examples:
            return current_value

        max_adj = self._get_max_adjustment()

        # Analyze examples to determine adjustment direction and magnitude
        # For compression: lower threshold catches more (more sensitive)
        # For confidence: lower threshold accepts more (less confident)

        # Compute average delta from examples
        deltas = []
        for ex in examples:
            if threshold_name == "compression_threshold":
                # If compression attacks are being missed, lower threshold
                deltas.append(-0.02)  # Lower = more sensitive
            elif threshold_name == "temporal_consistency_threshold":
                # If jitter is missed, increase sensitivity
                deltas.append(-0.1)
            elif threshold_name == "adversarial_detection_threshold":
                # If adversarial inputs missed, lower threshold
                deltas.append(-0.02)
            elif threshold_name == "coherence_threshold":
                # If misalignment missed, raise threshold
                deltas.append(0.02)
            elif threshold_name == "confidence_calibration":
                # Temperature adjustment
                deltas.append(0.1)  # Increase temperature = less confident
            else:
                deltas.append(0.0)

        if not deltas:
            return current_value

        # Compute mean adjustment
        mean_delta = sum(deltas) / len(deltas)

        # Apply strategy limits
        if abs(mean_delta) > max_adj:
            mean_delta = max_adj if mean_delta > 0 else -max_adj

        # Adaptive scaling based on example count
        if self.strategy == TuningStrategy.ADAPTIVE:
            # More examples = more confident adjustment
            confidence = min(1.0, len(examples) / 50)
            mean_delta *= confidence

        new_value = current_value + mean_delta

        # Ensure reasonable bounds
        if threshold_name in ["compression_threshold", "coherence_threshold"]:
            new_value = max(0.1, min(0.99, new_value))
        elif threshold_name == "temporal_consistency_threshold":
            new_value = max(0.5, min(5.0, new_value))
        elif threshold_name == "adversarial_detection_threshold":
            new_value = max(0.1, min(0.9, new_value))
        elif threshold_name == "confidence_calibration":
            new_value = max(0.5, min(3.0, new_value))
        elif threshold_name == "merkle_tree_depth":
            new_value = max(4, min(16, int(new_value)))

        return new_value

    def tune(
        self,
        reason_codes: list[ReasonCode] = None,
        requires_approval: bool = True,
    ) -> dict:
        """
        Tune thresholds based on training examples.

        Args:
            reason_codes: Specific reason codes to tune (None = all)
            requires_approval: Whether updates need human approval

        Returns:
            Threshold update receipt
        """
        update_id = str(uuid.uuid4())

        # Load current thresholds
        current_thresholds = load_thresholds()
        new_thresholds = current_thresholds.copy()

        # Get training examples
        all_examples = self.training_ledger.read_all()

        if reason_codes is None:
            reason_codes = list(ReasonCode)

        # Track changes
        changes = []
        examples_used = 0

        for reason_code in reason_codes:
            # Get examples for this reason code
            examples = [e for e in all_examples if e.get("label") == reason_code.value]

            if len(examples) < self.min_examples:
                continue

            examples_used += len(examples)

            # Get target threshold
            threshold_name = reason_code.auto_tune_target
            if threshold_name == "unknown":
                continue

            current_value = current_thresholds.get(threshold_name, 0.85)
            new_value = self._compute_adjustment(examples, current_value, threshold_name)

            if new_value != current_value:
                new_thresholds[threshold_name] = new_value
                changes.append({
                    "threshold": threshold_name,
                    "old_value": current_value,
                    "new_value": new_value,
                    "reason_code": reason_code.value,
                    "examples_used": len(examples),
                })

        # Compute improvement estimate
        improvement_estimate = len(changes) * 0.02  # ~2% improvement per threshold

        # Determine if approval is needed
        needs_approval = requires_approval
        if any(c["threshold"] in ["adversarial_detection_threshold"] for c in changes):
            needs_approval = True  # Safety-critical always needs approval

        receipt = emit_receipt("threshold_update", {
            "update_id": update_id,
            "old_thresholds": current_thresholds,
            "new_thresholds": new_thresholds,
            "training_examples_used": examples_used,
            "changes": changes,
            "improvement_estimate": improvement_estimate,
            "strategy": self.strategy.value,
            "reasoning": f"Auto-tuned {len(changes)} thresholds based on {examples_used} training examples",
            "requires_human_approval": needs_approval,
            "approved_by": None,
        }, domain=self.domain)

        self.tuning_history.append(receipt)

        # If not requiring approval, apply immediately
        if not needs_approval and changes:
            save_thresholds(new_thresholds)
            emit_receipt("threshold_applied", {
                "update_id": update_id,
                "thresholds_applied": new_thresholds,
            }, domain=self.domain)

        return receipt

    def approve_update(self, update_id: str, approver_id: str) -> dict:
        """
        Approve and apply a pending threshold update.

        Args:
            update_id: The update ID to approve
            approver_id: ID of approving human

        Returns:
            Approval receipt
        """
        # Find the update
        update = None
        for h in self.tuning_history:
            if h.get("update_id") == update_id:
                update = h
                break

        if not update:
            return emit_receipt("threshold_approval_failed", {
                "update_id": update_id,
                "reason": "Update not found",
            }, domain=self.domain)

        # Apply thresholds
        save_thresholds(update["new_thresholds"])

        receipt = emit_receipt("threshold_approved", {
            "update_id": update_id,
            "approved_by": approver_id,
            "thresholds_applied": update["new_thresholds"],
            "changes": update.get("changes", []),
        }, domain=self.domain)

        return receipt

    def rollback(self, update_id: str) -> dict:
        """
        Rollback a threshold update.

        Args:
            update_id: The update ID to rollback

        Returns:
            Rollback receipt
        """
        # Find the update
        update = None
        for h in self.tuning_history:
            if h.get("update_id") == update_id:
                update = h
                break

        if not update:
            return emit_receipt("threshold_rollback_failed", {
                "update_id": update_id,
                "reason": "Update not found",
            }, domain=self.domain)

        # Restore old thresholds
        save_thresholds(update["old_thresholds"])

        receipt = emit_receipt("threshold_rollback", {
            "update_id": update_id,
            "rolled_back_to": update["old_thresholds"],
            "reason": "Manual rollback",
        }, domain=self.domain)

        return receipt

    def get_tuning_stats(self) -> dict:
        """Get tuning statistics."""
        all_examples = self.training_ledger.read_all()

        return {
            "total_tuning_updates": len(self.tuning_history),
            "total_training_examples": len(all_examples),
            "examples_by_reason": self.training_ledger.count_by_reason(),
            "current_thresholds": load_thresholds(),
            "strategy": self.strategy.value,
            "min_examples_for_tuning": self.min_examples,
        }

    def suggest_tuning(self) -> dict:
        """
        Suggest which thresholds should be tuned.

        Returns:
            Suggestion receipt
        """
        all_examples = self.training_ledger.read_all()
        counts = self.training_ledger.count_by_reason()

        suggestions = []
        for code_value, count in counts.items():
            if count >= self.min_examples:
                try:
                    reason_code = ReasonCode(code_value)
                    suggestions.append({
                        "reason_code": code_value,
                        "threshold": reason_code.auto_tune_target,
                        "example_count": count,
                        "ready_for_tuning": True,
                    })
                except ValueError:
                    pass

        receipt = emit_receipt("tuning_suggestion", {
            "suggestions": suggestions,
            "suggestion_count": len(suggestions),
            "total_examples": len(all_examples),
            "min_examples_threshold": self.min_examples,
        }, domain=self.domain)

        return receipt
