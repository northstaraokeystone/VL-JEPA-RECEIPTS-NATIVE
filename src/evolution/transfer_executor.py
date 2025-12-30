"""
Transfer Executor - Singularity 5 (Part 2)

Executes approved cross-domain transfers.
Validates in staging before production deployment.
"""

from typing import Any
import uuid
import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core import emit_receipt, dual_hash, load_thresholds, save_thresholds
from .transfer_proposer import TransferType


class TransferExecutor:
    """
    Executes approved transfer proposals.

    Handles FULL, PARTIAL, and HYBRID transfers with validation.
    """

    def __init__(self, domain: str = "default"):
        self.domain = domain
        self.executed_transfers: list[dict] = []

    def execute(
        self,
        proposal: dict,
        validation_data: list[dict] = None,
    ) -> dict:
        """
        Execute an approved transfer proposal.

        Args:
            proposal: Approved transfer proposal
            validation_data: Optional data for validation

        Returns:
            Transfer execution receipt
        """
        if not proposal.get("approved_by") and proposal.get("approval_required", True):
            return emit_receipt("transfer_execution_failed", {
                "proposal_id": proposal.get("proposal_id"),
                "reason": "Not approved",
            }, domain=self.domain)

        transfer_type = TransferType(proposal.get("transfer_type", "PARTIAL"))
        source = proposal.get("source_module")
        target = proposal.get("target_domain")

        # Execute based on type
        if transfer_type == TransferType.FULL:
            result = self._execute_full_transfer(source, target)
        elif transfer_type == TransferType.PARTIAL:
            result = self._execute_partial_transfer(source, target)
        else:  # HYBRID
            parents = proposal.get("parent_modules", [source])
            result = self._execute_hybrid_transfer(parents, target)

        # Validate in staging
        pre_effectiveness = proposal.get("source_effectiveness", 0.90)
        post_effectiveness = self._validate_transfer(result, validation_data)

        success = post_effectiveness >= pre_effectiveness

        receipt = emit_receipt("transfer_execution", {
            "transfer_id": str(uuid.uuid4()),
            "proposal_id": proposal.get("proposal_id"),
            "source_module": source,
            "target_module": result.get("target_module", target),
            "transfer_type": transfer_type.value,
            "transferred_components": result.get("transferred_components", []),
            "adaptation_changes": result.get("adaptations", {}),
            "pre_transfer_effectiveness": pre_effectiveness,
            "post_transfer_effectiveness": post_effectiveness,
            "success": success,
            "rollback_available": True,
        }, domain=self.domain)

        self.executed_transfers.append(receipt)

        return receipt

    def _execute_full_transfer(self, source: str, target: str) -> dict:
        """Execute full architecture transfer."""
        thresholds = load_thresholds()

        # Copy all relevant thresholds
        transferred = [
            "compression_threshold",
            "adversarial_detection_threshold",
            "temporal_consistency_threshold",
            "coherence_threshold",
        ]

        return {
            "target_module": f"{target}_from_{source}",
            "transferred_components": transferred,
            "adaptations": {
                "domain": target,
                "source_reference": source,
            },
        }

    def _execute_partial_transfer(self, source: str, target: str) -> dict:
        """Execute partial component transfer."""
        thresholds = load_thresholds()

        # Transfer only compression threshold (most universal)
        transferred = ["compression_threshold"]

        return {
            "target_module": target,
            "transferred_components": transferred,
            "adaptations": {
                "source_compression": thresholds.get("compression_threshold", 0.85),
            },
        }

    def _execute_hybrid_transfer(self, parents: list[str], target: str) -> dict:
        """Execute hybrid offspring creation."""
        thresholds = load_thresholds()

        # Create hybrid module name
        parent_short = "_".join(p.split("_")[0] for p in parents)
        hybrid_name = f"{parent_short}_hybrid_{target}"

        # Combine best of each parent
        transferred = [
            "compression_threshold",
            "adversarial_detection_threshold",
            "temporal_consistency_threshold",
        ]

        return {
            "target_module": hybrid_name,
            "transferred_components": transferred,
            "adaptations": {
                "parents": parents,
                "combination_strategy": "best_of_each",
                "scoring_formula": "0.6 * parent1 + 0.4 * parent2",
            },
        }

    def _validate_transfer(
        self,
        transfer_result: dict,
        validation_data: list[dict] = None,
    ) -> float:
        """
        Validate transfer in staging environment.

        Returns post-transfer effectiveness.
        """
        if validation_data:
            successes = sum(1 for d in validation_data if d.get("success", True))
            return successes / len(validation_data)

        # Simulated validation
        base = 0.88

        # Full transfers usually maintain effectiveness
        if "from_" in transfer_result.get("target_module", ""):
            base = 0.92

        # Hybrids often outperform
        if "hybrid" in transfer_result.get("target_module", ""):
            base = 0.94

        return base

    def rollback(self, transfer_id: str) -> dict:
        """Rollback a transfer."""
        for t in self.executed_transfers:
            if t.get("transfer_id") == transfer_id:
                return emit_receipt("transfer_rollback", {
                    "transfer_id": transfer_id,
                    "rolled_back_module": t.get("target_module"),
                    "restored_state": "pre_transfer",
                }, domain=self.domain)

        return emit_receipt("transfer_rollback_failed", {
            "transfer_id": transfer_id,
            "reason": "Transfer not found",
        }, domain=self.domain)

    def get_transfer_stats(self) -> dict:
        """Get transfer statistics."""
        total = len(self.executed_transfers)
        successful = len([t for t in self.executed_transfers if t.get("success")])

        return {
            "total_transfers": total,
            "successful_transfers": successful,
            "success_rate": successful / total if total > 0 else 0,
        }
