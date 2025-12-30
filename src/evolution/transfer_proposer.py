"""
Transfer Proposer - Singularity 5 (Part 1)

Auto-proposes cross-domain pattern transfers.
High-performing patterns are shared between domains.

Transfer types:
- FULL: Copy entire module architecture
- PARTIAL: Copy specific components (thresholds, params)
- HYBRID: Create offspring combining 2+ sources
"""

from enum import Enum
from typing import Any
import uuid
import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core import emit_receipt, dual_hash


class TransferType(Enum):
    """Types of cross-domain transfer."""
    FULL = "FULL"          # Complete architecture transfer
    PARTIAL = "PARTIAL"    # Component transfer (thresholds, params)
    HYBRID = "HYBRID"      # Combine multiple sources into new offspring


# Cross-domain similarity matrix
SIMILARITY_MATRIX = {
    ("x_authenticity", "tesla_fsd"): 0.85,
    ("x_authenticity", "grok_verifiable"): 0.65,
    ("tesla_fsd", "spacex_mars"): 0.78,
    ("grok_verifiable", "neuralink_bci"): 0.72,
    ("x_authenticity", "starlink_edge"): 0.80,
    ("tesla_fsd", "spacex_mars"): 0.88,
}


class TransferProposer:
    """
    Proposes cross-domain transfers for high-performing modules.

    Analyzes similarity and proposes FULL, PARTIAL, or HYBRID transfers.
    """

    def __init__(
        self,
        similarity_threshold: float = 0.70,
        domain: str = "default",
    ):
        self.similarity_threshold = similarity_threshold
        self.domain = domain

        self.proposals: list[dict] = []

    def compute_similarity(self, source: str, target: str) -> float:
        """
        Compute similarity between source and target domains.

        Uses pre-computed matrix or string-based fallback.
        """
        # Check matrix
        key = (source, target)
        if key in SIMILARITY_MATRIX:
            return SIMILARITY_MATRIX[key]

        # Check reverse
        key_rev = (target, source)
        if key_rev in SIMILARITY_MATRIX:
            return SIMILARITY_MATRIX[key_rev]

        # String-based fallback
        source_parts = set(source.lower().split("_"))
        target_parts = set(target.lower().split("_"))

        common = len(source_parts & target_parts)
        total = len(source_parts | target_parts)

        return common / total if total > 0 else 0.0

    def propose_transfer(
        self,
        source_module: str,
        source_effectiveness: float,
        all_modules: list[str],
    ) -> list[dict]:
        """
        Propose transfers from a high-performing source module.

        Args:
            source_module: Source module ID
            source_effectiveness: Source module effectiveness score
            all_modules: List of all module IDs

        Returns:
            List of transfer proposal receipts
        """
        proposals = []

        for target in all_modules:
            if target == source_module:
                continue

            similarity = self.compute_similarity(source_module, target)

            if similarity > self.similarity_threshold:
                # Determine transfer type based on similarity
                if similarity > 0.85:
                    transfer_type = TransferType.FULL
                    expected_gain = 0.08
                elif similarity > 0.75:
                    transfer_type = TransferType.PARTIAL
                    expected_gain = 0.05
                else:
                    transfer_type = TransferType.HYBRID
                    expected_gain = 0.03

                # Risk assessment
                risk = self._assess_risk(source_module, target)

                proposal = emit_receipt("transfer_proposal", {
                    "proposal_id": str(uuid.uuid4()),
                    "source_module": source_module,
                    "source_effectiveness": source_effectiveness,
                    "target_domain": target,
                    "similarity_score": similarity,
                    "transfer_type": transfer_type.value,
                    "expected_gain": expected_gain,
                    "risk_assessment": risk,
                    "approval_required": risk["safety_critical"],
                    "expires_in_days": 30,
                }, domain=self.domain)

                proposals.append(proposal)
                self.proposals.append(proposal)

        return proposals

    def _assess_risk(self, source: str, target: str) -> dict:
        """Assess risk of transfer."""
        safety_critical = any(
            x in target.lower() for x in ["fsd", "mars", "bci", "neuralink"]
        )

        return {
            "safety_critical": safety_critical,
            "latency_impact": "low" if "edge" not in target else "medium",
            "cost_impact": "low",
            "rollback_complexity": "low" if not safety_critical else "high",
        }

    def propose_hybrid(
        self,
        sources: list[tuple[str, float]],  # (module_id, effectiveness)
        target_domain: str,
    ) -> dict:
        """
        Propose a hybrid offspring from multiple sources.

        Args:
            sources: List of (module_id, effectiveness) tuples
            target_domain: Target domain for hybrid

        Returns:
            Hybrid proposal receipt
        """
        source_ids = [s[0] for s in sources]
        avg_effectiveness = sum(s[1] for s in sources) / len(sources)

        # Compute combined similarity
        similarities = []
        for src, _ in sources:
            sim = self.compute_similarity(src, target_domain)
            similarities.append(sim)

        combined_similarity = sum(similarities) / len(similarities)

        proposal = emit_receipt("transfer_proposal", {
            "proposal_id": str(uuid.uuid4()),
            "source_module": "+".join(source_ids),
            "source_effectiveness": avg_effectiveness,
            "target_domain": target_domain,
            "similarity_score": combined_similarity,
            "transfer_type": TransferType.HYBRID.value,
            "expected_gain": 0.10,  # Hybrids typically outperform parents
            "parent_modules": source_ids,
            "parent_effectiveness": [s[1] for s in sources],
            "risk_assessment": {
                "safety_critical": False,
                "latency_impact": "low",
                "cost_impact": "low",
                "rollback_complexity": "medium",
            },
            "approval_required": False,  # Hybrids are exploratory
        }, domain=self.domain)

        self.proposals.append(proposal)
        return proposal

    def get_pending_proposals(self) -> list[dict]:
        """Get all pending proposals."""
        return [p for p in self.proposals if p.get("approved_by") is None]

    def get_approved_proposals(self) -> list[dict]:
        """Get all approved proposals."""
        return [p for p in self.proposals if p.get("approved_by") is not None]

    def approve_proposal(self, proposal_id: str, approver_id: str) -> dict:
        """Approve a transfer proposal."""
        for p in self.proposals:
            if p.get("proposal_id") == proposal_id:
                p["approved_by"] = approver_id

                return emit_receipt("transfer_approved", {
                    "proposal_id": proposal_id,
                    "approved_by": approver_id,
                    "source_module": p.get("source_module"),
                    "target_domain": p.get("target_domain"),
                    "transfer_type": p.get("transfer_type"),
                }, domain=self.domain)

        return emit_receipt("transfer_approval_failed", {
            "proposal_id": proposal_id,
            "reason": "Proposal not found",
        }, domain=self.domain)
