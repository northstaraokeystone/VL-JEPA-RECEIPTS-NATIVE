"""
Module Qualification Protocol - Singularity 1

Pre-deployment qualification gate for all modules.
Modules must qualify BEFORE building to avoid waste.

Qualification gates:
- >$50M problem?
- API buyer exists?
- T+48h provable?
- Elon pain point?

Zero-cost PASS → don't build, move to next opportunity
"""

from enum import Enum
from typing import Any
import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core import emit_receipt, dual_hash


class QualificationVerdict(Enum):
    """Qualification decision outcomes."""
    HUNT = "HUNT"      # Qualify → proceed to build
    PASS = "PASS"      # Disqualify → skip, zero cost
    DEFER = "DEFER"    # Qualify but not now → queue for later


# Pre-built qualification knowledge
QUALIFICATION_KNOWLEDGE = {
    "x_authenticity": {
        "estimated_roi": 300_000_000,  # $200-400M
        "api_buyers": ["x_corp", "regulators", "advertisers"],
        "t48h_provable": True,
        "elon_pain_points": ["deepfakes", "bots", "misinformation"],
        "default_verdict": "HUNT",
    },
    "tesla_fsd": {
        "estimated_roi": 400_000_000,  # $300-500M
        "api_buyers": ["tesla", "insurers", "nhtsa"],
        "t48h_provable": True,
        "elon_pain_points": ["nhtsa_scrutiny", "liability_proof", "safety_verification"],
        "default_verdict": "HUNT",
    },
    "grok_verifiable": {
        "estimated_roi": 500_000_000,  # $500M+
        "api_buyers": ["enterprises", "regulators", "x_premium"],
        "t48h_provable": True,
        "elon_pain_points": ["trust_gap", "hallucination_concerns", "enterprise_adoption"],
        "default_verdict": "HUNT",
    },
    "spacex_mars": {
        "estimated_roi": 100_000_000,  # Mission-critical, hard to quantify
        "api_buyers": ["nasa", "spacex"],
        "t48h_provable": True,
        "elon_pain_points": ["autonomy_proof", "delayed_verification", "mission_critical"],
        "default_verdict": "HUNT",
    },
    "starlink_edge": {
        "estimated_roi": 150_000_000,  # $100-200M
        "api_buyers": ["spacex"],
        "t48h_provable": False,  # Partial - needs more time
        "elon_pain_points": ["bandwidth_costs", "edge_latency"],
        "default_verdict": "DEFER",
    },
    "neuralink_bci": {
        "estimated_roi": 50_000_000,  # FDA pathway value
        "api_buyers": ["neuralink", "fda"],
        "t48h_provable": False,  # Needs >6 months
        "elon_pain_points": ["safety_proof", "fda_approval"],
        "default_verdict": "DEFER",
    },
}


def qualify_module(
    module_spec: dict,
    market_research: dict,
    proof_complexity: float,
) -> dict:
    """
    Pre-deployment qualification gate for a module.

    Args:
        module_spec: Module specification
            - name: str
            - target_company: str
            - estimated_savings: int
            - features: list
        market_research: Market research data
            - pain_points: list
            - competitors: list
            - regulatory_drivers: list
        proof_complexity: Time to prove in hours (T+0 to T+N)

    Returns:
        Qualification receipt with verdict
    """
    module_name = module_spec.get("name", "unknown")

    # Check pre-built knowledge first
    knowledge = QUALIFICATION_KNOWLEDGE.get(module_name, {})

    # Extract or compute qualification factors
    estimated_roi = module_spec.get("estimated_savings", 0) or knowledge.get("estimated_roi", 0)
    api_buyers = knowledge.get("api_buyers", [])
    t48h_provable = proof_complexity <= 48 and knowledge.get("t48h_provable", True)
    elon_pain_points = knowledge.get("elon_pain_points", [])

    # Compute pain point match
    research_pains = set(p.lower() for p in market_research.get("pain_points", []))
    known_pains = set(p.lower() for p in elon_pain_points)
    pain_match = bool(research_pains & known_pains) or len(known_pains) > 0

    # Qualification logic
    reasons = []
    verdict = QualificationVerdict.HUNT

    if estimated_roi < 50_000_000:
        verdict = QualificationVerdict.PASS
        reasons.append(f"ROI ${estimated_roi:,} < $50M threshold")

    if len(api_buyers) == 0:
        verdict = QualificationVerdict.PASS
        reasons.append("No API buyers identified")

    if not t48h_provable:
        # Only defer, not pass - still valuable
        if verdict == QualificationVerdict.HUNT:
            verdict = QualificationVerdict.DEFER
        reasons.append(f"Proof complexity {proof_complexity}h > T+48h")

    if not pain_match and verdict == QualificationVerdict.HUNT:
        verdict = QualificationVerdict.PASS
        reasons.append("No Elon pain point match")

    # Compute qualification score
    score = 0.0
    if estimated_roi >= 50_000_000:
        score += 0.25
    if len(api_buyers) > 0:
        score += 0.25
    if t48h_provable:
        score += 0.25
    if pain_match:
        score += 0.25

    # Override with default if no specific disqualifications
    if not reasons and knowledge.get("default_verdict"):
        verdict = QualificationVerdict[knowledge["default_verdict"]]

    receipt = emit_receipt("module_qualification", {
        "module_name": module_name,
        "target_company": module_spec.get("target_company", "unknown"),
        "verdict": verdict.value,
        "qualification_score": score,
        "estimated_roi": estimated_roi,
        "api_buyer_count": len(api_buyers),
        "api_buyers": api_buyers,
        "t48h_provable": t48h_provable,
        "proof_complexity_hours": proof_complexity,
        "elon_pain_match": pain_match,
        "elon_pain_points": elon_pain_points,
        "reasons": reasons if reasons else ["All criteria met"],
    }, domain="meta")

    return receipt


class ModuleQualifier:
    """
    Module qualification manager.

    Tracks qualification history and provides batch qualification.
    """

    def __init__(self):
        self.qualification_history: list[dict] = []
        self.verdicts: dict[str, QualificationVerdict] = {}

    def qualify(
        self,
        module_spec: dict,
        market_research: dict = None,
        proof_complexity: float = 48.0,
    ) -> dict:
        """
        Qualify a single module.

        Args:
            module_spec: Module specification
            market_research: Optional market research
            proof_complexity: Proof time in hours

        Returns:
            Qualification receipt
        """
        if market_research is None:
            market_research = {"pain_points": [], "competitors": [], "regulatory_drivers": []}

        receipt = qualify_module(module_spec, market_research, proof_complexity)

        self.qualification_history.append(receipt)
        self.verdicts[module_spec["name"]] = QualificationVerdict(receipt["verdict"])

        return receipt

    def qualify_all(
        self,
        module_specs: list[dict],
    ) -> tuple[list[dict], dict]:
        """
        Qualify all proposed modules.

        Args:
            module_specs: List of module specifications

        Returns:
            Tuple of (hunt_modules, summary_receipt)
        """
        hunt_modules = []
        pass_modules = []
        defer_modules = []

        for spec in module_specs:
            receipt = self.qualify(spec)
            verdict = QualificationVerdict(receipt["verdict"])

            if verdict == QualificationVerdict.HUNT:
                hunt_modules.append(spec["name"])
            elif verdict == QualificationVerdict.PASS:
                pass_modules.append(spec["name"])
            else:
                defer_modules.append(spec["name"])

        summary_receipt = emit_receipt("qualification_batch", {
            "total_modules": len(module_specs),
            "hunt_count": len(hunt_modules),
            "pass_count": len(pass_modules),
            "defer_count": len(defer_modules),
            "hunt_modules": hunt_modules,
            "pass_modules": pass_modules,
            "defer_modules": defer_modules,
            "qualification_rate": len(hunt_modules) / len(module_specs) if module_specs else 0,
        }, domain="meta")

        return hunt_modules, summary_receipt

    def get_build_order(self) -> list[str]:
        """
        Get recommended build order for qualified modules.

        Prioritizes by ROI and pain point urgency.
        """
        hunt_receipts = [
            r for r in self.qualification_history
            if r.get("verdict") == "HUNT"
        ]

        # Sort by qualification score and ROI
        sorted_receipts = sorted(
            hunt_receipts,
            key=lambda r: (r.get("qualification_score", 0), r.get("estimated_roi", 0)),
            reverse=True,
        )

        return [r["module_name"] for r in sorted_receipts]

    def get_deferred(self) -> list[str]:
        """Get list of deferred modules."""
        return [
            name for name, verdict in self.verdicts.items()
            if verdict == QualificationVerdict.DEFER
        ]

    def requalify(self, module_name: str, new_data: dict) -> dict:
        """
        Re-qualify a module with new data.

        Useful for deferred modules when conditions change.
        """
        # Find previous spec
        prev_receipts = [
            r for r in self.qualification_history
            if r.get("module_name") == module_name
        ]

        if prev_receipts:
            prev = prev_receipts[-1]
            spec = {
                "name": module_name,
                "target_company": prev.get("target_company", "unknown"),
                "estimated_savings": new_data.get("estimated_roi", prev.get("estimated_roi", 0)),
            }
        else:
            spec = {"name": module_name, "target_company": "unknown", "estimated_savings": 0}

        return self.qualify(
            spec,
            new_data.get("market_research"),
            new_data.get("proof_complexity", 48.0),
        )
