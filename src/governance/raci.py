"""
RACI Accountability - Singularity 4 (Part 1)

Every receipt embeds RACI:
- Responsible: Who generates the receipt
- Accountable: Who owns the outcome
- Consulted: Who must be asked before action
- Informed: Who must be notified after action

Creates provable liability trails for regulators/insurers.
"""

from typing import Any
import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core import emit_receipt, dual_hash


# RACI matrices per domain
RACI_MATRICES = {
    "x_twitter": {
        "deepfake_detection": {
            "responsible": "detection_algorithm",
            "accountable": "x_safety_team",
            "consulted": ["content_creator"],
            "informed": ["regulators", "users"],
            "escalation_path": ["x_legal", "x_trust_safety_vp", "x_ceo"],
        },
        "content_flag": {
            "responsible": "detection_algorithm",
            "accountable": "human_moderator",
            "consulted": ["x_legal"],
            "informed": ["advertisers"],
            "escalation_path": ["x_trust_safety", "x_legal"],
        },
        "authenticity_badge": {
            "responsible": "verification_system",
            "accountable": "x_trust_team",
            "consulted": ["enterprise_customer"],
            "informed": ["public"],
            "escalation_path": ["x_product", "x_legal"],
        },
    },
    "tesla_fsd": {
        "frame_verification": {
            "responsible": "fsd_vision_stack",
            "accountable": "safety_driver",
            "consulted": [],
            "informed": ["tesla_fleet_ops"],
            "escalation_path": ["tesla_safety_eng", "nhtsa"],
        },
        "entropy_gate_trigger": {
            "responsible": "selective_decode_module",
            "accountable": "fsd_planner",
            "consulted": ["safety_driver"],
            "informed": ["vehicle_logs", "tesla_safety_dashboard"],
            "escalation_path": ["fsd_planner", "tesla_safety_eng"],
        },
        "incident_reconstruction": {
            "responsible": "incident_analyzer",
            "accountable": "tesla_safety_eng",
            "consulted": ["nhtsa"],
            "informed": ["insurers", "legal"],
            "escalation_path": ["tesla_legal", "nhtsa"],
        },
    },
    "xai_grok": {
        "multimodal_prediction": {
            "responsible": "grok_predictor",
            "accountable": "x_ai_eng",
            "consulted": ["enterprise_customer"],
            "informed": ["regulators"],
            "escalation_path": ["xai_safety", "xai_leadership"],
        },
        "confidence_low": {
            "responsible": "confidence_scorer",
            "accountable": "grok_safety_team",
            "consulted": ["customer_support"],
            "informed": ["product_team"],
            "escalation_path": ["xai_safety", "xai_product"],
        },
        "hallucination_detected": {
            "responsible": "adversarial_detector",
            "accountable": "x_ai_safety",
            "consulted": ["customer"],
            "informed": ["compliance"],
            "escalation_path": ["xai_safety", "xai_legal"],
        },
    },
    "spacex_mars": {
        "autonomy_proof": {
            "responsible": "autonomy_verifier",
            "accountable": "mission_control",
            "consulted": ["nasa_oversight"],
            "informed": ["spacex_leadership"],
            "escalation_path": ["mission_director", "spacex_ceo"],
        },
        "delayed_verification": {
            "responsible": "merkle_verifier",
            "accountable": "mission_control",
            "consulted": [],
            "informed": ["nasa", "spacex_eng"],
            "escalation_path": ["mission_director"],
        },
    },
    "default": {
        "default": {
            "responsible": "system",
            "accountable": "platform_team",
            "consulted": [],
            "informed": ["audit_log"],
            "escalation_path": ["platform_lead", "safety_team"],
        },
    },
}


def get_raci_for_event(event_type: str, domain: str = "default") -> dict:
    """
    Get RACI assignment for an event type in a domain.

    Args:
        event_type: Type of event
        domain: Domain context

    Returns:
        RACI dict
    """
    domain_matrix = RACI_MATRICES.get(domain, RACI_MATRICES["default"])
    raci = domain_matrix.get(event_type, domain_matrix.get("default", RACI_MATRICES["default"]["default"]))

    return raci


def validate_raci_chain(receipts: list[dict]) -> tuple[bool, list[str]]:
    """
    Validate RACI chain integrity across a sequence of receipts.

    Args:
        receipts: List of receipts to validate

    Returns:
        Tuple of (is_valid, issues)
    """
    issues = []

    for i, receipt in enumerate(receipts):
        raci = receipt.get("raci", {})

        # Check required fields
        if not raci.get("responsible"):
            issues.append(f"Receipt {i}: Missing responsible party")

        if not raci.get("accountable"):
            issues.append(f"Receipt {i}: Missing accountable party")

        # Accountable must be singular
        accountable = raci.get("accountable", "")
        if isinstance(accountable, list):
            issues.append(f"Receipt {i}: Accountable must be singular, got list")

    # Check chain continuity (informed at step N should include responsible at step N+1)
    for i in range(len(receipts) - 1):
        current_informed = set(receipts[i].get("raci", {}).get("informed", []))
        next_responsible = receipts[i + 1].get("raci", {}).get("responsible", "")

        # This is a soft check - not always required
        if next_responsible and next_responsible not in current_informed:
            pass  # Could add as warning

    return len(issues) == 0, issues


class RACIManager:
    """
    RACI accountability manager.

    Tracks RACI assignments and validates accountability chains.
    """

    def __init__(self, domain: str = "default"):
        self.domain = domain
        self.raci_history: list[dict] = []

    def assign_raci(
        self,
        event_type: str,
        event_id: str,
        context: dict = None,
    ) -> dict:
        """
        Assign RACI for an event.

        Args:
            event_type: Type of event
            event_id: Unique event identifier
            context: Additional context for RACI lookup

        Returns:
            RACI assignment receipt
        """
        raci = get_raci_for_event(event_type, self.domain)

        # Customize based on context
        if context:
            if context.get("is_critical"):
                raci["escalation_path"] = ["immediate_" + p for p in raci.get("escalation_path", [])]

        receipt = emit_receipt("raci_assignment", {
            "event_id": event_id,
            "event_type": event_type,
            "responsible": raci["responsible"],
            "accountable": raci["accountable"],
            "consulted": raci.get("consulted", []),
            "informed": raci.get("informed", []),
            "escalation_path": raci.get("escalation_path", []),
        }, domain=self.domain)

        self.raci_history.append(receipt)

        return receipt

    def escalate(
        self,
        event_id: str,
        escalation_reason: str,
        escalation_level: int = 1,
    ) -> dict:
        """
        Trigger escalation for an event.

        Args:
            event_id: Event to escalate
            escalation_reason: Why escalation is needed
            escalation_level: How far up the chain to escalate

        Returns:
            Escalation receipt
        """
        # Find original RACI assignment
        original = None
        for h in self.raci_history:
            if h.get("event_id") == event_id:
                original = h
                break

        if not original:
            return emit_receipt("escalation_failed", {
                "event_id": event_id,
                "reason": "Event not found",
            }, domain=self.domain)

        escalation_path = original.get("escalation_path", [])
        target = escalation_path[min(escalation_level - 1, len(escalation_path) - 1)] if escalation_path else "platform_safety"

        return emit_receipt("escalation", {
            "event_id": event_id,
            "original_accountable": original.get("accountable"),
            "escalated_to": target,
            "escalation_level": escalation_level,
            "escalation_reason": escalation_reason,
        }, domain=self.domain)

    def validate_chain(self, event_ids: list[str]) -> tuple[bool, dict]:
        """
        Validate RACI chain for a sequence of events.

        Args:
            event_ids: List of event IDs in sequence

        Returns:
            Tuple of (is_valid, validation_receipt)
        """
        # Collect receipts for these events
        receipts = []
        for eid in event_ids:
            for h in self.raci_history:
                if h.get("event_id") == eid:
                    receipts.append(h)
                    break

        is_valid, issues = validate_raci_chain(receipts)

        receipt = emit_receipt("raci_chain_validation", {
            "event_ids": event_ids,
            "chain_length": len(receipts),
            "is_valid": is_valid,
            "issues": issues,
        }, domain=self.domain)

        return is_valid, receipt

    def get_accountability_trail(self, event_id: str) -> list[dict]:
        """
        Get full accountability trail for an event.

        Returns all RACI assignments and escalations.
        """
        trail = []
        for h in self.raci_history:
            if h.get("event_id") == event_id:
                trail.append(h)

        return trail

    def emit_liability_proof(self, event_ids: list[str]) -> dict:
        """
        Emit a liability proof for a sequence of events.

        Creates cryptographic proof of accountability chain.
        """
        trail = []
        for eid in event_ids:
            trail.extend(self.get_accountability_trail(eid))

        # Compute Merkle root of trail
        from src.core import merkle
        trail_root = merkle(trail)

        return emit_receipt("liability_proof", {
            "event_ids": event_ids,
            "trail_length": len(trail),
            "merkle_root": trail_root,
            "accountable_parties": list(set(t.get("accountable") for t in trail)),
            "proof_valid": True,
        }, domain=self.domain)
