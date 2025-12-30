"""
Model Provenance - Singularity 4 (Part 2)

Tracks model and policy provenance for every decision.
Creates audit trail for regulatory compliance.
"""

from typing import Any
import uuid
import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core import emit_receipt, dual_hash


def compute_model_provenance(
    model_id: str,
    model_version: str,
    model_weights: bytes = None,
    config: dict = None,
) -> dict:
    """
    Compute provenance hash for a model.

    Args:
        model_id: Model identifier
        model_version: Version string
        model_weights: Optional model weights bytes
        config: Optional model config

    Returns:
        Provenance dict with hashes
    """
    model_hash = dual_hash(model_weights) if model_weights else dual_hash(f"{model_id}:{model_version}")
    config_hash = dual_hash(json.dumps(config, sort_keys=True)) if config else None

    return {
        "model_id": model_id,
        "model_version": model_version,
        "model_hash": model_hash,
        "config_hash": config_hash,
    }


class ProvenanceTracker:
    """
    Tracks model and policy provenance for decisions.

    Creates audit trail for regulatory compliance.
    """

    def __init__(self, domain: str = "default"):
        self.domain = domain
        self.provenance_history: list[dict] = []

        # Current model state
        self.current_model: dict = {}
        self.current_policy: dict = {}

    def register_model(
        self,
        model_id: str,
        model_version: str,
        model_weights_hash: str = None,
        config: dict = None,
    ) -> dict:
        """
        Register a model for provenance tracking.

        Args:
            model_id: Model identifier
            model_version: Version string
            model_weights_hash: Pre-computed weights hash
            config: Model configuration

        Returns:
            Registration receipt
        """
        self.current_model = {
            "model_id": model_id,
            "model_version": model_version,
            "model_hash": model_weights_hash or dual_hash(f"{model_id}:{model_version}"),
            "config": config or {},
            "config_hash": dual_hash(json.dumps(config or {}, sort_keys=True)),
        }

        return emit_receipt("model_registered", {
            **self.current_model,
        }, domain=self.domain)

    def register_policy(
        self,
        policy_version: str,
        guardrail_thresholds: dict,
        tool_allowlist: list[str] = None,
        autonomy_level: str = "SUPERVISED",
    ) -> dict:
        """
        Register a policy for provenance tracking.

        Args:
            policy_version: Policy version string
            guardrail_thresholds: Safety thresholds
            tool_allowlist: Allowed tools
            autonomy_level: FULL, SUPERVISED, or MANUAL

        Returns:
            Registration receipt
        """
        self.current_policy = {
            "policy_version": policy_version,
            "policy_hash": dual_hash(json.dumps(guardrail_thresholds, sort_keys=True)),
            "guardrail_thresholds": guardrail_thresholds,
            "tool_allowlist": tool_allowlist or [],
            "autonomy_level": autonomy_level,
        }

        return emit_receipt("policy_registered", {
            **self.current_policy,
        }, domain=self.domain)

    def emit_decision_provenance(
        self,
        decision_id: str,
        decision_type: str,
        inputs_hash: str,
        output_hash: str,
    ) -> dict:
        """
        Emit provenance receipt for a decision.

        Args:
            decision_id: Unique decision identifier
            decision_type: Type of decision
            inputs_hash: Hash of decision inputs
            output_hash: Hash of decision outputs

        Returns:
            Provenance receipt
        """
        receipt = emit_receipt("model_provenance", {
            "decision_id": decision_id,
            "decision_type": decision_type,
            "model_id": self.current_model.get("model_id", "unknown"),
            "model_version": self.current_model.get("model_version", "unknown"),
            "model_hash": self.current_model.get("model_hash", "unknown"),
            "policy_version": self.current_policy.get("policy_version", "unknown"),
            "policy_hash": self.current_policy.get("policy_hash", "unknown"),
            "guardrail_thresholds": self.current_policy.get("guardrail_thresholds", {}),
            "tool_allowlist": self.current_policy.get("tool_allowlist", []),
            "autonomy_level": self.current_policy.get("autonomy_level", "SUPERVISED"),
            "inputs_hash": inputs_hash,
            "output_hash": output_hash,
        }, domain=self.domain)

        self.provenance_history.append(receipt)

        return receipt

    def verify_provenance(self, decision_id: str) -> tuple[bool, dict]:
        """
        Verify provenance for a past decision.

        Args:
            decision_id: Decision to verify

        Returns:
            Tuple of (is_valid, verification_receipt)
        """
        # Find original provenance
        original = None
        for p in self.provenance_history:
            if p.get("decision_id") == decision_id:
                original = p
                break

        if not original:
            return False, emit_receipt("provenance_verification_failed", {
                "decision_id": decision_id,
                "reason": "Decision not found",
            }, domain=self.domain)

        # Verify hashes
        model_valid = original.get("model_hash") == self.current_model.get("model_hash")
        policy_valid = original.get("policy_hash") == self.current_policy.get("policy_hash")

        is_valid = model_valid and policy_valid

        receipt = emit_receipt("provenance_verification", {
            "decision_id": decision_id,
            "model_valid": model_valid,
            "policy_valid": policy_valid,
            "is_valid": is_valid,
            "model_changed": not model_valid,
            "policy_changed": not policy_valid,
        }, domain=self.domain)

        return is_valid, receipt

    def get_decision_lineage(self, decision_id: str) -> list[dict]:
        """
        Get full lineage for a decision.

        Returns model and policy state at decision time.
        """
        for p in self.provenance_history:
            if p.get("decision_id") == decision_id:
                return [{
                    "decision_id": decision_id,
                    "model_id": p.get("model_id"),
                    "model_version": p.get("model_version"),
                    "policy_version": p.get("policy_version"),
                    "autonomy_level": p.get("autonomy_level"),
                    "timestamp": p.get("ts"),
                }]

        return []

    def emit_audit_report(self, start_ts: str, end_ts: str) -> dict:
        """
        Emit an audit report for a time period.

        Args:
            start_ts: Start timestamp (ISO8601)
            end_ts: End timestamp (ISO8601)

        Returns:
            Audit report receipt
        """
        # Filter decisions in time range
        decisions = [
            p for p in self.provenance_history
            if start_ts <= p.get("ts", "") <= end_ts
        ]

        # Compute statistics
        model_versions = set(p.get("model_version") for p in decisions)
        policy_versions = set(p.get("policy_version") for p in decisions)
        autonomy_levels = {}
        for p in decisions:
            level = p.get("autonomy_level", "UNKNOWN")
            autonomy_levels[level] = autonomy_levels.get(level, 0) + 1

        return emit_receipt("audit_report", {
            "period_start": start_ts,
            "period_end": end_ts,
            "total_decisions": len(decisions),
            "model_versions_used": list(model_versions),
            "policy_versions_used": list(policy_versions),
            "autonomy_distribution": autonomy_levels,
            "provenance_coverage": 1.0,  # All decisions have provenance
        }, domain=self.domain)
