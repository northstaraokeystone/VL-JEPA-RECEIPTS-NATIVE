"""
Topology Classifier - Singularity 3 (Part 1)

Classifies module topology based on effectiveness metrics.
Modules evolve through lifecycle: NASCENT -> MATURING -> GRADUATED -> CASCADE

Key metrics:
- E (Effectiveness): (problems_solved - problems_remaining) / total_receipts
- A (Autonomy): auto_approved_decisions / total_decisions
- T (Transfer Potential): cross_domain_similarity > 0.70
"""

from enum import Enum
from typing import Any
import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core import emit_receipt, dual_hash
from src.core.ledger import Ledger


class TopologyState(Enum):
    """Module topology states."""
    NASCENT = "NASCENT"       # Still learning, E < 0.50
    MATURING = "MATURING"     # Improving, 0.50 <= E < V_esc
    GRADUATED = "GRADUATED"   # Autonomous, E >= V_esc AND A > 0.75
    CASCADE = "CASCADE"       # Spawned variant
    TRANSFER = "TRANSFER"     # Cross-domain transfer in progress
    PRUNED = "PRUNED"         # Deactivated


class TopologyAction(Enum):
    """Actions triggered by topology classification."""
    CONTINUE = "CONTINUE"     # Keep optimizing
    GRADUATE = "GRADUATE"     # Ready for autonomous operation
    CASCADE = "CASCADE"       # Spawn variants
    TRANSFER = "TRANSFER"     # Propose cross-domain transfer
    PRUNE = "PRUNE"           # Deactivate module


# Escape velocities per domain
ESCAPE_VELOCITIES = {
    "x_authenticity": 0.95,
    "tesla_fsd": 0.98,
    "grok_verifiable": 0.90,
    "spacex_mars": 0.95,
    "starlink_edge": 0.85,
    "neuralink_bci": 0.90,
    "default": 0.90,
}


class TopologyClassifier:
    """
    Topology classifier for module evolution.

    Tracks effectiveness, autonomy, and transfer potential.
    """

    def __init__(self, domain: str = "default"):
        self.domain = domain
        self.ledger = Ledger()

        # Module state tracking
        self.module_states: dict[str, TopologyState] = {}
        self.module_metrics: dict[str, dict] = {}
        self.classification_history: list[dict] = []

    def compute_effectiveness(self, module_id: str, receipts: list[dict]) -> float:
        """
        Compute effectiveness score for a module.

        E = (successful_receipts) / total_receipts
        """
        if not receipts:
            return 0.0

        # Count successful receipts (those without errors/violations)
        successful = sum(
            1 for r in receipts
            if not r.get("is_adversarial", False)
            and r.get("verdict", "success") not in ["error", "violation", "failed"]
        )

        return successful / len(receipts)

    def compute_autonomy(self, module_id: str, receipts: list[dict]) -> float:
        """
        Compute autonomy score for a module.

        A = auto_approved / total_decisions
        """
        if not receipts:
            return 0.0

        # Count decisions that didn't require human intervention
        total_decisions = len([r for r in receipts if r.get("receipt_type") != "human_intervention"])
        interventions = len([r for r in receipts if r.get("receipt_type") == "human_intervention"])

        if total_decisions == 0:
            return 0.0

        return (total_decisions - interventions) / total_decisions

    def compute_transfer_potential(self, module_id: str, similar_modules: list[str]) -> float:
        """
        Compute transfer potential score.

        T = max(similarity with any other module)
        """
        if not similar_modules:
            return 0.0

        # Simplified: based on domain name similarity
        similarities = []
        for other in similar_modules:
            # Simple string-based similarity
            common_chars = len(set(module_id) & set(other))
            total_chars = len(set(module_id) | set(other))
            similarities.append(common_chars / total_chars if total_chars > 0 else 0)

        return max(similarities) if similarities else 0.0

    def classify(
        self,
        module_id: str,
        receipts: list[dict] = None,
        similar_modules: list[str] = None,
    ) -> dict:
        """
        Classify module topology based on metrics.

        Args:
            module_id: Module identifier
            receipts: Recent receipts for the module
            similar_modules: List of similar module IDs for transfer potential

        Returns:
            Topology classification receipt
        """
        if receipts is None:
            receipts = self.ledger.read_all()
            receipts = [r for r in receipts if r.get("module_id") == module_id]

        if similar_modules is None:
            similar_modules = list(self.module_states.keys())

        # Compute metrics
        e_score = self.compute_effectiveness(module_id, receipts)
        a_score = self.compute_autonomy(module_id, receipts)
        t_score = self.compute_transfer_potential(module_id, similar_modules)

        # Get escape velocity for domain
        domain_key = module_id.split("_")[0] if "_" in module_id else "default"
        v_esc = ESCAPE_VELOCITIES.get(module_id, ESCAPE_VELOCITIES.get(domain_key, 0.90))

        # Current state
        current_state = self.module_states.get(module_id, TopologyState.NASCENT)

        # Determine new state and action
        new_state = current_state
        action = TopologyAction.CONTINUE

        if e_score < 0.50:
            new_state = TopologyState.NASCENT
            action = TopologyAction.CONTINUE
        elif e_score < v_esc:
            new_state = TopologyState.MATURING
            action = TopologyAction.CONTINUE
        elif e_score >= v_esc and a_score > 0.75:
            if current_state != TopologyState.GRADUATED:
                new_state = TopologyState.GRADUATED
                action = TopologyAction.GRADUATE
            else:
                action = TopologyAction.CASCADE
        elif t_score > 0.70:
            new_state = TopologyState.TRANSFER
            action = TopologyAction.TRANSFER

        # Check for pruning (declining effectiveness over 3 classifications)
        history = [h for h in self.classification_history if h.get("module_id") == module_id]
        if len(history) >= 3:
            recent_e = [h.get("effectiveness_score", 0) for h in history[-3:]]
            if all(recent_e[i] > recent_e[i+1] for i in range(len(recent_e)-1)):
                new_state = TopologyState.PRUNED
                action = TopologyAction.PRUNE

        # Update state
        self.module_states[module_id] = new_state
        self.module_metrics[module_id] = {
            "effectiveness": e_score,
            "autonomy": a_score,
            "transfer_potential": t_score,
        }

        # Emit classification receipt
        receipt = emit_receipt("topology_classification", {
            "module_id": module_id,
            "old_topology": current_state.value,
            "new_topology": new_state.value,
            "effectiveness_score": e_score,
            "autonomy_score": a_score,
            "transfer_score": t_score,
            "escape_velocity": v_esc,
            "action": action.value,
            "reasoning": self._get_reasoning(e_score, a_score, t_score, v_esc, action),
            "receipts_analyzed": len(receipts),
        }, domain=self.domain)

        self.classification_history.append(receipt)

        return receipt

    def _get_reasoning(
        self,
        e: float,
        a: float,
        t: float,
        v_esc: float,
        action: TopologyAction,
    ) -> str:
        """Generate human-readable reasoning for classification."""
        if action == TopologyAction.PRUNE:
            return "Declining effectiveness over 3 consecutive classifications"
        elif action == TopologyAction.GRADUATE:
            return f"E={e:.2f} >= V_esc={v_esc:.2f} AND A={a:.2f} > 0.75"
        elif action == TopologyAction.CASCADE:
            return f"Already graduated, ready to spawn variants"
        elif action == TopologyAction.TRANSFER:
            return f"High transfer potential T={t:.2f} > 0.70"
        else:
            if e < 0.50:
                return f"E={e:.2f} < 0.50, still nascent"
            else:
                return f"E={e:.2f} < V_esc={v_esc:.2f}, still maturing"

    def get_module_status(self, module_id: str) -> dict:
        """Get current status for a module."""
        return {
            "module_id": module_id,
            "state": self.module_states.get(module_id, TopologyState.NASCENT).value,
            "metrics": self.module_metrics.get(module_id, {}),
        }

    def get_all_module_status(self) -> dict:
        """Get status for all tracked modules."""
        return {
            module_id: self.get_module_status(module_id)
            for module_id in self.module_states
        }

    def get_graduated_modules(self) -> list[str]:
        """Get list of graduated modules ready for cascade."""
        return [
            module_id for module_id, state in self.module_states.items()
            if state == TopologyState.GRADUATED
        ]

    def get_transfer_candidates(self) -> list[str]:
        """Get modules with high transfer potential."""
        return [
            module_id for module_id, metrics in self.module_metrics.items()
            if metrics.get("transfer_potential", 0) > 0.70
        ]
