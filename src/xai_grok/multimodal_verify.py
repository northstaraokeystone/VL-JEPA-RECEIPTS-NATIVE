"""
xAI Grok Multimodal Verification Module

Verifiable AI responses with confidence calibration.
Hallucination detection and enterprise-grade audit trails.

Target ROI: $500M+ annually
Pain points: Trust gap, hallucination concerns, enterprise adoption
"""

from typing import Any
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core import emit_receipt, dual_hash, load_thresholds
from src.reasoning import ConfidenceScorer
from src.verify import CrossModalVerifier
from src.detect import AdversarialDetector
from src.governance import RACIManager, ProvenanceTracker


def verify_grok_response(
    query: str,
    response: str,
    query_embedding: np.ndarray,
    response_embedding: np.ndarray,
    confidence_logits: np.ndarray = None,
) -> tuple[bool, float, dict]:
    """
    Verify a Grok response.

    Args:
        query: User query
        response: Grok response
        query_embedding: Query embedding
        response_embedding: Response embedding
        confidence_logits: Optional confidence logits

    Returns:
        Tuple of (is_verified, confidence, receipt)
    """
    verifier = GrokVerifier()
    return verifier.verify(
        query, response, query_embedding, response_embedding, confidence_logits
    )


class GrokVerifier:
    """
    Grok response verification with confidence calibration.

    Provides enterprise-grade verification receipts.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.85,
        coherence_threshold: float = 0.80,
    ):
        self.domain = "xai_grok"

        thresholds = load_thresholds()
        self.confidence_threshold = confidence_threshold
        self.coherence_threshold = coherence_threshold

        # Initialize sub-modules
        self.confidence_scorer = ConfidenceScorer(
            confidence_threshold=confidence_threshold,
            domain=self.domain,
        )
        self.cross_modal_verifier = CrossModalVerifier(
            coherence_threshold=coherence_threshold,
            domain=self.domain,
        )
        self.adversarial_detector = AdversarialDetector(domain=self.domain)
        self.raci_manager = RACIManager(domain=self.domain)
        self.provenance_tracker = ProvenanceTracker(domain=self.domain)

    def verify(
        self,
        query: str,
        response: str,
        query_embedding: np.ndarray,
        response_embedding: np.ndarray,
        confidence_logits: np.ndarray = None,
    ) -> tuple[bool, float, dict]:
        """
        Full verification pipeline for Grok response.

        Args:
            query: User query
            response: Grok response
            query_embedding: Query embedding
            response_embedding: Response embedding
            confidence_logits: Optional confidence logits

        Returns:
            Tuple of (is_verified, confidence, receipt)
        """
        import uuid
        decision_id = str(uuid.uuid4())

        # Assign RACI
        raci_receipt = self.raci_manager.assign_raci(
            "multimodal_prediction",
            decision_id,
        )

        # 1. Confidence scoring
        if confidence_logits is not None:
            raw_conf, calibrated_conf, conf_receipt = self.confidence_scorer.score(
                confidence_logits
            )
        else:
            # Estimate confidence from embedding similarity
            raw_conf = float(np.dot(query_embedding, response_embedding) /
                           (np.linalg.norm(query_embedding) * np.linalg.norm(response_embedding) + 1e-8))
            calibrated_conf = (raw_conf + 1) / 2  # Normalize to 0-1
            conf_receipt = {"raw_confidence": raw_conf, "calibrated_confidence": calibrated_conf}

        # 2. Cross-modal coherence
        is_coherent, coherence_receipt = self.cross_modal_verifier.verify(
            query_embedding, response_embedding, response
        )
        coherence_score = coherence_receipt.get("aggregate_score", 0)

        # 3. Hallucination detection (simplified)
        hallucination_score = self._detect_hallucination(
            query_embedding, response_embedding, calibrated_conf
        )

        # 4. Emit provenance
        self.provenance_tracker.emit_decision_provenance(
            decision_id,
            "grok_response",
            dual_hash(query),
            dual_hash(response),
        )

        # Compute final verification
        is_verified = (
            calibrated_conf >= self.confidence_threshold
            and is_coherent
            and hallucination_score < 0.3
        )

        # Emit verification receipt
        receipt = emit_receipt("grok_verification", {
            "decision_id": decision_id,
            "query_hash": dual_hash(query),
            "response_hash": dual_hash(response),
            "raw_confidence": raw_conf,
            "calibrated_confidence": calibrated_conf,
            "confidence_threshold": self.confidence_threshold,
            "coherence_score": coherence_score,
            "coherence_threshold": self.coherence_threshold,
            "hallucination_score": hallucination_score,
            "is_verified": is_verified,
            "raci": raci_receipt.get("raci", {}),
        }, domain=self.domain)

        return is_verified, calibrated_conf, receipt

    def _detect_hallucination(
        self,
        query_emb: np.ndarray,
        response_emb: np.ndarray,
        confidence: float,
    ) -> float:
        """
        Detect hallucination based on embedding analysis.

        Returns hallucination score (0-1, higher = more likely hallucination).
        """
        # Simple heuristic: low coherence + high confidence = possible hallucination
        coherence = float(np.dot(query_emb, response_emb) /
                         (np.linalg.norm(query_emb) * np.linalg.norm(response_emb) + 1e-8))
        coherence = (coherence + 1) / 2  # Normalize

        # High confidence but low coherence is suspicious
        if confidence > 0.9 and coherence < 0.5:
            return 0.7
        elif confidence > 0.8 and coherence < 0.4:
            return 0.5
        else:
            return max(0, 0.5 - coherence)

    def emit_confidence_alert(
        self,
        decision_id: str,
        confidence: float,
    ) -> dict:
        """
        Emit alert for low confidence response.

        Args:
            decision_id: Decision identifier
            confidence: Confidence score

        Returns:
            Alert receipt
        """
        raci_receipt = self.raci_manager.assign_raci(
            "confidence_low",
            decision_id,
        )

        return emit_receipt("confidence_alert", {
            "decision_id": decision_id,
            "confidence": confidence,
            "threshold": self.confidence_threshold,
            "action": "flag_for_review",
            "raci": raci_receipt.get("raci", {}),
        }, domain=self.domain)

    def emit_hallucination_alert(
        self,
        decision_id: str,
        hallucination_score: float,
    ) -> dict:
        """
        Emit alert for potential hallucination.

        Args:
            decision_id: Decision identifier
            hallucination_score: Hallucination probability

        Returns:
            Alert receipt
        """
        raci_receipt = self.raci_manager.assign_raci(
            "hallucination_detected",
            decision_id,
        )

        return emit_receipt("hallucination_alert", {
            "decision_id": decision_id,
            "hallucination_score": hallucination_score,
            "action": "escalate_to_safety",
            "raci": raci_receipt.get("raci", {}),
        }, domain=self.domain)
