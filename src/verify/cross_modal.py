"""
Cross-Modal Verification Module - Core Receipt Gap #4

VL-JEPA fuses video and language but doesn't provide receipts
for cross-modal coherence verification. This module provides:

1. Embedding alignment scoring
2. Semantic coherence measurement
3. Cross-modal consistency receipts
"""

from typing import Any
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core import emit_receipt, dual_hash, StopRule, load_thresholds


def compute_cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    if isinstance(a, list):
        a = np.array(a)
    if isinstance(b, list):
        b = np.array(b)

    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return float(np.dot(a, b) / (norm_a * norm_b))


def compute_coherence_score(
    video_embedding: np.ndarray,
    text_embedding: np.ndarray,
    method: str = "cosine",
) -> float:
    """
    Compute coherence score between video and text embeddings.

    Args:
        video_embedding: Video modality embedding
        text_embedding: Text modality embedding
        method: Scoring method ("cosine", "euclidean", "projection")

    Returns:
        Coherence score (0-1, higher = more coherent)
    """
    if method == "cosine":
        return (compute_cosine_similarity(video_embedding, text_embedding) + 1) / 2
    elif method == "euclidean":
        # Normalize to [0, 1] using exponential decay
        dist = np.linalg.norm(video_embedding - text_embedding)
        return float(np.exp(-dist / 10))
    elif method == "projection":
        # Project text onto video space and measure alignment
        v_norm = video_embedding / (np.linalg.norm(video_embedding) + 1e-8)
        projection = np.dot(text_embedding, v_norm) * v_norm
        alignment = compute_cosine_similarity(projection, text_embedding)
        return (alignment + 1) / 2
    else:
        raise ValueError(f"Unknown method: {method}")


class CrossModalVerifier:
    """
    Cross-modal verification for video-language coherence.

    Provides receipts for all cross-modal verification operations.
    """

    def __init__(
        self,
        coherence_threshold: float | None = None,
        methods: list[str] = None,
        domain: str = "default",
    ):
        thresholds = load_thresholds()

        self.coherence_threshold = coherence_threshold or thresholds.get("coherence_threshold", 0.80)
        self.methods = methods or ["cosine"]
        self.domain = domain

        self.verification_count = 0
        self.coherent_count = 0

    def verify(
        self,
        video_embedding: np.ndarray,
        text_embedding: np.ndarray,
        text_content: str = None,
    ) -> tuple[bool, dict]:
        """
        Verify cross-modal coherence.

        Args:
            video_embedding: Video modality embedding
            text_embedding: Text modality embedding
            text_content: Optional original text for audit

        Returns:
            Tuple of (is_coherent, receipt)
        """
        # Compute coherence using all methods
        scores = {}
        for method in self.methods:
            scores[method] = compute_coherence_score(
                video_embedding, text_embedding, method
            )

        # Aggregate score (mean of all methods)
        aggregate_score = sum(scores.values()) / len(scores)

        is_coherent = aggregate_score >= self.coherence_threshold

        self.verification_count += 1
        if is_coherent:
            self.coherent_count += 1

        receipt = emit_receipt("cross_modal", {
            "modalities": ["video", "text"],
            "coherence_scores": scores,
            "aggregate_score": aggregate_score,
            "coherence_threshold": self.coherence_threshold,
            "alignment_verified": is_coherent,
            "text_hash": dual_hash(text_content) if text_content else None,
            "video_embedding_hash": dual_hash(video_embedding.tobytes()),
            "text_embedding_hash": dual_hash(text_embedding.tobytes()),
        }, domain=self.domain)

        return is_coherent, receipt

    def verify_multimodal(
        self,
        embeddings: dict[str, np.ndarray],
        primary_modality: str = "video",
    ) -> tuple[bool, dict]:
        """
        Verify coherence across multiple modalities.

        Args:
            embeddings: Dict of modality_name -> embedding
            primary_modality: Primary modality to compare others against

        Returns:
            Tuple of (all_coherent, receipt)
        """
        if primary_modality not in embeddings:
            raise ValueError(f"Primary modality {primary_modality} not in embeddings")

        primary_emb = embeddings[primary_modality]
        scores = {}

        for modality, emb in embeddings.items():
            if modality != primary_modality:
                scores[f"{primary_modality}_to_{modality}"] = compute_coherence_score(
                    primary_emb, emb
                )

        aggregate_score = sum(scores.values()) / len(scores) if scores else 1.0
        all_coherent = aggregate_score >= self.coherence_threshold

        receipt = emit_receipt("multimodal_coherence", {
            "modalities": list(embeddings.keys()),
            "primary_modality": primary_modality,
            "pairwise_scores": scores,
            "aggregate_score": aggregate_score,
            "coherence_threshold": self.coherence_threshold,
            "all_aligned": all_coherent,
        }, domain=self.domain)

        return all_coherent, receipt

    def detect_misalignment(
        self,
        video_embedding: np.ndarray,
        text_embedding: np.ndarray,
        expected_alignment_vector: np.ndarray = None,
    ) -> tuple[list[str], dict]:
        """
        Detect specific misalignment issues between modalities.

        Args:
            video_embedding: Video modality embedding
            text_embedding: Text modality embedding
            expected_alignment_vector: Optional expected alignment direction

        Returns:
            Tuple of (misalignment_issues, receipt)
        """
        issues = []

        # Check direct coherence
        coherence = compute_coherence_score(video_embedding, text_embedding)
        if coherence < self.coherence_threshold:
            issues.append(f"low_coherence_{coherence:.2f}")

        # Check norm discrepancy (may indicate missing content)
        v_norm = np.linalg.norm(video_embedding)
        t_norm = np.linalg.norm(text_embedding)
        norm_ratio = v_norm / t_norm if t_norm > 0 else float("inf")

        if norm_ratio > 2.0 or norm_ratio < 0.5:
            issues.append(f"norm_mismatch_{norm_ratio:.2f}")

        # Check expected alignment if provided
        if expected_alignment_vector is not None:
            v_align = compute_cosine_similarity(video_embedding, expected_alignment_vector)
            t_align = compute_cosine_similarity(text_embedding, expected_alignment_vector)

            if abs(v_align - t_align) > 0.3:
                issues.append(f"alignment_divergence_{abs(v_align - t_align):.2f}")

        receipt = emit_receipt("misalignment_detection", {
            "issues_detected": issues,
            "issue_count": len(issues),
            "coherence_score": coherence,
            "video_norm": float(v_norm),
            "text_norm": float(t_norm),
            "norm_ratio": norm_ratio,
        }, domain=self.domain)

        return issues, receipt

    def get_stats(self) -> dict:
        """Get verifier statistics."""
        return {
            "verification_count": self.verification_count,
            "coherent_count": self.coherent_count,
            "coherence_rate": self.coherent_count / self.verification_count if self.verification_count > 0 else 0,
        }


def stoprule_cross_modal_mismatch(coherence: float, threshold: float) -> None:
    """Stoprule for critical cross-modal misalignment."""
    if coherence < threshold:
        raise StopRule(
            f"Cross-modal mismatch: coherence {coherence} < threshold {threshold}",
            metric="coherence",
            delta=coherence - threshold,
            action="escalate",
        )
