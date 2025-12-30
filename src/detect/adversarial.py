"""
Adversarial Detection Module - Core Receipt Gap #6

VL-JEPA doesn't natively detect adversarial inputs or deepfakes.
This module provides:

1. Compression asymmetry detection
2. Synthetic pattern recognition
3. Deepfake probability scoring
4. Adversarial input receipts
"""

import gzip
from typing import Any
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core import emit_receipt, dual_hash, StopRule, load_thresholds


def compute_compression_ratio(data: bytes) -> float:
    """
    Compute compression ratio.

    Real content typically compresses to 0.4-0.8.
    Synthetic/adversarial content often compresses differently.
    """
    if len(data) == 0:
        return 1.0

    compressed = gzip.compress(data, compresslevel=9)
    return len(compressed) / len(data)


def detect_compression_asymmetry(
    frame1: np.ndarray,
    frame2: np.ndarray,
    asymmetry_threshold: float = 0.15,
) -> tuple[bool, float, dict]:
    """
    Detect compression asymmetry between consecutive frames.

    Adversarial/synthetic videos often show unusual compression patterns.

    Args:
        frame1: First frame
        frame2: Second frame
        asymmetry_threshold: Threshold for flagging asymmetry

    Returns:
        Tuple of (is_asymmetric, asymmetry_score, details)
    """
    ratio1 = compute_compression_ratio(frame1.tobytes())
    ratio2 = compute_compression_ratio(frame2.tobytes())

    asymmetry = abs(ratio1 - ratio2)
    is_asymmetric = asymmetry > asymmetry_threshold

    return is_asymmetric, asymmetry, {
        "ratio1": ratio1,
        "ratio2": ratio2,
        "asymmetry": asymmetry,
        "threshold": asymmetry_threshold,
    }


def detect_synthetic_patterns(
    frame: np.ndarray,
    frequency_threshold: float = 0.3,
) -> tuple[bool, float, dict]:
    """
    Detect synthetic generation patterns in a frame.

    GAN-generated content often shows characteristic frequency patterns.

    Args:
        frame: Frame to analyze
        frequency_threshold: Threshold for synthetic detection

    Returns:
        Tuple of (is_synthetic, synthetic_score, details)
    """
    # Analyze frequency domain characteristics
    if len(frame.shape) == 3:
        # Use grayscale for frequency analysis
        gray = np.mean(frame, axis=-1) if frame.shape[-1] == 3 else frame[:, :, 0]
    else:
        gray = frame

    # Compute 2D FFT
    fft = np.fft.fft2(gray)
    fft_shift = np.fft.fftshift(fft)
    magnitude = np.abs(fft_shift)

    # Normalize
    magnitude = magnitude / (magnitude.max() + 1e-8)

    # Check for periodic artifacts (common in GANs)
    center = np.array(magnitude.shape) // 2
    radial_profile = []

    for r in range(min(center)):
        # Compute mean at radius r
        y, x = np.ogrid[:magnitude.shape[0], :magnitude.shape[1]]
        mask = np.abs(np.sqrt((x - center[1])**2 + (y - center[0])**2) - r) < 1
        radial_profile.append(float(magnitude[mask].mean()))

    radial_profile = np.array(radial_profile)

    # Check for unusual peaks (artifact of GAN generation)
    if len(radial_profile) > 10:
        mean_energy = radial_profile.mean()
        peak_ratio = radial_profile.max() / (mean_energy + 1e-8)
        synthetic_score = min(1.0, (peak_ratio - 1) / 10)
    else:
        synthetic_score = 0.0

    is_synthetic = synthetic_score > frequency_threshold

    return is_synthetic, synthetic_score, {
        "frequency_threshold": frequency_threshold,
        "radial_profile_length": len(radial_profile),
        "peak_ratio": peak_ratio if len(radial_profile) > 10 else 0,
    }


class AdversarialDetector:
    """
    Adversarial and synthetic content detector.

    Provides receipts for all detection operations.
    """

    def __init__(
        self,
        compression_threshold: float | None = None,
        synthetic_threshold: float | None = None,
        adversarial_threshold: float | None = None,
        domain: str = "default",
    ):
        thresholds = load_thresholds()

        self.compression_threshold = compression_threshold or thresholds.get("compression_threshold", 0.85)
        self.synthetic_threshold = synthetic_threshold or thresholds.get("synthetic_threshold", 0.30)
        self.adversarial_threshold = adversarial_threshold or thresholds.get("adversarial_detection_threshold", 0.30)
        self.domain = domain

        self.detection_count = 0
        self.adversarial_count = 0

    def detect_adversarial(
        self,
        frames: list[np.ndarray],
        embeddings: list[np.ndarray] = None,
    ) -> tuple[bool, float, dict]:
        """
        Comprehensive adversarial detection on video frames.

        Args:
            frames: List of video frames
            embeddings: Optional VL-JEPA embeddings

        Returns:
            Tuple of (is_adversarial, detection_score, receipt)
        """
        scores = []
        issues = []

        # 1. Check compression ratios
        compression_ratios = []
        for frame in frames:
            ratio = compute_compression_ratio(frame.tobytes())
            compression_ratios.append(ratio)

            if ratio < self.compression_threshold:
                issues.append(f"low_compression_{ratio:.2f}")

        # Check compression asymmetry between consecutive frames
        asymmetry_scores = []
        for i in range(len(frames) - 1):
            is_asym, asym_score, _ = detect_compression_asymmetry(
                frames[i], frames[i + 1]
            )
            asymmetry_scores.append(asym_score)
            if is_asym:
                issues.append(f"asymmetry_at_frame_{i}")

        # 2. Check synthetic patterns (sample frames)
        synthetic_scores = []
        sample_indices = list(range(0, len(frames), max(1, len(frames) // 5)))[:5]
        for idx in sample_indices:
            is_synth, synth_score, _ = detect_synthetic_patterns(frames[idx])
            synthetic_scores.append(synth_score)
            if is_synth:
                issues.append(f"synthetic_pattern_frame_{idx}")

        # 3. Compute aggregate detection score
        compression_score = 1.0 - (sum(compression_ratios) / len(compression_ratios))
        asymmetry_score = max(asymmetry_scores) if asymmetry_scores else 0.0
        synthetic_score = max(synthetic_scores) if synthetic_scores else 0.0

        # Weighted combination
        detection_score = (
            0.4 * compression_score +
            0.3 * asymmetry_score +
            0.3 * synthetic_score
        )

        is_adversarial = detection_score > self.adversarial_threshold

        self.detection_count += 1
        if is_adversarial:
            self.adversarial_count += 1

        # Classify attack type
        attack_type = "unknown"
        if synthetic_score > 0.5:
            attack_type = "synthetic_generation"
        elif asymmetry_score > 0.3:
            attack_type = "frame_manipulation"
        elif compression_score > 0.3:
            attack_type = "compression_attack"

        receipt = emit_receipt("adversarial", {
            "frame_count": len(frames),
            "detection_score": detection_score,
            "adversarial_threshold": self.adversarial_threshold,
            "is_adversarial": is_adversarial,
            "attack_classification": attack_type,
            "compression_asymmetry": max(asymmetry_scores) if asymmetry_scores else 0,
            "compression_scores": {
                "mean": float(np.mean(compression_ratios)),
                "min": float(np.min(compression_ratios)),
                "max": float(np.max(compression_ratios)),
            },
            "synthetic_score": synthetic_score,
            "issues_detected": issues[:10],  # Limit to first 10
            "mitigation": "flag_for_review" if is_adversarial else "none",
        }, domain=self.domain)

        return is_adversarial, detection_score, receipt

    def detect_deepfake(
        self,
        frame: np.ndarray,
        face_regions: list[tuple[int, int, int, int]] = None,
    ) -> tuple[bool, float, dict]:
        """
        Detect deepfake in a single frame.

        Args:
            frame: Video frame
            face_regions: Optional list of (x, y, w, h) face bounding boxes

        Returns:
            Tuple of (is_deepfake, confidence, receipt)
        """
        # Compression analysis
        compression_ratio = compute_compression_ratio(frame.tobytes())

        # Synthetic pattern analysis
        is_synthetic, synthetic_score, _ = detect_synthetic_patterns(frame)

        # Analyze face regions if provided
        face_scores = []
        if face_regions:
            for x, y, w, h in face_regions:
                face_crop = frame[y:y+h, x:x+w]
                if face_crop.size > 0:
                    face_ratio = compute_compression_ratio(face_crop.tobytes())
                    face_scores.append(1.0 - face_ratio)

        # Combine scores
        face_score = max(face_scores) if face_scores else 0.0
        deepfake_score = (
            0.4 * (1.0 - compression_ratio) +
            0.3 * synthetic_score +
            0.3 * face_score
        )

        is_deepfake = deepfake_score > self.adversarial_threshold

        receipt = emit_receipt("deepfake_detection", {
            "compression_ratio": compression_ratio,
            "synthetic_score": synthetic_score,
            "face_regions_analyzed": len(face_regions) if face_regions else 0,
            "face_anomaly_score": face_score,
            "deepfake_score": deepfake_score,
            "threshold": self.adversarial_threshold,
            "is_deepfake": is_deepfake,
            "confidence": deepfake_score if is_deepfake else 1.0 - deepfake_score,
        }, domain=self.domain)

        return is_deepfake, deepfake_score, receipt

    def get_stats(self) -> dict:
        """Get detector statistics."""
        return {
            "detection_count": self.detection_count,
            "adversarial_count": self.adversarial_count,
            "adversarial_rate": self.adversarial_count / self.detection_count if self.detection_count > 0 else 0,
        }


def stoprule_adversarial_detected(score: float, threshold: float, attack_type: str) -> None:
    """Stoprule for adversarial input detection."""
    if score > threshold:
        raise StopRule(
            f"Adversarial input detected: {attack_type} (score: {score})",
            metric="adversarial_score",
            delta=score - threshold,
            action="halt",
        )


def stoprule_novel_attack(attack_signature: str) -> None:
    """Stoprule for novel attack patterns."""
    raise StopRule(
        f"Novel attack pattern detected: {attack_signature}",
        metric="novel_attack",
        delta=-1.0,
        action="escalate",
    )
