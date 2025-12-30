"""
Selective Decode Module - Core Receipt Gap #1

VL-JEPA's predictor uses masked modeling but lacks receipts for
WHICH frames were decoded and WHY. This module provides:

1. Entropy-based frame selection
2. Compression ratio tracking
3. Decode decision receipts
"""

import math
from typing import Any
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core import emit_receipt, dual_hash, StopRule, load_thresholds


def compute_entropy(data: np.ndarray | list) -> float:
    """
    Compute Shannon entropy of data.

    Args:
        data: Numpy array or list of values

    Returns:
        Entropy value (bits)
    """
    if isinstance(data, list):
        data = np.array(data)

    # Flatten if needed
    data = data.flatten()

    # Compute histogram
    if data.dtype in [np.float32, np.float64]:
        # For floats, bin into 256 bins
        hist, _ = np.histogram(data, bins=256, density=True)
    else:
        # For integers, use unique values
        _, counts = np.unique(data, return_counts=True)
        hist = counts / counts.sum()

    # Remove zeros to avoid log(0)
    hist = hist[hist > 0]

    # Compute entropy
    entropy = -np.sum(hist * np.log2(hist))

    return float(entropy)


def compute_compression_ratio(original: bytes, compressed: bytes | None = None) -> float:
    """
    Compute compression ratio.

    If compressed is None, simulates compression using gzip.

    Args:
        original: Original data bytes
        compressed: Compressed data bytes (optional)

    Returns:
        Compression ratio (0.0 to 1.0, lower = more compressible)
    """
    import gzip

    if compressed is None:
        compressed = gzip.compress(original, compresslevel=9)

    ratio = len(compressed) / len(original) if len(original) > 0 else 1.0

    return min(ratio, 1.0)


def entropy_gate(
    frame_data: np.ndarray,
    threshold: float | None = None,
    domain: str = "default",
) -> tuple[bool, dict]:
    """
    Entropy gate for frame selection.

    Frames with entropy below threshold are candidates for selective decoding.

    Args:
        frame_data: Frame data as numpy array
        threshold: Entropy threshold (default from config)
        domain: Domain for RACI assignment

    Returns:
        Tuple of (should_decode, receipt)
    """
    thresholds = load_thresholds()

    if threshold is None:
        # Default: entropy threshold based on typical video frame
        threshold = thresholds.get("entropy_threshold", 5.0)

    entropy = compute_entropy(frame_data)

    # Higher entropy = more information = should decode
    should_decode = entropy >= threshold

    receipt = emit_receipt("entropy_gate", {
        "frame_entropy": entropy,
        "threshold": threshold,
        "should_decode": should_decode,
        "frame_shape": list(frame_data.shape) if hasattr(frame_data, "shape") else None,
        "decision_reason": "high_entropy" if should_decode else "low_entropy_skip",
    }, domain=domain)

    return should_decode, receipt


class SelectiveDecoder:
    """
    Selective decoder with entropy-based frame selection and compression analysis.

    Provides receipts for all decode decisions.
    """

    def __init__(
        self,
        entropy_threshold: float | None = None,
        compression_threshold: float | None = None,
        domain: str = "default",
    ):
        thresholds = load_thresholds()

        self.entropy_threshold = entropy_threshold or thresholds.get("entropy_threshold", 5.0)
        self.compression_threshold = compression_threshold or thresholds.get("compression_threshold", 0.85)
        self.domain = domain

        self.decoded_count = 0
        self.skipped_count = 0
        self.frame_receipts: list[dict] = []

    def should_decode_frame(self, frame: np.ndarray, frame_idx: int) -> tuple[bool, dict]:
        """
        Decide whether to decode a frame.

        Args:
            frame: Frame data
            frame_idx: Frame index in sequence

        Returns:
            Tuple of (should_decode, receipt)
        """
        entropy = compute_entropy(frame)

        # Compute compression ratio (simulated)
        frame_bytes = frame.tobytes() if hasattr(frame, "tobytes") else bytes(frame)
        compression_ratio = compute_compression_ratio(frame_bytes)

        # Decode if high entropy OR low compression (anomaly detection)
        high_entropy = entropy >= self.entropy_threshold
        suspicious_compression = compression_ratio < self.compression_threshold

        should_decode = high_entropy or suspicious_compression

        decision_reason = []
        if high_entropy:
            decision_reason.append("high_entropy")
        if suspicious_compression:
            decision_reason.append("suspicious_compression")
        if not decision_reason:
            decision_reason.append("skip_low_priority")

        receipt = emit_receipt("selective_decode", {
            "frame_idx": frame_idx,
            "entropy": entropy,
            "entropy_threshold": self.entropy_threshold,
            "compression_ratio": compression_ratio,
            "compression_threshold": self.compression_threshold,
            "should_decode": should_decode,
            "decision_reason": ",".join(decision_reason),
            "frame_hash": dual_hash(frame_bytes),
        }, domain=self.domain)

        self.frame_receipts.append(receipt)

        if should_decode:
            self.decoded_count += 1
        else:
            self.skipped_count += 1

        return should_decode, receipt

    def process_video(self, frames: list[np.ndarray]) -> tuple[list[int], dict]:
        """
        Process a video, selecting frames to decode.

        Args:
            frames: List of frame arrays

        Returns:
            Tuple of (decoded_frame_indices, summary_receipt)
        """
        decoded_indices = []

        for idx, frame in enumerate(frames):
            should_decode, _ = self.should_decode_frame(frame, idx)
            if should_decode:
                decoded_indices.append(idx)

        # Emit summary receipt
        summary_receipt = emit_receipt("selective_decode_batch", {
            "total_frames": len(frames),
            "decoded_frames": len(decoded_indices),
            "skipped_frames": len(frames) - len(decoded_indices),
            "decode_ratio": len(decoded_indices) / len(frames) if frames else 0,
            "decoded_indices": decoded_indices[:100],  # Limit to first 100
            "entropy_threshold": self.entropy_threshold,
            "compression_threshold": self.compression_threshold,
        }, domain=self.domain)

        return decoded_indices, summary_receipt

    def get_stats(self) -> dict:
        """Get decoder statistics."""
        total = self.decoded_count + self.skipped_count
        return {
            "decoded_count": self.decoded_count,
            "skipped_count": self.skipped_count,
            "total_processed": total,
            "decode_ratio": self.decoded_count / total if total > 0 else 0,
            "receipt_count": len(self.frame_receipts),
        }


def stoprule_entropy_anomaly(entropy: float, expected_min: float, expected_max: float) -> None:
    """
    Stoprule for entropy anomalies.

    Triggers if entropy is outside expected range (potential adversarial input).
    """
    if entropy < expected_min or entropy > expected_max:
        raise StopRule(
            f"Entropy anomaly: {entropy} outside [{expected_min}, {expected_max}]",
            metric="entropy",
            delta=entropy - (expected_min + expected_max) / 2,
            action="escalate",
        )


def stoprule_compression_anomaly(ratio: float, threshold: float) -> None:
    """
    Stoprule for compression ratio anomalies.

    Very low compression ratio may indicate synthetic/adversarial content.
    """
    if ratio < threshold:
        raise StopRule(
            f"Compression anomaly: ratio {ratio} < threshold {threshold}",
            metric="compression_ratio",
            delta=ratio - threshold,
            action="escalate",
        )
