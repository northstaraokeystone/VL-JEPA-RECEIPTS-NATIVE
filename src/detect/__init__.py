"""Detect module: Adversarial detection and synthetic content detection."""

from .adversarial import (
    AdversarialDetector,
    detect_compression_asymmetry,
    detect_synthetic_patterns,
)

__all__ = [
    "AdversarialDetector",
    "detect_compression_asymmetry",
    "detect_synthetic_patterns",
]
