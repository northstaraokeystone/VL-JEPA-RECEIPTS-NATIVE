"""Tests for adversarial detection module."""

import pytest
import numpy as np


def test_detect_compression_asymmetry():
    """Test compression asymmetry detection."""
    from detect.adversarial import detect_compression_asymmetry

    frame1 = np.random.randn(64, 64, 3).astype(np.float32)
    frame2 = np.random.randn(64, 64, 3).astype(np.float32)

    is_asym, score, details = detect_compression_asymmetry(frame1, frame2)

    assert isinstance(is_asym, bool)
    assert 0 <= score


def test_detect_synthetic_patterns():
    """Test synthetic pattern detection."""
    from detect.adversarial import detect_synthetic_patterns

    frame = np.random.randn(64, 64, 3).astype(np.float32)

    is_synthetic, score, details = detect_synthetic_patterns(frame)

    assert isinstance(is_synthetic, bool)
    assert 0 <= score <= 1


def test_adversarial_detector(sample_frames):
    """Test AdversarialDetector class."""
    from detect.adversarial import AdversarialDetector

    detector = AdversarialDetector()

    is_adv, score, receipt = detector.detect_adversarial(sample_frames)

    assert isinstance(is_adv, bool)
    assert receipt["receipt_type"] == "adversarial"
    assert "attack_classification" in receipt


def test_deepfake_detection():
    """Test deepfake detection."""
    from detect.adversarial import AdversarialDetector

    detector = AdversarialDetector()

    frame = np.random.randn(128, 128, 3).astype(np.float32)

    is_deepfake, score, receipt = detector.detect_deepfake(frame)

    assert isinstance(is_deepfake, bool)
    assert receipt["receipt_type"] == "deepfake_detection"
