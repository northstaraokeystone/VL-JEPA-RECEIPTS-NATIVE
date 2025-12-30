"""Tests for confidence scoring module."""

import pytest
import numpy as np


def test_compute_ece():
    """Test ECE computation."""
    from reasoning.confidence import compute_ece

    # Perfect calibration
    confidences = [0.1, 0.5, 0.9]
    accuracies = [False, True, True]

    ece = compute_ece(confidences, accuracies)
    assert 0 <= ece <= 1


def test_calibrate_confidence():
    """Test confidence calibration."""
    from reasoning.confidence import calibrate_confidence

    raw = 0.9
    calibrated, receipt = calibrate_confidence(raw, temperature=1.5)

    # Higher temperature should reduce confidence
    assert calibrated < raw
    assert receipt["receipt_type"] == "confidence_calibration"


def test_confidence_scorer():
    """Test ConfidenceScorer class."""
    from reasoning.confidence import ConfidenceScorer

    scorer = ConfidenceScorer(temperature=1.0)

    logits = np.array([1.0, 2.0, 0.5])

    raw, calibrated, receipt = scorer.score(logits)

    assert 0 <= raw <= 1
    assert 0 <= calibrated <= 1
    assert receipt["receipt_type"] == "confidence"


def test_ece_receipt():
    """Test ECE receipt emission."""
    from reasoning.confidence import ConfidenceScorer

    scorer = ConfidenceScorer()

    # Score some predictions with ground truth
    for i in range(10):
        logits = np.array([0.1, 0.9, 0.1])
        scorer.score(logits, prediction=1, ground_truth=1)

    receipt = scorer.emit_ece_receipt()

    assert receipt["receipt_type"] == "ece_check"
    assert "ece" in receipt
