"""Tests for threshold tuner."""

import pytest


def test_threshold_tuner_tune():
    """Test threshold tuning."""
    from learning.threshold_tuner import ThresholdTuner, TuningStrategy
    from learning.intervention_capture import InterventionCapture, ReasonCode

    # Create some training examples first
    capture = InterventionCapture()
    for i in range(15):
        capture.capture(
            {"verdict": "wrong", "confidence": 0.9, "compression_ratio": 0.8},
            "correct",
            ReasonCode.COMPRESSION_FAILURE,
            f"Test {i}",
            f"operator_{i}",
        )

    tuner = ThresholdTuner(strategy=TuningStrategy.CONSERVATIVE, min_examples=10)
    result = tuner.tune(requires_approval=False)

    assert result["receipt_type"] == "threshold_update"
    assert result["training_examples_used"] >= 10


def test_tuning_strategies():
    """Test different tuning strategies."""
    from learning.threshold_tuner import ThresholdTuner, TuningStrategy

    conservative = ThresholdTuner(strategy=TuningStrategy.CONSERVATIVE)
    aggressive = ThresholdTuner(strategy=TuningStrategy.AGGRESSIVE)

    assert conservative._get_max_adjustment() == 0.05
    assert aggressive._get_max_adjustment() == 0.20


def test_suggest_tuning():
    """Test tuning suggestions."""
    from learning.threshold_tuner import ThresholdTuner

    tuner = ThresholdTuner()
    result = tuner.suggest_tuning()

    assert result["receipt_type"] == "tuning_suggestion"
    assert "suggestions" in result
