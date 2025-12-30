"""Tests for intervention capture."""

import pytest


def test_capture_intervention():
    """Test capturing a human intervention."""
    from learning.intervention_capture import capture_intervention, ReasonCode

    original = {"verdict": "authentic", "confidence": 0.9, "receipt_type": "test"}

    intervention, training_example = capture_intervention(
        original,
        "deepfake",
        ReasonCode.COMPRESSION_FAILURE,
        "Compression was too good",
        "operator_1",
    )

    assert intervention["receipt_type"] == "human_intervention"
    assert intervention["reason_code"] == "RC001"
    assert training_example["label"] == "RC001"


def test_intervention_capture_class():
    """Test InterventionCapture class."""
    from learning.intervention_capture import InterventionCapture, ReasonCode

    capture = InterventionCapture(auto_tune_threshold=3)

    # Capture 3 interventions with same reason
    for i in range(3):
        intervention, example, should_tune = capture.capture(
            {"verdict": "wrong", "receipt_type": "test"},
            "correct",
            ReasonCode.COMPRESSION_FAILURE,
            f"Test {i}",
            f"operator_{i}",
        )

    assert should_tune is True
    assert capture.intervention_count == 3


def test_reason_code_properties():
    """Test reason code properties."""
    from learning.intervention_capture import ReasonCode

    rc = ReasonCode.COMPRESSION_FAILURE
    assert rc.severity == "HIGH"
    assert rc.auto_tune_target == "compression_threshold"

    rc_critical = ReasonCode.ADVERSARIAL_NOVEL
    assert rc_critical.severity == "CRITICAL"
