"""Tests for Tesla FSD frame verification."""

import pytest
import numpy as np


def test_verify_fsd_frame():
    """Test FSD frame verification."""
    from tesla_fsd.frame_verify import verify_fsd_frame

    frame = np.random.randn(128, 128, 3).astype(np.float32)

    is_valid, receipt = verify_fsd_frame(frame, 0, "test_vehicle")

    assert isinstance(is_valid, bool)
    assert receipt["receipt_type"] == "fsd_frame_verification"
    assert "latency_ms" in receipt


def test_fsd_frame_verifier_sequence():
    """Test FSD frame verification sequence."""
    from tesla_fsd.frame_verify import FSDFrameVerifier

    verifier = FSDFrameVerifier(vehicle_id="test_vehicle")

    frames = [np.random.randn(64, 64, 3).astype(np.float32) for _ in range(10)]

    for i, frame in enumerate(frames):
        is_valid, receipt = verifier.verify_frame(frame, i)
        assert "frame_hash" in receipt


def test_fsd_latency_slo():
    """Test FSD latency SLO."""
    from tesla_fsd.frame_verify import FSDFrameVerifier

    verifier = FSDFrameVerifier(vehicle_id="test_vehicle", max_latency_ms=100.0)

    frame = np.random.randn(32, 32, 3).astype(np.float32)

    is_valid, receipt = verifier.verify_frame(frame, 0)

    # Should pass with small frame and generous latency
    assert receipt["latency_slo_met"] is True
