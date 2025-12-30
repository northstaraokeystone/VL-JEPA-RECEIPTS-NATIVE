"""Tests for selective decode module."""

import pytest
import numpy as np


def test_compute_entropy():
    """Test entropy computation."""
    from gate.selective_decode import compute_entropy

    # Random data should have high entropy
    random_data = np.random.rand(100, 100)
    entropy = compute_entropy(random_data)
    assert entropy > 0

    # Constant data should have low entropy
    constant_data = np.ones((100, 100))
    low_entropy = compute_entropy(constant_data)
    assert low_entropy < entropy


def test_compute_compression_ratio():
    """Test compression ratio computation."""
    from gate.selective_decode import compute_compression_ratio

    # Random data compresses poorly
    random_data = np.random.bytes(1000)
    ratio = compute_compression_ratio(random_data)
    assert 0 <= ratio <= 1

    # Repetitive data compresses well
    repetitive_data = b"a" * 1000
    low_ratio = compute_compression_ratio(repetitive_data)
    assert low_ratio < ratio


def test_entropy_gate(sample_frames):
    """Test entropy gate."""
    from gate.selective_decode import entropy_gate

    should_decode, receipt = entropy_gate(sample_frames[0])

    assert isinstance(should_decode, bool)
    assert receipt["receipt_type"] == "entropy_gate"


def test_selective_decoder():
    """Test SelectiveDecoder class."""
    from gate.selective_decode import SelectiveDecoder

    decoder = SelectiveDecoder()

    frames = [np.random.randn(64, 64, 3).astype(np.float32) for _ in range(10)]

    decoded_indices, summary = decoder.process_video(frames)

    assert len(decoded_indices) <= len(frames)
    assert summary["total_frames"] == 10
