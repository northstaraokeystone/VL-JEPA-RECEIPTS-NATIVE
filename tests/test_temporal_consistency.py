"""Tests for temporal consistency module."""

import pytest
import numpy as np


def test_compute_temporal_merkle_tree(sample_frames):
    """Test temporal Merkle tree computation."""
    from verify.temporal_consistency import compute_temporal_merkle_tree

    root, window_hashes = compute_temporal_merkle_tree(sample_frames)

    assert root is not None
    assert len(window_hashes) >= 1


def test_detect_temporal_jitter(sample_embeddings):
    """Test temporal jitter detection."""
    from verify.temporal_consistency import detect_temporal_jitter

    anomalies, mean_dist = detect_temporal_jitter(sample_embeddings)

    assert isinstance(anomalies, list)
    assert mean_dist >= 0


def test_temporal_verifier(sample_frames, sample_embeddings):
    """Test TemporalVerifier class."""
    from verify.temporal_consistency import TemporalVerifier

    verifier = TemporalVerifier()

    is_consistent, receipt = verifier.verify_sequence(sample_frames, sample_embeddings)

    assert isinstance(is_consistent, bool)
    assert receipt["receipt_type"] == "temporal_consistency"
    assert "merkle_root" in receipt


def test_temporal_chain(sample_frames):
    """Test temporal chain generation."""
    from verify.temporal_consistency import TemporalVerifier

    verifier = TemporalVerifier()

    # Process multiple segments
    for _ in range(3):
        verifier.verify_sequence(sample_frames)

    root, receipt = verifier.get_merkle_chain()

    assert root is not None
    assert receipt["segment_count"] == 3
