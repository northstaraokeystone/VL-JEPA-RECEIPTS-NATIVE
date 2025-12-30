"""Tests for X/Twitter authenticity module."""

import pytest
import numpy as np


def test_verify_media_authenticity(sample_frames):
    """Test media authenticity verification."""
    from x_twitter.authenticity import verify_media_authenticity

    verdict, score, receipt = verify_media_authenticity(sample_frames)

    assert verdict in ["AUTHENTIC", "SUSPICIOUS", "MANIPULATED"]
    assert 0 <= score <= 1
    assert receipt["receipt_type"] == "x_authenticity"


def test_x_authenticity_verifier(sample_frames, sample_embeddings):
    """Test XAuthenticityVerifier class."""
    from x_twitter.authenticity import XAuthenticityVerifier

    verifier = XAuthenticityVerifier(tenant_id="test_tenant")

    verdict, score, receipt = verifier.verify(sample_frames, sample_embeddings)

    assert "event_id" in receipt
    assert "raci" in receipt
    assert receipt["tenant_id"] == "test_tenant"


def test_batch_processor(sample_frames):
    """Test batch processing."""
    from x_twitter.batch_processor import XBatchProcessor

    processor = XBatchProcessor(batch_size=10)

    items = [{"frames": sample_frames} for _ in range(5)]

    results, batch_receipt = processor.process_batch(items)

    assert len(results) == 5
    assert batch_receipt["item_count"] == 5
