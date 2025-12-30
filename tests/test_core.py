"""Tests for core module."""

import pytest
import json
from pathlib import Path


def test_dual_hash():
    """Test dual hash function."""
    from core import dual_hash

    result = dual_hash(b"test")
    assert ":" in result
    parts = result.split(":")
    assert len(parts) == 2
    assert len(parts[0]) == 64  # SHA256 hex
    assert len(parts[1]) == 64  # BLAKE3 hex


def test_dual_hash_string():
    """Test dual hash with string input."""
    from core import dual_hash

    result = dual_hash("test string")
    assert ":" in result


def test_emit_receipt():
    """Test receipt emission."""
    from core import emit_receipt

    receipt = emit_receipt("test", {"value": 42}, write_to_ledger=False)

    assert receipt["receipt_type"] == "test"
    assert "ts" in receipt
    assert "payload_hash" in receipt
    assert receipt["value"] == 42


def test_emit_receipt_with_raci():
    """Test receipt emission with RACI."""
    from core import emit_receipt

    receipt = emit_receipt("test", {"value": 42}, include_raci=True, write_to_ledger=False)

    assert "raci" in receipt
    assert "responsible" in receipt["raci"]
    assert "accountable" in receipt["raci"]


def test_merkle_empty():
    """Test Merkle tree with empty list."""
    from core import merkle

    result = merkle([])
    assert result is not None


def test_merkle_single():
    """Test Merkle tree with single item."""
    from core import merkle

    result = merkle(["item"])
    assert result is not None


def test_merkle_multiple():
    """Test Merkle tree with multiple items."""
    from core import merkle

    result = merkle(["a", "b", "c", "d"])
    assert result is not None


def test_merkle_deterministic():
    """Test Merkle tree is deterministic."""
    from core import merkle

    items = ["a", "b", "c"]
    result1 = merkle(items)
    result2 = merkle(items)
    assert result1 == result2


def test_stoprule():
    """Test StopRule exception."""
    from core import StopRule

    with pytest.raises(StopRule) as exc_info:
        raise StopRule("test error", metric="test", delta=0.1, action="halt")

    assert "test error" in str(exc_info.value)
    assert exc_info.value.metric == "test"
    assert exc_info.value.delta == 0.1
