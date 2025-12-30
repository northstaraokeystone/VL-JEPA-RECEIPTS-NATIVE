"""Tests for topology classifier."""

import pytest


def test_topology_classifier_nascent():
    """Test nascent topology classification."""
    from evolution.topology_classifier import TopologyClassifier, TopologyState

    classifier = TopologyClassifier()

    # Low effectiveness receipts
    receipts = [{"success": False} for _ in range(10)]

    result = classifier.classify("test_module", receipts)

    assert result["new_topology"] == "NASCENT"
    assert result["effectiveness_score"] < 0.5


def test_topology_classifier_graduate():
    """Test graduation topology classification."""
    from evolution.topology_classifier import TopologyClassifier, TopologyState

    classifier = TopologyClassifier()

    # High effectiveness receipts
    receipts = [{"success": True} for _ in range(100)]

    result = classifier.classify("x_authenticity", receipts)

    # Should graduate if E >= 0.95 and A > 0.75
    assert result["effectiveness_score"] >= 0.95


def test_escape_velocities():
    """Test escape velocity lookup."""
    from evolution.topology_classifier import ESCAPE_VELOCITIES

    assert ESCAPE_VELOCITIES["x_authenticity"] == 0.95
    assert ESCAPE_VELOCITIES["tesla_fsd"] == 0.98
    assert ESCAPE_VELOCITIES["grok_verifiable"] == 0.90
