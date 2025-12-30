"""Tests for transfer proposer."""

import pytest


def test_propose_transfer():
    """Test proposing a transfer."""
    from evolution.transfer_proposer import TransferProposer

    proposer = TransferProposer()

    proposals = proposer.propose_transfer(
        "x_authenticity",
        0.98,
        ["tesla_fsd", "grok_verifiable"],
    )

    # Should propose transfer to tesla_fsd (high similarity)
    assert len(proposals) >= 1
    tesla_proposal = [p for p in proposals if "tesla" in p.get("target_domain", "")]
    assert len(tesla_proposal) >= 1


def test_similarity_matrix():
    """Test similarity computation."""
    from evolution.transfer_proposer import TransferProposer

    proposer = TransferProposer()

    # Check pre-computed similarity
    sim = proposer.compute_similarity("x_authenticity", "tesla_fsd")
    assert sim > 0.7


def test_hybrid_proposal():
    """Test hybrid offspring proposal."""
    from evolution.transfer_proposer import TransferProposer

    proposer = TransferProposer()

    proposal = proposer.propose_hybrid(
        [("x_authenticity", 0.98), ("tesla_fsd", 0.92)],
        "adversarial_detector",
    )

    assert proposal["transfer_type"] == "HYBRID"
    assert "x_authenticity" in proposal["source_module"]
