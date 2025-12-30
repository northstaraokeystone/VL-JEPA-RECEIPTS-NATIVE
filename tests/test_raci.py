"""Tests for RACI accountability."""

import pytest


def test_get_raci_for_event():
    """Test RACI lookup."""
    from governance.raci import get_raci_for_event

    raci = get_raci_for_event("deepfake_detection", "x_twitter")

    assert raci["responsible"] == "detection_algorithm"
    assert raci["accountable"] == "x_safety_team"
    assert "escalation_path" in raci


def test_raci_manager_assign():
    """Test RACI assignment."""
    from governance.raci import RACIManager

    manager = RACIManager(domain="x_twitter")

    receipt = manager.assign_raci("deepfake_detection", "event_123")

    assert receipt["receipt_type"] == "raci_assignment"
    assert receipt["responsible"] is not None
    assert receipt["accountable"] is not None


def test_raci_escalation():
    """Test RACI escalation."""
    from governance.raci import RACIManager

    manager = RACIManager(domain="x_twitter")

    # First assign RACI
    manager.assign_raci("deepfake_detection", "event_123")

    # Then escalate
    receipt = manager.escalate("event_123", "Critical threat detected", 1)

    assert receipt["receipt_type"] == "escalation"
    assert "escalated_to" in receipt


def test_validate_raci_chain():
    """Test RACI chain validation."""
    from governance.raci import validate_raci_chain

    receipts = [
        {"raci": {"responsible": "sys", "accountable": "team"}},
        {"raci": {"responsible": "sys2", "accountable": "team2"}},
    ]

    is_valid, issues = validate_raci_chain(receipts)

    assert is_valid is True
    assert len(issues) == 0


def test_validate_raci_chain_missing_accountable():
    """Test RACI chain validation with missing accountable."""
    from governance.raci import validate_raci_chain

    receipts = [
        {"raci": {"responsible": "sys"}},  # Missing accountable
    ]

    is_valid, issues = validate_raci_chain(receipts)

    assert is_valid is False
    assert len(issues) > 0
