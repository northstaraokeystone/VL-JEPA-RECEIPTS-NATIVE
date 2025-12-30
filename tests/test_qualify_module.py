"""Tests for module qualification."""

import pytest


def test_qualify_module_hunt():
    """Test module qualifies as HUNT."""
    from meta.qualify_module import qualify_module

    result = qualify_module(
        {"name": "x_authenticity", "target_company": "x", "estimated_savings": 200_000_000},
        {"pain_points": ["deepfakes"]},
        48.0,
    )

    assert result["verdict"] == "HUNT"
    assert result["qualification_score"] >= 0.75


def test_qualify_module_pass_low_roi():
    """Test module disqualifies due to low ROI."""
    from meta.qualify_module import qualify_module

    result = qualify_module(
        {"name": "low_value", "target_company": "test", "estimated_savings": 10_000_000},
        {"pain_points": []},
        48.0,
    )

    assert result["verdict"] in ["PASS", "DEFER"]


def test_qualify_module_defer_slow_proof():
    """Test module deferred due to slow proof time."""
    from meta.qualify_module import qualify_module

    result = qualify_module(
        {"name": "neuralink_bci", "target_company": "neuralink", "estimated_savings": 100_000_000},
        {"pain_points": ["safety"]},
        200.0,  # >T+48h
    )

    assert result["verdict"] == "DEFER"


def test_module_qualifier_batch():
    """Test batch qualification."""
    from meta.qualify_module import ModuleQualifier

    qualifier = ModuleQualifier()

    modules = [
        {"name": "x_authenticity", "target_company": "x"},
        {"name": "tesla_fsd", "target_company": "tesla"},
        {"name": "low_value", "target_company": "test", "estimated_savings": 1000},
    ]

    hunt_modules, summary = qualifier.qualify_all(modules)

    assert "x_authenticity" in hunt_modules
    assert "tesla_fsd" in hunt_modules
    assert summary["hunt_count"] >= 2


def test_module_qualifier_build_order():
    """Test build order prioritization."""
    from meta.qualify_module import ModuleQualifier

    qualifier = ModuleQualifier()

    modules = [
        {"name": "x_authenticity", "target_company": "x"},
        {"name": "tesla_fsd", "target_company": "tesla"},
        {"name": "grok_verifiable", "target_company": "xai"},
    ]

    qualifier.qualify_all(modules)
    order = qualifier.get_build_order()

    assert len(order) >= 2
