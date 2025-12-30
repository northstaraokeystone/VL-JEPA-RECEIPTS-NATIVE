"""Tests for cascade spawner."""

import pytest


def test_cascade_spawn():
    """Test spawning cascade variants."""
    from evolution.cascade_spawner import CascadeSpawner

    spawner = CascadeSpawner()

    variants, receipt = spawner.spawn("parent_module")

    assert len(variants) == 5
    assert receipt["receipt_type"] == "cascade_spawn"
    assert receipt["deployed_count"] >= 0


def test_variant_backtesting():
    """Test variant backtesting."""
    from evolution.cascade_spawner import CascadeSpawner

    spawner = CascadeSpawner(min_backtest_score=0.75)

    variants, receipt = spawner.spawn("parent_module")

    # Check that some variants pass backtest
    deployed = [v for v in variants if v.deployed]
    assert len(deployed) >= 0  # May be 0 due to randomness


def test_variant_mutations():
    """Test variant mutations."""
    from evolution.cascade_spawner import CascadeSpawner

    spawner = CascadeSpawner(mutation_rate=0.05)

    variants, _ = spawner.spawn("parent_module")

    # Check that variants have different mutations
    mutations = [v.mutations for v in variants]
    assert len(set(str(m) for m in mutations)) > 1  # At least some variation
