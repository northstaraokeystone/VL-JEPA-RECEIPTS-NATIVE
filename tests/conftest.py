"""Pytest configuration and fixtures."""

import pytest
import sys
from pathlib import Path
import tempfile
import os

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture(autouse=True)
def temp_ledger(tmp_path):
    """Use temporary ledger for each test."""
    ledger_path = tmp_path / "receipts.jsonl"
    training_path = tmp_path / "training_examples.jsonl"

    os.environ["VLJEPA_LEDGER_PATH"] = str(ledger_path)
    os.environ["VLJEPA_TRAINING_PATH"] = str(training_path)

    yield ledger_path

    # Cleanup
    if "VLJEPA_LEDGER_PATH" in os.environ:
        del os.environ["VLJEPA_LEDGER_PATH"]
    if "VLJEPA_TRAINING_PATH" in os.environ:
        del os.environ["VLJEPA_TRAINING_PATH"]


@pytest.fixture
def sample_frames():
    """Generate sample video frames."""
    import numpy as np
    return [np.random.randn(64, 64, 3).astype(np.float32) for _ in range(5)]


@pytest.fixture
def sample_embeddings():
    """Generate sample embeddings."""
    import numpy as np
    return [np.random.randn(768).astype(np.float32) for _ in range(5)]
