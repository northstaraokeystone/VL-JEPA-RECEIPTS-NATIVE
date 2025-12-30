"""
Pytest configuration and fixtures for Receipts-Native Compliance Tests.

This module provides fixtures for testing both compliant and non-compliant systems.
"""

import os
import sys
import pytest
import tempfile
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--system",
        action="store",
        default="examples.receipts_minimal",
        help="Module path to system under test (e.g., examples.receipts_minimal)"
    )


@pytest.fixture(autouse=True)
def temp_ledger(tmp_path):
    """Use temporary ledger for each test."""
    ledger_path = tmp_path / "receipts.jsonl"

    # Set environment variable for ledger path
    os.environ["RECEIPTS_LEDGER_PATH"] = str(ledger_path)

    yield ledger_path

    # Cleanup
    if "RECEIPTS_LEDGER_PATH" in os.environ:
        del os.environ["RECEIPTS_LEDGER_PATH"]


@pytest.fixture
def system_under_test(request, temp_ledger):
    """
    Fixture that provides the system under test.

    Uses --system command line option to determine which module to load.
    Default: examples.receipts_minimal
    """
    system_path = request.config.getoption("--system")

    # Import the system module
    try:
        # Try importing as a module path
        parts = system_path.split(".")
        module_name = ".".join(parts[:-1]) if len(parts) > 1 else parts[0]
        class_name = parts[-1] if len(parts) > 1 else None

        if class_name and class_name[0].isupper():
            # It's a class reference
            module = __import__(module_name, fromlist=[class_name])
            system_class = getattr(module, class_name)
        else:
            # It's a module with a default system
            module = __import__(system_path, fromlist=["SystemUnderTest"])
            if hasattr(module, "SystemUnderTest"):
                system_class = module.SystemUnderTest
            elif hasattr(module, "ReceiptsMinimal"):
                system_class = module.ReceiptsMinimal
            elif hasattr(module, "SimpleLogger"):
                system_class = module.SimpleLogger
            else:
                # Look for any class that looks like a system
                for name in dir(module):
                    obj = getattr(module, name)
                    if isinstance(obj, type) and hasattr(obj, "run_cycle"):
                        system_class = obj
                        break
                else:
                    pytest.fail(f"No system class found in {system_path}")

    except ImportError as e:
        # Try with starter prefix
        try:
            full_path = f"starter.{system_path}"
            parts = full_path.split(".")
            module_name = ".".join(parts[:-1])
            class_name = parts[-1]

            if class_name[0].isupper():
                module = __import__(module_name, fromlist=[class_name])
                system_class = getattr(module, class_name)
            else:
                module = __import__(full_path, fromlist=["SystemUnderTest"])
                system_class = getattr(module, "SystemUnderTest", None) or \
                              getattr(module, "ReceiptsMinimal", None)
        except ImportError:
            pytest.fail(f"Could not import system: {system_path}. Error: {e}")

    # Create and configure system instance
    try:
        from starter.core.receipt import set_ledger_path, reset_lineage
    except ImportError:
        try:
            from core.receipt import set_ledger_path, reset_lineage
        except ImportError:
            set_ledger_path = None
            reset_lineage = None

    if set_ledger_path:
        set_ledger_path(temp_ledger)
    if reset_lineage:
        reset_lineage()

    return system_class(ledger_path=temp_ledger)


@pytest.fixture
def sample_receipts(temp_ledger):
    """
    Generate a sample receipt chain for unit tests.

    Returns:
        List of 10 sample receipts with valid lineage
    """
    try:
        from starter.core.receipt import emit_receipt, reset_lineage, set_ledger_path
    except ImportError:
        from core.receipt import emit_receipt, reset_lineage, set_ledger_path

    set_ledger_path(temp_ledger)
    reset_lineage()

    receipts = []

    # Genesis receipt
    r0 = emit_receipt("ingest", {
        "source_type": "test",
        "data_size": 100,
    })
    receipts.append(r0)

    # Additional receipts
    for i in range(9):
        r = emit_receipt("ingest", {
            "source_type": "test",
            "data_size": 100 + i,
        })
        receipts.append(r)

    return receipts


@pytest.fixture
def passing_system(temp_ledger):
    """Fixture that provides a system that PASSES all compliance tests."""
    try:
        from starter.examples.receipts_minimal import ReceiptsMinimal
    except ImportError:
        from examples.receipts_minimal import ReceiptsMinimal

    try:
        from starter.core.receipt import set_ledger_path, reset_lineage
    except ImportError:
        from core.receipt import set_ledger_path, reset_lineage

    set_ledger_path(temp_ledger)
    reset_lineage()

    return ReceiptsMinimal(ledger_path=temp_ledger)


@pytest.fixture
def failing_system(temp_ledger):
    """Fixture that provides a system that FAILS all compliance tests."""
    try:
        from starter.examples.simple_logger import SimpleLogger
    except ImportError:
        from examples.simple_logger import SimpleLogger

    return SimpleLogger()
