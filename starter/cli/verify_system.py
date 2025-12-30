#!/usr/bin/env python3
"""
Receipts-Native Compliance Verification CLI

Command-line tool to verify if a system is receipts-native compliant.

Usage:
    python -m starter.cli.verify_system examples.receipts_minimal
    python -m starter.cli.verify_system examples.simple_logger
    python -m starter.cli.verify_system /path/to/your/system.py

Output:
    Running Receipts-Native Compliance Suite v1.1

    System: examples.receipts_minimal

    P1: Native Provenance         PASS
    P2: Cryptographic Lineage     PASS
    P3: Verifiable Causality      PASS
    P4: Query-as-Proof            PASS
    P5: Thermodynamic Governance  PASS
    P6: Receipts-Gated Progress   PASS

    RESULT: System is receipts-native (passed 6/6 tests)
"""

import argparse
import importlib
import os
import sys
import tempfile
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def load_system_class(module_path: str):
    """
    Load a system class from a module path.

    Args:
        module_path: Dotted module path (e.g., "examples.receipts_minimal")

    Returns:
        System class
    """
    try:
        # Try direct import
        parts = module_path.split(".")
        module_name = ".".join(parts[:-1]) if len(parts) > 1 else parts[0]

        if len(parts) > 1 and parts[-1][0].isupper():
            # Class specified
            module = importlib.import_module(module_name)
            return getattr(module, parts[-1])

        # Try to find a system class
        module = importlib.import_module(module_path)

        # Look for known class names
        for name in ["SystemUnderTest", "ReceiptsMinimal", "SimpleLogger"]:
            if hasattr(module, name):
                return getattr(module, name)

        # Look for any class with run_cycle method
        for name in dir(module):
            obj = getattr(module, name)
            if isinstance(obj, type) and hasattr(obj, "run_cycle"):
                return obj

        raise ImportError(f"No system class found in {module_path}")

    except ImportError:
        # Try with starter prefix
        try:
            full_path = f"starter.{module_path}"
            return load_system_class(full_path)
        except ImportError:
            raise ImportError(f"Could not import: {module_path}")


def run_test(system, test_name: str, test_func) -> tuple[bool, str]:
    """
    Run a single compliance test.

    Args:
        system: System under test
        test_name: Name of the test
        test_func: Test function to run

    Returns:
        Tuple of (passed, error_message)
    """
    try:
        test_func(system)
        return True, ""
    except AssertionError as e:
        return False, str(e)
    except Exception as e:
        return False, f"Error: {e}"


def verify_system(module_path: str, verbose: bool = False) -> dict:
    """
    Verify a system against the 6 receipts-native principles.

    Args:
        module_path: Path to system module
        verbose: Whether to print detailed output

    Returns:
        Results dictionary
    """
    from starter.tests.compliance_suite import (
        test_principle_1_native_provenance,
        test_principle_2_cryptographic_lineage,
        test_principle_3_verifiable_causality,
        test_principle_4_query_as_proof,
        test_principle_5_thermodynamic_governance,
        test_principle_6_receipts_gated_progress,
    )
    from starter.core.receipt import set_ledger_path, reset_lineage

    # Load system class
    system_class = load_system_class(module_path)

    # Create temp directory for ledger
    with tempfile.TemporaryDirectory() as tmp_dir:
        ledger_path = Path(tmp_dir) / "receipts.jsonl"
        os.environ["RECEIPTS_LEDGER_PATH"] = str(ledger_path)
        set_ledger_path(ledger_path)
        reset_lineage()

        # Create system instance
        system = system_class(ledger_path=ledger_path)

        # Define tests
        tests = [
            ("P1: Native Provenance", test_principle_1_native_provenance),
            ("P2: Cryptographic Lineage", test_principle_2_cryptographic_lineage),
            ("P3: Verifiable Causality", test_principle_3_verifiable_causality),
            ("P4: Query-as-Proof", test_principle_4_query_as_proof),
            ("P5: Thermodynamic Governance", test_principle_5_thermodynamic_governance),
            ("P6: Receipts-Gated Progress", test_principle_6_receipts_gated_progress),
        ]

        results = {
            "system": module_path,
            "passed": 0,
            "failed": 0,
            "principles": {},
        }

        # Run each test
        for name, test_func in tests:
            # Reset for each test
            reset_lineage()
            if ledger_path.exists():
                ledger_path.unlink()
            system = system_class(ledger_path=ledger_path)

            passed, error = run_test(system, name, test_func)

            if passed:
                results["passed"] += 1
                results["principles"][name] = {"status": "PASS", "error": None}
            else:
                results["failed"] += 1
                results["principles"][name] = {"status": "FAIL", "error": error}

        results["is_receipts_native"] = results["failed"] == 0

    return results


def print_results(results: dict, verbose: bool = False) -> None:
    """Print verification results."""
    print("\nRunning Receipts-Native Compliance Suite v1.1")
    print("=" * 50)
    print(f"\nSystem: {results['system']}\n")

    for name, info in results["principles"].items():
        status = info["status"]
        if status == "PASS":
            status_str = "\033[92mPASS\033[0m"  # Green
        else:
            status_str = "\033[91mFAIL\033[0m"  # Red

        print(f"  {name:<35} {status_str}")

        if verbose and info["error"]:
            error_lines = info["error"].split("\n")
            for line in error_lines[:3]:  # Show first 3 lines
                print(f"      {line}")

    print()

    if results["is_receipts_native"]:
        print(f"\033[92mRESULT: System is receipts-native (passed {results['passed']}/6 tests)\033[0m")
    else:
        print(f"\033[91mRESULT: System is NOT receipts-native (failed {results['failed']}/6 tests)\033[0m")
        if not verbose:
            print("\nRun with -v for detailed error messages")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Verify if a system is receipts-native compliant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m starter.cli.verify_system examples.receipts_minimal
  python -m starter.cli.verify_system examples.simple_logger -v
  python -m starter.cli.verify_system myproject.MySystem
        """,
    )

    parser.add_argument(
        "system",
        help="Module path to system under test (e.g., examples.receipts_minimal)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show detailed error messages",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON",
    )

    args = parser.parse_args()

    try:
        results = verify_system(args.system, args.verbose)

        if args.json:
            import json
            print(json.dumps(results, indent=2))
        else:
            print_results(results, args.verbose)

        # Exit code: 0 if compliant, 1 if not
        sys.exit(0 if results["is_receipts_native"] else 1)

    except ImportError as e:
        print(f"\033[91mError: {e}\033[0m", file=sys.stderr)
        sys.exit(2)
    except Exception as e:
        print(f"\033[91mError: {e}\033[0m", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
