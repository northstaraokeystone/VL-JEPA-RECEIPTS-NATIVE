#!/usr/bin/env python3
"""
VL-JEPA Receipts-Native v3.0 CLI

Command-line interface for the self-evolving verification framework.
Every command emits a receipt. No exceptions.
"""

import argparse
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from core import dual_hash, emit_receipt, merkle, StopRule


def cmd_test(args):
    """Run basic test to verify receipt emission."""
    receipt = emit_receipt("ingest", {
        "tenant_id": "test",
        "payload_hash": dual_hash(b"test_payload"),
        "source_type": "cli_test",
        "redactions": [],
    }, domain="system")

    print(json.dumps(receipt, indent=2))
    return 0


def cmd_ingest(args):
    """Ingest a file and emit receipt."""
    file_path = Path(args.file)
    if not file_path.exists():
        print(f"Error: File not found: {file_path}", file=sys.stderr)
        return 1

    content = file_path.read_bytes()

    receipt = emit_receipt("ingest", {
        "tenant_id": args.tenant,
        "payload_hash": dual_hash(content),
        "source_type": args.source_type,
        "file_path": str(file_path),
        "file_size": len(content),
        "redactions": [],
    }, domain="system")

    print(json.dumps(receipt, indent=2))
    return 0


def cmd_anchor(args):
    """Create anchor receipt from pending receipts."""
    ledger_path = Path("receipts.jsonl")
    if not ledger_path.exists():
        print("Error: No receipts to anchor", file=sys.stderr)
        return 1

    # Load recent receipts
    receipts = []
    with open(ledger_path) as f:
        for line in f:
            if line.strip():
                receipts.append(json.loads(line))

    # Filter to unanchored receipts
    anchored_hashes = set()
    for r in receipts:
        if r.get("receipt_type") == "anchor":
            anchored_hashes.update(r.get("anchored_hashes", []))

    unanchored = [r for r in receipts
                  if r.get("receipt_type") != "anchor"
                  and r.get("payload_hash") not in anchored_hashes]

    if not unanchored:
        print("No unanchored receipts found")
        return 0

    # Compute Merkle root
    root = merkle(unanchored)

    anchor_receipt = emit_receipt("anchor", {
        "merkle_root": root,
        "hash_algos": ["SHA256", "BLAKE3"],
        "batch_size": len(unanchored),
        "anchored_hashes": [r.get("payload_hash") for r in unanchored],
    }, domain="system")

    print(json.dumps(anchor_receipt, indent=2))
    return 0


def cmd_verify(args):
    """Verify a receipt exists in the ledger."""
    ledger_path = Path("receipts.jsonl")
    if not ledger_path.exists():
        print("Error: No ledger found", file=sys.stderr)
        return 1

    target_hash = args.hash

    with open(ledger_path) as f:
        for line in f:
            if line.strip():
                receipt = json.loads(line)
                if receipt.get("payload_hash") == target_hash:
                    print(json.dumps({
                        "verified": True,
                        "receipt": receipt,
                    }, indent=2))
                    return 0

    print(json.dumps({
        "verified": False,
        "hash": target_hash,
    }, indent=2))
    return 1


def cmd_qualify(args):
    """Run module qualification protocol."""
    from meta.qualify_module import qualify_module

    result = qualify_module({
        "name": args.module,
        "target_company": args.company,
        "estimated_savings": args.roi,
        "features": [],
    }, {
        "pain_points": [],
        "competitors": [],
        "regulatory_drivers": [],
    }, args.proof_time)

    print(json.dumps(result, indent=2))
    return 0 if result["verdict"] == "HUNT" else 1


def cmd_status(args):
    """Show system status."""
    ledger_path = Path("receipts.jsonl")

    stats = {
        "ledger_exists": ledger_path.exists(),
        "receipt_count": 0,
        "receipt_types": {},
        "latest_receipt": None,
    }

    if ledger_path.exists():
        with open(ledger_path) as f:
            for line in f:
                if line.strip():
                    receipt = json.loads(line)
                    stats["receipt_count"] += 1
                    rtype = receipt.get("receipt_type", "unknown")
                    stats["receipt_types"][rtype] = stats["receipt_types"].get(rtype, 0) + 1
                    stats["latest_receipt"] = receipt

    print(json.dumps(stats, indent=2))
    return 0


def cmd_topology(args):
    """Show module topology status."""
    from evolution.topology_classifier import TopologyClassifier

    classifier = TopologyClassifier()
    status = classifier.get_all_module_status()

    print(json.dumps(status, indent=2))
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="VL-JEPA Receipts-Native v3.0 CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Test command
    test_parser = subparsers.add_parser("test", help="Run basic test")
    test_parser.set_defaults(func=cmd_test)

    # Ingest command
    ingest_parser = subparsers.add_parser("ingest", help="Ingest a file")
    ingest_parser.add_argument("file", help="File to ingest")
    ingest_parser.add_argument("--tenant", default="default", help="Tenant ID")
    ingest_parser.add_argument("--source-type", default="file", help="Source type")
    ingest_parser.set_defaults(func=cmd_ingest)

    # Anchor command
    anchor_parser = subparsers.add_parser("anchor", help="Create anchor receipt")
    anchor_parser.set_defaults(func=cmd_anchor)

    # Verify command
    verify_parser = subparsers.add_parser("verify", help="Verify a receipt")
    verify_parser.add_argument("hash", help="Receipt payload hash to verify")
    verify_parser.set_defaults(func=cmd_verify)

    # Qualify command
    qualify_parser = subparsers.add_parser("qualify", help="Qualify a module")
    qualify_parser.add_argument("module", help="Module name")
    qualify_parser.add_argument("--company", default="default", help="Target company")
    qualify_parser.add_argument("--roi", type=int, default=0, help="Estimated ROI")
    qualify_parser.add_argument("--proof-time", type=float, default=48.0, help="Proof time in hours")
    qualify_parser.set_defaults(func=cmd_qualify)

    # Status command
    status_parser = subparsers.add_parser("status", help="Show system status")
    status_parser.set_defaults(func=cmd_status)

    # Topology command
    topology_parser = subparsers.add_parser("topology", help="Show module topology")
    topology_parser.set_defaults(func=cmd_topology)

    # Parse and execute
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    try:
        return args.func(args)
    except StopRule as e:
        print(f"StopRule triggered: {e}", file=sys.stderr)
        return 2
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
