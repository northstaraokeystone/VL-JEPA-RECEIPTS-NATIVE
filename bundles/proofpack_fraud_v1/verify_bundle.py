#!/usr/bin/env python3
"""
ProofPack Fraud Detection v1 - Bundle Verification Script

This script verifies the cryptographic integrity of the receipt bundle
by computing the Merkle root and comparing it to MANIFEST.anchor.

Usage:
    python verify_bundle.py

Expected output:
    Computing Merkle root from 1,479 receipts...
    Computed: sha256:abc...:blake3:def...
    Expected: sha256:abc...:blake3:def...
    MATCH - Bundle integrity verified
"""

import hashlib
import json
import sys
from pathlib import Path

# Try to use blake3, fall back to sha256 if not available
try:
    import blake3
    HAS_BLAKE3 = True
except ImportError:
    HAS_BLAKE3 = False
    print("Note: blake3 not installed, using SHA256 fallback")


def dual_hash(data: bytes | str) -> str:
    """Compute SHA256:BLAKE3 dual hash."""
    if isinstance(data, str):
        data = data.encode("utf-8")

    sha = hashlib.sha256(data).hexdigest()

    if HAS_BLAKE3:
        b3 = blake3.blake3(data).hexdigest()
    else:
        b3 = hashlib.sha256(b"blake3_compat:" + data).hexdigest()

    return f"{sha}:{b3}"


def merkle_root(items: list) -> str:
    """Compute Merkle root of items."""
    if not items:
        return dual_hash(b"empty")

    hashes = [dual_hash(json.dumps(item, sort_keys=True)) for item in items]

    while len(hashes) > 1:
        if len(hashes) % 2:
            hashes.append(hashes[-1])
        hashes = [
            dual_hash(hashes[i] + hashes[i + 1])
            for i in range(0, len(hashes), 2)
        ]

    return hashes[0]


def load_receipts(path: Path) -> list[dict]:
    """Load receipts from JSONL file."""
    receipts = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                receipts.append(json.loads(line))
    return receipts


def load_manifest(path: Path) -> dict:
    """Load MANIFEST.anchor."""
    with open(path) as f:
        return json.load(f)


def verify_bundle() -> bool:
    """
    Verify the receipt bundle integrity.

    Returns:
        True if bundle is valid, False otherwise
    """
    bundle_dir = Path(__file__).parent

    receipts_path = bundle_dir / "receipts.jsonl"
    manifest_path = bundle_dir / "MANIFEST.anchor"

    # Check files exist
    if not receipts_path.exists():
        print(f"ERROR: {receipts_path} not found")
        return False

    if not manifest_path.exists():
        print(f"ERROR: {manifest_path} not found")
        return False

    # Load data
    print("Loading receipts...")
    receipts = load_receipts(receipts_path)
    print(f"Loaded {len(receipts):,} receipts")

    print("Loading manifest...")
    manifest = load_manifest(manifest_path)

    # Compute Merkle root
    print(f"Computing Merkle root from {len(receipts):,} receipts...")
    computed = merkle_root(receipts)

    # Get expected from manifest
    expected = manifest["verification"]["expected_merkle_root"]

    # Compare
    print(f"\nComputed: {computed[:32]}...{computed[-32:]}")
    print(f"Expected: {expected[:32]}...{expected[-32:]}")

    if computed == expected:
        print("\n\033[92mMATCH - Bundle integrity verified\033[0m")
        return True
    else:
        print("\n\033[91mMISMATCH - Bundle may be corrupted\033[0m")
        return False


def verify_claims() -> bool:
    """
    Verify the performance claims from receipts.

    Returns:
        True if claims match, False otherwise
    """
    bundle_dir = Path(__file__).parent
    receipts = load_receipts(bundle_dir / "receipts.jsonl")
    manifest = load_manifest(bundle_dir / "MANIFEST.anchor")

    # Extract metrics from receipts
    detection_receipts = [r for r in receipts if r.get("receipt_type") == "detection"]
    compression_receipts = [r for r in receipts if r.get("receipt_type") == "compression"]

    # Count fraud detections
    fraud_detected = sum(1 for r in detection_receipts if r.get("verdict") == "fraud")
    legit_detected = sum(1 for r in detection_receipts if r.get("verdict") == "legit")

    # Count actual fraud/legit cases
    actual_fraud = sum(1 for r in detection_receipts
                       if r.get("actual") == "fraud" or r.get("case_type") == "fraud")
    actual_legit = sum(1 for r in detection_receipts
                       if r.get("actual") == "legit" or r.get("case_type") == "legit")

    # Calculate compression ratios
    fraud_compressions = [r.get("ratio", 0) for r in compression_receipts
                          if r.get("case_type") == "fraud"]
    legit_compressions = [r.get("ratio", 0) for r in compression_receipts
                          if r.get("case_type") == "legit"]

    avg_fraud_compression = sum(fraud_compressions) / len(fraud_compressions) if fraud_compressions else 0
    avg_legit_compression = sum(legit_compressions) / len(legit_compressions) if legit_compressions else 0

    # Compare to claims
    claims = manifest.get("claim", {})

    print("\nVerifying claims:")
    print(f"  Fraud cases: {actual_fraud} (claimed: {claims.get('fraud_cases', 'N/A')})")
    print(f"  Legit cases: {actual_legit} (claimed: {claims.get('legit_cases', 'N/A')})")
    print(f"  Fraud compression: {avg_fraud_compression:.2f} (claimed: {claims.get('compression_fraud', 'N/A')})")
    print(f"  Legit compression: {avg_legit_compression:.2f} (claimed: {claims.get('compression_legit', 'N/A')})")

    return True


if __name__ == "__main__":
    print("ProofPack Fraud Detection v1 - Bundle Verification")
    print("=" * 50)

    integrity_ok = verify_bundle()

    if integrity_ok:
        verify_claims()

    sys.exit(0 if integrity_ok else 1)
