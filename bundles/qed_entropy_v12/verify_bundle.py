#!/usr/bin/env python3
"""
QED Entropy Conservation v12 - Bundle Verification Script

Verifies Merkle root integrity and entropy conservation claims.

Usage:
    python verify_bundle.py
"""

import hashlib
import json
import sys
from pathlib import Path

try:
    import blake3
    HAS_BLAKE3 = True
except ImportError:
    HAS_BLAKE3 = False


def dual_hash(data: bytes | str) -> str:
    if isinstance(data, str):
        data = data.encode("utf-8")
    sha = hashlib.sha256(data).hexdigest()
    if HAS_BLAKE3:
        b3 = blake3.blake3(data).hexdigest()
    else:
        b3 = hashlib.sha256(b"blake3_compat:" + data).hexdigest()
    return f"{sha}:{b3}"


def merkle_root(items: list) -> str:
    if not items:
        return dual_hash(b"empty")
    hashes = [dual_hash(json.dumps(item, sort_keys=True)) for item in items]
    while len(hashes) > 1:
        if len(hashes) % 2:
            hashes.append(hashes[-1])
        hashes = [dual_hash(hashes[i] + hashes[i + 1]) for i in range(0, len(hashes), 2)]
    return hashes[0]


def load_receipts(path: Path) -> list[dict]:
    receipts = []
    with open(path) as f:
        for line in f:
            if line.strip():
                receipts.append(json.loads(line))
    return receipts


def verify_bundle() -> bool:
    bundle_dir = Path(__file__).parent
    receipts_path = bundle_dir / "receipts.jsonl"
    manifest_path = bundle_dir / "MANIFEST.anchor"

    if not receipts_path.exists():
        print(f"ERROR: {receipts_path} not found")
        return False

    if not manifest_path.exists():
        print(f"ERROR: {manifest_path} not found")
        return False

    print("Loading receipts...")
    receipts = load_receipts(receipts_path)
    print(f"Loaded {len(receipts):,} receipts")

    with open(manifest_path) as f:
        manifest = json.load(f)

    print(f"Computing Merkle root...")
    computed = merkle_root(receipts)
    expected = manifest["verification"]["expected_merkle_root"]

    print(f"\nComputed: {computed[:32]}...{computed[-32:]}")
    print(f"Expected: {expected[:32]}...{expected[-32:]}")

    if computed == expected:
        print("\n\033[92mMATCH - Bundle integrity verified\033[0m")
        return True
    else:
        print("\n\033[91mMISMATCH - Bundle may be corrupted\033[0m")
        return False


def verify_entropy_bounds() -> bool:
    bundle_dir = Path(__file__).parent
    receipts = load_receipts(bundle_dir / "receipts.jsonl")

    entropy_receipts = [r for r in receipts if r.get("receipt_type") == "entropy"]
    threshold = 0.01
    violations = 0
    max_delta = 0.0

    print(f"\nAnalyzing {len(entropy_receipts)} entropy receipts...")

    for r in entropy_receipts:
        delta = r.get("delta", 0)
        if abs(delta) >= threshold:
            violations += 1
        if abs(delta) > max_delta:
            max_delta = abs(delta)

    print(f"  Max |ΔS| observed: {max_delta:.6f}")
    print(f"  Threshold: {threshold}")
    print(f"  Violations: {violations}")

    if violations == 0:
        print("\n\033[92mAll cycles within entropy bounds\033[0m")
        return True
    else:
        print(f"\n\033[91m{violations} cycles exceeded entropy bounds\033[0m")
        return False


if __name__ == "__main__":
    print("QED Entropy Conservation v12 - Bundle Verification")
    print("=" * 50)

    integrity_ok = verify_bundle()
    if integrity_ok:
        verify_entropy_bounds()

    sys.exit(0 if integrity_ok else 1)
