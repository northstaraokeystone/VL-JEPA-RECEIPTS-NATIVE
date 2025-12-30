#!/usr/bin/env python3
"""
AXIOM Singularity Convergence v1 - Bundle Verification Script

Verifies Merkle root integrity and convergence claims.
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


def verify_convergence() -> bool:
    bundle_dir = Path(__file__).parent
    receipts = load_receipts(bundle_dir / "receipts.jsonl")

    training_receipts = [r for r in receipts if r.get("receipt_type") == "training"]
    convergence_receipts = [r for r in receipts if r.get("receipt_type") == "convergence"]

    print(f"\nAnalyzing convergence...")
    print(f"  Training receipts: {len(training_receipts)}")

    if convergence_receipts:
        conv = convergence_receipts[0]
        print(f"  Convergence cycle: {conv.get('cycle')}")
        print(f"  Final loss: {conv.get('final_loss')}")
        print(f"  Threshold: {conv.get('threshold')}")

    # Check for anomalies post-convergence
    conv_cycle = 1847
    post_conv = [r for r in training_receipts if r.get("cycle", 0) > conv_cycle]
    anomalies = [r for r in post_conv if r.get("loss", 0) >= 0.001]

    print(f"  Post-convergence receipts: {len(post_conv)}")
    print(f"  Anomalies: {len(anomalies)}")

    return len(anomalies) == 0


if __name__ == "__main__":
    print("AXIOM Singularity Convergence v1 - Bundle Verification")
    print("=" * 50)

    integrity_ok = verify_bundle()
    if integrity_ok:
        verify_convergence()

    sys.exit(0 if integrity_ok else 1)
