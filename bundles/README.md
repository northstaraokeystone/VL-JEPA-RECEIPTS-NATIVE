# Receipts-Native Receipt Bundles

Downloadable proof packages for verifying performance claims.

## Available Bundles

| Bundle | Description | Receipts | Claims |
|--------|-------------|----------|--------|
| [proofpack_fraud_v1](proofpack_fraud_v1/) | Medicare fraud detection | 1,479 | 100% recall, 0% FPR |
| [qed_entropy_v12](qed_entropy_v12/) | Entropy conservation | 1,002 | |ΔS| < 0.01, 0 violations |
| [axiom_singularity_v1](axiom_singularity_v1/) | Training convergence | 631,994 | Convergence at cycle 1847 |

## Quick Start

```bash
# Clone bundles
git clone https://github.com/receipts-native-standard/bundles

# Verify a bundle
cd bundles/proofpack_fraud_v1
python verify_bundle.py     # Check Merkle root
bash reproduce.sh           # Reproduce claims
```

## Bundle Structure

Each bundle contains:

```
bundle_name/
├── README.md              # What this proves
├── MANIFEST.anchor        # Merkle root + metadata
├── receipts.jsonl         # Complete receipt chain
├── verify_bundle.py       # Merkle verification
├── reproduce.sh           # Claim reproduction
└── dataset_identifiers.json (optional)
```

## Verification Protocol

1. **Download** the bundle
2. **Verify Merkle root** matches MANIFEST.anchor
3. **Reproduce metrics** from receipts
4. **Compare** to claimed values

## Creating Your Own Bundle

```python
from starter.core import emit_receipt, merkle_root, load_receipts

# Generate receipts during your run
emit_receipt("ingest", {"data": "..."})
emit_receipt("decision", {"verdict": "..."})
# ... more receipts

# Create MANIFEST
receipts = load_receipts()
root = merkle_root(receipts)
manifest = {
    "merkle_root": root,
    "receipt_count": len(receipts),
    "claim": {...}
}
```

## Citation

```bibtex
@dataset{receipts_native_bundles,
  title = {Receipts-Native Standard Receipt Bundles},
  year = {2025},
  url = {https://github.com/receipts-native-standard/bundles}
}
```
