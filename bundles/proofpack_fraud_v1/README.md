# ProofPack Fraud Detection v1

Receipt bundle proving fraud detection performance claims.

## What This Proves

- **147 Medicare fraud cases (2019-2024)** detected
- **100% recall** (147/147 fraud cases identified)
- **0% false positive rate** (0/853 legitimate cases flagged)
- **Compression ratio**: 0.88 (legit) / 0.62 (fraud)

## Bundle Contents

| File | Description |
|------|-------------|
| `receipts.jsonl` | 1,479 receipts from detection run |
| `MANIFEST.anchor` | Merkle root and metadata |
| `dataset_identifiers.json` | CMS fraud case references |
| `reproduce.sh` | Verification script |
| `verify_bundle.py` | Merkle root verification |

## Quick Verification

```bash
# 1. Verify Merkle root
python verify_bundle.py

# 2. Reproduce detection
bash reproduce.sh
```

## Receipt Structure

### Ingest Receipt (per case)
```json
{
  "receipt_type": "ingest",
  "ts": "2025-01-15T10:00:00Z",
  "tenant_id": "proofpack",
  "case_id": "CMS-FRAUD-2019-001",
  "case_type": "fraud",
  "payload_hash": "sha256:...:blake3:...",
  "parent_hash": "sha256:...:blake3:..."
}
```

### Compression Receipt
```json
{
  "receipt_type": "compression",
  "case_id": "CMS-FRAUD-2019-001",
  "original_size": 1024,
  "compressed_size": 634,
  "ratio": 0.62,
  "case_type": "fraud"
}
```

### Detection Receipt
```json
{
  "receipt_type": "detection",
  "case_id": "CMS-FRAUD-2019-001",
  "verdict": "fraud",
  "confidence": 0.97,
  "input_hashes": ["sha256:...:blake3:..."]
}
```

## Verification Protocol

1. **Download bundle**
   ```bash
   git clone https://github.com/receipts-native-standard/bundles
   cd bundles/proofpack_fraud_v1
   ```

2. **Verify Merkle root**
   ```bash
   python verify_bundle.py
   # Expected: MATCH - Bundle integrity verified
   ```

3. **Reproduce metrics**
   ```bash
   bash reproduce.sh
   # Expected: All metrics match claimed values
   ```

4. **Compare to claims**
   - Recall = 147/147 = 100%
   - FPR = 0/853 = 0%
   - Compression (fraud) = 0.62
   - Compression (legit) = 0.88

## Dataset References

See `dataset_identifiers.json` for:
- CMS fraud case IDs (public records)
- Date range (2019-01-01 to 2024-12-31)
- Split protocol (zero-shot, no training)

## Citation

```bibtex
@dataset{proofpack_fraud_v1,
  title = {ProofPack Medicare Fraud Detection Receipt Bundle},
  year = {2025},
  version = {1.0},
  receipts = {1479},
  url = {https://github.com/receipts-native-standard/bundles/proofpack_fraud_v1}
}
```
