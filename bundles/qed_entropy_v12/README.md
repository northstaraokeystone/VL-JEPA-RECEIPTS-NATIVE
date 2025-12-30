# QED Entropy Conservation v12

Receipt bundle proving entropy conservation claims per P5 (Thermodynamic Governance).

## What This Proves

- **1,000 cycles** tested with entropy tracking
- **Max |ΔS| observed**: 0.003 (well under 0.01 threshold)
- **0 violations** of entropy bounds
- **100% compliance** with P5 requirements

## Bundle Contents

| File | Description |
|------|-------------|
| `receipts.jsonl` | 1,000,000+ entropy receipts |
| `MANIFEST.anchor` | Merkle root and metadata |
| `reproduce.sh` | Verification script |
| `verify_bundle.py` | Merkle root verification |

## Quick Verification

```bash
# 1. Verify Merkle root
python verify_bundle.py

# 2. Reproduce entropy analysis
bash reproduce.sh
```

## Receipt Structure

### Entropy Receipt (per cycle)
```json
{
  "receipt_type": "entropy",
  "ts": "2025-01-15T10:00:00Z",
  "tenant_id": "qed",
  "cycle_id": 1,
  "s_before": 0.0,
  "s_after": 0.005,
  "work": 0.005,
  "delta": 0.0,
  "within_bounds": true,
  "payload_hash": "sha256:...:blake3:..."
}
```

## Verification Protocol

1. **Download bundle**
   ```bash
   git clone https://github.com/receipts-native-standard/bundles
   cd bundles/qed_entropy_v12
   ```

2. **Verify Merkle root**
   ```bash
   python verify_bundle.py
   # Expected: MATCH - Bundle integrity verified
   ```

3. **Verify entropy bounds**
   ```bash
   bash reproduce.sh
   # Expected: All cycles within |ΔS| < 0.01 bounds
   ```

## Entropy Conservation Formula

```
P5 Compliance: |ΔS_total - work_done| < 0.01

Where:
  ΔS_total = S_after - S_before
  work_done = expected entropy change
  threshold = 0.01 (configurable)
```

## Citation

```bibtex
@dataset{qed_entropy_v12,
  title = {QED Entropy Conservation Receipt Bundle},
  year = {2025},
  version = {12},
  cycles = {1000},
  url = {https://github.com/receipts-native-standard/bundles/qed_entropy_v12}
}
```
