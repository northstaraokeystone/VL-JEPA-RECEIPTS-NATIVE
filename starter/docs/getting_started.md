# Getting Started with Receipts-Native

Complete this tutorial in 10 minutes to understand receipts-native architecture.

## Prerequisites

- Python 3.10+
- pip

## Step 1: Installation (2 min)

```bash
# Clone the starter kit
git clone https://github.com/receipts-native-standard/starter
cd starter

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "from core.receipt import dual_hash; print(dual_hash(b'test'))"
```

Expected output: A dual-hash string like `abc123...:def456...`

## Step 2: Run the Example (3 min)

```bash
# Run the minimal receipts-native system
python examples/receipts_minimal.py
```

Expected output:
```
Receipts-Native Minimal System Demo
========================================
Cycle 1: 1 cycles completed
Cycle 2: 2 cycles completed
Cycle 3: 3 cycles completed

Total receipts: 10
  - genesis: a3f2c4e8...
  - ingest: b4c5d6e7...
  - decision: c6d7e8f9...
  ...

Anomaly detection proof: 0 anomalies found
```

### Inspect the Receipts

```bash
cat receipts.jsonl | head -3
```

Each line is a JSON receipt with:
- `receipt_type`: What kind of operation
- `ts`: ISO8601 timestamp
- `payload_hash`: SHA256:BLAKE3 dual hash
- `parent_hash`: Link to previous receipt

## Step 3: Run Compliance Tests (3 min)

```bash
# Run the 6 compliance tests against the example
cd starter
pytest tests/compliance_suite.py --system=examples.receipts_minimal -v
```

Expected output:
```
test_principle_1_native_provenance PASSED
test_principle_2_cryptographic_lineage PASSED
test_principle_3_verifiable_causality PASSED
test_principle_4_query_as_proof PASSED
test_principle_5_thermodynamic_governance PASSED
test_principle_6_receipts_gated_progress PASSED

6 passed in 2.34s
```

All 6 tests PASS = System is receipts-native.

## Step 4: See What Fails (2 min)

```bash
# Run tests against the non-compliant logger
pytest tests/compliance_suite.py --system=examples.simple_logger -v
```

Expected output:
```
test_principle_1_native_provenance FAILED
test_principle_2_cryptographic_lineage FAILED
test_principle_3_verifiable_causality FAILED
test_principle_4_query_as_proof FAILED
test_principle_5_thermodynamic_governance FAILED
test_principle_6_receipts_gated_progress FAILED

6 failed in 1.23s
```

All 6 tests FAIL = System is NOT receipts-native.

### Why Does simple_logger Fail?

| Principle | Why It Fails |
|-----------|--------------|
| P1 | State stored in variables, not reconstructable |
| P2 | No parent_hash links, no chain |
| P3 | Decisions missing input_hashes |
| P4 | Pre-computed _fraud_alerts table |
| P5 | No entropy tracking |
| P6 | No gates, no StopRule |

## Next Steps

1. **Understand the principles**: Read [principles_explained.md](principles_explained.md)
2. **Migrate your system**: Follow [migration_guide.md](migration_guide.md)
3. **Verify claims**: Check [bundles/](../../bundles/) for receipt bundles

## Quick Reference

### Core Functions

```python
from core.receipt import dual_hash, emit_receipt, merkle_root

# Hash any data
hash = dual_hash(b"my data")

# Emit a receipt
receipt = emit_receipt("ingest", {
    "source": "api",
    "data_size": 1024
})

# Compute Merkle root
root = merkle_root([receipt1, receipt2, receipt3])
```

### CLI Verification

```bash
# Verify any system
python -m starter.cli.verify_system your_module.YourSystem
```

## FAQ

**Q: Do I need BLAKE3 installed?**
A: No, it falls back to SHA256. But BLAKE3 is recommended for production.

**Q: Where are receipts stored?**
A: By default in `receipts.jsonl`. Set `RECEIPTS_LEDGER_PATH` to change.

**Q: How do I know if my system is receipts-native?**
A: Run the compliance tests. Pass all 6 = receipts-native.
