# Receipts-Native Starter Kit v1.1

Reference implementation and compliance tests for the Receipts-Native Standard.

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Run example
python examples/receipts_minimal.py

# Run compliance tests
pytest tests/compliance_suite.py --system=examples.receipts_minimal -v
```

## What's Included

- **core/**: Core primitives (dual_hash, emit_receipt, merkle_root)
- **tests/**: 6-principle compliance test suite
- **examples/**: Passing (receipts_minimal) and failing (simple_logger) examples
- **cli/**: Command-line verification tool
- **docs/**: Getting started, migration guide, principles explained

## The 6 Principles

| Principle | Description |
|-----------|-------------|
| P1 | Native Provenance - State reconstructable from receipts |
| P2 | Cryptographic Lineage - Chain traceable to genesis |
| P3 | Verifiable Causality - Decisions auditable without code |
| P4 | Query-as-Proof - Proofs derived, not stored |
| P5 | Thermodynamic Governance - Entropy conserved |
| P6 | Receipts-Gated Progress - Gates enforced |

## Verify Your System

```bash
python -m cli.verify_system your_module.YourSystem
```

## Documentation

- [Getting Started](docs/getting_started.md) - 10-minute tutorial
- [Migration Guide](docs/migration_guide.md) - Logging to receipts-native
- [Principles Explained](docs/principles_explained.md) - Deep dive

## License

MIT
