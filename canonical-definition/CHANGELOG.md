# RECEIPTS-NATIVE STANDARD CHANGELOG

All notable changes to the Receipts-Native Architecture standard are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [1.1] - 2025-01-15

### Added

- **NON-GOALS Section**: Explicit documentation of what receipts-native does NOT guarantee
  - Clarifies that receipts prove verifiability, not correctness/fairness/safety
  - Anti-pattern warning: "Has receipts ≠ good"
  - Correct pattern: "Has receipts = verifiable"

- **APPENDIX: Performance Claims**: Receipt requirements for all metrics
  - ProofPack fraud detection with verification protocol
  - QED entropy conservation with bundle reference
  - AXIOM convergence with reproducibility steps
  - Standard receipt bundle requirements

- **Receipt Bundles**: Downloadable proof packages
  - `bundles/proofpack_fraud_v1/` - 1,479 receipts
  - `bundles/qed_entropy_v12/` - 1,000,000 receipts
  - `bundles/axiom_singularity_v1/` - 631,994 receipts

- **Executable Compliance Tests**: pytest-compatible test suite
  - `starter/tests/compliance_suite.py` - 6 runnable tests
  - `receipts_minimal.py` - passing example
  - `simple_logger.py` - failing counter-example

### Changed

- **P4 (Query-as-Proof)**: Clarified that proofs are derived at query time
  - Added: "not pre-stored" to definition
  - Added: Note explaining proofs as derived artifacts
  - Added: Pre-computed results table as explicit violation

- **P5 (Thermodynamic Governance)**: Clarified entropy bound scope
  - Added: "over execution windows" to definition
  - Added: Note that bounds are per-window, not infinite time
  - Changed: Test formula to |ΔS_total - work_done| < 0.01

### Fixed

- Removed ambiguity in P4 about where proofs are stored
- Removed confusion in P5 about infinite-time entropy constraints

---

## [1.0] - 2025-01-01

### Added

- **Six Core Principles**
  - P1: Native Provenance
  - P2: Cryptographic Lineage
  - P3: Verifiable Causality
  - P4: Query-as-Proof
  - P5: Thermodynamic Governance
  - P6: Receipts-Gated Progress

- **Compliance Test Framework**
  - Pseudocode for each principle test
  - Pass/fail criteria
  - Violation examples

- **DISTINCTIONS Table**
  - Comparison: Logging vs Observability vs Receipts-Native
  - Clear differentiation criteria

- **Hash Strategy**
  - Dual-hash requirement (SHA256:BLAKE3)
  - Merkle tree specification

- **SLO Thresholds**
  - Entropy delta: |ΔS| < 0.01
  - Bias disparity: < 0.5%
  - Chain integrity: 100%

- **Receipt Schemas**
  - Core fields (required on all receipts)
  - Standard receipt types

- **Citation Format**
  - BibTeX entry for academic references

---

## Migration Guide: v1.0 → v1.1

### No Breaking Changes

v1.1 is fully backward compatible with v1.0. Systems compliant with v1.0 remain compliant with v1.1.

### Recommended Updates

1. **Add NON-GOALS acknowledgment** to your documentation
   - Clarify that receipts prove verifiability, not correctness

2. **Run new compliance tests** from starter kit
   - `pytest tests/compliance_suite.py --system=your_system`

3. **Publish receipt bundles** for performance claims
   - Follow bundle structure in `bundles/` directory

4. **Update P4 implementation** if using pre-computed results
   - Migrate to query-time proof derivation

5. **Update P5 implementation** if using unbounded entropy
   - Add per-window bounds (default: 0.01 threshold)

---

## Versioning Policy

- **MAJOR** (x.0): Breaking changes to principles or tests
- **MINOR** (1.x): Clarifications, additions, new features
- **PATCH** (1.1.x): Typos, documentation fixes

---

## Contributors

See repository CONTRIBUTORS file for full list.

---

*Receipts-Native Standard is maintained by the working group.*
