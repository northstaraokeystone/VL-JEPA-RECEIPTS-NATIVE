# RECEIPTS-NATIVE ARCHITECTURE: CANONICAL DEFINITION v1.1

**Version:** 1.1
**Status:** ACTIVE
**Hash:** COMPUTE_ON_SAVE

---

## ABSTRACT

Receipts-Native Architecture is a design paradigm where every computational action produces a cryptographically verifiable receipt, and all system state is reconstructable from the receipt chain alone. This document defines the six core principles, compliance tests, and verification protocols that distinguish receipts-native systems from traditional logging approaches.

---

## THE SIX PRINCIPLES

### P1: Native Provenance
**Definition:** All state is reconstructable from receipts alone.

**Test:** Delete all state except receipts.jsonl. Reconstruct. State matches original.

**Violation:** Any state stored outside receipt chain that affects system behavior.

**Implementation:**
- Every state change emits a receipt before taking effect
- State reconstruction function exists and is tested
- No shadow state or side channels

### P2: Cryptographic Lineage
**Definition:** Any receipt is traceable to genesis via hash chain.

**Test:** Select random receipt. Trace parent_hash backward. Reaches genesis.

**Violation:** Orphan receipts, missing genesis, broken hash chains.

**Implementation:**
- Each receipt includes parent_hash linking to previous
- Genesis receipt is first in chain with parent_hash = null
- Hash verification function exists for full chain

### P3: Verifiable Causality
**Definition:** Every decision is auditable without source code.

**Test:** Extract decision. Reconstruct inputs from receipts. Verify deterministic.

**Violation:** Decision logic embedded in code, not expressed in receipts.

**Implementation:**
- Decision receipts include all input receipt hashes
- Decision function hash included in receipt
- Third-party can verify decision from receipts alone

### P4: Query-as-Proof
**Definition:** Proofs are derived from receipts at query time, not pre-stored.

**Test:** Query returns proof computed deterministically from receipt chain. No pre-computed results table exists.

**Violation:** Pre-computed fraud alerts stored in separate table. Cached decisions without derivation path.

**Note:** Proofs are derived artifacts from the receipt chain. They are computed on-demand, not stored alongside receipts. This ensures proofs reflect current receipt state and cannot become stale or inconsistent.

**Implementation:**
- Query function computes proof from receipt scan
- No stored "results" or "alerts" tables
- Same query always produces same proof from same receipts

### P5: Thermodynamic Governance
**Definition:** Entropy is conserved within bounds over execution windows.

**Test:** Run cycle with entropy tracking. |ΔS_total - work_done| < 0.01 per bounded cycle.

**Violation:** Unbounded entropy growth. No conservation check. Missing entropy receipts.

**Note:** Entropy bounds are evaluated per execution window, not across infinite time. Each window starts fresh. The 0.01 threshold applies to the delta between measured entropy change and work performed.

**Implementation:**
- Entropy tracking on all state mutations
- Entropy receipt emitted per cycle
- StopRule triggers when delta exceeds threshold

### P6: Receipts-Gated Progress
**Definition:** System halts without required gate receipts.

**Test:** Delete gate receipt (e.g., T+2h). Attempt advance. StopRule raised.

**Violation:** System progresses without gates. Gates are advisory not enforced.

**Implementation:**
- Gate receipts required before phase transitions
- StopRule exception on missing gate
- No bypass mechanism in production

---

## NON-GOALS

Receipts-native architecture provides specific guarantees. It is critical to understand what it does **NOT** guarantee:

### What Receipts-Native Does NOT Guarantee

| Non-Goal | Explanation |
|----------|-------------|
| **Correctness** | A system may make wrong decisions with full receipts. Receipts prove *what* happened, not that it was *right*. |
| **Fairness** | Biased algorithms remain biased with receipts. The receipts prove the bias is *verifiable*, not absent. |
| **Safety** | Unsafe operations with receipts are still unsafe. Receipts enable *post-hoc analysis*, not prevention. |
| **Optimality** | Suboptimal paths are fully receipted. Receipts don't improve performance, they document it. |
| **Privacy** | Receipts may expose sensitive operations unless explicitly redacted. |

### What Receipts-Native DOES Guarantee

| Guarantee | Explanation |
|-----------|-------------|
| **Verifiability** | Any claim can be independently verified from receipt chain |
| **Traceability** | Any state can be traced to its causal chain |
| **Reconstructability** | System state is recoverable from receipts alone |
| **Determinism** | Same receipts produce same derived proofs |
| **Accountability** | Every operation has an auditable record |

### Anti-Pattern to Avoid

**WRONG:** "This system has receipts, therefore it is good/safe/fair."

**CORRECT:** "This system has receipts, therefore its behavior is verifiable."

**Example:** A biased fraud detection model with receipts is still biased. However, the bias is now *provable* from the receipt chain rather than hidden in opaque model weights.

---

## DISTINCTIONS

| Property | Logging | Observability | Receipts-Native |
|----------|---------|---------------|-----------------|
| State reconstruction | No | Partial | Complete |
| Cryptographic integrity | No | Optional | Required |
| Audit without code | No | No | Yes |
| Proof derivation | N/A | Stored metrics | Query-time from chain |
| Entropy tracking | No | No | Yes |
| Progress gating | No | No | Enforced |

---

## VERIFICATION PROTOCOL

### Compliance Test Suite

Any system claiming receipts-native status MUST pass all six tests:

```python
# Test 1: Native Provenance
def test_principle_1_native_provenance(system_under_test):
    """State reconstructable from receipts alone."""
    state_before = system_under_test.get_state()
    system_under_test.delete_all_state_except_receipts()
    state_after = system_under_test.reconstruct_from_receipts()
    assert state_before == state_after, "State reconstruction failed"

# Test 2: Cryptographic Lineage
def test_principle_2_cryptographic_lineage(system_under_test):
    """Any receipt traceable to genesis."""
    receipts = system_under_test.load_receipts()
    random_receipt = random.choice(receipts[1:])  # Skip genesis
    chain = trace_to_genesis(random_receipt, receipts)
    assert chain[0]["parent_hash"] is None, "Genesis missing"
    assert verify_chain_integrity(chain), "Chain integrity failure"

# Test 3: Verifiable Causality
def test_principle_3_verifiable_causality(system_under_test):
    """Decisions auditable without code."""
    decision = system_under_test.get_random_decision()
    inputs = extract_input_receipts(decision)
    recomputed = deterministic_verify(decision, inputs)
    assert recomputed, "Decision not verifiable from inputs"

# Test 4: Query-as-Proof
def test_principle_4_query_as_proof(system_under_test):
    """Proofs derived, not stored."""
    proof1 = system_under_test.query("detect_anomalies")
    proof2 = system_under_test.query("detect_anomalies")
    assert proof1 == proof2, "Non-deterministic proof"
    assert not system_under_test.has_precomputed_results(), "Pre-stored proofs"

# Test 5: Thermodynamic Governance
def test_principle_5_thermodynamic_governance(system_under_test):
    """Entropy conserved within bounds."""
    entropy_before, entropy_after, work = system_under_test.run_with_entropy()
    delta = abs((entropy_after - entropy_before) - work)
    assert delta < 0.01, f"Entropy violation: delta={delta}"

# Test 6: Receipts-Gated Progress
def test_principle_6_receipts_gated_progress(system_under_test):
    """System halts without gates."""
    system_under_test.delete_gate_receipt("t2h")
    with pytest.raises(StopRule):
        system_under_test.advance_to_next_gate()
```

### Executable Tests

**Starter Kit:** `github.com/receipts-native-standard/starter`
- Clone → Run → Verify your system in 10 minutes
- 6 automated compliance tests
- Passing and failing examples included

**Command:**
```bash
pytest tests/compliance_suite.py --system=your_system
```

---

## RECEIPT SCHEMAS

### Core Receipt Fields (Required on ALL receipts)

```json
{
  "receipt_type": "string (required)",
  "ts": "ISO8601 timestamp (required)",
  "tenant_id": "string (required)",
  "payload_hash": "sha256:blake3 (required)",
  "parent_hash": "sha256:blake3 or null (required for lineage)"
}
```

### Standard Receipt Types

| Type | Purpose | Additional Fields |
|------|---------|-------------------|
| ingest | Data ingestion | source_type, redactions |
| anchor | Merkle batch | merkle_root, batch_size |
| decision | Choice made | inputs[], output, confidence |
| entropy | Cycle entropy | s_before, s_after, work |
| gate | Progress gate | gate_id, gate_type |
| anomaly | StopRule trigger | metric, delta, action |

---

## HASH STRATEGY

```json
{
  "hash_strategy": {
    "algorithm": ["SHA256", "BLAKE3"],
    "format": "sha256_hex:blake3_hex",
    "merkle_algorithm": "BLAKE3",
    "dual_hash_required": true
  }
}
```

**Implementation:**
```python
def dual_hash(data: bytes | str) -> str:
    if isinstance(data, str):
        data = data.encode()
    sha = hashlib.sha256(data).hexdigest()
    b3 = blake3.blake3(data).hexdigest()
    return f"{sha}:{b3}"
```

---

## SLO THRESHOLDS

| Metric | Threshold | Stoprule Action |
|--------|-----------|-----------------|
| Entropy delta | |ΔS| < 0.01 | halt |
| Chain integrity | 100% | halt |
| Bias disparity | < 0.5% | halt + escalate |
| Reconstruction match | 100% | halt |
| Gate presence | required | halt |

---

## APPENDIX: PERFORMANCE CLAIMS

All performance claims in receipts-native systems are **PROVISIONAL** until receipt bundles are published and verified.

### ProofPack Fraud Detection v1

| Metric | Claimed Value | Receipt Requirement |
|--------|---------------|---------------------|
| Fraud cases analyzed | 147 | ingest_receipts for each case |
| Recall | 100% (147/147) | detection_receipts with verdicts |
| False positive rate | 0% (0/853) | decision_receipts for negatives |
| Compression (legit) | 0.88 | compression_receipts |
| Compression (fraud) | 0.62 | compression_receipts |

**Verification:** Download bundle from `bundles/proofpack_fraud_v1/`, run `reproduce.sh`, verify Merkle root matches `MANIFEST.anchor`.

### QED Entropy Conservation v12

| Metric | Claimed Value | Receipt Requirement |
|--------|---------------|---------------------|
| Cycles tested | 1,000 | entropy_receipts per cycle |
| Max |ΔS| observed | 0.003 | entropy_receipts with deltas |
| Violations | 0 | anomaly_receipts (none expected) |

**Verification:** Download bundle from `bundles/qed_entropy_v12/`, run `reproduce.sh`, verify all entropy deltas.

### AXIOM Singularity Convergence v1

| Metric | Claimed Value | Receipt Requirement |
|--------|---------------|---------------------|
| Convergence cycle | 1847/10000 | convergence_receipts |
| Final loss | < 0.001 | training_receipts |
| Stability | 100% post-convergence | no anomaly_receipts after 1847 |

**Verification:** Download bundle from `bundles/axiom_singularity_v1/`, run `reproduce.sh`, verify convergence trajectory.

### Receipt Bundle Requirements

Every performance claim MUST include:

1. **Dataset identifiers** - Which data was used (hashes or public IDs)
2. **Split protocol** - How train/test/validation divided
3. **Evaluation script hash** - Exact code used to compute metrics
4. **Receipt chain** - Complete receipts.jsonl for the run
5. **MANIFEST.anchor** - Merkle root of receipt chain
6. **reproduce.sh** - Script to verify claims from receipts

---

## CITATION

```bibtex
@standard{receipts_native_v1.1,
  title = {Receipts-Native Architecture: Canonical Definition},
  version = {1.1},
  year = {2025},
  url = {https://github.com/receipts-native-standard/definition},
  note = {Cryptographically verifiable AI system architecture}
}
```

---

## REFERENCE IMPLEMENTATION

- **Starter Kit:** `starter/` - Minimal receipts-native system
- **Compliance Tests:** `starter/tests/compliance_suite.py`
- **Receipt Bundles:** `bundles/` - Verified performance claims
- **Documentation:** `starter/docs/` - Tutorials and guides

---

**Hash of this document:** COMPUTE_ON_SAVE
**Version:** 1.1
**Status:** ACTIVE

*No receipt → not real. No test → not shipped.*
