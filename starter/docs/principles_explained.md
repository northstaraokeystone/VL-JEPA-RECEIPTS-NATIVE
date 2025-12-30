# The Six Principles Explained

Deep dive into each principle of receipts-native architecture.

---

## P1: Native Provenance

> **All state is reconstructable from receipts alone.**

### What It Means

Your entire system state can be rebuilt by replaying the receipt chain. If you lose everything except `receipts.jsonl`, you can recover.

### Why It Matters

- **Disaster recovery**: State is never truly lost
- **Debugging**: Replay to any point in time
- **Verification**: Third parties can reconstruct and verify

### How to Implement

```python
class MySystem:
    def get_state(self):
        return self._state.copy()

    def delete_all_state_except_receipts(self):
        self._state = {}

    def reconstruct_from_receipts(self):
        receipts = load_receipts()
        for receipt in receipts:
            # Rebuild state from each receipt
            if receipt["receipt_type"] == "ingest":
                self._state["last_ingest"] = receipt["payload_hash"]
        return self._state
```

### Common Mistakes

- Storing state in variables without receipts
- Having "shadow state" not captured in receipts
- Side effects that don't emit receipts

### The Test

```python
def test_principle_1(system):
    state_before = system.get_state()
    system.delete_all_state_except_receipts()
    state_after = system.reconstruct_from_receipts()
    assert state_before == state_after
```

---

## P2: Cryptographic Lineage

> **Any receipt is traceable to genesis via hash chain.**

### What It Means

Every receipt links to its parent via `parent_hash`. Following these links backwards always reaches the genesis receipt (where `parent_hash` is null).

### Why It Matters

- **Tamper evidence**: Can't insert or remove receipts
- **Ordering**: Receipt order is cryptographically enforced
- **Integrity**: Chain verification catches corruption

### How to Implement

```python
def emit_receipt(receipt_type, data):
    global _last_hash

    receipt = {
        "receipt_type": receipt_type,
        "parent_hash": _last_hash,  # Link to previous
        **data
    }

    receipt["payload_hash"] = dual_hash(json.dumps(receipt))
    _last_hash = receipt["payload_hash"]

    return receipt
```

### Common Mistakes

- Orphan receipts (no parent_hash)
- Missing genesis receipt
- Hash mismatches from data mutation
- Circular references

### The Test

```python
def test_principle_2(system):
    receipts = system.load_receipts()
    random_receipt = random.choice(receipts[1:])
    chain = trace_to_genesis(random_receipt, receipts)
    assert chain[0]["parent_hash"] is None  # Genesis
    assert verify_chain_integrity(chain)
```

---

## P3: Verifiable Causality

> **Every decision is auditable without source code.**

### What It Means

Given a decision receipt, you can verify the decision by looking only at the receipts it references. You don't need to see the source code.

### Why It Matters

- **External audit**: Auditors don't need codebase access
- **Determinism**: Same inputs = same decision
- **Accountability**: Clear chain of causation

### How to Implement

```python
def make_decision(inputs, input_receipts):
    result = decide(inputs)

    receipt = emit_receipt("decision", {
        "input_hashes": [r["payload_hash"] for r in input_receipts],
        "output": result,
        "decision_hash": dual_hash(str(decide)),  # Optional
    })

    return result, receipt
```

### Common Mistakes

- Decision receipts without `input_hashes`
- Non-deterministic decisions
- Hiding decision logic in code

### The Test

```python
def test_principle_3(system):
    decision = system.get_random_decision()
    assert decision.get("input_hashes"), "Missing input_hashes"

    receipts = system.load_receipts()
    inputs = [r for r in receipts if r["payload_hash"] in decision["input_hashes"]]
    assert len(inputs) == len(decision["input_hashes"])
```

---

## P4: Query-as-Proof

> **Proofs are derived at query time, not pre-stored.**

### What It Means

When you query the system for results, those results are computed from the receipt chain on-demand. There's no table of pre-computed answers.

### Why It Matters

- **Freshness**: Proofs always reflect current state
- **Consistency**: No stale cached results
- **Transparency**: Derivation is visible

### How to Implement

```python
class MySystem:
    # WRONG: Pre-computed
    def __init__(self):
        self.fraud_alerts = {}  # Pre-stored!

    # RIGHT: Query-time derivation
    def query(self, query_str):
        receipts = load_receipts()
        # Derive from receipts NOW
        if query_str == "detect_fraud":
            fraud = [r for r in receipts
                     if r["receipt_type"] == "detection"
                     and r["verdict"] == "fraud"]
            return {"fraud_cases": [r["payload_hash"] for r in fraud]}
```

### Common Mistakes

- Pre-computed results tables
- Cached decisions
- Stored alerts

### The Test

```python
def test_principle_4(system):
    proof1 = system.query("detect_anomalies")
    proof2 = system.query("detect_anomalies")
    assert proof1 == proof2, "Non-deterministic"
    assert not system.has_precomputed_results()
```

---

## P5: Thermodynamic Governance

> **Entropy is conserved within bounds per execution window.**

### What It Means

Each cycle tracks entropy (disorder/randomness). The change in entropy should match the work performed, within a threshold (default: 0.01).

### Why It Matters

- **Bounded behavior**: System doesn't run away
- **Energy accounting**: Track computational "energy"
- **Anomaly detection**: Entropy violations indicate problems

### How to Implement

```python
class MySystem:
    def __init__(self):
        self.entropy = 0.0

    def run_with_entropy(self):
        s_before = self.entropy
        work = self.do_work()  # Returns 0.005 for example
        self.entropy = s_before + work

        delta = abs((self.entropy - s_before) - work)
        emit_receipt("entropy", {
            "s_before": s_before,
            "s_after": self.entropy,
            "work": work,
            "delta": delta,
        })

        if delta >= 0.01:
            raise StopRule("Entropy violation", metric="entropy", delta=delta)

        return s_before, self.entropy, work
```

### Common Mistakes

- No entropy tracking
- Unbounded entropy growth
- Missing StopRule on violation

### The Test

```python
def test_principle_5(system):
    s_before, s_after, work = system.run_with_entropy()
    delta = abs((s_after - s_before) - work)
    assert delta < 0.01, f"Entropy violation: delta={delta}"
```

---

## P6: Receipts-Gated Progress

> **System halts without required gate receipts.**

### What It Means

Progress through lifecycle phases requires gate receipts. Missing a gate triggers a StopRule exception.

### Why It Matters

- **Quality gates**: Can't skip required checkpoints
- **Enforcement**: Gates are mandatory, not advisory
- **Audit trail**: Gate passage is recorded

### How to Implement

```python
class MySystem:
    def __init__(self):
        self.gates = {}

    def get_gates(self):
        return ["t2h", "t24h", "t48h"]

    def emit_gate(self, gate_id):
        receipt = emit_receipt("gate", {"gate_id": gate_id, "passed": True})
        self.gates[gate_id] = receipt["payload_hash"]

    def delete_gate_receipt(self, gate_id):
        if gate_id in self.gates:
            del self.gates[gate_id]

    def advance_to_next_gate(self):
        if "t2h" not in self.gates:
            raise StopRule(
                "Missing gate: t2h",
                gate_id="t2h",
                action="halt"
            )
```

### Common Mistakes

- No gates defined
- Gates are advisory (don't enforce)
- Wrong exception type

### The Test

```python
def test_principle_6(system):
    gates = system.get_gates()
    assert gates, "No gates defined"

    system.delete_gate_receipt(gates[0])
    with pytest.raises(StopRule):
        system.advance_to_next_gate()
```

---

## Summary Table

| Principle | Key Method | Key Field | StopRule Trigger |
|-----------|-----------|-----------|------------------|
| P1: Native Provenance | `reconstruct_from_receipts()` | All fields | State mismatch |
| P2: Cryptographic Lineage | `trace_to_genesis()` | `parent_hash` | Broken chain |
| P3: Verifiable Causality | Decision receipts | `input_hashes` | Missing inputs |
| P4: Query-as-Proof | `query()` | Derived proof | Pre-computed found |
| P5: Thermodynamic Governance | `run_with_entropy()` | `delta` | delta >= 0.01 |
| P6: Receipts-Gated Progress | `advance_to_next_gate()` | `gate_id` | Missing gate |

---

## FAQ

**Q: Do I need all 6 principles?**
A: Yes. Partial compliance is not receipts-native.

**Q: What if P5 doesn't apply to my system?**
A: Implement minimal entropy tracking (return small constant work).

**Q: Can I use different thresholds?**
A: The 0.01 entropy threshold is default. Document if you use different.

**Q: What about offline/edge systems?**
A: Same principles apply. Sync receipts when online.
