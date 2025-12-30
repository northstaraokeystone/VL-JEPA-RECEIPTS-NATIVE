# Migration Guide: Logging to Receipts-Native

Follow these 10 steps to migrate from traditional logging to receipts-native architecture.

## Before You Start

**Audit your current system:**
- How many `logger.info/debug/warning/error` calls?
- What state is stored in variables vs databases?
- Do you have pre-computed results tables?
- Are decisions traceable to their inputs?

## Step 1: Install Core Primitives

```python
# Copy these to your project
from starter.core import (
    dual_hash,      # Always use, never single hash
    emit_receipt,   # Replace logger calls
    merkle_root,    # For batch anchoring
    StopRule,       # For gate enforcement
)
```

## Step 2: Replace State Changes

### Before (Logging)
```python
class MySystem:
    def process(self, data):
        self.state = transform(data)
        logger.info(f"Processed data, state updated")
```

### After (Receipts-Native)
```python
class MySystem:
    def process(self, data):
        result = transform(data)
        receipt = emit_receipt("state_change", {
            "operation": "process",
            "input_hash": dual_hash(data),
            "output_hash": dual_hash(result),
        })
        # State IS the receipt chain
        return receipt
```

## Step 3: Replace Decisions

### Before (Logging)
```python
def decide(self, inputs):
    decision = self.model.predict(inputs)
    logger.info(f"Decision: {decision}")
    return decision
```

### After (Receipts-Native)
```python
def decide(self, inputs, input_receipts):
    decision = self.model.predict(inputs)
    receipt = emit_receipt("decision", {
        "input_hashes": [r["payload_hash"] for r in input_receipts],
        "output": decision,
        "confidence": self.model.confidence,
    })
    return decision, receipt
```

**Key difference**: Decision receipts include `input_hashes` for verifiability.

## Step 4: Add Lineage

```python
# Track parent_hash for chain integrity
class MySystem:
    def __init__(self):
        # Genesis receipt
        self.genesis = emit_receipt("genesis", {
            "system": "MySystem",
            "version": "1.0",
        })

    def process(self, data):
        # Each receipt links to previous
        receipt = emit_receipt("ingest", {
            "data_hash": dual_hash(data),
        })
        # parent_hash is automatically tracked
```

## Step 5: Add Gates

```python
from starter.core import StopRule

class MySystem:
    def __init__(self):
        self.gates = {}

    def emit_gate(self, gate_id):
        receipt = emit_receipt("gate", {
            "gate_id": gate_id,
            "passed": True,
        })
        self.gates[gate_id] = receipt["payload_hash"]

    def require_gate(self, gate_id):
        if gate_id not in self.gates:
            raise StopRule(
                f"Missing required gate: {gate_id}",
                gate_id=gate_id,
            )
```

## Step 6: Add Entropy Tracking

```python
class MySystem:
    def __init__(self):
        self.entropy = 0.0

    def run_cycle(self):
        s_before = self.entropy
        work = self.do_work()  # Returns work performed
        self.entropy += work

        emit_receipt("entropy", {
            "s_before": s_before,
            "s_after": self.entropy,
            "work": work,
            "delta": abs((self.entropy - s_before) - work),
        })

        # Check bounds (P5)
        if abs((self.entropy - s_before) - work) >= 0.01:
            raise StopRule("Entropy violation", metric="entropy")
```

## Step 7: Replace Pre-Computed Results

### Before (Pre-stored)
```python
class FraudDetector:
    def __init__(self):
        self.fraud_alerts = {}  # Pre-computed!

    def detect(self, case_id):
        return self.fraud_alerts.get(case_id)
```

### After (Query-Time Derivation)
```python
class FraudDetector:
    def query(self, query_str):
        receipts = load_receipts()
        # Derive from receipts at query time
        if query_str == "detect_fraud":
            detection_receipts = [
                r for r in receipts
                if r["receipt_type"] == "detection"
            ]
            fraud_cases = [
                r for r in detection_receipts
                if r["verdict"] == "fraud"
            ]
            return {
                "query": query_str,
                "derived_from": [r["payload_hash"] for r in detection_receipts],
                "fraud_count": len(fraud_cases),
            }
```

## Step 8: Implement State Reconstruction

```python
class MySystem:
    def reconstruct_from_receipts(self):
        """P1: State must be reconstructable from receipts alone."""
        receipts = load_receipts()
        state = {}

        for receipt in receipts:
            rtype = receipt["receipt_type"]

            if rtype == "genesis":
                state["version"] = receipt.get("version")

            elif rtype == "ingest":
                state["last_ingest"] = receipt["payload_hash"]

            elif rtype == "decision":
                state["last_decision"] = receipt["payload_hash"]

            elif rtype == "gate":
                state.setdefault("gates", {})[receipt["gate_id"]] = True

        return state
```

## Step 9: Run Compliance Tests

```bash
# Create a test adapter
# tests/test_my_system.py

from starter.tests.compliance_suite import (
    test_principle_1_native_provenance,
    test_principle_2_cryptographic_lineage,
    # ... etc
)

def test_my_system():
    system = MySystem()
    test_principle_1_native_provenance(system)
    test_principle_2_cryptographic_lineage(system)
    # ... etc
```

Run tests:
```bash
pytest tests/test_my_system.py -v
```

Fix failures one by one until all 6 pass.

## Step 10: Continuous Verification

Add to CI/CD:

```yaml
# .github/workflows/compliance.yml
- name: Run compliance tests
  run: pytest tests/compliance_suite.py --system=my_system
```

Gate deployments on compliance:
```yaml
- name: Verify receipts-native
  run: |
    python -m starter.cli.verify_system my_system
    if [ $? -ne 0 ]; then
      echo "System is not receipts-native!"
      exit 1
    fi
```

## Common Migration Challenges

### Challenge: Too Many Logger Calls

**Solution**: Not all logs become receipts. Only:
- State changes
- Decisions
- Gates
- Entropy measurements

Debug logs can stay as logs.

### Challenge: Large State Objects

**Solution**: Hash the state, don't store it in receipts:
```python
emit_receipt("state_change", {
    "state_hash": dual_hash(json.dumps(large_state)),
})
```

### Challenge: External Dependencies

**Solution**: Emit receipts for external calls:
```python
response = external_api.call(params)
emit_receipt("external_call", {
    "api": "external_api",
    "params_hash": dual_hash(json.dumps(params)),
    "response_hash": dual_hash(json.dumps(response)),
})
```

### Challenge: Existing Database

**Solution**: Receipts don't replace your database. They provide an audit trail:
```python
db.save(record)
emit_receipt("db_write", {
    "table": "records",
    "record_hash": dual_hash(json.dumps(record)),
})
```

## Checklist

- [ ] Core primitives installed
- [ ] State changes emit receipts
- [ ] Decisions include input_hashes
- [ ] Parent_hash lineage implemented
- [ ] Genesis receipt exists
- [ ] Gates implemented with StopRule
- [ ] Entropy tracking (if applicable)
- [ ] Pre-computed results removed
- [ ] State reconstruction works
- [ ] All 6 compliance tests pass
- [ ] CI/CD gates on compliance
