#!/bin/bash
# gate_t2h.sh - T+2h Gate: SKELETON
# RUN THIS OR KILL PROJECT
#
# Required at T+2h:
# - spec.md exists
# - ledger_schema.json exists
# - cli.py exists and emits valid receipts
# - core.py has emit_receipt and dual_hash

echo "=== VL-JEPA Receipts-Native T+2h Gate ==="
echo "Checking skeleton requirements..."
echo ""

PASS=0
FAIL=0

check() {
    if [ $1 -eq 0 ]; then
        echo "[PASS] $2"
        ((PASS++)) || true
    else
        echo "[FAIL] $2"
        ((FAIL++)) || true
    fi
}

# Check spec.md exists
if [ -f spec.md ]; then check 0 "spec.md exists"; else check 1 "spec.md exists"; fi

# Check ledger_schema.json exists
if [ -f ledger_schema.json ]; then check 0 "ledger_schema.json exists"; else check 1 "ledger_schema.json exists"; fi

# Check cli.py exists
if [ -f cli.py ]; then check 0 "cli.py exists"; else check 1 "cli.py exists"; fi

# Check src/core/core.py exists
if [ -f src/core/core.py ]; then check 0 "src/core/core.py exists"; else check 1 "src/core/core.py exists"; fi

# Check cli.py emits valid receipt JSON
if python cli.py test 2>&1 | grep -q '"receipt_type"'; then check 0 "cli.py emits valid receipt JSON"; else check 1 "cli.py emits valid receipt JSON"; fi

# Check dual_hash is defined
if grep -q "def dual_hash" src/core/core.py; then check 0 "dual_hash function defined"; else check 1 "dual_hash function defined"; fi

# Check emit_receipt is defined
if grep -q "def emit_receipt" src/core/core.py; then check 0 "emit_receipt function defined"; else check 1 "emit_receipt function defined"; fi

# Check merkle is defined
if grep -q "def merkle" src/core/core.py; then check 0 "merkle function defined"; else check 1 "merkle function defined"; fi

# Check StopRule is defined
if grep -q "class StopRule" src/core/core.py; then check 0 "StopRule class defined"; else check 1 "StopRule class defined"; fi

# Check RACI support
if grep -q "raci" src/core/core.py; then check 0 "RACI support in core"; else check 1 "RACI support in core"; fi

# Check ledger_schema has hash_strategy
if grep -q '"hash_strategy"' ledger_schema.json; then check 0 "ledger_schema has hash_strategy"; else check 1 "ledger_schema has hash_strategy"; fi

# Check ledger_schema has receipt_types
if grep -q '"receipt_types"' ledger_schema.json; then check 0 "ledger_schema has receipt_types"; else check 1 "ledger_schema has receipt_types"; fi

echo ""
echo "=== Gate Results ==="
echo "PASS: $PASS"
echo "FAIL: $FAIL"

if [ $FAIL -eq 0 ]; then
    echo ""
    echo "PASS: T+2h gate - Skeleton complete"
    exit 0
else
    echo ""
    echo "FAIL: T+2h gate - Fix failures before proceeding"
    exit 1
fi
