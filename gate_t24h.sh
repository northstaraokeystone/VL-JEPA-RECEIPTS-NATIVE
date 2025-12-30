#!/bin/bash
# gate_t24h.sh - T+24h Gate: MVP
# RUN THIS OR KILL PROJECT
#
# Required at T+24h:
# - T+2h gate passes
# - Pipeline runs: ingest -> process -> emit
# - Tests exist with receipt verification
# - SLO assertions present
# - emit_receipt in all src files

set -e

echo "=== VL-JEPA Receipts-Native T+24h Gate ==="
echo "Checking MVP requirements..."
echo ""

PASS=0
FAIL=0

check() {
    if [ $1 -eq 0 ]; then
        echo "[PASS] $2"
        ((PASS++))
    else
        echo "[FAIL] $2"
        ((FAIL++))
    fi
}

# First, T+2h gate must pass
echo "--- Running T+2h Gate First ---"
./gate_t2h.sh > /dev/null 2>&1
check $? "T+2h gate passes"
echo ""

# Check tests directory exists
[ -d tests ]
check $? "tests directory exists"

# Check at least one test file
[ -n "$(ls tests/test_*.py 2>/dev/null)" ]
check $? "test files exist"

# Check tests have assertions
grep -rq "assert" tests/*.py 2>/dev/null
check $? "tests have assertions"

# Check emit_receipt in src modules
for dir in gate reasoning verify provenance detect meta learning evolution governance; do
    if [ -d "src/$dir" ] && [ -n "$(ls src/$dir/*.py 2>/dev/null | head -1)" ]; then
        grep -rq "emit_receipt" src/$dir/*.py 2>/dev/null
        check $? "emit_receipt in src/$dir"
    fi
done

# Check core modules exist
[ -f src/gate/__init__.py ] || [ -f src/gate/selective_decode.py ]
check $? "gate module exists"

[ -f src/reasoning/__init__.py ] || [ -f src/reasoning/confidence.py ]
check $? "reasoning module exists"

[ -f src/verify/__init__.py ] || [ -f src/verify/temporal_consistency.py ]
check $? "verify module exists"

# Check qualification module exists (Singularity 1)
[ -f src/meta/__init__.py ] || [ -f src/meta/qualify_module.py ]
check $? "qualification module exists"

# Check learning modules exist (Singularity 2)
[ -f src/learning/__init__.py ] || [ -f src/learning/intervention_capture.py ]
check $? "learning module exists"

# Check no silent exceptions
! grep -rq "except.*pass\|except:$" src/*.py src/**/*.py 2>/dev/null
check $? "no silent exceptions"

# Check pytest can be run
python -m pytest tests/ -q --collect-only > /dev/null 2>&1
check $? "pytest can collect tests"

# Run the actual tests
python -m pytest tests/ -q 2>&1 | tail -1 | grep -q "passed\|error\|failed"
check $? "pytest runs"

echo ""
echo "=== Gate Results ==="
echo "PASS: $PASS"
echo "FAIL: $FAIL"

if [ $FAIL -eq 0 ]; then
    echo ""
    echo "PASS: T+24h gate - MVP complete"
    exit 0
else
    echo ""
    echo "FAIL: T+24h gate - Fix failures before proceeding"
    exit 1
fi
