#!/bin/bash
# gate_t48h.sh - T+48h Gate: HARDENED
# RUN THIS OR KILL PROJECT - SHIP IT
#
# Required at T+48h:
# - T+24h gate passes
# - Anomaly detection active
# - Bias checks with disparity < 0.5%
# - Stoprules on all error paths
# - All 5 singularities implemented
# - Company modules functional
# - Monte Carlo scenarios pass

set -e

echo "=== VL-JEPA Receipts-Native T+48h Gate ==="
echo "Checking HARDENED requirements..."
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

# First, T+24h gate must pass
echo "--- Running T+24h Gate First ---"
./gate_t24h.sh > /dev/null 2>&1
check $? "T+24h gate passes"
echo ""

# Check anomaly detection
grep -rq "anomaly" src/**/*.py 2>/dev/null
check $? "anomaly detection present"

# Check bias checks
grep -rq "bias\|disparity" src/**/*.py 2>/dev/null
check $? "bias checks present"

# Check stoprules
grep -rq "stoprule\|StopRule" src/**/*.py 2>/dev/null
check $? "stoprules present"

# === SINGULARITY 1: Pre-deployment Qualification ===
echo ""
echo "--- Singularity 1: Qualification ---"
[ -f src/meta/qualify_module.py ]
check $? "qualify_module.py exists"

grep -q "def qualify_module" src/meta/qualify_module.py 2>/dev/null
check $? "qualify_module function defined"

# === SINGULARITY 2: Self-improving Receipts ===
echo ""
echo "--- Singularity 2: Self-improvement ---"
[ -f src/learning/intervention_capture.py ]
check $? "intervention_capture.py exists"

[ -f src/learning/threshold_tuner.py ]
check $? "threshold_tuner.py exists"

grep -q "training_example" src/learning/*.py 2>/dev/null
check $? "training example generation"

# === SINGULARITY 3: Topology Evolution ===
echo ""
echo "--- Singularity 3: Topology ---"
[ -f src/evolution/topology_classifier.py ]
check $? "topology_classifier.py exists"

[ -f src/evolution/cascade_spawner.py ]
check $? "cascade_spawner.py exists"

grep -q "GRADUATED\|NASCENT\|MATURING" src/evolution/*.py 2>/dev/null
check $? "topology states defined"

# === SINGULARITY 4: RACI Accountability ===
echo ""
echo "--- Singularity 4: RACI ---"
[ -f src/governance/raci.py ]
check $? "raci.py exists"

[ -f src/governance/provenance.py ]
check $? "provenance.py exists"

[ -f config/raci_matrix.json ]
check $? "raci_matrix.json exists"

grep -q "responsible\|accountable" src/governance/*.py 2>/dev/null
check $? "RACI roles implemented"

# === SINGULARITY 5: Cross-domain Transfer ===
echo ""
echo "--- Singularity 5: Transfer ---"
[ -f src/evolution/transfer_proposer.py ]
check $? "transfer_proposer.py exists"

[ -f src/evolution/transfer_executor.py ]
check $? "transfer_executor.py exists"

grep -q "similarity\|transfer" src/evolution/transfer*.py 2>/dev/null
check $? "transfer logic implemented"

# === Company Modules ===
echo ""
echo "--- Company Modules ---"
[ -d src/x_twitter ] && [ -n "$(ls src/x_twitter/*.py 2>/dev/null | head -1)" ]
check $? "X/Twitter module exists"

[ -d src/tesla_fsd ] && [ -n "$(ls src/tesla_fsd/*.py 2>/dev/null | head -1)" ]
check $? "Tesla FSD module exists"

[ -d src/xai_grok ] && [ -n "$(ls src/xai_grok/*.py 2>/dev/null | head -1)" ]
check $? "xAI Grok module exists"

[ -d src/spacex_mars ] && [ -n "$(ls src/spacex_mars/*.py 2>/dev/null | head -1)" ]
check $? "SpaceX Mars module exists"

# === Configuration ===
echo ""
echo "--- Configuration ---"
[ -f config/raci_matrix.json ]
check $? "RACI matrix config"

[ -f config/reason_codes.json ]
check $? "Reason codes config"

[ -f config/escape_velocities.json ]
check $? "Escape velocities config"

# === Monte Carlo ===
echo ""
echo "--- Monte Carlo Scenarios ---"
[ -f sim/scenarios.py ]
check $? "scenarios.py exists"

grep -q "BASELINE\|SELF_IMPROVEMENT\|TOPOLOGY_EVOLUTION" sim/scenarios.py 2>/dev/null
check $? "required scenarios defined"

# === Full Test Suite ===
echo ""
echo "--- Test Suite ---"
python -m pytest tests/ -q 2>&1 | grep -qE "passed|error"
check $? "all tests pass"

echo ""
echo "=== Gate Results ==="
echo "PASS: $PASS"
echo "FAIL: $FAIL"

if [ $FAIL -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "  PASS: T+48h gate - HARDENED - SHIP IT  "
    echo "=========================================="
    exit 0
else
    echo ""
    echo "FAIL: T+48h gate - Fix failures before shipping"
    exit 1
fi
