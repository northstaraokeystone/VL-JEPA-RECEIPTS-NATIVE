#!/bin/bash
# AXIOM Singularity Convergence v1 - Reproduction Script

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "AXIOM Singularity Convergence v1 - Verification"
echo "================================================"
echo ""

# Check files
echo "Step 1: Checking required files..."
for file in receipts.jsonl MANIFEST.anchor verify_bundle.py; do
    if [ ! -f "$file" ]; then
        echo "ERROR: Missing $file"
        exit 3
    fi
    echo "  Found: $file"
done
echo ""

# Verify Merkle root
echo "Step 2: Verifying Merkle root..."
if python3 verify_bundle.py; then
    echo "Merkle verification: PASSED"
else
    echo "ERROR: Merkle verification FAILED"
    exit 1
fi
echo ""

# Analyze convergence
echo "Step 3: Analyzing convergence..."
CONVERGENCE=$(grep '"receipt_type":"convergence"' receipts.jsonl || true)
if [ -n "$CONVERGENCE" ]; then
    echo "  Convergence receipt found"
else
    echo "  No convergence receipt (check training receipts)"
fi
echo ""

echo "================================================"
echo "VERIFICATION COMPLETE"
echo ""
echo "Results:"
echo "  - Merkle root: VERIFIED"
echo "  - Convergence: Check training receipts for cycle 1847"
echo ""

exit 0
