#!/bin/bash
# QED Entropy Conservation v12 - Reproduction Script

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "QED Entropy Conservation v12 - Verification"
echo "============================================"
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

# Analyze entropy
echo "Step 3: Analyzing entropy bounds..."
ENTROPY_COUNT=$(grep '"receipt_type":"entropy"' receipts.jsonl | wc -l)
echo "  Entropy receipts: $ENTROPY_COUNT"

# Check for violations
VIOLATIONS=$(grep '"receipt_type":"entropy"' receipts.jsonl | grep -c '"within_bounds":false' || true)
echo "  Violations: $VIOLATIONS"
echo ""

echo "============================================"
echo "VERIFICATION COMPLETE"
echo ""
echo "Results:"
echo "  - Merkle root: VERIFIED"
echo "  - Entropy cycles: $ENTROPY_COUNT"
echo "  - Violations: $VIOLATIONS"
echo "  - P5 Compliance: $([ "$VIOLATIONS" -eq 0 ] && echo 'PASSED' || echo 'FAILED')"
echo ""

exit 0
