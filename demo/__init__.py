"""
Provable Autonomy Demo

Side-by-side comparison showing how receipts-native AI detects
adversarial inputs that fool black-box classifiers.

Modules:
- adversarial_scenarios: Adversarial example generation and detection
- visualize_comparison: Side-by-side visualization
- provable_autonomy: Main demo orchestration
- verify_receipts: Standalone receipt verifier
"""

from demo.adversarial_scenarios import (
    load_clean_baseline,
    load_adversarial_example,
    generate_black_box_prediction,
    generate_receipts_native_detection,
    run_adversarial_scenario,
    ATTACK_TYPES,
)
from demo.visualize_comparison import (
    create_comparison_frame,
    animate_entropy_gauge,
    render_receipt_preview,
    animate_merkle_tree,
    create_demo_frame_ascii,
    create_summary_ascii,
)
from demo.provable_autonomy import (
    run_demo,
    run_all_scenarios,
    verify_demo_integrity,
)
from demo.verify_receipts import (
    verify_receipt_chain,
    verify_receipt_hash,
)

__all__ = [
    # adversarial_scenarios
    "load_clean_baseline",
    "load_adversarial_example",
    "generate_black_box_prediction",
    "generate_receipts_native_detection",
    "run_adversarial_scenario",
    "ATTACK_TYPES",
    # visualize_comparison
    "create_comparison_frame",
    "animate_entropy_gauge",
    "render_receipt_preview",
    "animate_merkle_tree",
    "create_demo_frame_ascii",
    "create_summary_ascii",
    # provable_autonomy
    "run_demo",
    "run_all_scenarios",
    "verify_demo_integrity",
    # verify_receipts
    "verify_receipt_chain",
    "verify_receipt_hash",
]
