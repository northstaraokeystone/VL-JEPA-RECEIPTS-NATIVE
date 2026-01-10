#!/usr/bin/env python3
"""
Provable Autonomy Demo - Main Orchestration

Narrative Arc:
1. FEAR: Show black-box confident failure on adversarial input
2. DETECTION: Show receipts-native entropy spike catches it
3. PROOF: Show receipt chain with cryptographic verification
4. SCALE: Show fleet aggregation (optional)

Runtime: <60 seconds

Usage:
    python demo/provable_autonomy.py --scenario stop_sign_patch
    python demo/provable_autonomy.py --scenario stop_sign_patch --display_mode headless
    python demo/provable_autonomy.py --help
"""

import argparse
import sys
import time
import json
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent))

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
from src.core.core import emit_receipt, merkle, dual_hash
from src.verify.temporal_consistency import TemporalVerifier


def run_demo(
    scenario: str = "stop_sign_patch",
    display_mode: str = "live",
    generate_artifact: bool = True,
    verbose: bool = True,
) -> Dict:
    """
    Run complete Provable Autonomy demo.

    Args:
        scenario: Which adversarial example to use
        display_mode: "live" (terminal) | "headless" (save video) | "screenshots"
        generate_artifact: Whether to generate PDF incident report
        verbose: Whether to print progress messages

    Sequence:
        1. Load baseline (clean frames)
        2. Load adversarial example
        3. Run black-box prediction (confident failure)
        4. Run receipts-native detection (catches anomaly)
        5. Display side-by-side comparison
        6. Show Merkle chain verification
        7. Generate PDF artifact (if requested)
        8. Emit demo_completion_receipt

    Returns:
    {
        "baseline": dict,
        "adversarial_result": dict,
        "black_box_result": dict,
        "receipts_result": dict,
        "merkle_root": str,
        "artifact_path": str (if generated),
        "demo_completion_receipt": dict
    }
    """
    start_time = time.time()
    receipts: List[Dict] = []

    if verbose:
        print("\n" + "=" * 70)
        print("PROVABLE AUTONOMY DEMO")
        print("Making Black-Box AI Architecturally Obsolete")
        print("=" * 70)

    # Validate scenario
    if scenario not in ATTACK_TYPES:
        raise ValueError(f"Unknown scenario: {scenario}. Valid: {list(ATTACK_TYPES.keys())}")

    # =========================================================================
    # PHASE 1: BASELINE (5 seconds narrative time)
    # =========================================================================
    if verbose:
        print("\n[PHASE 1] Establishing baseline from clean traffic signs...")

    baseline = load_clean_baseline(n_frames=50)
    receipts.append(baseline["baseline_receipt"])

    if verbose:
        print(f"    Baseline entropy: {baseline['baseline_mean']:.4f} ± {baseline['baseline_std']:.4f}")
        print(f"    Frames analyzed: {baseline['n_frames']}")
        print(f"    Merkle root: {baseline['baseline_receipt']['merkle_root'][:32]}...")

    # Narrative pause: establish normal
    if display_mode == "live":
        time.sleep(1.5)  # 1.5s pause per DEMO_STEALTH_BOMBER

    # =========================================================================
    # PHASE 2: THREAT (2 seconds narrative time)
    # =========================================================================
    if verbose:
        print("\n[PHASE 2] Loading adversarial input...")

    adversarial_frame, adversarial_metadata = load_adversarial_example(scenario)
    receipts.append(adversarial_metadata["load_receipt"])

    if verbose:
        print(f"    Attack type: {adversarial_metadata['attack_type']}")
        print(f"    Target sign: {adversarial_metadata['target_sign_name']}")
        print(f"    Expected misclassification: {adversarial_metadata['misclassified_as']}")

    # Narrative pause: fear
    if display_mode == "live":
        time.sleep(1.5)  # 1.5s pause before detection (fear)

    # =========================================================================
    # PHASE 3: BLACK-BOX FAILURE
    # =========================================================================
    if verbose:
        print("\n[PHASE 3] Black-box AI prediction...")

    black_box_result = generate_black_box_prediction(adversarial_frame, adversarial_metadata)
    receipts.append(black_box_result["receipt"])

    bb_wrong = black_box_result["classification"] != black_box_result["true_label"]

    if verbose:
        print(f"    Classification: {black_box_result['classification']}")
        print(f"    Confidence: {black_box_result['confidence']:.1%}")
        print(f"    True label: {black_box_result['true_label']}")
        print(f"    Result: {'WRONG - FAILURE' if bb_wrong else 'Correct'}")

    # =========================================================================
    # PHASE 4: RECEIPTS-NATIVE DETECTION (3 seconds narrative time)
    # =========================================================================
    if verbose:
        print("\n[PHASE 4] Receipts-native AI detection...")

    receipts_result = generate_receipts_native_detection(
        adversarial_frame,
        baseline,
        threshold_sigma=2.0
    )
    receipts.append(receipts_result["receipt"])

    if verbose:
        print(f"    Entropy score: {receipts_result['entropy_score']:.4f}")
        print(f"    Sigma delta: {receipts_result['sigma_delta']:+.2f}σ")
        print(f"    Verdict: {receipts_result['verdict']}")
        print(f"    Safe mode: {'TRIGGERED' if receipts_result['safe_mode_triggered'] else 'OFF'}")

    # Narrative pause: relief/detection
    if display_mode == "live":
        time.sleep(2.0)  # 2.0s pause after detection (relief)

    # =========================================================================
    # PHASE 5: DISPLAY COMPARISON (5 seconds narrative time)
    # =========================================================================
    if verbose:
        print("\n[PHASE 5] Side-by-side comparison...")
        ascii_frame = create_demo_frame_ascii(black_box_result, receipts_result)
        print(ascii_frame)

    # =========================================================================
    # PHASE 6: MERKLE VERIFICATION (5 seconds narrative time)
    # =========================================================================
    if verbose:
        print("\n[PHASE 6] Cryptographic verification...")

    # Compute Merkle root of all receipts
    merkle_root = merkle(receipts)

    # Verify chain integrity
    is_valid = verify_demo_integrity(receipts)

    if verbose:
        print(f"    Receipts in chain: {len(receipts)}")
        print(f"    Merkle root: {merkle_root[:48]}...")
        print(f"    Chain integrity: {'VERIFIED' if is_valid else 'FAILED'}")

    # =========================================================================
    # PHASE 7: ARTIFACT GENERATION
    # =========================================================================
    artifact_path = None
    if generate_artifact:
        if verbose:
            print("\n[PHASE 7] Generating incident report...")

        try:
            from artifacts.incident_report import generate_incident_report
            artifact_path = generate_incident_report(
                baseline=baseline,
                adversarial_result=adversarial_metadata,
                receipts=receipts,
                merkle_chain=[r.get("payload_hash", "") for r in receipts],
                output_path=f"incident_report_{scenario}.pdf"
            )
            if verbose:
                print(f"    Artifact: {artifact_path}")
        except ImportError:
            if verbose:
                print("    Artifact generation skipped (module not available)")
        except Exception as e:
            if verbose:
                print(f"    Artifact generation failed: {e}")

    # =========================================================================
    # PHASE 8: COMPLETION RECEIPT
    # =========================================================================
    elapsed = time.time() - start_time

    demo_completion_receipt = emit_receipt("demo_completion", {
        "scenario": scenario,
        "display_mode": display_mode,
        "elapsed_seconds": elapsed,
        "total_receipts": len(receipts),
        "merkle_root": merkle_root,
        "black_box_failed": bb_wrong,
        "receipts_native_caught": receipts_result["verdict"] == "ANOMALY",
        "chain_verified": is_valid,
        "artifact_generated": artifact_path is not None,
    }, domain="tesla_fsd")

    receipts.append(demo_completion_receipt)

    # =========================================================================
    # SUMMARY
    # =========================================================================
    if verbose:
        print("\n" + "=" * 70)
        print("DEMO COMPLETE")
        print("=" * 70)
        print(f"\nRuntime: {elapsed:.2f} seconds")
        print(f"Receipts generated: {len(receipts)}")
        print(f"\nResult:")
        print(f"  Black-box AI: {'FAILED (confident wrong prediction)' if bb_wrong else 'Passed'}")
        print(f"  Receipts-native AI: {'CAUGHT ANOMALY' if receipts_result['verdict'] == 'ANOMALY' else 'Missed'}")
        print(f"\nConclusion: Receipts-native AI provides PROVABLE safety guarantees.")
        print("            Black-box AI is architecturally inadequate for safety-critical systems.")
        print("")

    return {
        "baseline": baseline,
        "adversarial_result": adversarial_metadata,
        "adversarial_frame": adversarial_frame,
        "black_box_result": black_box_result,
        "receipts_result": receipts_result,
        "merkle_root": merkle_root,
        "all_receipts": receipts,
        "artifact_path": artifact_path,
        "demo_completion_receipt": demo_completion_receipt,
        "elapsed_seconds": elapsed,
    }


def display_narrative_sequence(
    frames: List,
    receipts: List[Dict],
    fps: int = 30,
) -> None:
    """
    Display demo with narrative pacing.

    Pacing per DEMO_STEALTH_BOMBER:
        - Phase 1 (baseline): Show clean frames, establish normal (5s)
        - Phase 2 (threat): Show adversarial input (2s)
        - Phase 3 (detection): Show entropy spike + verdict (3s)
        - Phase 4 (proof): Show receipt chain (5s)

    Total: ~15 seconds display time

    Uses:
        - 1.5s pause before detection (fear)
        - 2.0s pause after detection (relief)
        - Per DEMO_STEALTH_BOMBER doctrine
    """
    print("\n" + "=" * 70)
    print("NARRATIVE SEQUENCE")
    print("=" * 70)

    # Phase 1: Baseline
    print("\n[1/4] Establishing baseline (5s)...")
    time.sleep(1.0)  # Abbreviated for demo

    # Phase 2: Threat
    print("\n[2/4] Adversarial threat detected (2s)...")
    time.sleep(0.5)  # Abbreviated

    # Fear pause
    print("      [1.5s FEAR pause]")
    time.sleep(0.5)

    # Phase 3: Detection
    print("\n[3/4] Anomaly detected (3s)...")
    time.sleep(0.5)

    # Relief pause
    print("      [2.0s RELIEF pause]")
    time.sleep(0.5)

    # Phase 4: Proof
    print("\n[4/4] Receipt chain verification (5s)...")
    time.sleep(0.5)

    print("\nNarrative sequence complete.")


def verify_demo_integrity(receipts: List[Dict]) -> bool:
    """
    Verify all demo receipts form valid Merkle chain.

    Uses existing verification logic.

    Args:
        receipts: List of receipts from demo run

    Returns:
        True if all receipts verified, False otherwise
    """
    if not receipts:
        return True

    # Verify each receipt has required fields
    for i, receipt in enumerate(receipts):
        if "payload_hash" not in receipt:
            return False
        if "ts" not in receipt:
            return False
        if "receipt_type" not in receipt:
            return False

    # Verify Merkle chain consistency
    # Recompute hashes and verify they match
    for receipt in receipts:
        # Extract stored hash
        stored_hash = receipt.get("payload_hash", "")

        # Recompute hash
        payload_for_hash = {k: v for k, v in receipt.items() if k != "payload_hash"}
        recomputed_hash = dual_hash(json.dumps(payload_for_hash, sort_keys=True))

        if stored_hash != recomputed_hash:
            return False

    return True


def run_all_scenarios(verbose: bool = True) -> Dict:
    """
    Run all available adversarial scenarios.

    Args:
        verbose: Print progress

    Returns:
        Summary of all scenario results
    """
    results = {}

    for scenario in ATTACK_TYPES.keys():
        if verbose:
            print(f"\n{'='*70}")
            print(f"SCENARIO: {scenario}")
            print(f"{'='*70}")

        result = run_demo(
            scenario=scenario,
            display_mode="headless",
            generate_artifact=False,
            verbose=verbose,
        )

        results[scenario] = {
            "black_box_failed": result["black_box_result"]["classification"] != result["black_box_result"]["true_label"],
            "receipts_caught": result["receipts_result"]["verdict"] == "ANOMALY",
            "sigma_delta": result["receipts_result"]["sigma_delta"],
            "elapsed": result["elapsed_seconds"],
        }

    # Summary
    if verbose:
        print("\n" + "=" * 70)
        print("ALL SCENARIOS SUMMARY")
        print("=" * 70)
        for scenario, r in results.items():
            status = "PASS" if r["receipts_caught"] and r["black_box_failed"] else "FAIL"
            print(f"  {scenario}: {status} (σ={r['sigma_delta']:.2f}, {r['elapsed']:.2f}s)")

    return results


def main():
    """Main entry point with CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Provable Autonomy Demo - Making Black-Box AI Architecturally Obsolete"
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default="stop_sign_patch",
        choices=list(ATTACK_TYPES.keys()),
        help="Adversarial scenario to run"
    )
    parser.add_argument(
        "--display_mode",
        type=str,
        default="live",
        choices=["live", "headless", "screenshots"],
        help="Display mode"
    )
    parser.add_argument(
        "--no-artifact",
        action="store_true",
        help="Skip PDF artifact generation"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all scenarios"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose output"
    )

    args = parser.parse_args()

    if args.all:
        results = run_all_scenarios(verbose=not args.quiet)
    else:
        result = run_demo(
            scenario=args.scenario,
            display_mode=args.display_mode,
            generate_artifact=not args.no_artifact,
            verbose=not args.quiet,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
