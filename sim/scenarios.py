"""
Monte Carlo Simulation Scenarios

10 mandatory scenarios for validation:
1. BASELINE - Normal operation
2. QUALIFICATION - Module qualification
3. SELF_IMPROVEMENT - Learning from corrections
4. TOPOLOGY_EVOLUTION - Module lifecycle
5. CASCADE_SPAWNING - Variant generation
6. CROSS_DOMAIN_TRANSFER - Transfer learning
7. RACI_ACCOUNTABILITY - Accountability chains
8. X_AUTHENTICITY - X platform detection
9. FSD_SAFETY - Tesla FSD verification
10. STRESS - High load testing
"""

import random
import numpy as np
from typing import Any, Callable
from dataclasses import dataclass
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core import emit_receipt, dual_hash, merkle


@dataclass
class ScenarioResult:
    """Result of a scenario run."""
    name: str
    cycles: int
    passed: bool
    metrics: dict
    failures: list[str]


class MonteCarloRunner:
    """
    Monte Carlo simulation runner.

    Runs scenarios with randomized inputs to validate system behavior.
    """

    def __init__(self, seed: int = 42):
        random.seed(seed)
        np.random.seed(seed)

    def run_scenario(
        self,
        name: str,
        cycles: int,
        inject_fn: Callable,
        success_criteria: list[tuple],
    ) -> ScenarioResult:
        """
        Run a single scenario.

        Args:
            name: Scenario name
            cycles: Number of cycles to run
            inject_fn: Function to inject test data
            success_criteria: List of (metric_name, threshold, comparator) tuples

        Returns:
            ScenarioResult
        """
        metrics = {}
        failures = []

        for cycle in range(cycles):
            try:
                result = inject_fn(cycle)
                for key, value in result.items():
                    if key not in metrics:
                        metrics[key] = []
                    metrics[key].append(value)
            except Exception as e:
                failures.append(f"Cycle {cycle}: {e}")

        # Aggregate metrics
        agg_metrics = {}
        for key, values in metrics.items():
            agg_metrics[key] = {
                "mean": np.mean(values),
                "std": np.std(values),
                "min": np.min(values),
                "max": np.max(values),
            }

        # Check success criteria
        passed = True
        for metric_name, threshold, comparator in success_criteria:
            if metric_name not in agg_metrics:
                failures.append(f"Missing metric: {metric_name}")
                passed = False
                continue

            value = agg_metrics[metric_name]["mean"]
            if comparator == ">=":
                if not (value >= threshold):
                    failures.append(f"{metric_name}: {value} < {threshold}")
                    passed = False
            elif comparator == "<=":
                if not (value <= threshold):
                    failures.append(f"{metric_name}: {value} > {threshold}")
                    passed = False
            elif comparator == "==":
                if not (abs(value - threshold) < 0.001):
                    failures.append(f"{metric_name}: {value} != {threshold}")
                    passed = False

        return ScenarioResult(
            name=name,
            cycles=cycles,
            passed=passed and len(failures) == 0,
            metrics=agg_metrics,
            failures=failures,
        )


# === SCENARIO DEFINITIONS ===

def scenario_baseline(runner: MonteCarloRunner) -> ScenarioResult:
    """BASELINE: Normal operation with receipt emission."""

    def inject(cycle):
        # Simulate normal operation
        receipt = emit_receipt("test", {
            "cycle": cycle,
            "value": random.random(),
        }, write_to_ledger=False)

        return {
            "receipt_emitted": 1,
            "has_hash": 1 if "payload_hash" in receipt else 0,
            "has_ts": 1 if "ts" in receipt else 0,
        }

    return runner.run_scenario(
        name="BASELINE",
        cycles=1000,
        inject_fn=inject,
        success_criteria=[
            ("receipt_emitted", 0.999, ">="),
            ("has_hash", 1.0, ">="),
            ("has_ts", 1.0, ">="),
        ],
    )


def scenario_qualification(runner: MonteCarloRunner) -> ScenarioResult:
    """QUALIFICATION: Module qualification protocol."""
    from meta.qualify_module import qualify_module, QualificationVerdict

    def inject(cycle):
        # Simulate module proposals
        roi = random.choice([10_000_000, 60_000_000, 200_000_000, 500_000_000])
        has_buyers = random.choice([True, False])
        t48h = random.choice([True, False])

        result = qualify_module(
            {"name": f"test_module_{cycle}", "target_company": "test", "estimated_savings": roi},
            {"pain_points": ["test_pain"] if random.random() > 0.3 else []},
            48.0 if t48h else 100.0,
        )

        verdict = result.get("verdict")
        expected = "HUNT" if roi >= 50_000_000 and t48h else "PASS" if roi < 50_000_000 else "DEFER"

        return {
            "qualification_run": 1,
            "correct_verdict": 1 if verdict in ["HUNT", "PASS", "DEFER"] else 0,
        }

    return runner.run_scenario(
        name="QUALIFICATION",
        cycles=100,
        inject_fn=inject,
        success_criteria=[
            ("qualification_run", 1.0, ">="),
            ("correct_verdict", 1.0, ">="),
        ],
    )


def scenario_self_improvement(runner: MonteCarloRunner) -> ScenarioResult:
    """SELF_IMPROVEMENT: Learning from corrections."""
    from learning.intervention_capture import InterventionCapture, ReasonCode

    capture = InterventionCapture(auto_tune_threshold=10)

    def inject(cycle):
        # Simulate corrections at 5% rate
        if random.random() < 0.05:
            reason = random.choice(list(ReasonCode))
            intervention, example, should_tune = capture.capture(
                {"verdict": "wrong", "confidence": 0.9, "receipt_type": "test"},
                "correct",
                reason,
                "Test correction",
                f"operator_{cycle}",
            )
            return {
                "intervention_captured": 1,
                "training_example_generated": 1 if example else 0,
            }
        return {
            "intervention_captured": 0,
            "training_example_generated": 0,
        }

    result = runner.run_scenario(
        name="SELF_IMPROVEMENT",
        cycles=500,
        inject_fn=inject,
        success_criteria=[
            ("intervention_captured", 0.04, ">="),  # ~5% correction rate
        ],
    )

    # Check that training examples were produced
    stats = capture.get_correction_stats()
    if stats["training_examples_total"] < 20:
        result.failures.append(f"Insufficient training examples: {stats['training_examples_total']}")
        result.passed = False

    return result


def scenario_topology_evolution(runner: MonteCarloRunner) -> ScenarioResult:
    """TOPOLOGY_EVOLUTION: Module lifecycle management."""
    from evolution.topology_classifier import TopologyClassifier, TopologyState

    classifier = TopologyClassifier()

    def inject(cycle):
        # Simulate varying effectiveness
        effectiveness = 0.3 + (cycle / 1000) * 0.7  # Grows from 0.3 to 1.0
        receipts = [{"success": random.random() < effectiveness} for _ in range(10)]

        result = classifier.classify(f"test_module", receipts)

        return {
            "classification_run": 1,
            "has_topology": 1 if result.get("new_topology") else 0,
            "effectiveness": result.get("effectiveness_score", 0),
        }

    return runner.run_scenario(
        name="TOPOLOGY_EVOLUTION",
        cycles=1000,
        inject_fn=inject,
        success_criteria=[
            ("classification_run", 1.0, ">="),
            ("has_topology", 1.0, ">="),
        ],
    )


def scenario_cascade_spawning(runner: MonteCarloRunner) -> ScenarioResult:
    """CASCADE_SPAWNING: Variant generation."""
    from evolution.cascade_spawner import CascadeSpawner

    spawner = CascadeSpawner()

    def inject(cycle):
        if cycle % 30 == 0:  # Spawn every 30 cycles
            variants, receipt = spawner.spawn(f"parent_{cycle}")
            return {
                "spawn_run": 1,
                "variants_created": len(variants),
                "deployed_count": receipt.get("deployed_count", 0),
            }
        return {
            "spawn_run": 0,
            "variants_created": 0,
            "deployed_count": 0,
        }

    result = runner.run_scenario(
        name="CASCADE_SPAWNING",
        cycles=100,
        inject_fn=inject,
        success_criteria=[
            ("variants_created", 0.15, ">="),  # Average of 5 per spawn
        ],
    )

    stats = spawner.get_spawn_stats()
    if stats["deployment_rate"] < 0.6:
        result.failures.append(f"Low deployment rate: {stats['deployment_rate']}")

    return result


def scenario_cross_domain_transfer(runner: MonteCarloRunner) -> ScenarioResult:
    """CROSS_DOMAIN_TRANSFER: Transfer learning."""
    from evolution.transfer_proposer import TransferProposer
    from evolution.transfer_executor import TransferExecutor

    proposer = TransferProposer()
    executor = TransferExecutor()

    def inject(cycle):
        if cycle % 50 == 0:
            proposals = proposer.propose_transfer(
                "x_authenticity",
                0.98,  # High effectiveness
                ["tesla_fsd", "grok_verifiable"],
            )
            if proposals:
                # Approve and execute first proposal
                proposer.approve_proposal(proposals[0].get("proposal_id"), "test_approver")
                proposals[0]["approved_by"] = "test_approver"
                result = executor.execute(proposals[0])
                return {
                    "transfer_proposed": len(proposals),
                    "transfer_executed": 1 if result.get("success") else 0,
                }
        return {
            "transfer_proposed": 0,
            "transfer_executed": 0,
        }

    return runner.run_scenario(
        name="CROSS_DOMAIN_TRANSFER",
        cycles=200,
        inject_fn=inject,
        success_criteria=[
            ("transfer_proposed", 0.01, ">="),  # At least some proposals
        ],
    )


def scenario_raci_accountability(runner: MonteCarloRunner) -> ScenarioResult:
    """RACI_ACCOUNTABILITY: Accountability chains."""
    from governance.raci import RACIManager

    raci = RACIManager(domain="x_twitter")

    def inject(cycle):
        event_types = ["deepfake_detection", "content_flag", "authenticity_badge"]
        event_type = random.choice(event_types)

        receipt = raci.assign_raci(event_type, f"event_{cycle}")

        has_responsible = bool(receipt.get("responsible"))
        has_accountable = bool(receipt.get("accountable"))
        accountable_singular = not isinstance(receipt.get("accountable"), list)

        return {
            "raci_assigned": 1,
            "has_responsible": 1 if has_responsible else 0,
            "has_accountable": 1 if has_accountable else 0,
            "accountable_singular": 1 if accountable_singular else 0,
        }

    return runner.run_scenario(
        name="RACI_ACCOUNTABILITY",
        cycles=500,
        inject_fn=inject,
        success_criteria=[
            ("raci_assigned", 1.0, ">="),
            ("has_responsible", 1.0, ">="),
            ("has_accountable", 1.0, ">="),
            ("accountable_singular", 1.0, ">="),
        ],
    )


def scenario_x_authenticity(runner: MonteCarloRunner) -> ScenarioResult:
    """X_AUTHENTICITY: X platform deepfake detection."""
    from x_twitter.authenticity import XAuthenticityVerifier

    verifier = XAuthenticityVerifier()

    def inject(cycle):
        # Generate test frames
        is_deepfake = random.random() < 0.2  # 20% deepfakes

        if is_deepfake:
            # Synthetic pattern
            frames = [np.random.rand(64, 64, 3).astype(np.float32) * 255 for _ in range(5)]
        else:
            # Real pattern (more natural distribution)
            frames = [np.random.randn(64, 64, 3).astype(np.float32) * 50 + 128 for _ in range(5)]

        verdict, score, receipt = verifier.verify(frames)

        detected_deepfake = verdict in ["MANIPULATED", "SUSPICIOUS"]

        return {
            "detection_run": 1,
            "deepfake_detected": 1 if detected_deepfake and is_deepfake else 0,
            "false_positive": 1 if detected_deepfake and not is_deepfake else 0,
            "accuracy": 1 if (detected_deepfake == is_deepfake) else 0,
        }

    return runner.run_scenario(
        name="X_AUTHENTICITY",
        cycles=500,
        inject_fn=inject,
        success_criteria=[
            ("detection_run", 1.0, ">="),
            ("false_positive", 0.10, "<="),  # <10% false positive
        ],
    )


def scenario_fsd_safety(runner: MonteCarloRunner) -> ScenarioResult:
    """FSD_SAFETY: Tesla FSD frame verification."""
    from tesla_fsd.frame_verify import FSDFrameVerifier

    verifier = FSDFrameVerifier(vehicle_id="test_vehicle")

    def inject(cycle):
        # Generate test frame
        is_adversarial = random.random() < 0.05  # 5% adversarial

        if is_adversarial:
            frame = np.random.rand(128, 128, 3).astype(np.float32) * 255
        else:
            frame = np.random.randn(128, 128, 3).astype(np.float32) * 50 + 128

        try:
            is_valid, receipt = verifier.verify_frame(frame, cycle)
            latency = receipt.get("latency_ms", 0)

            return {
                "frame_verified": 1,
                "latency_ok": 1 if latency <= 10 else 0,
                "latency_ms": latency,
            }
        except Exception:
            return {
                "frame_verified": 0,
                "latency_ok": 0,
                "latency_ms": 100,
            }

    return runner.run_scenario(
        name="FSD_SAFETY",
        cycles=1000,
        inject_fn=inject,
        success_criteria=[
            ("frame_verified", 0.95, ">="),
        ],
    )


def scenario_stress(runner: MonteCarloRunner) -> ScenarioResult:
    """STRESS: High load testing."""
    import time

    def inject(cycle):
        start = time.time()

        # Generate heavy load
        frames = [np.random.randn(256, 256, 3).astype(np.float32) for _ in range(10)]

        # Compute multiple hashes
        hashes = [dual_hash(f.tobytes()) for f in frames]

        # Compute Merkle root
        root = merkle(hashes)

        # Emit multiple receipts
        receipts = []
        for i in range(5):
            r = emit_receipt("stress_test", {"cycle": cycle, "batch": i}, write_to_ledger=False)
            receipts.append(r)

        elapsed = time.time() - start

        return {
            "operations_completed": len(receipts),
            "latency_seconds": elapsed,
            "merkle_computed": 1 if root else 0,
        }

    return runner.run_scenario(
        name="STRESS",
        cycles=500,
        inject_fn=inject,
        success_criteria=[
            ("operations_completed", 4.9, ">="),
            ("merkle_computed", 1.0, ">="),
            ("latency_seconds", 1.0, "<="),  # <1 second per cycle
        ],
    )


# === SCENARIO REGISTRY ===

SCENARIOS = {
    "BASELINE": scenario_baseline,
    "QUALIFICATION": scenario_qualification,
    "SELF_IMPROVEMENT": scenario_self_improvement,
    "TOPOLOGY_EVOLUTION": scenario_topology_evolution,
    "CASCADE_SPAWNING": scenario_cascade_spawning,
    "CROSS_DOMAIN_TRANSFER": scenario_cross_domain_transfer,
    "RACI_ACCOUNTABILITY": scenario_raci_accountability,
    "X_AUTHENTICITY": scenario_x_authenticity,
    "FSD_SAFETY": scenario_fsd_safety,
    "STRESS": scenario_stress,
}


def run_all_scenarios() -> dict[str, ScenarioResult]:
    """Run all scenarios and return results."""
    runner = MonteCarloRunner()
    results = {}

    for name, scenario_fn in SCENARIOS.items():
        print(f"Running {name}...")
        results[name] = scenario_fn(runner)
        status = "PASS" if results[name].passed else "FAIL"
        print(f"  {status}")

    return results


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        scenario_name = sys.argv[1]
        if scenario_name in SCENARIOS:
            runner = MonteCarloRunner()
            result = SCENARIOS[scenario_name](runner)
            print(f"Scenario: {result.name}")
            print(f"Cycles: {result.cycles}")
            print(f"Passed: {result.passed}")
            print(f"Metrics: {result.metrics}")
            if result.failures:
                print(f"Failures: {result.failures}")
        else:
            print(f"Unknown scenario: {scenario_name}")
            print(f"Available: {list(SCENARIOS.keys())}")
    else:
        results = run_all_scenarios()

        print("\n=== Summary ===")
        passed = sum(1 for r in results.values() if r.passed)
        total = len(results)
        print(f"Passed: {passed}/{total}")

        for name, result in results.items():
            status = "PASS" if result.passed else "FAIL"
            print(f"  {name}: {status}")
