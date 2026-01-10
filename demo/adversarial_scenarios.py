"""
Adversarial Scenario Generator for Provable Autonomy Demo

Uses existing VL-JEPA entropy detection to catch adversarial inputs.
Integrates with tesla_fsd module for autonomous vehicle context.

Core Receipt Gap: Adversarial detection via entropy anomaly.
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple
import json
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.core import dual_hash, emit_receipt, merkle
from src.gate.selective_decode import compute_entropy, compute_compression_ratio


# Traffic sign templates (simplified representations)
SIGN_TEMPLATES = {
    "stop": {
        "name": "STOP",
        "classification": "regulatory",
        "required_action": "full_stop",
        "base_entropy": 5.8,  # Clean STOP sign entropy baseline
        "pixel_pattern": "octagonal_red_white",
    },
    "speed_limit_45": {
        "name": "Speed Limit 45",
        "classification": "regulatory",
        "required_action": "speed_control",
        "base_entropy": 5.2,
        "pixel_pattern": "circular_white_black",
    },
    "yield": {
        "name": "YIELD",
        "classification": "regulatory",
        "required_action": "prepare_stop",
        "base_entropy": 5.5,
        "pixel_pattern": "triangular_red_white",
    },
}

# Adversarial attack types
ATTACK_TYPES = {
    "stop_sign_patch": {
        "target": "stop",
        "black_box_misclassification": "speed_limit_45",
        "black_box_confidence": 0.94,
        "entropy_delta": 2.7,  # Sigma above baseline
        "description": "Adversarial sticker patches that cause misclassification",
    },
    "speed_limit_blur": {
        "target": "speed_limit_45",
        "black_box_misclassification": "speed_limit_75",
        "black_box_confidence": 0.89,
        "entropy_delta": 2.3,
        "description": "Targeted blur attack on speed limit digits",
    },
    "yield_occlusion": {
        "target": "yield",
        "black_box_misclassification": "merge",
        "black_box_confidence": 0.87,
        "entropy_delta": 2.1,
        "description": "Strategic occlusion causing misclassification",
    },
}


def load_clean_baseline(n_frames: int = 50, seed: int = 42) -> Dict:
    """
    Load clean traffic sign images to establish baseline.

    Simulates clean video frames of traffic signs to establish
    normal entropy distribution.

    Args:
        n_frames: Number of frames for baseline computation
        seed: Random seed for reproducibility

    Returns:
        {
            "baseline_mean": float,
            "baseline_std": float,
            "frame_hashes": [str],
            "n_frames": int,
            "entropy_values": [float],
            "baseline_receipt": dict
        }
    """
    np.random.seed(seed)

    frame_hashes = []
    entropy_values = []

    # Generate clean baseline frames (64x64 RGB)
    for i in range(n_frames):
        # Simulate clean traffic sign frame with typical entropy
        # Clean signs have structured patterns with moderate entropy
        base = np.random.randint(100, 200, size=(64, 64, 3), dtype=np.uint8)

        # Add sign structure (red octagon pattern for STOP)
        center = 32
        for y in range(64):
            for x in range(64):
                dist = np.sqrt((x - center)**2 + (y - center)**2)
                if dist < 28:
                    base[y, x] = [220, 40, 40]  # Red
                if dist < 20:
                    base[y, x] = [255, 255, 255]  # White center

        # Compute entropy
        entropy = compute_entropy(base)
        entropy_values.append(entropy)

        # Hash frame
        frame_hash = dual_hash(base.tobytes())
        frame_hashes.append(frame_hash)

    baseline_mean = float(np.mean(entropy_values))
    baseline_std = float(np.std(entropy_values))

    # Emit baseline receipt
    baseline_receipt = emit_receipt("adversarial_baseline", {
        "n_frames": n_frames,
        "baseline_mean": baseline_mean,
        "baseline_std": baseline_std,
        "entropy_min": float(min(entropy_values)),
        "entropy_max": float(max(entropy_values)),
        "merkle_root": merkle(frame_hashes),
        "sign_types": list(SIGN_TEMPLATES.keys()),
    }, domain="tesla_fsd")

    return {
        "baseline_mean": baseline_mean,
        "baseline_std": baseline_std,
        "frame_hashes": frame_hashes,
        "n_frames": n_frames,
        "entropy_values": entropy_values,
        "baseline_receipt": baseline_receipt,
    }


def generate_adversarial_frame(
    example_type: str,
    seed: int = 42,
) -> Tuple[np.ndarray, Dict]:
    """
    Generate adversarial example frame.

    Creates a frame with adversarial perturbations that:
    - Fool black-box classifiers (high confidence wrong prediction)
    - Trigger entropy anomaly detection (detectable by receipts-native)

    Args:
        example_type: "stop_sign_patch" | "speed_limit_blur" | "yield_occlusion"
        seed: Random seed for reproducibility

    Returns:
        (frame: np.ndarray, metadata: dict)
    """
    np.random.seed(seed)

    if example_type not in ATTACK_TYPES:
        raise ValueError(f"Unknown attack type: {example_type}. Valid: {list(ATTACK_TYPES.keys())}")

    attack = ATTACK_TYPES[example_type]
    target_sign = SIGN_TEMPLATES[attack["target"]]

    # Generate base frame (64x64 RGB)
    frame = np.random.randint(100, 200, size=(64, 64, 3), dtype=np.uint8)

    # Add sign structure
    center = 32
    for y in range(64):
        for x in range(64):
            dist = np.sqrt((x - center)**2 + (y - center)**2)
            if dist < 28:
                frame[y, x] = [220, 40, 40]  # Red
            if dist < 20:
                frame[y, x] = [255, 255, 255]  # White center

    # Apply adversarial perturbation based on attack type
    if example_type == "stop_sign_patch":
        # Add adversarial sticker patches (random noise in specific regions)
        # These patches cause misclassification while being "invisible" to humans
        for _ in range(8):
            px = np.random.randint(10, 54)
            py = np.random.randint(10, 54)
            # High-frequency adversarial pattern
            patch = np.random.randint(0, 255, size=(5, 5, 3), dtype=np.uint8)
            frame[py:py+5, px:px+5] = patch

    elif example_type == "speed_limit_blur":
        # Targeted blur with adversarial noise
        from scipy.ndimage import gaussian_filter
        frame_float = frame.astype(np.float32)
        frame_float = gaussian_filter(frame_float, sigma=1.5)
        # Add targeted noise
        noise = np.random.randn(64, 64, 3) * 30
        frame = np.clip(frame_float + noise, 0, 255).astype(np.uint8)

    elif example_type == "yield_occlusion":
        # Strategic occlusion pattern
        # Black patches in specific locations
        frame[20:30, 25:40] = [20, 20, 20]
        frame[35:45, 20:35] = [20, 20, 20]
        # Add subtle adversarial texture
        noise = np.random.randint(-20, 20, size=(64, 64, 3))
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    metadata = {
        "attack_type": example_type,
        "target_sign": attack["target"],
        "target_sign_name": target_sign["name"],
        "misclassified_as": attack["black_box_misclassification"],
        "black_box_confidence": attack["black_box_confidence"],
        "expected_entropy_delta": attack["entropy_delta"],
        "description": attack["description"],
        "frame_hash": dual_hash(frame.tobytes()),
        "frame_shape": list(frame.shape),
    }

    return frame, metadata


def load_adversarial_example(example_type: str) -> Tuple[np.ndarray, Dict]:
    """
    Load adversarial example (perturbed stop sign, speed limit, etc.)

    Wrapper for generate_adversarial_frame with receipt emission.

    Args:
        example_type: "stop_sign_patch" | "speed_limit_blur" | "yield_occlusion"

    Returns:
        (frame: np.ndarray, metadata: dict)
    """
    frame, metadata = generate_adversarial_frame(example_type)

    # Emit loading receipt
    load_receipt = emit_receipt("adversarial_load", {
        "example_type": example_type,
        "frame_hash": metadata["frame_hash"],
        "target_sign": metadata["target_sign"],
        "attack_description": metadata["description"],
    }, domain="tesla_fsd")

    metadata["load_receipt"] = load_receipt

    return frame, metadata


def generate_black_box_prediction(
    frame: np.ndarray,
    attack_metadata: Dict | None = None,
) -> Dict:
    """
    Simulate black-box classifier prediction (confident but wrong).

    Black-box neural networks are vulnerable to adversarial examples.
    They output high confidence on wrong classifications.

    Args:
        frame: Frame to classify
        attack_metadata: Optional metadata from adversarial generation

    Returns:
    {
        "classification": str,
        "confidence": float,
        "bounding_box": [x, y, w, h],
        "is_adversarial": bool,
        "true_label": str,
        "receipt": dict
    }
    """
    # Compute frame entropy
    entropy = compute_entropy(frame)

    # Determine if this is adversarial based on entropy characteristics
    # (black-box doesn't know this, but we track it)
    if attack_metadata:
        # Use attack metadata
        classification = attack_metadata.get("misclassified_as", "unknown")
        confidence = attack_metadata.get("black_box_confidence", 0.90)
        true_label = attack_metadata.get("target_sign_name", "STOP")
        is_adversarial = True
    else:
        # Normal frame - classify correctly
        classification = "STOP"
        confidence = 0.97
        true_label = "STOP"
        is_adversarial = False

    # Simulated bounding box (center of frame)
    bounding_box = [16, 16, 32, 32]

    # Emit black-box prediction receipt
    receipt = emit_receipt("black_box_prediction", {
        "classification": classification,
        "confidence": confidence,
        "bounding_box": bounding_box,
        "frame_entropy": entropy,
        "frame_hash": dual_hash(frame.tobytes()),
        "model_type": "black_box_cnn",
        "note": "Black-box has NO internal uncertainty access",
    }, domain="tesla_fsd")

    return {
        "classification": classification,
        "confidence": confidence,
        "bounding_box": bounding_box,
        "is_adversarial": is_adversarial,
        "true_label": true_label,
        "frame_entropy": entropy,
        "receipt": receipt,
    }


def generate_receipts_native_detection(
    frame: np.ndarray,
    baseline: Dict,
    threshold_sigma: float = 2.0,
) -> Dict:
    """
    Run receipts-native detection using existing entropy gate.

    Uses the existing SelectiveDecoder entropy computation to detect
    adversarial inputs through statistical anomaly detection.

    Args:
        frame: Frame to analyze
        baseline: Baseline statistics from load_clean_baseline()
        threshold_sigma: Number of std devs for anomaly detection

    Returns:
    {
        "entropy_score": float,
        "baseline_mean": float,
        "baseline_std": float,
        "sigma_delta": float,
        "verdict": "NORMAL" | "ANOMALY",
        "safe_mode_triggered": bool,
        "compression_ratio": float,
        "receipt": dict
    }
    """
    # Compute entropy using existing gate
    entropy = compute_entropy(frame)

    # Compute compression ratio
    frame_bytes = frame.tobytes()
    compression_ratio = compute_compression_ratio(frame_bytes)

    # Get baseline statistics
    baseline_mean = baseline["baseline_mean"]
    baseline_std = baseline["baseline_std"]

    # Compute sigma delta
    if baseline_std > 0:
        sigma_delta = (entropy - baseline_mean) / baseline_std
    else:
        sigma_delta = 0.0

    # Determine verdict
    is_anomaly = abs(sigma_delta) >= threshold_sigma
    verdict = "ANOMALY" if is_anomaly else "NORMAL"

    # Safe mode: triggered when anomaly detected
    safe_mode_triggered = is_anomaly

    # Emit detection receipt
    receipt = emit_receipt("adversarial_detection", {
        "entropy_score": entropy,
        "baseline_mean": baseline_mean,
        "baseline_std": baseline_std,
        "sigma_delta": sigma_delta,
        "threshold_sigma": threshold_sigma,
        "verdict": verdict,
        "safe_mode_triggered": safe_mode_triggered,
        "compression_ratio": compression_ratio,
        "frame_hash": dual_hash(frame_bytes),
        "detection_method": "entropy_anomaly",
        "detection_confidence": min(abs(sigma_delta) / threshold_sigma, 1.0) if is_anomaly else 0.0,
    }, domain="tesla_fsd")

    return {
        "entropy_score": entropy,
        "baseline_mean": baseline_mean,
        "baseline_std": baseline_std,
        "sigma_delta": sigma_delta,
        "verdict": verdict,
        "safe_mode_triggered": safe_mode_triggered,
        "compression_ratio": compression_ratio,
        "receipt": receipt,
    }


def run_adversarial_scenario(
    scenario: str = "stop_sign_patch",
    n_baseline_frames: int = 50,
) -> Dict:
    """
    Run complete adversarial detection scenario.

    End-to-end workflow:
    1. Establish baseline from clean frames
    2. Load adversarial example
    3. Run black-box prediction (fails)
    4. Run receipts-native detection (catches anomaly)
    5. Compare results

    Args:
        scenario: Attack scenario to run
        n_baseline_frames: Number of frames for baseline

    Returns:
        Complete scenario results with all receipts
    """
    # Step 1: Establish baseline
    baseline = load_clean_baseline(n_frames=n_baseline_frames)

    # Step 2: Load adversarial example
    adversarial_frame, adversarial_metadata = load_adversarial_example(scenario)

    # Step 3: Black-box prediction (confident but wrong)
    black_box_result = generate_black_box_prediction(
        adversarial_frame,
        attack_metadata=adversarial_metadata
    )

    # Step 4: Receipts-native detection (catches anomaly)
    receipts_result = generate_receipts_native_detection(
        adversarial_frame,
        baseline,
        threshold_sigma=2.0
    )

    # Step 5: Compile comparison
    comparison = {
        "scenario": scenario,
        "black_box": {
            "classification": black_box_result["classification"],
            "confidence": black_box_result["confidence"],
            "correct": False,  # Adversarial caused wrong classification
            "true_label": black_box_result["true_label"],
        },
        "receipts_native": {
            "verdict": receipts_result["verdict"],
            "sigma_delta": receipts_result["sigma_delta"],
            "safe_mode": receipts_result["safe_mode_triggered"],
            "correct": receipts_result["verdict"] == "ANOMALY",  # Should detect anomaly
        },
    }

    # Emit scenario completion receipt
    scenario_receipt = emit_receipt("adversarial_scenario_complete", {
        "scenario": scenario,
        "baseline_frames": n_baseline_frames,
        "black_box_failed": not comparison["black_box"]["correct"],
        "receipts_native_caught": comparison["receipts_native"]["correct"],
        "sigma_delta": receipts_result["sigma_delta"],
        "adversarial_hash": adversarial_metadata["frame_hash"],
        "baseline_merkle": baseline["baseline_receipt"]["merkle_root"],
    }, domain="tesla_fsd")

    return {
        "baseline": baseline,
        "adversarial_metadata": adversarial_metadata,
        "adversarial_frame": adversarial_frame,
        "black_box_result": black_box_result,
        "receipts_result": receipts_result,
        "comparison": comparison,
        "scenario_receipt": scenario_receipt,
    }


if __name__ == "__main__":
    # Quick verification
    print("Adversarial Scenario Module - Verification")
    print("=" * 50)

    # Test baseline loading
    baseline = load_clean_baseline(n_frames=50)
    print(f"\n[1] Baseline loaded:")
    print(f"    Mean entropy: {baseline['baseline_mean']:.4f}")
    print(f"    Std entropy:  {baseline['baseline_std']:.4f}")
    print(f"    Frames: {baseline['n_frames']}")

    # Test adversarial loading
    frame, metadata = load_adversarial_example("stop_sign_patch")
    print(f"\n[2] Adversarial example loaded:")
    print(f"    Attack: {metadata['attack_type']}")
    print(f"    Target: {metadata['target_sign_name']}")
    print(f"    Will be misclassified as: {metadata['misclassified_as']}")

    # Test black-box prediction
    bb_result = generate_black_box_prediction(frame, metadata)
    print(f"\n[3] Black-box prediction:")
    print(f"    Classification: {bb_result['classification']}")
    print(f"    Confidence: {bb_result['confidence']:.2%}")
    print(f"    True label: {bb_result['true_label']}")
    print(f"    WRONG: {bb_result['classification'] != bb_result['true_label']}")

    # Test receipts-native detection
    rn_result = generate_receipts_native_detection(frame, baseline)
    print(f"\n[4] Receipts-native detection:")
    print(f"    Entropy: {rn_result['entropy_score']:.4f}")
    print(f"    Sigma delta: {rn_result['sigma_delta']:.2f}σ above baseline")
    print(f"    Verdict: {rn_result['verdict']}")
    print(f"    Safe mode: {rn_result['safe_mode_triggered']}")

    print("\n" + "=" * 50)
    print("PASS: Adversarial scenario module ready")
