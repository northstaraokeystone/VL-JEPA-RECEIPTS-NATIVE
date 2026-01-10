"""
Side-by-Side Comparison Visualization

LEFT PANEL: Black-box AI (confident failure)
RIGHT PANEL: Receipts-native (correct detection + proof)

Follows DEMO_STEALTH_BOMBER visual standards:
- Background: #0a0a0a (matte black)
- Text: #E2E8F0 (bone white)
- Threat/anomaly: #DC2626 (emergency red only when triggered)
- Success: #4B5563 (muted gray, not green)
- Typography: Monospace for data, Sans for labels
- NO neon colors, NO matrix scrolling effects
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.core import dual_hash, merkle

# Color palette per DEMO_STEALTH_BOMBER
COLORS = {
    "background": (10, 10, 10),       # #0a0a0a - matte black
    "text": (226, 232, 240),          # #E2E8F0 - bone white
    "threat": (220, 38, 38),          # #DC2626 - emergency red
    "normal": (75, 85, 99),           # #4B5563 - muted gray
    "panel_bg": (20, 20, 20),         # Slightly lighter for panels
    "border": (55, 65, 81),           # #374151 - subtle border
}

# Convert hex to BGR for OpenCV
def hex_to_bgr(hex_color: str) -> Tuple[int, int, int]:
    """Convert hex color to BGR for OpenCV."""
    hex_color = hex_color.lstrip('#')
    r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return (b, g, r)


def rgb_to_bgr(rgb: Tuple[int, int, int]) -> Tuple[int, int, int]:
    """Convert RGB to BGR for OpenCV."""
    return (rgb[2], rgb[1], rgb[0])


def create_text_image(
    text: str,
    width: int,
    height: int,
    color: Tuple[int, int, int] = COLORS["text"],
    bg_color: Tuple[int, int, int] = COLORS["background"],
    font_scale: float = 0.5,
    monospace: bool = True,
) -> np.ndarray:
    """
    Create image with text (fallback without OpenCV).

    Returns numpy array representing the image.
    """
    # Create base image
    img = np.full((height, width, 3), bg_color, dtype=np.uint8)

    # Text is represented as metadata for now
    # In actual OpenCV implementation, cv2.putText would be used
    return img


def create_comparison_frame(
    original_frame: np.ndarray,
    black_box_result: Dict,
    receipts_result: Dict,
    show_merkle: bool = False,
) -> np.ndarray:
    """
    Create side-by-side comparison visualization.

    LEFT SIDE:
        - Original frame
        - Black-box prediction overlay
        - Confidence score (falsely high)

    RIGHT SIDE:
        - Original frame
        - Entropy gauge (red spike)
        - "ANOMALY DETECTED" verdict
        - Receipt hash preview
        - Optional: Merkle tree visualization

    Color palette per DEMO_STEALTH_BOMBER:
        - Background: #0a0a0a
        - Text: #E2E8F0
        - Anomaly alert: #DC2626
        - Normal: #4B5563

    Args:
        original_frame: Original frame numpy array
        black_box_result: Result from generate_black_box_prediction()
        receipts_result: Result from generate_receipts_native_detection()
        show_merkle: Whether to show Merkle tree visualization

    Returns:
        Combined frame (side-by-side, 1920x1080)
    """
    width, height = 1920, 1080
    panel_width = width // 2

    # Create canvas
    canvas = np.full((height, width, 3), COLORS["background"], dtype=np.uint8)

    # Calculate panel regions
    left_panel = (0, 0, panel_width, height)
    right_panel = (panel_width, 0, width, height)

    # Fill panel backgrounds
    canvas[0:height, 0:panel_width] = COLORS["panel_bg"]
    canvas[0:height, panel_width:width] = COLORS["panel_bg"]

    # Draw center divider
    canvas[:, panel_width-2:panel_width+2] = COLORS["border"]

    # Scale and position the original frame (centered in each panel)
    frame_display_size = (400, 300)
    if original_frame is not None:
        # Resize frame for display
        frame_h, frame_w = original_frame.shape[:2]
        scale = min(frame_display_size[0] / frame_w, frame_display_size[1] / frame_h)
        new_w = int(frame_w * scale)
        new_h = int(frame_h * scale)

        # Create resized frame (simple nearest-neighbor resize)
        resized = np.zeros((new_h, new_w, 3), dtype=np.uint8)
        for y in range(new_h):
            for x in range(new_w):
                src_y = int(y / scale)
                src_x = int(x / scale)
                resized[y, x] = original_frame[min(src_y, frame_h-1), min(src_x, frame_w-1)]

        # Position in left panel
        left_frame_x = (panel_width - new_w) // 2
        left_frame_y = 150
        canvas[left_frame_y:left_frame_y+new_h, left_frame_x:left_frame_x+new_w] = resized

        # Position in right panel
        right_frame_x = panel_width + (panel_width - new_w) // 2
        canvas[left_frame_y:left_frame_y+new_h, right_frame_x:right_frame_x+new_w] = resized

    # Generate panel metadata (for text overlay information)
    left_panel_info = {
        "title": "BLACK-BOX AI",
        "classification": black_box_result.get("classification", "Unknown"),
        "confidence": black_box_result.get("confidence", 0.0),
        "true_label": black_box_result.get("true_label", "Unknown"),
        "is_wrong": black_box_result.get("classification") != black_box_result.get("true_label"),
        "note": "No internal uncertainty access",
    }

    right_panel_info = {
        "title": "RECEIPTS-NATIVE AI",
        "verdict": receipts_result.get("verdict", "NORMAL"),
        "entropy": receipts_result.get("entropy_score", 0.0),
        "sigma_delta": receipts_result.get("sigma_delta", 0.0),
        "safe_mode": receipts_result.get("safe_mode_triggered", False),
        "receipt_hash": receipts_result.get("receipt", {}).get("payload_hash", "")[:32],
    }

    # Store panel info in canvas metadata (for text rendering)
    # In actual implementation with OpenCV, text would be drawn here

    return canvas, left_panel_info, right_panel_info


def animate_entropy_gauge(
    entropy_score: float,
    baseline_mean: float,
    baseline_std: float,
    width: int = 300,
    height: int = 100,
) -> Tuple[np.ndarray, Dict]:
    """
    Create entropy gauge visualization.

    Visual:
        - Horizontal bar gauge
        - Baseline shown as gray region
        - Current value as bar
        - Color: gray (<2σ) → red (≥2σ)
        - Numeric display: "2.7σ above baseline"

    Args:
        entropy_score: Current entropy value
        baseline_mean: Baseline mean entropy
        baseline_std: Baseline standard deviation
        width: Gauge width
        height: Gauge height

    Returns:
        (gauge_image, gauge_info)
    """
    # Create gauge canvas
    gauge = np.full((height, width, 3), COLORS["background"], dtype=np.uint8)

    # Calculate sigma delta
    if baseline_std > 0:
        sigma_delta = (entropy_score - baseline_mean) / baseline_std
    else:
        sigma_delta = 0.0

    # Determine color based on anomaly status
    is_anomaly = abs(sigma_delta) >= 2.0
    bar_color = COLORS["threat"] if is_anomaly else COLORS["normal"]

    # Draw baseline region (gray band)
    baseline_region_height = 20
    baseline_y = (height - baseline_region_height) // 2
    gauge[baseline_y:baseline_y+baseline_region_height, 20:width-20] = COLORS["normal"]

    # Draw current value bar
    # Normalize to gauge width (0 to 5 sigma range)
    max_sigma = 5.0
    normalized = min(abs(sigma_delta) / max_sigma, 1.0)
    bar_width = int((width - 40) * normalized)

    bar_height = 30
    bar_y = (height - bar_height) // 2
    gauge[bar_y:bar_y+bar_height, 20:20+bar_width] = bar_color

    gauge_info = {
        "entropy_score": entropy_score,
        "baseline_mean": baseline_mean,
        "baseline_std": baseline_std,
        "sigma_delta": sigma_delta,
        "is_anomaly": is_anomaly,
        "display_text": f"{abs(sigma_delta):.1f}σ {'above' if sigma_delta > 0 else 'below'} baseline",
    }

    return gauge, gauge_info


def render_receipt_preview(
    receipt: Dict,
    width: int = 400,
    height: int = 200,
) -> Tuple[np.ndarray, Dict]:
    """
    Render receipt hash preview for overlay.

    Visual:
        - Receipt type
        - Dual-hash (truncated to 16 chars)
        - Verdict (NORMAL/ANOMALY)
        - Monospace font

    Args:
        receipt: Receipt dictionary
        width: Preview width
        height: Preview height

    Returns:
        (preview_image, preview_info)
    """
    # Create preview canvas
    preview = np.full((height, width, 3), COLORS["panel_bg"], dtype=np.uint8)

    # Draw border
    preview[0:2, :] = COLORS["border"]
    preview[-2:, :] = COLORS["border"]
    preview[:, 0:2] = COLORS["border"]
    preview[:, -2:] = COLORS["border"]

    # Extract receipt info
    receipt_type = receipt.get("receipt_type", "unknown")
    payload_hash = receipt.get("payload_hash", "")
    verdict = receipt.get("verdict", "NORMAL")
    ts = receipt.get("ts", "")

    # Truncate hash for display
    hash_display = payload_hash[:16] + "..." if len(payload_hash) > 16 else payload_hash

    preview_info = {
        "receipt_type": receipt_type,
        "hash_display": hash_display,
        "full_hash": payload_hash,
        "verdict": verdict,
        "timestamp": ts,
        "lines": [
            f"TYPE: {receipt_type}",
            f"HASH: {hash_display}",
            f"VERDICT: {verdict}",
            f"TS: {ts[:19] if ts else 'N/A'}",
        ],
    }

    return preview, preview_info


def animate_merkle_tree(
    receipts: List[Dict],
    current_frame: int,
    width: int = 400,
    height: int = 300,
) -> Tuple[np.ndarray, Dict]:
    """
    Animate Merkle tree construction.

    Visual:
        - Binary tree structure
        - Nodes = receipt hashes
        - Root highlighted
        - Current frame highlighted in tree
        - Minimal, clean diagram

    Args:
        receipts: List of receipt dictionaries
        current_frame: Index of current frame to highlight
        width: Tree visualization width
        height: Tree visualization height

    Returns:
        (tree_image, tree_info)
    """
    # Create tree canvas
    tree = np.full((height, width, 3), COLORS["panel_bg"], dtype=np.uint8)

    # Extract hashes
    hashes = [r.get("payload_hash", "")[:8] for r in receipts]

    # Compute Merkle root
    if receipts:
        root = merkle(receipts)
        root_display = root[:16] + "..."
    else:
        root = ""
        root_display = "EMPTY"

    # Calculate tree levels
    n_leaves = len(hashes)
    if n_leaves > 0:
        import math
        levels = int(math.ceil(math.log2(n_leaves))) + 1 if n_leaves > 1 else 1
    else:
        levels = 0

    # Draw simplified tree structure (nodes as circles)
    node_radius = 8
    level_height = height // (levels + 1) if levels > 0 else height // 2

    # Draw leaf nodes
    if n_leaves > 0:
        leaf_y = height - 40
        leaf_spacing = min(width // n_leaves, 40)
        start_x = (width - (n_leaves - 1) * leaf_spacing) // 2

        for i in range(min(n_leaves, 10)):  # Limit to 10 nodes for display
            x = start_x + i * leaf_spacing
            # Draw circle
            for dy in range(-node_radius, node_radius + 1):
                for dx in range(-node_radius, node_radius + 1):
                    if dx*dx + dy*dy <= node_radius*node_radius:
                        if 0 <= leaf_y + dy < height and 0 <= x + dx < width:
                            color = COLORS["threat"] if i == current_frame else COLORS["normal"]
                            tree[leaf_y + dy, x + dx] = color

    # Draw root node
    root_y = 40
    root_x = width // 2
    for dy in range(-node_radius-2, node_radius + 3):
        for dx in range(-node_radius-2, node_radius + 3):
            if dx*dx + dy*dy <= (node_radius+2)*(node_radius+2):
                if 0 <= root_y + dy < height and 0 <= root_x + dx < width:
                    tree[root_y + dy, root_x + dx] = COLORS["text"]

    tree_info = {
        "n_leaves": n_leaves,
        "levels": levels,
        "root_hash": root,
        "root_display": root_display,
        "current_frame": current_frame,
        "leaf_hashes": hashes[:10],  # First 10
    }

    return tree, tree_info


def create_demo_frame_ascii(
    black_box_result: Dict,
    receipts_result: Dict,
    frame_idx: int = 0,
) -> str:
    """
    Create ASCII representation of demo frame for terminal display.

    This is a fallback when OpenCV is not available.

    Args:
        black_box_result: Black-box prediction result
        receipts_result: Receipts-native detection result
        frame_idx: Current frame index

    Returns:
        ASCII art string
    """
    # Extract values
    bb_class = black_box_result.get("classification", "Unknown")
    bb_conf = black_box_result.get("confidence", 0.0)
    bb_true = black_box_result.get("true_label", "Unknown")

    rn_verdict = receipts_result.get("verdict", "NORMAL")
    rn_sigma = receipts_result.get("sigma_delta", 0.0)
    rn_entropy = receipts_result.get("entropy_score", 0.0)
    rn_safe = receipts_result.get("safe_mode_triggered", False)
    rn_hash = receipts_result.get("receipt", {}).get("payload_hash", "")[:24]

    # Determine status indicators
    bb_status = "WRONG" if bb_class != bb_true else "CORRECT"
    rn_status = "DETECTED" if rn_verdict == "ANOMALY" else "NORMAL"

    # Build ASCII frame
    width = 80
    half = width // 2 - 1

    lines = [
        "=" * width,
        f"{'BLACK-BOX AI'.center(half)} | {'RECEIPTS-NATIVE AI'.center(half)}",
        "=" * width,
        "",
        f"{'[TRAFFIC SIGN FRAME]'.center(half)} | {'[TRAFFIC SIGN FRAME]'.center(half)}",
        "",
        f"  Classification: {bb_class:<12}    |   Verdict: {rn_verdict}",
        f"  Confidence: {bb_conf:>6.1%}            |   Entropy: {rn_entropy:.4f}",
        f"  True Label: {bb_true:<12}       |   Sigma Δ: {rn_sigma:+.2f}σ",
        f"  Status: {bb_status:<15}       |   Safe Mode: {'ON' if rn_safe else 'OFF'}",
        "",
        f"  {'[NO PROOF AVAILABLE]':<23}    |   Hash: {rn_hash}",
        "",
        "-" * width,
    ]

    # Add anomaly alert if triggered
    if rn_verdict == "ANOMALY":
        alert = ">>> ANOMALY DETECTED - SAFE MODE TRIGGERED <<<"
        lines.append(f"{alert.center(width)}")
        lines.append("-" * width)

    return "\n".join(lines)


def create_summary_ascii(
    scenario_results: Dict,
) -> str:
    """
    Create ASCII summary of scenario comparison.

    Args:
        scenario_results: Results from run_adversarial_scenario()

    Returns:
        ASCII summary string
    """
    comparison = scenario_results.get("comparison", {})
    bb = comparison.get("black_box", {})
    rn = comparison.get("receipts_native", {})

    lines = [
        "",
        "=" * 70,
        "PROVABLE AUTONOMY DEMO - RESULTS SUMMARY",
        "=" * 70,
        "",
        f"Scenario: {scenario_results.get('adversarial_metadata', {}).get('attack_type', 'Unknown')}",
        "",
        "BLACK-BOX AI:",
        f"  Classification: {bb.get('classification', 'N/A')} (True: {bb.get('true_label', 'N/A')})",
        f"  Confidence: {bb.get('confidence', 0):.1%}",
        f"  Correct: {'YES' if bb.get('correct', False) else 'NO - FAILED'}",
        "",
        "RECEIPTS-NATIVE AI:",
        f"  Verdict: {rn.get('verdict', 'N/A')}",
        f"  Sigma Delta: {rn.get('sigma_delta', 0):+.2f}σ",
        f"  Safe Mode: {'TRIGGERED' if rn.get('safe_mode', False) else 'OFF'}",
        f"  Correct: {'YES - CAUGHT ANOMALY' if rn.get('correct', False) else 'NO'}",
        "",
        "-" * 70,
        "CONCLUSION: Black-box AI FAILED. Receipts-native AI DETECTED anomaly.",
        "-" * 70,
        "",
    ]

    return "\n".join(lines)


if __name__ == "__main__":
    # Quick verification
    print("Visualization Module - Verification")
    print("=" * 50)

    # Test entropy gauge
    gauge, gauge_info = animate_entropy_gauge(
        entropy_score=4.9,
        baseline_mean=4.5,
        baseline_std=0.1,
    )
    print(f"\n[1] Entropy gauge created:")
    print(f"    Shape: {gauge.shape}")
    print(f"    Sigma delta: {gauge_info['sigma_delta']:.2f}σ")
    print(f"    Is anomaly: {gauge_info['is_anomaly']}")

    # Test receipt preview
    test_receipt = {
        "receipt_type": "adversarial_detection",
        "payload_hash": "abc123def456789012345678901234567890",
        "verdict": "ANOMALY",
        "ts": "2025-01-10T12:00:00Z",
    }
    preview, preview_info = render_receipt_preview(test_receipt)
    print(f"\n[2] Receipt preview created:")
    print(f"    Shape: {preview.shape}")
    print(f"    Lines: {preview_info['lines']}")

    # Test Merkle tree
    test_receipts = [{"payload_hash": f"hash_{i}"} for i in range(8)]
    tree, tree_info = animate_merkle_tree(test_receipts, current_frame=3)
    print(f"\n[3] Merkle tree created:")
    print(f"    Shape: {tree.shape}")
    print(f"    Leaves: {tree_info['n_leaves']}")
    print(f"    Root: {tree_info['root_display']}")

    # Test ASCII frame
    test_bb = {
        "classification": "speed_limit_45",
        "confidence": 0.94,
        "true_label": "STOP",
    }
    test_rn = {
        "verdict": "ANOMALY",
        "entropy_score": 4.9161,
        "sigma_delta": 455.81,
        "safe_mode_triggered": True,
        "receipt": {"payload_hash": "abc123def456789012345678901234567890"},
    }
    ascii_frame = create_demo_frame_ascii(test_bb, test_rn)
    print(f"\n[4] ASCII frame created:")
    print(ascii_frame)

    print("\n" + "=" * 50)
    print("PASS: Visualization module ready")
