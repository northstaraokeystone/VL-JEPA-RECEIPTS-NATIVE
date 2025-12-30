"""
Tesla FSD Frame Verification Module

Real-time frame verification for Full Self-Driving.
Critical latency requirement: <10ms p99.

Target ROI: $300-500M annually
Pain points: NHTSA scrutiny, liability proof, safety verification
"""

import time
from typing import Any
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core import emit_receipt, dual_hash, StopRule, load_thresholds
from src.gate import SelectiveDecoder, entropy_gate
from src.verify import TemporalVerifier
from src.governance import RACIManager


def verify_fsd_frame(
    frame: np.ndarray,
    frame_idx: int,
    vehicle_id: str = "unknown",
) -> tuple[bool, dict]:
    """
    Verify a single FSD frame.

    Args:
        frame: Camera frame
        frame_idx: Frame index
        vehicle_id: Vehicle identifier

    Returns:
        Tuple of (is_valid, receipt)
    """
    verifier = FSDFrameVerifier(vehicle_id=vehicle_id)
    return verifier.verify_frame(frame, frame_idx)


class FSDFrameVerifier:
    """
    FSD frame verification with entropy gating and temporal consistency.

    Optimized for <10ms latency.
    """

    def __init__(
        self,
        vehicle_id: str = "unknown",
        max_latency_ms: float = 10.0,
    ):
        self.vehicle_id = vehicle_id
        self.max_latency_ms = max_latency_ms
        self.domain = "tesla_fsd"

        thresholds = load_thresholds()

        # Initialize sub-modules
        self.selective_decoder = SelectiveDecoder(domain=self.domain)
        self.temporal_verifier = TemporalVerifier(domain=self.domain)
        self.raci_manager = RACIManager(domain=self.domain)

        # Frame buffer for temporal consistency
        self.frame_buffer: list[np.ndarray] = []
        self.embedding_buffer: list[np.ndarray] = []
        self.frame_count = 0

    def verify_frame(
        self,
        frame: np.ndarray,
        frame_idx: int,
    ) -> tuple[bool, dict]:
        """
        Verify a single frame with latency constraint.

        Args:
            frame: Camera frame
            frame_idx: Frame index

        Returns:
            Tuple of (is_valid, receipt)
        """
        start_time = time.time()

        import uuid
        event_id = str(uuid.uuid4())

        # Assign RACI
        raci_receipt = self.raci_manager.assign_raci(
            "frame_verification",
            event_id,
        )

        # Entropy gate check
        should_decode, gate_receipt = entropy_gate(frame, domain=self.domain)

        # Compute frame hash
        frame_hash = dual_hash(frame.tobytes())

        # Add to buffer
        self.frame_buffer.append(frame)
        if len(self.frame_buffer) > 8:
            self.frame_buffer.pop(0)

        # Temporal check (if we have enough frames)
        temporal_valid = True
        consistency_score = 1.0
        if len(self.frame_buffer) >= 4:
            temporal_valid, temp_receipt = self.temporal_verifier.verify_sequence(
                self.frame_buffer[-4:]
            )
            consistency_score = temp_receipt.get("consistency_score", 1.0)

        # Compute latency
        latency_ms = (time.time() - start_time) * 1000

        # Check latency SLO
        latency_ok = latency_ms <= self.max_latency_ms

        # Overall validity
        is_valid = temporal_valid and latency_ok

        self.frame_count += 1

        # Emit verification receipt
        receipt = emit_receipt("fsd_frame_verification", {
            "event_id": event_id,
            "vehicle_id": self.vehicle_id,
            "frame_idx": frame_idx,
            "frame_hash": frame_hash,
            "should_decode": should_decode,
            "entropy": gate_receipt.get("frame_entropy", 0),
            "temporal_valid": temporal_valid,
            "consistency_score": consistency_score,
            "latency_ms": latency_ms,
            "latency_slo_met": latency_ok,
            "is_valid": is_valid,
            "raci": raci_receipt.get("raci", {}),
        }, domain=self.domain)

        # Trigger stoprule if latency exceeded
        if not latency_ok:
            stoprule_fsd_latency(latency_ms, self.max_latency_ms)

        return is_valid, receipt

    def trigger_entropy_gate(
        self,
        frame: np.ndarray,
        reason: str = "high_entropy",
    ) -> dict:
        """
        Trigger entropy gate action.

        Called when selective decode determines frame needs attention.
        """
        import uuid
        event_id = str(uuid.uuid4())

        # Assign RACI for gate trigger
        raci_receipt = self.raci_manager.assign_raci(
            "entropy_gate_trigger",
            event_id,
        )

        return emit_receipt("entropy_gate_trigger", {
            "event_id": event_id,
            "vehicle_id": self.vehicle_id,
            "frame_hash": dual_hash(frame.tobytes()),
            "trigger_reason": reason,
            "action": "alert_safety_driver",
            "raci": raci_receipt.get("raci", {}),
        }, domain=self.domain)

    def get_verification_chain(self) -> tuple[str, dict]:
        """Get Merkle chain of all verified frames."""
        return self.temporal_verifier.get_merkle_chain()


def stoprule_fsd_latency(actual_ms: float, max_ms: float) -> None:
    """Stoprule for FSD latency violations."""
    raise StopRule(
        f"FSD latency violation: {actual_ms:.2f}ms > {max_ms}ms",
        metric="fsd_latency",
        delta=actual_ms - max_ms,
        action="halt",
    )
