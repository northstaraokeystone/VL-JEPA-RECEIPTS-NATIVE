"""
Encoder Provenance Module - Core Receipt Gap #5

VL-JEPA uses DINOv2 encoder but doesn't audit encoder outputs.
This module provides:

1. Layer-by-layer checksums
2. Encoder integrity verification
3. Model provenance receipts
"""

from typing import Any
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core import emit_receipt, dual_hash, StopRule


def compute_layer_checksum(layer_output: np.ndarray) -> str:
    """
    Compute checksum for a layer output.

    Args:
        layer_output: Layer activation as numpy array

    Returns:
        Checksum string
    """
    if hasattr(layer_output, "tobytes"):
        return dual_hash(layer_output.tobytes())
    return dual_hash(str(layer_output))


def compute_statistical_fingerprint(tensor: np.ndarray) -> dict:
    """
    Compute statistical fingerprint of a tensor.

    Captures distribution characteristics without storing full tensor.
    """
    flat = tensor.flatten()

    return {
        "mean": float(np.mean(flat)),
        "std": float(np.std(flat)),
        "min": float(np.min(flat)),
        "max": float(np.max(flat)),
        "median": float(np.median(flat)),
        "l2_norm": float(np.linalg.norm(flat)),
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
    }


def verify_encoder_integrity(
    model_id: str,
    expected_hash: str,
    actual_hash: str,
) -> tuple[bool, dict]:
    """
    Verify encoder model integrity.

    Args:
        model_id: Model identifier
        expected_hash: Expected model hash
        actual_hash: Actual computed hash

    Returns:
        Tuple of (is_valid, receipt)
    """
    is_valid = expected_hash == actual_hash

    receipt = emit_receipt("encoder_integrity", {
        "model_id": model_id,
        "expected_hash": expected_hash,
        "actual_hash": actual_hash,
        "integrity_verified": is_valid,
    }, domain="system")

    if not is_valid:
        stoprule_encoder_tampered(model_id, expected_hash, actual_hash)

    return is_valid, receipt


class EncoderAuditor:
    """
    Encoder auditing with layer-level provenance.

    Tracks encoder behavior and provides integrity receipts.
    """

    def __init__(
        self,
        encoder_id: str = "dinov2_base",
        encoder_version: str = "1.0",
        domain: str = "default",
    ):
        self.encoder_id = encoder_id
        self.encoder_version = encoder_version
        self.domain = domain

        self.layer_checksums: dict[str, str] = {}
        self.statistical_fingerprints: dict[str, dict] = {}
        self.audit_count = 0

    def audit_layer(
        self,
        layer_name: str,
        layer_output: np.ndarray,
    ) -> dict:
        """
        Audit a single encoder layer.

        Args:
            layer_name: Name of the layer
            layer_output: Layer output tensor

        Returns:
            Layer audit receipt
        """
        checksum = compute_layer_checksum(layer_output)
        fingerprint = compute_statistical_fingerprint(layer_output)

        self.layer_checksums[layer_name] = checksum
        self.statistical_fingerprints[layer_name] = fingerprint
        self.audit_count += 1

        receipt = emit_receipt("encoder_layer_audit", {
            "encoder_id": self.encoder_id,
            "encoder_version": self.encoder_version,
            "layer_name": layer_name,
            "checksum": checksum,
            "fingerprint": fingerprint,
        }, domain=self.domain)

        return receipt

    def audit_full_forward(
        self,
        layer_outputs: dict[str, np.ndarray],
        input_hash: str = None,
    ) -> dict:
        """
        Audit a complete forward pass.

        Args:
            layer_outputs: Dict of layer_name -> output tensor
            input_hash: Optional hash of input for traceability

        Returns:
            Forward pass audit receipt
        """
        # Audit each layer
        for layer_name, output in layer_outputs.items():
            self.audit_layer(layer_name, output)

        # Compute aggregate checksum
        all_checksums = "|".join(f"{k}:{v}" for k, v in sorted(self.layer_checksums.items()))
        aggregate_checksum = dual_hash(all_checksums)

        receipt = emit_receipt("encoder_forward_audit", {
            "encoder_id": self.encoder_id,
            "encoder_version": self.encoder_version,
            "layer_count": len(layer_outputs),
            "layer_names": list(layer_outputs.keys()),
            "aggregate_checksum": aggregate_checksum,
            "input_hash": input_hash,
            "layer_checksums": self.layer_checksums.copy(),
        }, domain=self.domain)

        return receipt

    def verify_reproducibility(
        self,
        input_data: np.ndarray,
        expected_output: np.ndarray,
        actual_output: np.ndarray,
        tolerance: float = 1e-5,
    ) -> tuple[bool, dict]:
        """
        Verify encoder output reproducibility.

        Args:
            input_data: Input tensor
            expected_output: Expected output
            actual_output: Actual output
            tolerance: Numerical tolerance

        Returns:
            Tuple of (is_reproducible, receipt)
        """
        diff = np.abs(expected_output - actual_output)
        max_diff = float(np.max(diff))
        mean_diff = float(np.mean(diff))

        is_reproducible = max_diff <= tolerance

        receipt = emit_receipt("encoder_reproducibility", {
            "encoder_id": self.encoder_id,
            "encoder_version": self.encoder_version,
            "input_hash": dual_hash(input_data.tobytes()),
            "expected_output_hash": dual_hash(expected_output.tobytes()),
            "actual_output_hash": dual_hash(actual_output.tobytes()),
            "max_diff": max_diff,
            "mean_diff": mean_diff,
            "tolerance": tolerance,
            "is_reproducible": is_reproducible,
        }, domain=self.domain)

        if not is_reproducible:
            stoprule_encoder_drift(max_diff, tolerance)

        return is_reproducible, receipt

    def get_encoder_fingerprint(self) -> tuple[str, dict]:
        """
        Get the current encoder fingerprint.

        Returns:
            Tuple of (fingerprint_hash, receipt)
        """
        fingerprint_data = {
            "encoder_id": self.encoder_id,
            "encoder_version": self.encoder_version,
            "layer_checksums": self.layer_checksums,
            "audit_count": self.audit_count,
        }

        fingerprint_hash = dual_hash(str(fingerprint_data))

        receipt = emit_receipt("encoder_fingerprint", {
            **fingerprint_data,
            "fingerprint_hash": fingerprint_hash,
        }, domain=self.domain)

        return fingerprint_hash, receipt

    def emit_provenance_receipt(
        self,
        model_weights_hash: str = None,
        config_hash: str = None,
        training_data_hash: str = None,
    ) -> dict:
        """
        Emit a complete encoder provenance receipt.

        Args:
            model_weights_hash: Hash of model weights
            config_hash: Hash of model config
            training_data_hash: Hash of training data reference

        Returns:
            Provenance receipt
        """
        receipt = emit_receipt("encoder_provenance", {
            "encoder_id": self.encoder_id,
            "encoder_version": self.encoder_version,
            "layer_checksums": self.layer_checksums,
            "model_weights_hash": model_weights_hash,
            "config_hash": config_hash,
            "training_data_hash": training_data_hash,
            "audit_count": self.audit_count,
        }, domain=self.domain)

        return receipt


def stoprule_encoder_tampered(model_id: str, expected: str, actual: str) -> None:
    """Stoprule for encoder tampering detection."""
    raise StopRule(
        f"Encoder tampered: {model_id} hash mismatch",
        metric="encoder_integrity",
        delta=-1.0,
        action="halt",
    )


def stoprule_encoder_drift(diff: float, tolerance: float) -> None:
    """Stoprule for encoder numerical drift."""
    raise StopRule(
        f"Encoder drift detected: diff {diff} > tolerance {tolerance}",
        metric="encoder_drift",
        delta=diff - tolerance,
        action="escalate",
    )
