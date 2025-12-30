"""
Confidence Scoring Module - Core Receipt Gap #2

VL-JEPA outputs predictions but doesn't provide calibrated confidence
scores with SLO-linked receipts. This module provides:

1. Raw confidence extraction
2. Temperature scaling calibration
3. Expected Calibration Error (ECE) tracking
4. Confidence receipts with SLO assertions
"""

import math
from typing import Any
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core import emit_receipt, dual_hash, StopRule, load_thresholds


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Apply softmax with temperature scaling."""
    scaled = logits / temperature
    exp_x = np.exp(scaled - np.max(scaled))
    return exp_x / exp_x.sum()


def compute_ece(
    confidences: list[float],
    accuracies: list[bool],
    n_bins: int = 10,
) -> float:
    """
    Compute Expected Calibration Error.

    ECE measures how well confidence matches accuracy.
    Lower is better. Target: < 0.05

    Args:
        confidences: List of confidence scores (0-1)
        accuracies: List of whether predictions were correct
        n_bins: Number of bins for calibration

    Returns:
        ECE value (0-1)
    """
    if not confidences or not accuracies:
        return 0.0

    confidences = np.array(confidences)
    accuracies = np.array(accuracies, dtype=float)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        prop_in_bin = in_bin.mean()

        if prop_in_bin > 0:
            avg_confidence = confidences[in_bin].mean()
            avg_accuracy = accuracies[in_bin].mean()
            ece += np.abs(avg_accuracy - avg_confidence) * prop_in_bin

    return float(ece)


def calibrate_confidence(
    raw_confidence: float,
    temperature: float = 1.0,
    domain: str = "default",
) -> tuple[float, dict]:
    """
    Calibrate a raw confidence score using temperature scaling.

    Args:
        raw_confidence: Raw model confidence (0-1)
        temperature: Calibration temperature (>1 = more uncertain)
        domain: Domain for RACI assignment

    Returns:
        Tuple of (calibrated_confidence, receipt)
    """
    # Apply temperature scaling in log space
    if raw_confidence <= 0 or raw_confidence >= 1:
        # Clip to valid range
        raw_confidence = max(0.001, min(0.999, raw_confidence))

    log_odds = math.log(raw_confidence / (1 - raw_confidence))
    scaled_log_odds = log_odds / temperature
    calibrated = 1 / (1 + math.exp(-scaled_log_odds))

    receipt = emit_receipt("confidence_calibration", {
        "raw_confidence": raw_confidence,
        "temperature": temperature,
        "calibrated_confidence": calibrated,
        "calibration_method": "temperature_scaling",
    }, domain=domain)

    return calibrated, receipt


class ConfidenceScorer:
    """
    Confidence scoring with calibration and ECE tracking.

    Provides receipts for all confidence computations with SLO assertions.
    """

    def __init__(
        self,
        temperature: float = 1.0,
        confidence_threshold: float | None = None,
        ece_threshold: float = 0.05,
        domain: str = "default",
    ):
        thresholds = load_thresholds()

        self.temperature = temperature
        self.confidence_threshold = confidence_threshold or thresholds.get("confidence_threshold", 0.85)
        self.ece_threshold = ece_threshold
        self.domain = domain

        # Tracking for ECE computation
        self.confidences: list[float] = []
        self.predictions: list[Any] = []
        self.ground_truths: list[Any] = []

    def score(
        self,
        logits: np.ndarray | list[float],
        prediction: Any = None,
        ground_truth: Any = None,
    ) -> tuple[float, float, dict]:
        """
        Score confidence from logits.

        Args:
            logits: Raw model logits
            prediction: Optional prediction for tracking
            ground_truth: Optional ground truth for ECE tracking

        Returns:
            Tuple of (raw_confidence, calibrated_confidence, receipt)
        """
        if isinstance(logits, list):
            logits = np.array(logits)

        # Apply softmax to get probabilities
        probs = softmax(logits, temperature=1.0)  # Raw probs first
        raw_confidence = float(probs.max())

        # Apply temperature calibration
        calibrated_probs = softmax(logits, temperature=self.temperature)
        calibrated_confidence = float(calibrated_probs.max())

        # Track for ECE
        self.confidences.append(calibrated_confidence)
        if prediction is not None:
            self.predictions.append(prediction)
        if ground_truth is not None:
            self.ground_truths.append(ground_truth)

        # Check confidence SLO
        meets_threshold = calibrated_confidence >= self.confidence_threshold

        receipt = emit_receipt("confidence", {
            "raw_confidence": raw_confidence,
            "calibrated_confidence": calibrated_confidence,
            "temperature": self.temperature,
            "confidence_threshold": self.confidence_threshold,
            "meets_threshold": meets_threshold,
            "predicted_class": int(probs.argmax()) if len(probs) > 1 else None,
            "ece": self.compute_current_ece() if ground_truth is not None else None,
        }, domain=self.domain)

        return raw_confidence, calibrated_confidence, receipt

    def compute_current_ece(self) -> float:
        """Compute current ECE from tracked predictions."""
        if not self.confidences or not self.ground_truths:
            return 0.0

        accuracies = [p == g for p, g in zip(self.predictions, self.ground_truths)]
        return compute_ece(self.confidences, accuracies)

    def emit_ece_receipt(self) -> dict:
        """Emit ECE receipt with current calibration state."""
        ece = self.compute_current_ece()

        receipt = emit_receipt("ece_check", {
            "ece": ece,
            "ece_threshold": self.ece_threshold,
            "meets_threshold": ece <= self.ece_threshold,
            "sample_count": len(self.confidences),
            "temperature": self.temperature,
        }, domain=self.domain)

        # Check SLO
        if ece > self.ece_threshold * 2:  # Critical threshold
            stoprule_calibration(ece, self.ece_threshold)

        return receipt

    def suggest_temperature(self) -> tuple[float, dict]:
        """
        Suggest optimal temperature based on collected data.

        Uses simple grid search to minimize ECE.
        """
        if len(self.confidences) < 10:
            return self.temperature, emit_receipt("temperature_suggestion", {
                "suggested_temperature": self.temperature,
                "current_temperature": self.temperature,
                "reason": "insufficient_data",
                "sample_count": len(self.confidences),
            }, domain=self.domain)

        # Grid search for optimal temperature
        best_temp = self.temperature
        best_ece = float("inf")

        for temp in [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]:
            # Recalibrate all confidences with new temperature
            recalibrated = []
            for c in self.confidences:
                if c <= 0 or c >= 1:
                    c = max(0.001, min(0.999, c))
                log_odds = math.log(c / (1 - c))
                scaled = log_odds / temp
                recalibrated.append(1 / (1 + math.exp(-scaled)))

            accuracies = [p == g for p, g in zip(self.predictions, self.ground_truths)]
            ece = compute_ece(recalibrated, accuracies)

            if ece < best_ece:
                best_ece = ece
                best_temp = temp

        receipt = emit_receipt("temperature_suggestion", {
            "suggested_temperature": best_temp,
            "current_temperature": self.temperature,
            "current_ece": self.compute_current_ece(),
            "projected_ece": best_ece,
            "sample_count": len(self.confidences),
        }, domain=self.domain)

        return best_temp, receipt

    def reset(self) -> None:
        """Reset tracking state."""
        self.confidences = []
        self.predictions = []
        self.ground_truths = []


def stoprule_calibration(ece: float, threshold: float) -> None:
    """Stoprule for calibration failures."""
    raise StopRule(
        f"Calibration failure: ECE {ece} > threshold {threshold}",
        metric="ece",
        delta=ece - threshold,
        action="escalate",
    )


def stoprule_low_confidence(confidence: float, threshold: float) -> None:
    """Stoprule for critically low confidence on important decisions."""
    raise StopRule(
        f"Low confidence: {confidence} < threshold {threshold}",
        metric="confidence",
        delta=confidence - threshold,
        action="escalate",
    )
