"""Gate module: Selective decode and entropy gating."""

from .selective_decode import (
    SelectiveDecoder,
    compute_entropy,
    entropy_gate,
)

__all__ = ["SelectiveDecoder", "compute_entropy", "entropy_gate"]
