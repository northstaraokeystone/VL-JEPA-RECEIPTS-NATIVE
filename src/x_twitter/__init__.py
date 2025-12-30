"""X/Twitter module: Authenticity verification and deepfake detection."""

from .authenticity import (
    XAuthenticityVerifier,
    verify_media_authenticity,
)
from .batch_processor import (
    XBatchProcessor,
)

__all__ = [
    "XAuthenticityVerifier",
    "verify_media_authenticity",
    "XBatchProcessor",
]
