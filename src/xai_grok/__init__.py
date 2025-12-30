"""xAI Grok module: Multimodal verification and confidence tracking."""

from .multimodal_verify import (
    GrokVerifier,
    verify_grok_response,
)

__all__ = [
    "GrokVerifier",
    "verify_grok_response",
]
