"""
X/Twitter Batch Processor

High-throughput processing for platform-scale verification.
"""

from typing import Any
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core import emit_receipt, dual_hash, merkle
from .authenticity import XAuthenticityVerifier


class XBatchProcessor:
    """
    Batch processor for X platform verification.

    Handles high-throughput verification with receipts.
    """

    def __init__(
        self,
        tenant_id: str = "x_platform",
        max_workers: int = 4,
        batch_size: int = 100,
    ):
        self.tenant_id = tenant_id
        self.max_workers = max_workers
        self.batch_size = batch_size
        self.domain = "x_twitter"

        self.verifier = XAuthenticityVerifier(tenant_id=tenant_id)
        self.processed_count = 0
        self.batch_receipts: list[str] = []

    def process_batch(
        self,
        media_items: list[dict],
    ) -> tuple[list[dict], dict]:
        """
        Process a batch of media items.

        Args:
            media_items: List of media items to verify

        Returns:
            Tuple of (results, batch_receipt)
        """
        results = []

        # Process in chunks
        for i in range(0, len(media_items), self.batch_size):
            chunk = media_items[i:i + self.batch_size]
            chunk_results, _ = self.verifier.batch_verify(chunk)
            results.extend(chunk_results)

        self.processed_count += len(media_items)

        # Emit batch receipt
        verdicts = [r["verdict"] for r in results]
        batch_receipt = emit_receipt("x_batch_processed", {
            "tenant_id": self.tenant_id,
            "total_items": len(media_items),
            "batch_size": self.batch_size,
            "authentic_count": verdicts.count("AUTHENTIC"),
            "suspicious_count": verdicts.count("SUSPICIOUS"),
            "manipulated_count": verdicts.count("MANIPULATED"),
            "merkle_root": merkle([r.get("receipt_hash", "") for r in results]),
        }, domain=self.domain, tenant_id=self.tenant_id)

        self.batch_receipts.append(batch_receipt.get("payload_hash", ""))

        return results, batch_receipt

    def get_stats(self) -> dict:
        """Get processor statistics."""
        return {
            "processed_count": self.processed_count,
            "batch_count": len(self.batch_receipts),
            "batch_size": self.batch_size,
        }
