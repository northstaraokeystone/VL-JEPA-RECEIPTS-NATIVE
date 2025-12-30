"""
Example systems for Receipts-Native Standard v1.1.

Contains:
- receipts_minimal: Minimal system that PASSES all 6 compliance tests
- simple_logger: Traditional logging system that FAILS all 6 tests
"""

from .receipts_minimal import ReceiptsMinimal
from .simple_logger import SimpleLogger

__all__ = ["ReceiptsMinimal", "SimpleLogger"]
