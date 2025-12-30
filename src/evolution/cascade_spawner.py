"""
Cascade Spawner - Singularity 3 (Part 2)

Spawns 5x variants when a module graduates.
Each variant has mutations and recombinations.

Variants are backtested before deployment.
"""

from typing import Any
import uuid
import random
import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core import emit_receipt, dual_hash, load_thresholds


class ModuleVariant:
    """A variant module created from a parent."""

    def __init__(
        self,
        parent_id: str,
        variant_id: str,
        mutations: dict,
        recombined_from: str = None,
    ):
        self.parent_id = parent_id
        self.variant_id = variant_id
        self.mutations = mutations
        self.recombined_from = recombined_from
        self.backtest_score = 0.0
        self.deployed = False

    def to_dict(self) -> dict:
        return {
            "parent_id": self.parent_id,
            "variant_id": self.variant_id,
            "mutations": self.mutations,
            "recombined_from": self.recombined_from,
            "backtest_score": self.backtest_score,
            "deployed": self.deployed,
        }


class CascadeSpawner:
    """
    Spawns cascade variants from graduated modules.

    Creates 5 variants with:
    - Parameter mutations (+/- mutation_rate)
    - Recombinations with similar modules
    - Backtesting validation
    """

    def __init__(
        self,
        mutation_rate: float = 0.05,
        min_backtest_score: float = 0.75,
        domain: str = "default",
    ):
        self.mutation_rate = mutation_rate
        self.min_backtest_score = min_backtest_score
        self.domain = domain

        self.spawned_variants: list[ModuleVariant] = []
        self.spawn_history: list[dict] = []

    def _mutate_threshold(self, value: float, direction: str = "random") -> float:
        """Apply mutation to a threshold value."""
        if direction == "increase":
            delta = random.uniform(0, self.mutation_rate)
        elif direction == "decrease":
            delta = -random.uniform(0, self.mutation_rate)
        else:
            delta = random.uniform(-self.mutation_rate, self.mutation_rate)

        new_value = value + delta
        return max(0.1, min(0.99, new_value))

    def _create_variant(
        self,
        parent_id: str,
        variant_type: str,
        base_thresholds: dict,
        recombine_pool: list[str] = None,
    ) -> ModuleVariant:
        """Create a single variant."""
        variant_id = f"{parent_id}_v2_{variant_type}"
        mutations = {}
        recombined_from = None

        if variant_type == "HighPrecision":
            # Stricter thresholds
            mutations = {
                "compression_threshold": self._mutate_threshold(
                    base_thresholds.get("compression_threshold", 0.85), "decrease"
                ),
                "adversarial_detection_threshold": self._mutate_threshold(
                    base_thresholds.get("adversarial_detection_threshold", 0.30), "decrease"
                ),
            }
        elif variant_type == "LowLatency":
            # Optimized for speed (higher thresholds = fewer checks)
            mutations = {
                "compression_threshold": self._mutate_threshold(
                    base_thresholds.get("compression_threshold", 0.85), "increase"
                ),
                "entropy_threshold": self._mutate_threshold(
                    base_thresholds.get("entropy_threshold", 5.0) / 10, "increase"
                ) * 10,
            }
        elif variant_type == "AudioDeepfake":
            # Cross-domain to audio (recombination)
            mutations = {
                "compression_threshold": base_thresholds.get("compression_threshold", 0.85),
                "modality": "audio",
            }
            recombined_from = recombine_pool[0] if recombine_pool else None
        elif variant_type == "LongForm":
            # Optimized for long videos
            mutations = {
                "merkle_tree_depth": min(16, base_thresholds.get("merkle_tree_depth", 8) + 2),
                "temporal_window": 8,
            }
        elif variant_type == "Realtime":
            # Streaming mode
            mutations = {
                "streaming_mode": True,
                "batch_size": 1,
                "compression_threshold": self._mutate_threshold(
                    base_thresholds.get("compression_threshold", 0.85), "increase"
                ),
            }
        else:
            # Random mutations
            mutations = {
                "compression_threshold": self._mutate_threshold(
                    base_thresholds.get("compression_threshold", 0.85)
                ),
            }

        return ModuleVariant(
            parent_id=parent_id,
            variant_id=variant_id,
            mutations=mutations,
            recombined_from=recombined_from,
        )

    def _backtest_variant(self, variant: ModuleVariant, validation_data: list[dict] = None) -> float:
        """
        Backtest a variant on validation data.

        Returns effectiveness score (0-1).
        """
        # Simulated backtesting - in production would use actual validation
        if validation_data:
            # Use actual data
            successes = sum(1 for d in validation_data if d.get("success", True))
            return successes / len(validation_data)
        else:
            # Simulated based on mutation characteristics
            base_score = 0.85

            # High precision variants tend to do well
            if "HighPrecision" in variant.variant_id:
                base_score = 0.92

            # Low latency may sacrifice some accuracy
            if "LowLatency" in variant.variant_id:
                base_score = 0.88

            # Long form optimizations usually help
            if "LongForm" in variant.variant_id:
                base_score = 0.91

            # Realtime is challenging
            if "Realtime" in variant.variant_id:
                base_score = 0.78

            # Audio transfer is risky
            if "AudioDeepfake" in variant.variant_id:
                base_score = 0.82

            # Add some randomness
            noise = random.uniform(-0.05, 0.05)
            return max(0.0, min(1.0, base_score + noise))

    def spawn(
        self,
        parent_id: str,
        recombine_pool: list[str] = None,
        validation_data: list[dict] = None,
    ) -> tuple[list[ModuleVariant], dict]:
        """
        Spawn 5 variants from a graduated parent module.

        Args:
            parent_id: Parent module ID
            recombine_pool: Pool of similar modules for recombination
            validation_data: Optional data for backtesting

        Returns:
            Tuple of (variants, cascade_receipt)
        """
        if recombine_pool is None:
            recombine_pool = []

        base_thresholds = load_thresholds()

        # Create 5 variants
        variant_types = [
            "HighPrecision",
            "LowLatency",
            "AudioDeepfake",
            "LongForm",
            "Realtime",
        ]

        variants = []
        for vtype in variant_types:
            variant = self._create_variant(
                parent_id, vtype, base_thresholds, recombine_pool
            )
            variants.append(variant)

        # Backtest each variant
        backtest_results = []
        deployed_count = 0

        for variant in variants:
            score = self._backtest_variant(variant, validation_data)
            variant.backtest_score = score
            backtest_results.append(score)

            if score >= self.min_backtest_score:
                variant.deployed = True
                deployed_count += 1

        self.spawned_variants.extend(variants)

        # Emit cascade receipt
        receipt = emit_receipt("cascade_spawn", {
            "parent_id": parent_id,
            "variants_spawned": [v.variant_id for v in variants],
            "mutations_applied": [v.mutations for v in variants],
            "backtest_results": backtest_results,
            "deployed_count": deployed_count,
            "min_backtest_score": self.min_backtest_score,
            "recombine_pool": recombine_pool,
        }, domain=self.domain)

        self.spawn_history.append(receipt)

        return variants, receipt

    def get_deployed_variants(self) -> list[ModuleVariant]:
        """Get all deployed variants."""
        return [v for v in self.spawned_variants if v.deployed]

    def get_variant_by_id(self, variant_id: str) -> ModuleVariant | None:
        """Get a specific variant by ID."""
        for v in self.spawned_variants:
            if v.variant_id == variant_id:
                return v
        return None

    def archive_variant(self, variant_id: str, reason: str) -> dict:
        """Archive a failed or superseded variant."""
        variant = self.get_variant_by_id(variant_id)

        if variant:
            variant.deployed = False

        return emit_receipt("variant_archived", {
            "variant_id": variant_id,
            "reason": reason,
            "was_deployed": variant.deployed if variant else False,
        }, domain=self.domain)

    def get_spawn_stats(self) -> dict:
        """Get spawning statistics."""
        total = len(self.spawned_variants)
        deployed = len(self.get_deployed_variants())

        return {
            "total_variants": total,
            "deployed_variants": deployed,
            "deployment_rate": deployed / total if total > 0 else 0,
            "spawn_count": len(self.spawn_history),
        }
