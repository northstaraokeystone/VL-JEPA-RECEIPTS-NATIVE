"""Evolution module: Topology-driven module evolution and transfer learning."""

from .topology_classifier import (
    TopologyClassifier,
    TopologyState,
    TopologyAction,
)
from .cascade_spawner import (
    CascadeSpawner,
    ModuleVariant,
)
from .transfer_proposer import (
    TransferProposer,
    TransferType,
)
from .transfer_executor import (
    TransferExecutor,
)

__all__ = [
    "TopologyClassifier",
    "TopologyState",
    "TopologyAction",
    "CascadeSpawner",
    "ModuleVariant",
    "TransferProposer",
    "TransferType",
    "TransferExecutor",
]
