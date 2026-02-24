"""CNN architecture for PyRat."""

from alpharat.nn.architectures.cnn.blocks import (
    BlockConfig,
    GPoolBlockConfig,
    InterleavedBlockConfig,
    ResBlockConfig,
    TrunkConfig,
)
from alpharat.nn.architectures.cnn.config import (
    CNNModelConfig,
    CNNOptimConfig,
    KataGoCNNModelConfig,
    KataGoCNNOptimConfig,
)
from alpharat.nn.architectures.cnn.heads import (
    MLPPolicyHeadConfig,
    PointValueHeadConfig,
    PolicyHeadConfig,
    PooledValueHeadConfig,
    ValueHeadConfig,
)

__all__ = [
    "BlockConfig",
    "CNNModelConfig",
    "CNNOptimConfig",
    "GPoolBlockConfig",
    "InterleavedBlockConfig",
    "KataGoCNNModelConfig",
    "KataGoCNNOptimConfig",
    "MLPPolicyHeadConfig",
    "PointValueHeadConfig",
    "PolicyHeadConfig",
    "PooledValueHeadConfig",
    "ResBlockConfig",
    "TrunkConfig",
    "ValueHeadConfig",
]
