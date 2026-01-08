"""Model architectures for PyRat neural networks.

Each architecture owns its model, config, and loss computation.
"""

from __future__ import annotations

# Import architectures for registration
from alpharat.nn.architectures import local_value, mlp, symmetric

__all__ = ["local_value", "mlp", "symmetric"]
