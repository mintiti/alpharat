# Neural Network Module

Training infrastructure for PyRat neural networks with pluggable architectures.

## Quick Start

```bash
# Train any architecture (auto-detected from config)
uv run python scripts/train.py configs/train.yaml --epochs 100

# Train and benchmark against baselines
uv run python scripts/train_and_benchmark.py configs/train.yaml --games 50
```

## Config Format

All architectures use the same YAML schema. Pydantic discriminated unions auto-dispatch based on `architecture`:

```yaml
model:
  architecture: mlp  # or "symmetric" or "local_value"
  hidden_dim: 256
  dropout: 0.0
  p_augment: 0.5     # mlp/local_value only

optim:
  architecture: mlp  # must match model
  lr: 1e-3
  batch_size: 4096
  policy_weight: 1.0
  value_weight: 1.0
  # Architecture-specific:
  ownership_weight: 1.0  # local_value only

data:
  train_dir: data/train
  val_dir: data/val

seed: 42
resume_from: null  # or path to checkpoint
```

## Architecture Comparison

| Architecture | Description | Augmentation |
|--------------|-------------|--------------|
| `mlp` | Flat observation, shared trunk, separate policy/value heads | Player swap |
| `symmetric` | Structural P1/P2 symmetry in network design | None needed |
| `local_value` | Per-cell ownership logits + scalar value heads | Player swap |

## Module Structure

```
nn/
├── config.py           # TrainConfig with discriminated unions (imports architectures)
│
├── training/           # Generic training infrastructure (no arch knowledge)
│   ├── loop.py        # run_training() - works with any architecture
│   ├── base.py        # BaseModelConfig, BaseOptimConfig, DataConfig
│   ├── protocols.py   # TrainableModel, LossFunction, AugmentationStrategy
│   └── keys.py        # ModelOutput, LossKey, BatchKey (StrEnum)
│
├── architectures/      # Architecture-specific configs + losses
│   ├── mlp/
│   │   ├── config.py  # MLPModelConfig, MLPOptimConfig
│   │   └── loss.py    # compute_mlp_losses()
│   ├── symmetric/
│   │   ├── config.py  # SymmetricModelConfig, SymmetricOptimConfig
│   │   └── loss.py    # compute_symmetric_losses()
│   └── local_value/
│       ├── config.py  # LocalValueModelConfig, LocalValueOptimConfig
│       └── loss.py    # compute_local_value_losses()
│
├── models/            # nn.Module implementations
│   ├── mlp.py         # PyRatMLP
│   ├── symmetric.py   # SymmetricMLP
│   └── local_value.py # LocalValueMLP
│
├── losses/            # Shared loss utilities (imported via __init__)
│   └── ownership.py         # compute_ownership_loss()
│
├── builders/          # Observation builders
│   └── flat.py        # FlatObservationBuilder - 1D encoding
│
└── augmentation.py    # PlayerSwapStrategy, NoAugmentation
```

## Data Pipeline: Batches → Shards

The data pipeline has two stages:

1. **Batches** (`experiments/batches/`) — Raw game recordings from sampling. Contains game state arrays, MCTS policies, outcomes. Architecture-agnostic.

2. **Shards** (`experiments/shards/`) — Pre-tensorized data ready for training. The goal is maximum training speed: load shards → train, no per-sample transformation.

**Currently:** `FlatObservationBuilder` converts batches to shards with a 1D flat encoding:
```
[maze H×W×4] [p1_pos H×W] [p2_pos H×W] [cheese H×W] [score_diff, progress, p1_mud, p2_mud, p1_score, p2_score]
```

All current architectures (MLP, Symmetric, LocalValue) consume this flat format. The model parses the flat vector internally.

**When you need a new observation builder:**

- **GNN architecture** — needs graph structure (adjacency, node features), not flat vectors
- **Transformer** — might want sequence tokenization or different spatial encoding
- **CNN** — needs 2D/3D tensor layout `[C, H, W]` instead of flat

If your architecture can parse a flat vector, use `FlatObservationBuilder`. If it needs fundamentally different input structure, create a new builder in `builders/` and a corresponding sharding pipeline.

## Adding a New Architecture

1. Create `architectures/myarch/config.py`:

```python
from typing import Literal
from alpharat.nn.training.base import BaseModelConfig, BaseOptimConfig

class MyArchModelConfig(BaseModelConfig):
    architecture: Literal["myarch"] = "myarch"  # Discriminator
    # Architecture params...

    def set_data_dimensions(self, width: int, height: int) -> None:
        # Set any dimension fields needed by build_model()
        pass

    def build_model(self) -> TrainableModel:
        from alpharat.nn.models.myarch import MyArchMLP
        return MyArchMLP(...)

    def build_loss_fn(self) -> LossFunction:
        from alpharat.nn.architectures.myarch.loss import compute_myarch_losses
        return compute_myarch_losses

    def build_augmentation(self) -> AugmentationStrategy:
        return PlayerSwapStrategy(p_swap=self.p_augment)

    def build_observation_builder(self, width: int, height: int) -> ObservationBuilder:
        # Return builder for this architecture's input format.
        # Use FlatObservationBuilder if your model can parse flat vectors.
        # Create a new builder if you need different structure (graphs, 2D, etc).
        from alpharat.nn.builders.flat import FlatObservationBuilder
        return FlatObservationBuilder(width=width, height=height)

class MyArchOptimConfig(BaseOptimConfig):
    architecture: Literal["myarch"] = "myarch"
    # Loss weights...
```

2. Create `architectures/myarch/loss.py`:

```python
def compute_myarch_losses(
    model_output: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    config: MyArchOptimConfig,
) -> dict[str, torch.Tensor]:
    # Compute losses, return dict with LossKey.TOTAL
    ...
```

3. Create `models/myarch.py` implementing the model.

4. Register in `nn/config.py`:

```python
ModelConfig = Annotated[
    MLPModelConfig | SymmetricModelConfig | LocalValueModelConfig | MyArchModelConfig,
    Discriminator("architecture"),
]
# Same for OptimConfig
```

## Reproducibility

Checkpoints save everything needed to reproduce training:
- Model state dict
- Optimizer state dict
- Full config (model + optim + data)
- Data dimensions (width, height)
- Best validation loss

Resume training:
```yaml
resume_from: checkpoints/train_20260101_120000/best_model.pt
```

## Key Protocols

Models must implement `TrainableModel`:
- `forward(x, **kwargs) -> dict[str, Tensor]` — returns logits + value
- `predict(x, **kwargs) -> dict[str, Tensor]` — returns probs + value

Loss functions must return `dict[str, Tensor]` with at least `LossKey.TOTAL`.

Augmentation strategies must implement `__call__(batch, width, height) -> batch`.
