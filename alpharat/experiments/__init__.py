"""Experiment management for alpharat.

This module provides the ExperimentManager class for managing experiment artifacts:
- Batches: Raw game recordings from sampling
- Shards: Processed train/val splits
- Runs: Training runs with checkpoints
- Benchmarks: Tournament and benchmark results

Usage:
    from alpharat.experiments import ExperimentManager

    exp = ExperimentManager()
    batch_dir = exp.create_batch(group="uniform_5x5", mcts_config=..., game=...)

The experiments folder (not in git) has the structure:
    experiments/
    ├── manifest.yaml       # Central index
    ├── batches/            # Raw game data
    ├── shards/             # Processed training data
    ├── runs/               # Training runs
    └── benchmarks/         # Evaluation results
"""

from alpharat.experiments.manager import ExperimentManager
from alpharat.experiments.schema import (
    BatchEntry,
    BenchmarkEntry,
    Manifest,
    RunEntry,
    ShardEntry,
)

__all__ = [
    "BatchEntry",
    "BenchmarkEntry",
    "ExperimentManager",
    "Manifest",
    "RunEntry",
    "ShardEntry",
]
