"""Game data recording and persistence for MCTS training."""

from alpharat.config.game import GameConfig
from alpharat.data.batch import (
    BatchMetadata,
    BatchMetadataError,
    BatchStats,
    create_batch,
    get_batch_stats,
    load_batch_metadata,
    save_batch_metadata,
)
from alpharat.data.loader import load_game_data
from alpharat.data.maze import build_maze_array
from alpharat.data.recorder import GameRecorder
from alpharat.data.sampling import (
    GameStats,
    SamplingConfig,
    SamplingMetrics,
    SamplingParams,
    run_sampling,
)
from alpharat.data.sharding import (
    TrainingSetManifest,
    load_training_set_manifest,
    prepare_training_set,
)

# Backward compatibility alias (deprecated, use GameConfig)
GameParams = GameConfig

__all__ = [
    "BatchMetadata",
    "BatchMetadataError",
    "BatchStats",
    "GameConfig",
    "GameParams",  # Deprecated alias for GameConfig
    "GameRecorder",
    "GameStats",
    "SamplingConfig",
    "SamplingMetrics",
    "SamplingParams",
    "TrainingSetManifest",
    "build_maze_array",
    "create_batch",
    "get_batch_stats",
    "load_batch_metadata",
    "load_game_data",
    "load_training_set_manifest",
    "prepare_training_set",
    "run_sampling",
    "save_batch_metadata",
]
