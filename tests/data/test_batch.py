"""Tests for batch metadata."""

from __future__ import annotations

import tempfile
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from alpharat.data.batch import (
    BatchMetadata,
    GameParams,
    create_batch,
    get_batch_stats,
    load_batch_metadata,
    save_batch_metadata,
)
from alpharat.mcts.config import DecoupledPUCTConfig, PriorSamplingConfig


class TestMCTSConfig:
    """Tests for MCTS config discriminated union."""

    def test_prior_sampling_from_dict(self) -> None:
        """PriorSamplingConfig parses from dict."""
        data = {"variant": "prior_sampling", "simulations": 800, "gamma": 0.95}
        config = PriorSamplingConfig.model_validate(data)

        assert config.variant == "prior_sampling"
        assert config.simulations == 800
        assert config.gamma == 0.95

    def test_prior_sampling_defaults(self) -> None:
        """PriorSamplingConfig uses default gamma."""
        config = PriorSamplingConfig(simulations=100)

        assert config.gamma == 0.99

    def test_decoupled_puct_from_dict(self) -> None:
        """DecoupledPUCTConfig parses from dict."""
        data = {
            "variant": "decoupled_puct",
            "simulations": 400,
            "gamma": 0.98,
            "c_puct": 2.0,
        }
        config = DecoupledPUCTConfig.model_validate(data)

        assert config.variant == "decoupled_puct"
        assert config.simulations == 400
        assert config.gamma == 0.98
        assert config.c_puct == 2.0

    def test_decoupled_puct_defaults(self) -> None:
        """DecoupledPUCTConfig uses default gamma and c_puct."""
        config = DecoupledPUCTConfig(simulations=100)

        assert config.gamma == 0.99
        assert config.c_puct == 1.5

    def test_discriminated_union_prior_sampling(self) -> None:
        """BatchMetadata parses prior_sampling via discriminator."""
        data = {
            "batch_id": "test-id",
            "created_at": "2024-01-01T00:00:00Z",
            "checkpoint_path": None,
            "mcts_config": {"variant": "prior_sampling", "simulations": 800},
            "game_params": {"width": 10, "height": 10, "max_turns": 200},
        }
        metadata = BatchMetadata.model_validate(data)

        assert isinstance(metadata.mcts_config, PriorSamplingConfig)
        assert metadata.mcts_config.simulations == 800

    def test_discriminated_union_decoupled_puct(self) -> None:
        """BatchMetadata parses decoupled_puct via discriminator."""
        data = {
            "batch_id": "test-id",
            "created_at": "2024-01-01T00:00:00Z",
            "checkpoint_path": "/path/to/model.pt",
            "mcts_config": {"variant": "decoupled_puct", "simulations": 400, "c_puct": 2.5},
            "game_params": {"width": 10, "height": 10, "max_turns": 200},
        }
        metadata = BatchMetadata.model_validate(data)

        assert isinstance(metadata.mcts_config, DecoupledPUCTConfig)
        assert metadata.mcts_config.c_puct == 2.5


class TestBatchMetadata:
    """Tests for BatchMetadata model."""

    def test_checkpoint_path_none(self) -> None:
        """BatchMetadata accepts None checkpoint for random policy."""
        metadata = BatchMetadata(
            batch_id="test",
            created_at=datetime.now(UTC),
            checkpoint_path=None,
            mcts_config=PriorSamplingConfig(simulations=100),
            game_params=GameParams(width=10, height=10, max_turns=200),
        )

        assert metadata.checkpoint_path is None

    def test_checkpoint_path_string(self) -> None:
        """BatchMetadata accepts string checkpoint path."""
        metadata = BatchMetadata(
            batch_id="test",
            created_at=datetime.now(UTC),
            checkpoint_path="/models/checkpoint_100.pt",
            mcts_config=PriorSamplingConfig(simulations=100),
            game_params=GameParams(width=10, height=10, max_turns=200),
        )

        assert metadata.checkpoint_path == "/models/checkpoint_100.pt"


class TestCreateBatch:
    """Tests for create_batch function."""

    def test_creates_directory_structure(self) -> None:
        """create_batch creates batch dir and games subdir."""
        with tempfile.TemporaryDirectory() as tmpdir:
            batch_dir = create_batch(
                parent_dir=tmpdir,
                checkpoint_path=None,
                mcts_config=PriorSamplingConfig(simulations=100),
                game_params=GameParams(width=10, height=10, max_turns=200),
            )

            assert batch_dir.exists()
            assert (batch_dir / "games").exists()
            assert (batch_dir / "metadata.json").exists()

    def test_creates_valid_metadata(self) -> None:
        """create_batch writes valid metadata file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            batch_dir = create_batch(
                parent_dir=tmpdir,
                checkpoint_path="/test/checkpoint.pt",
                mcts_config=DecoupledPUCTConfig(simulations=400, c_puct=2.0),
                game_params=GameParams(width=15, height=12, max_turns=300),
            )

            metadata = load_batch_metadata(batch_dir)

            assert metadata.checkpoint_path == "/test/checkpoint.pt"
            assert isinstance(metadata.mcts_config, DecoupledPUCTConfig)
            assert metadata.mcts_config.simulations == 400
            assert metadata.mcts_config.c_puct == 2.0
            assert metadata.game_params.width == 15
            assert metadata.game_params.height == 12
            assert metadata.game_params.max_turns == 300

    def test_batch_id_is_uuid(self) -> None:
        """create_batch generates UUID batch_id."""
        with tempfile.TemporaryDirectory() as tmpdir:
            batch_dir = create_batch(
                parent_dir=tmpdir,
                checkpoint_path=None,
                mcts_config=PriorSamplingConfig(simulations=100),
                game_params=GameParams(width=10, height=10, max_turns=200),
            )

            metadata = load_batch_metadata(batch_dir)

            # UUID format: 8-4-4-4-12 hex chars
            parts = metadata.batch_id.split("-")
            assert len(parts) == 5
            assert len(parts[0]) == 8
            assert len(parts[1]) == 4
            assert len(parts[2]) == 4
            assert len(parts[3]) == 4
            assert len(parts[4]) == 12

    def test_returns_batch_dir_path(self) -> None:
        """create_batch returns path to batch directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            batch_dir = create_batch(
                parent_dir=tmpdir,
                checkpoint_path=None,
                mcts_config=PriorSamplingConfig(simulations=100),
                game_params=GameParams(width=10, height=10, max_turns=200),
            )

            assert batch_dir.parent == Path(tmpdir)
            metadata = load_batch_metadata(batch_dir)
            assert batch_dir.name == metadata.batch_id


class TestSaveLoadRoundtrip:
    """Tests for save/load metadata roundtrip."""

    def test_roundtrip_prior_sampling(self) -> None:
        """Metadata with PriorSamplingConfig survives roundtrip."""
        with tempfile.TemporaryDirectory() as tmpdir:
            batch_dir = Path(tmpdir) / "test-batch"
            batch_dir.mkdir()

            original = BatchMetadata(
                batch_id="test-batch",
                created_at=datetime(2024, 6, 15, 12, 30, 0, tzinfo=UTC),
                checkpoint_path=None,
                mcts_config=PriorSamplingConfig(simulations=800, gamma=0.95),
                game_params=GameParams(width=20, height=15, max_turns=400),
            )

            save_batch_metadata(batch_dir, original)
            loaded = load_batch_metadata(batch_dir)

            assert loaded.batch_id == original.batch_id
            assert loaded.created_at == original.created_at
            assert loaded.checkpoint_path == original.checkpoint_path
            assert isinstance(loaded.mcts_config, PriorSamplingConfig)
            assert loaded.mcts_config.simulations == 800
            assert loaded.mcts_config.gamma == 0.95
            assert loaded.game_params.width == 20

    def test_roundtrip_decoupled_puct(self) -> None:
        """Metadata with DecoupledPUCTConfig survives roundtrip."""
        with tempfile.TemporaryDirectory() as tmpdir:
            batch_dir = Path(tmpdir) / "test-batch"
            batch_dir.mkdir()

            original = BatchMetadata(
                batch_id="test-batch",
                created_at=datetime(2024, 6, 15, 12, 30, 0, tzinfo=UTC),
                checkpoint_path="/models/best.pt",
                mcts_config=DecoupledPUCTConfig(simulations=400, gamma=0.98, c_puct=2.5),
                game_params=GameParams(width=10, height=10, max_turns=200),
            )

            save_batch_metadata(batch_dir, original)
            loaded = load_batch_metadata(batch_dir)

            assert loaded.checkpoint_path == "/models/best.pt"
            assert isinstance(loaded.mcts_config, DecoupledPUCTConfig)
            assert loaded.mcts_config.c_puct == 2.5


class TestGetBatchStats:
    """Tests for get_batch_stats function."""

    def test_empty_games_dir(self) -> None:
        """get_batch_stats returns zeros for empty games dir."""
        with tempfile.TemporaryDirectory() as tmpdir:
            batch_dir = Path(tmpdir)
            (batch_dir / "games").mkdir()

            stats = get_batch_stats(batch_dir)

            assert stats.game_count == 0
            assert stats.total_positions == 0

    def test_counts_npz_files(self) -> None:
        """get_batch_stats counts game files correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            batch_dir = Path(tmpdir)
            games_dir = batch_dir / "games"
            games_dir.mkdir()

            # Create fake game files with num_positions
            for i, num_pos in enumerate([10, 25, 15]):
                np.savez(games_dir / f"game_{i}.npz", num_positions=np.int32(num_pos))

            stats = get_batch_stats(batch_dir)

            assert stats.game_count == 3
            assert stats.total_positions == 50

    def test_ignores_non_npz_files(self) -> None:
        """get_batch_stats ignores non-.npz files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            batch_dir = Path(tmpdir)
            games_dir = batch_dir / "games"
            games_dir.mkdir()

            np.savez(games_dir / "game.npz", num_positions=np.int32(20))
            (games_dir / "readme.txt").write_text("ignore me")
            (games_dir / "data.json").write_text("{}")

            stats = get_batch_stats(batch_dir)

            assert stats.game_count == 1
            assert stats.total_positions == 20
