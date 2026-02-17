"""Tests for batch metadata."""

from __future__ import annotations

import json
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from alpharat.config.game import GameConfig
from alpharat.data.batch import (
    BatchMetadata,
    BatchMetadataError,
    create_batch,
    get_batch_stats,
    load_batch_metadata,
    save_batch_metadata,
)
from alpharat.mcts import PythonMCTSConfig


class TestMCTSConfig:
    """Tests for MCTS config."""

    def test_decoupled_puct_from_dict(self) -> None:
        """PythonMCTSConfig parses from dict."""
        data = {
            "simulations": 400,
            "c_puct": 2.0,
        }
        config = PythonMCTSConfig.model_validate(data)

        assert config.simulations == 400
        assert config.c_puct == 2.0

    def test_decoupled_puct_defaults(self) -> None:
        """PythonMCTSConfig uses default c_puct."""
        config = PythonMCTSConfig(simulations=100)

        assert config.c_puct == 1.5

    def test_batch_metadata_parses_config(self) -> None:
        """BatchMetadata parses decoupled_puct config."""
        data = {
            "batch_id": "test-id",
            "created_at": "2024-01-01T00:00:00Z",
            "checkpoint_path": "/path/to/model.pt",
            "mcts_config": {"backend": "python", "simulations": 400, "c_puct": 2.5},
            "game": {"width": 10, "height": 10, "max_turns": 200, "cheese_count": 21},
        }
        metadata = BatchMetadata.model_validate(data)

        assert isinstance(metadata.mcts_config, PythonMCTSConfig)
        assert metadata.mcts_config.c_puct == 2.5


class TestBatchMetadata:
    """Tests for BatchMetadata model."""

    def test_checkpoint_path_none(self) -> None:
        """BatchMetadata accepts None checkpoint for random policy."""
        metadata = BatchMetadata(
            batch_id="test",
            created_at=datetime.now(UTC),
            checkpoint_path=None,
            mcts_config=PythonMCTSConfig(simulations=100),
            game=GameConfig(width=10, height=10, max_turns=200, cheese_count=21),
        )

        assert metadata.checkpoint_path is None

    def test_checkpoint_path_string(self) -> None:
        """BatchMetadata accepts string checkpoint path."""
        metadata = BatchMetadata(
            batch_id="test",
            created_at=datetime.now(UTC),
            checkpoint_path="/models/checkpoint_100.pt",
            mcts_config=PythonMCTSConfig(simulations=100),
            game=GameConfig(width=10, height=10, max_turns=200, cheese_count=21),
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
                mcts_config=PythonMCTSConfig(simulations=100),
                game=GameConfig(width=10, height=10, max_turns=200, cheese_count=21),
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
                mcts_config=PythonMCTSConfig(simulations=400, c_puct=2.0),
                game=GameConfig(width=15, height=12, max_turns=300, cheese_count=21),
            )

            metadata = load_batch_metadata(batch_dir)

            assert metadata.checkpoint_path == "/test/checkpoint.pt"
            assert isinstance(metadata.mcts_config, PythonMCTSConfig)
            assert metadata.mcts_config.simulations == 400
            assert metadata.mcts_config.c_puct == 2.0
            assert metadata.game.width == 15
            assert metadata.game.height == 12
            assert metadata.game.max_turns == 300

    def test_batch_id_is_uuid(self) -> None:
        """create_batch generates UUID batch_id."""
        with tempfile.TemporaryDirectory() as tmpdir:
            batch_dir = create_batch(
                parent_dir=tmpdir,
                checkpoint_path=None,
                mcts_config=PythonMCTSConfig(simulations=100),
                game=GameConfig(width=10, height=10, max_turns=200, cheese_count=21),
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
                mcts_config=PythonMCTSConfig(simulations=100),
                game=GameConfig(width=10, height=10, max_turns=200, cheese_count=21),
            )

            assert batch_dir.parent == Path(tmpdir)
            metadata = load_batch_metadata(batch_dir)
            assert batch_dir.name == metadata.batch_id


class TestSaveLoadRoundtrip:
    """Tests for save/load metadata roundtrip."""

    def test_roundtrip_decoupled_puct(self) -> None:
        """Metadata with PythonMCTSConfig survives roundtrip."""
        with tempfile.TemporaryDirectory() as tmpdir:
            batch_dir = Path(tmpdir) / "test-batch"
            batch_dir.mkdir()

            original = BatchMetadata(
                batch_id="test-batch",
                created_at=datetime(2024, 6, 15, 12, 30, 0, tzinfo=UTC),
                checkpoint_path="/models/best.pt",
                mcts_config=PythonMCTSConfig(simulations=400, c_puct=2.5),
                game=GameConfig(width=10, height=10, max_turns=200, cheese_count=21),
            )

            save_batch_metadata(batch_dir, original)
            loaded = load_batch_metadata(batch_dir)

            assert loaded.checkpoint_path == "/models/best.pt"
            assert isinstance(loaded.mcts_config, PythonMCTSConfig)
            assert loaded.mcts_config.simulations == 400
            assert loaded.mcts_config.c_puct == 2.5
            assert loaded.game.width == 10


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


class TestBatchMetadataError:
    """Tests for actionable errors from load_batch_metadata."""

    def _write_metadata(self, batch_dir: Path, data: dict[str, object]) -> None:
        (batch_dir / "metadata.json").write_text(json.dumps(data))

    # All fields for each config, so _field_diff reports "OK" when section is clean.
    _VALID_MCTS = {
        "simulations": 100,
        "gamma": 1.0,
        "c_puct": 1.5,
        "force_k": 2.0,
        "fpu_reduction": 0.2,
    }
    _VALID_GAME = {
        "width": 5,
        "height": 5,
        "max_turns": 30,
        "cheese_count": 5,
        "wall_density": None,
        "mud_density": None,
        "symmetric": True,
    }

    def test_extra_mcts_config_field(self) -> None:
        """Extra field in mcts_config reports drift and marks game OK."""
        with tempfile.TemporaryDirectory() as tmpdir:
            batch_dir = Path(tmpdir)
            mcts = {**self._VALID_MCTS, "dirichlet_alpha": 0.3}
            self._write_metadata(
                batch_dir,
                {
                    "batch_id": "test",
                    "created_at": "2024-01-01T00:00:00Z",
                    "checkpoint_path": None,
                    "mcts_config": mcts,
                    "game": self._VALID_GAME,
                },
            )

            with pytest.raises(BatchMetadataError, match="dirichlet_alpha") as exc_info:
                load_batch_metadata(batch_dir)
            assert "game: OK" in str(exc_info.value)

    def test_extra_game_field(self) -> None:
        """Extra field in game reports drift and marks mcts_config OK."""
        with tempfile.TemporaryDirectory() as tmpdir:
            batch_dir = Path(tmpdir)
            game = {**self._VALID_GAME, "fog_of_war": True}
            self._write_metadata(
                batch_dir,
                {
                    "batch_id": "test",
                    "created_at": "2024-01-01T00:00:00Z",
                    "checkpoint_path": None,
                    "mcts_config": self._VALID_MCTS,
                    "game": game,
                },
            )

            with pytest.raises(BatchMetadataError, match="fog_of_war") as exc_info:
                load_batch_metadata(batch_dir)
            assert "mcts_config: OK" in str(exc_info.value)

    def test_missing_and_extra_in_one_section(self) -> None:
        """Swapped field (removed + added) reports both missing and extra, game OK."""
        with tempfile.TemporaryDirectory() as tmpdir:
            batch_dir = Path(tmpdir)
            # Remove simulations, add unknown — the extra triggers ValidationError,
            # and _field_diff reports both missing and extra.
            mcts = {k: v for k, v in self._VALID_MCTS.items() if k != "simulations"}
            mcts["exploration_weight"] = 0.5
            self._write_metadata(
                batch_dir,
                {
                    "batch_id": "test",
                    "created_at": "2024-01-01T00:00:00Z",
                    "checkpoint_path": None,
                    "mcts_config": mcts,
                    "game": self._VALID_GAME,
                },
            )

            with pytest.raises(BatchMetadataError) as exc_info:
                load_batch_metadata(batch_dir)
            msg = str(exc_info.value)
            assert "missing fields: simulations" in msg
            assert "extra fields: exploration_weight" in msg
            assert "game: OK" in msg

    def test_extra_and_missing_both_sections(self) -> None:
        """Drift in both sections reports both."""
        with tempfile.TemporaryDirectory() as tmpdir:
            batch_dir = Path(tmpdir)
            mcts = {**self._VALID_MCTS, "dirichlet_alpha": 0.3}
            game = {**self._VALID_GAME, "fog_of_war": True}
            self._write_metadata(
                batch_dir,
                {
                    "batch_id": "test",
                    "created_at": "2024-01-01T00:00:00Z",
                    "checkpoint_path": None,
                    "mcts_config": mcts,
                    "game": game,
                },
            )

            with pytest.raises(BatchMetadataError) as exc_info:
                load_batch_metadata(batch_dir)
            msg = str(exc_info.value)
            assert "dirichlet_alpha" in msg
            assert "fog_of_war" in msg

    def test_value_validation_chains_original_error(self) -> None:
        """Value errors (valid keys, invalid values) chain the original ValidationError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            batch_dir = Path(tmpdir)
            game = {**self._VALID_GAME, "cheese_count": 9999}  # fails check_cheese_fits
            self._write_metadata(
                batch_dir,
                {
                    "batch_id": "test",
                    "created_at": "2024-01-01T00:00:00Z",
                    "checkpoint_path": None,
                    "mcts_config": self._VALID_MCTS,
                    "game": game,
                },
            )

            with pytest.raises(BatchMetadataError) as exc_info:
                load_batch_metadata(batch_dir)
            assert exc_info.value.__cause__ is not None
            assert isinstance(exc_info.value.__cause__, ValidationError)
