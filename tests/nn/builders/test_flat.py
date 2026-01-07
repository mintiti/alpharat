"""Tests for FlatObservationBuilder."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from alpharat.data.sharding import prepare_training_set
from alpharat.data.types import CheeseOutcome
from alpharat.nn.builders.flat import (
    MAX_MUD_COST,
    MAX_MUD_TURNS,
    MAX_SCORE,
    FlatDataset,
    FlatObservationBuilder,
)
from alpharat.nn.types import ObservationInput


class TestFlatObservationBuilder:
    """Tests for FlatObservationBuilder."""

    def test_obs_shape_5x5(self) -> None:
        """Test observation shape for 5x5 maze."""
        builder = FlatObservationBuilder(width=5, height=5)
        # 5*5*4 (maze) + 5*5*3 (positions + cheese) + 6 (scalars)
        # = 100 + 75 + 6 = 181
        assert builder.obs_shape == (181,)

    def test_obs_shape_3x3(self) -> None:
        """Test observation shape for 3x3 maze."""
        builder = FlatObservationBuilder(width=3, height=3)
        # 3*3*4 + 3*3*3 + 6 = 36 + 27 + 6 = 69
        assert builder.obs_shape == (69,)

    def test_obs_shape_rectangular(self) -> None:
        """Test observation shape for non-square maze."""
        builder = FlatObservationBuilder(width=7, height=4)
        # 7*4*4 + 7*4*3 + 6 = 112 + 84 + 6 = 202
        assert builder.obs_shape == (202,)

    def test_version(self) -> None:
        """Test version string."""
        builder = FlatObservationBuilder(width=5, height=5)
        assert builder.version == "flat_v2"

    def test_build_output_shape(self) -> None:
        """Test that build() returns correct shape."""
        builder = FlatObservationBuilder(width=5, height=5)
        input_data = _make_observation_input(width=5, height=5)

        obs = builder.build(input_data)

        assert obs.shape == (181,)
        assert obs.dtype == np.float32

    def test_build_maze_walls_encoded_as_minus_one(self) -> None:
        """Test that walls in maze are encoded as -1."""
        builder = FlatObservationBuilder(width=3, height=3)

        # Create maze with a wall (value -1) at position [0,0,0]
        maze = np.ones((3, 3, 4), dtype=np.int8)
        maze[0, 0, 0] = -1  # Wall in UP direction from (0,0)

        input_data = _make_observation_input(width=3, height=3, maze=maze)
        obs = builder.build(input_data)

        # Maze is first in concatenation, walls should stay -1
        assert obs[0] == -1.0

    def test_build_maze_costs_normalized(self) -> None:
        """Test that mud costs are normalized by MAX_MUD_COST."""
        builder = FlatObservationBuilder(width=3, height=3)

        maze = np.ones((3, 3, 4), dtype=np.int8)
        maze[0, 0, 0] = 5  # Cost 5 in UP direction

        input_data = _make_observation_input(width=3, height=3, maze=maze)
        obs = builder.build(input_data)

        assert obs[0] == pytest.approx(5.0 / MAX_MUD_COST)

    def test_build_player_positions_one_hot(self) -> None:
        """Test that player positions are one-hot encoded."""
        builder = FlatObservationBuilder(width=3, height=3)

        input_data = _make_observation_input(
            width=3,
            height=3,
            p1_pos=(1, 2),  # x=1, y=2
            p2_pos=(0, 0),  # x=0, y=0
        )
        obs = builder.build(input_data)

        # Maze takes 3*3*4 = 36 elements
        # P1 position starts at index 36, shape (3,3) flattened
        # Position (1,2) means row=2, col=1 -> index 2*3 + 1 = 7
        p1_start = 36
        p1_section = obs[p1_start : p1_start + 9]
        assert p1_section[7] == 1.0
        assert p1_section.sum() == 1.0

        # P2 position starts at index 45
        # Position (0,0) means row=0, col=0 -> index 0
        p2_start = 45
        p2_section = obs[p2_start : p2_start + 9]
        assert p2_section[0] == 1.0
        assert p2_section.sum() == 1.0

    def test_build_cheese_mask(self) -> None:
        """Test that cheese mask is preserved."""
        builder = FlatObservationBuilder(width=3, height=3)

        cheese_mask = np.zeros((3, 3), dtype=bool)
        cheese_mask[1, 1] = True  # Cheese at (1,1)
        cheese_mask[2, 0] = True  # Cheese at (0,2) - col=0, row=2

        input_data = _make_observation_input(width=3, height=3, cheese_mask=cheese_mask)
        obs = builder.build(input_data)

        # Cheese starts at index 36 + 9 + 9 = 54
        cheese_start = 54
        cheese_section = obs[cheese_start : cheese_start + 9]
        assert cheese_section[4] == 1.0  # row=1, col=1 -> idx 4
        assert cheese_section[6] == 1.0  # row=2, col=0 -> idx 6
        assert cheese_section.sum() == 2.0

    def test_build_score_diff(self) -> None:
        """Test score difference encoding."""
        builder = FlatObservationBuilder(width=3, height=3)

        input_data = _make_observation_input(width=3, height=3, p1_score=3.5, p2_score=1.0)
        obs = builder.build(input_data)

        # Scalars start after spatial features: 36 + 9 + 9 + 9 = 63
        # Score diff is first scalar
        assert obs[63] == pytest.approx(2.5)

    def test_build_progress(self) -> None:
        """Test game progress encoding."""
        builder = FlatObservationBuilder(width=3, height=3)

        input_data = _make_observation_input(width=3, height=3, turn=15, max_turns=30)
        obs = builder.build(input_data)

        # Progress is second scalar (index 64)
        assert obs[64] == pytest.approx(0.5)

    def test_build_mud_turns_normalized(self) -> None:
        """Test mud turns are normalized."""
        builder = FlatObservationBuilder(width=3, height=3)

        input_data = _make_observation_input(width=3, height=3, p1_mud=5, p2_mud=3)
        obs = builder.build(input_data)

        # P1 mud is third scalar (index 65)
        # P2 mud is fourth scalar (index 66)
        assert obs[65] == pytest.approx(5.0 / MAX_MUD_TURNS)
        assert obs[66] == pytest.approx(3.0 / MAX_MUD_TURNS)

    def test_build_scores_normalized(self) -> None:
        """Test player scores are normalized."""
        builder = FlatObservationBuilder(width=3, height=3)

        input_data = _make_observation_input(width=3, height=3, p1_score=7.0, p2_score=3.0)
        obs = builder.build(input_data)

        # P1 score is fifth scalar (index 67)
        # P2 score is sixth scalar (index 68)
        assert obs[67] == pytest.approx(7.0 / MAX_SCORE)
        assert obs[68] == pytest.approx(3.0 / MAX_SCORE)


class TestFlatObservationBuilderSerialization:
    """Tests for save/load functionality."""

    def test_save_to_arrays(self) -> None:
        """Test saving observations to arrays."""
        builder = FlatObservationBuilder(width=3, height=3)

        obs1 = np.ones(69, dtype=np.float32)
        obs2 = np.ones(69, dtype=np.float32) * 2

        arrays = builder.save_to_arrays([obs1, obs2])

        assert "observations" in arrays
        assert arrays["observations"].shape == (2, 69)
        assert arrays["observations"][0, 0] == 1.0
        assert arrays["observations"][1, 0] == 2.0

    def test_load_from_arrays(self) -> None:
        """Test loading observation from arrays."""
        builder = FlatObservationBuilder(width=3, height=3)

        obs1 = np.ones(69, dtype=np.float32)
        obs2 = np.ones(69, dtype=np.float32) * 2
        arrays = builder.save_to_arrays([obs1, obs2])

        loaded = builder.load_from_arrays(arrays, idx=1)

        assert loaded.shape == (69,)
        assert loaded[0] == 2.0

    def test_roundtrip(self) -> None:
        """Test save -> load roundtrip preserves data."""
        builder = FlatObservationBuilder(width=5, height=5)
        input_data = _make_observation_input(width=5, height=5)

        original = builder.build(input_data)
        arrays = builder.save_to_arrays([original])
        loaded = builder.load_from_arrays(arrays, idx=0)

        np.testing.assert_array_equal(original, loaded)


def _make_observation_input(
    *,
    width: int,
    height: int,
    maze: np.ndarray | None = None,
    p1_pos: tuple[int, int] = (0, 0),
    p2_pos: tuple[int, int] = (1, 1),
    cheese_mask: np.ndarray | None = None,
    p1_score: float = 0.0,
    p2_score: float = 0.0,
    turn: int = 0,
    max_turns: int = 30,
    p1_mud: int = 0,
    p2_mud: int = 0,
) -> ObservationInput:
    """Create ObservationInput with defaults for testing."""
    if maze is None:
        maze = np.ones((height, width, 4), dtype=np.int8)
    if cheese_mask is None:
        cheese_mask = np.zeros((height, width), dtype=bool)

    return ObservationInput(
        maze=maze,
        p1_pos=p1_pos,
        p2_pos=p2_pos,
        cheese_mask=cheese_mask,
        p1_score=p1_score,
        p2_score=p2_score,
        turn=turn,
        max_turns=max_turns,
        p1_mud=p1_mud,
        p2_mud=p2_mud,
        width=width,
        height=height,
    )


# =============================================================================
# FlatDataset tests
# =============================================================================


def _create_game_npz(
    path: Path,
    *,
    width: int = 5,
    height: int = 5,
    num_positions: int = 3,
    final_p1_score: float = 2.0,
    final_p2_score: float = 1.0,
) -> None:
    """Create a minimal valid game npz file for testing."""
    n = num_positions

    maze = np.ones((height, width, 4), dtype=np.int8)
    maze[:, 0, 3] = -1
    maze[:, width - 1, 1] = -1
    maze[0, :, 0] = -1
    maze[height - 1, :, 2] = -1

    initial_cheese = np.zeros((height, width), dtype=bool)
    initial_cheese[2, 2] = True

    cheese_outcomes = np.full((height, width), CheeseOutcome.UNCOLLECTED, dtype=np.int8)
    cheese_outcomes[2, 2] = CheeseOutcome.P1_WIN

    p1_pos = np.zeros((n, 2), dtype=np.int8)
    p1_pos[:, 0] = 1
    p1_pos[:, 1] = 1

    p2_pos = np.zeros((n, 2), dtype=np.int8)
    p2_pos[:, 0] = 3
    p2_pos[:, 1] = 3

    p1_score = np.linspace(0, final_p1_score, n, dtype=np.float32)
    p2_score = np.linspace(0, final_p2_score, n, dtype=np.float32)

    p1_mud = np.zeros(n, dtype=np.int8)
    p2_mud = np.zeros(n, dtype=np.int8)

    cheese_mask = np.zeros((n, height, width), dtype=bool)
    cheese_mask[:, 2, 2] = True

    turn = np.arange(n, dtype=np.int16)

    payout_matrix = np.zeros((n, 5, 5), dtype=np.float32)
    visit_counts = np.ones((n, 5, 5), dtype=np.int32) * 10

    prior_p1 = np.ones((n, 5), dtype=np.float32) / 5
    prior_p2 = np.ones((n, 5), dtype=np.float32) / 5
    policy_p1 = np.ones((n, 5), dtype=np.float32) / 5
    policy_p2 = np.ones((n, 5), dtype=np.float32) / 5
    action_p1 = np.zeros(n, dtype=np.int8)
    action_p2 = np.zeros(n, dtype=np.int8)

    np.savez_compressed(
        path,
        maze=maze,
        initial_cheese=initial_cheese,
        cheese_outcomes=cheese_outcomes,
        max_turns=np.array(100, dtype=np.int16),
        result=np.array(1 if final_p1_score > final_p2_score else 0, dtype=np.int8),
        final_p1_score=np.array(final_p1_score, dtype=np.float32),
        final_p2_score=np.array(final_p2_score, dtype=np.float32),
        num_positions=np.array(n, dtype=np.int32),
        p1_pos=p1_pos,
        p2_pos=p2_pos,
        p1_score=p1_score,
        p2_score=p2_score,
        p1_mud=p1_mud,
        p2_mud=p2_mud,
        cheese_mask=cheese_mask,
        turn=turn,
        payout_matrix=payout_matrix,
        visit_counts=visit_counts,
        prior_p1=prior_p1,
        prior_p2=prior_p2,
        policy_p1=policy_p1,
        policy_p2=policy_p2,
        action_p1=action_p1,
        action_p2=action_p2,
    )


def _create_training_set(tmp_path: Path, num_games: int = 2) -> Path:
    """Create a training set for testing."""
    from alpharat.data.sharding import prepare_training_set

    batch_dir = tmp_path / "batch1"
    games_dir = batch_dir / "games"
    games_dir.mkdir(parents=True)

    for i in range(num_games):
        game_path = games_dir / f"game_{i}.npz"
        _create_game_npz(game_path, num_positions=3)

    output_dir = tmp_path / "training_sets"
    output_dir.mkdir()

    builder = FlatObservationBuilder(width=5, height=5)
    return prepare_training_set(
        batch_dirs=[batch_dir],
        output_dir=output_dir,
        builder=builder,
        positions_per_shard=100,
        seed=42,
    )


class TestFlatDataset:
    """Tests for FlatDataset."""

    def test_len_matches_total_positions(self) -> None:
        """len() should return total positions from manifest."""
        with tempfile.TemporaryDirectory() as tmpdir:
            training_set_dir = _create_training_set(Path(tmpdir), num_games=2)
            dataset = FlatDataset(training_set_dir)

            assert len(dataset) == 6  # 2 games × 3 positions

    def test_getitem_returns_correct_keys(self) -> None:
        """__getitem__ should return dict with obs, policies, values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            training_set_dir = _create_training_set(Path(tmpdir))
            dataset = FlatDataset(training_set_dir)

            item = dataset[0]

            assert "observation" in item
            assert "policy_p1" in item
            assert "policy_p2" in item
            assert "p1_value" in item
            assert "p2_value" in item

    def test_getitem_observation_shape(self) -> None:
        """Observation should have correct shape."""
        with tempfile.TemporaryDirectory() as tmpdir:
            training_set_dir = _create_training_set(Path(tmpdir))
            dataset = FlatDataset(training_set_dir)

            item = dataset[0]

            assert item["observation"].shape == (181,)  # 5×5 flat obs

    def test_getitem_policy_shapes(self) -> None:
        """Policies should have shape (5,)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            training_set_dir = _create_training_set(Path(tmpdir))
            dataset = FlatDataset(training_set_dir)

            item = dataset[0]

            assert item["policy_p1"].shape == (5,)
            assert item["policy_p2"].shape == (5,)

    def test_getitem_value_shapes(self) -> None:
        """Values should be 1D arrays with single element."""
        with tempfile.TemporaryDirectory() as tmpdir:
            training_set_dir = _create_training_set(Path(tmpdir))
            dataset = FlatDataset(training_set_dir)

            item = dataset[0]

            assert item["p1_value"].shape == (1,)
            assert item["p2_value"].shape == (1,)

    def test_getitem_dtypes(self) -> None:
        """All arrays should be float32."""
        with tempfile.TemporaryDirectory() as tmpdir:
            training_set_dir = _create_training_set(Path(tmpdir))
            dataset = FlatDataset(training_set_dir)

            item = dataset[0]

            assert item["observation"].dtype == np.float32
            assert item["policy_p1"].dtype == np.float32
            assert item["policy_p2"].dtype == np.float32
            assert item["p1_value"].dtype == np.float32
            assert item["p2_value"].dtype == np.float32

    def test_obs_shape_property(self) -> None:
        """obs_shape should match builder."""
        with tempfile.TemporaryDirectory() as tmpdir:
            training_set_dir = _create_training_set(Path(tmpdir))
            dataset = FlatDataset(training_set_dir)

            assert dataset.obs_shape == (181,)

    def test_manifest_property(self) -> None:
        """manifest should return TrainingSetManifest."""
        with tempfile.TemporaryDirectory() as tmpdir:
            training_set_dir = _create_training_set(Path(tmpdir))
            dataset = FlatDataset(training_set_dir)

            manifest = dataset.manifest

            assert manifest.width == 5
            assert manifest.height == 5
            assert manifest.total_positions == 6

    def test_loads_multiple_shards(self) -> None:
        """Should load data from multiple shards."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)

            # Create training set with small shard size
            batch_dir = tmp_path / "batch1"
            games_dir = batch_dir / "games"
            games_dir.mkdir(parents=True)

            for i in range(3):
                _create_game_npz(games_dir / f"game_{i}.npz", num_positions=3)

            output_dir = tmp_path / "training_sets"
            output_dir.mkdir()

            builder = FlatObservationBuilder(width=5, height=5)
            training_set_dir = prepare_training_set(
                batch_dirs=[batch_dir],
                output_dir=output_dir,
                builder=builder,
                positions_per_shard=4,  # Will create 3 shards for 9 positions
                seed=42,
            )

            dataset = FlatDataset(training_set_dir)

            # Should have all 9 positions accessible
            assert len(dataset) == 9
            # Should be able to access all indices
            for i in range(9):
                item = dataset[i]
                assert item["observation"].shape == (181,)
