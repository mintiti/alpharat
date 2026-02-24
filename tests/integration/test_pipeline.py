"""Integration tests for full training data pipeline.

Verifies data integrity from game creation through to training shards.
Runs a real game with real MCTS, then checks semantic invariants at each stage.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from pyrat_engine.core.types import Direction

from alpharat.config.game import CheeseConfig, GameConfig
from alpharat.data.loader import is_bundle_file, iter_games_from_bundle, load_game_data
from alpharat.data.sampling import SamplingConfig, SamplingParams, run_sampling
from alpharat.data.sharding import load_training_set_manifest, prepare_training_set
from alpharat.mcts import PythonMCTSConfig
from alpharat.nn.builders.flat import FlatObservationBuilder
from alpharat.nn.extraction import from_game_arrays
from alpharat.nn.targets import build_targets

if TYPE_CHECKING:
    from pathlib import Path

    from alpharat.data.types import GameData, PositionData
    from alpharat.nn.types import ObservationInput, TargetBundle

# =============================================================================
# Loading Helpers
# =============================================================================


def _load_first_game(game_files: list[Path]) -> GameData:
    """Load the first game from a list of game files (handles bundles or single-game files)."""
    game_file = game_files[0]
    if is_bundle_file(game_file):
        return next(iter_games_from_bundle(game_file))
    return load_game_data(game_file)


# =============================================================================
# Verification Helpers
# =============================================================================


def _verify_maze_edges(maze: np.ndarray, height: int, width: int) -> None:
    """Verify edge cells have walls (-1) in appropriate directions."""
    # Y-up coordinate system: y=0 is BOTTOM, y=height-1 is TOP
    # Bottom edge (y=0): DOWN blocked
    assert np.all(maze[0, :, Direction.DOWN] == -1), "Bottom edge should block DOWN"
    # Top edge (y=height-1): UP blocked
    assert np.all(maze[height - 1, :, Direction.UP] == -1), "Top edge should block UP"
    # Left edge (x=0): LEFT blocked
    assert np.all(maze[:, 0, Direction.LEFT] == -1), "Left edge should block LEFT"
    # Right edge (x=width-1): RIGHT blocked
    assert np.all(maze[:, width - 1, Direction.RIGHT] == -1), "Right edge should block RIGHT"


def _verify_obs_input_matches_position(
    obs_input: ObservationInput,
    pos: PositionData,
    game_data: GameData,
) -> None:
    """Verify ObservationInput fields match source position data."""
    assert obs_input.p1_pos == pos.p1_pos, f"P1 pos mismatch: {obs_input.p1_pos} != {pos.p1_pos}"
    assert obs_input.p2_pos == pos.p2_pos, f"P2 pos mismatch: {obs_input.p2_pos} != {pos.p2_pos}"
    assert obs_input.p1_score == pos.p1_score, "P1 score mismatch"
    assert obs_input.p2_score == pos.p2_score, "P2 score mismatch"
    assert obs_input.p1_mud == pos.p1_mud, "P1 mud mismatch"
    assert obs_input.p2_mud == pos.p2_mud, "P2 mud mismatch"
    assert obs_input.turn == pos.turn, "Turn mismatch"
    assert obs_input.max_turns == game_data.max_turns, "Max turns mismatch"
    assert obs_input.width == game_data.width, "Width mismatch"
    assert obs_input.height == game_data.height, "Height mismatch"

    # Cheese mask matches position list
    for x, y in pos.cheese_positions:
        assert obs_input.cheese_mask[y, x], f"Cheese at ({x}, {y}) not in mask"
    assert obs_input.cheese_mask.sum() == len(pos.cheese_positions), (
        f"Cheese count mismatch: {obs_input.cheese_mask.sum()} != {len(pos.cheese_positions)}"
    )


def _verify_observation_encoding(
    obs: np.ndarray,
    obs_input: ObservationInput,
    width: int,
    height: int,
) -> None:
    """Verify flat observation encodes input correctly.

    Layout: [maze(H*W*4), p1(H*W), p2(H*W), cheese(H*W), score_diff, progress, p1_mud, p2_mud]
    """
    spatial = width * height

    # P1 position one-hot
    p1_start = spatial * 4
    p1_x, p1_y = obs_input.p1_pos
    p1_idx = p1_start + p1_y * width + p1_x
    assert obs[p1_idx] == 1.0, f"P1 one-hot wrong at idx {p1_idx}"
    assert obs[p1_start : p1_start + spatial].sum() == 1.0, "P1 should be exactly one-hot"

    # P2 position one-hot
    p2_start = spatial * 5
    p2_x, p2_y = obs_input.p2_pos
    p2_idx = p2_start + p2_y * width + p2_x
    assert obs[p2_idx] == 1.0, f"P2 one-hot wrong at idx {p2_idx}"
    assert obs[p2_start : p2_start + spatial].sum() == 1.0, "P2 should be exactly one-hot"

    # Cheese mask
    cheese_start = spatial * 6
    cheese_section = obs[cheese_start : cheese_start + spatial].reshape(height, width)
    np.testing.assert_array_equal(
        cheese_section,
        obs_input.cheese_mask.astype(np.float32),
        err_msg="Cheese mask encoding mismatch",
    )

    # Scalars
    scalars_start = spatial * 7
    expected_score_diff = obs_input.p1_score - obs_input.p2_score
    expected_progress = obs_input.turn / obs_input.max_turns if obs_input.max_turns > 0 else 0
    expected_p1_mud = obs_input.p1_mud / 10  # MAX_MUD_TURNS
    expected_p2_mud = obs_input.p2_mud / 10

    np.testing.assert_allclose(
        obs[scalars_start], expected_score_diff, err_msg="Score diff encoding mismatch"
    )
    np.testing.assert_allclose(
        obs[scalars_start + 1], expected_progress, err_msg="Progress encoding mismatch"
    )
    np.testing.assert_allclose(
        obs[scalars_start + 2], expected_p1_mud, err_msg="P1 mud encoding mismatch"
    )
    np.testing.assert_allclose(
        obs[scalars_start + 3], expected_p2_mud, err_msg="P2 mud encoding mismatch"
    )


def _verify_targets(
    targets: TargetBundle,
    pos: PositionData,
    game_data: GameData,
) -> None:
    """Verify targets match source data."""
    np.testing.assert_allclose(targets.policy_p1, pos.policy_p1, err_msg="Policy P1 mismatch")
    np.testing.assert_allclose(targets.policy_p2, pos.policy_p2, err_msg="Policy P2 mismatch")

    # p1_value = remaining score P1 can get = final_p1_score - current_p1_score
    # p2_value = remaining score P2 can get = final_p2_score - current_p2_score
    expected_p1_value = game_data.final_p1_score - pos.p1_score
    expected_p2_value = game_data.final_p2_score - pos.p2_score
    assert targets.p1_value == pytest.approx(expected_p1_value), (
        f"P1 value mismatch: {targets.p1_value} != {expected_p1_value}"
    )
    assert targets.p2_value == pytest.approx(expected_p2_value), (
        f"P2 value mismatch: {targets.p2_value} != {expected_p2_value}"
    )


def _verify_training_shards(
    training_dir: Path,
    game_data: GameData,
    builder: FlatObservationBuilder,
) -> None:
    """Verify training shards contain all positions with correct data."""
    manifest = load_training_set_manifest(training_dir)
    assert manifest.total_positions == len(game_data.positions), (
        f"Position count mismatch: {manifest.total_positions} != {len(game_data.positions)}"
    )
    assert manifest.width == game_data.width
    assert manifest.height == game_data.height

    # Load all shard data
    all_policies_p1 = []
    all_policies_p2 = []
    all_p1_values = []
    all_p2_values = []
    for i in range(manifest.shard_count):
        with np.load(training_dir / f"shard_{i:04d}.npz") as data:
            all_policies_p1.append(data["policy_p1"])
            all_policies_p2.append(data["policy_p2"])
            all_p1_values.append(data["value_p1"])
            all_p2_values.append(data["value_p2"])

    all_policies_p1 = np.concatenate(all_policies_p1)
    all_policies_p2 = np.concatenate(all_policies_p2)
    all_p1_values = np.concatenate(all_p1_values)
    all_p2_values = np.concatenate(all_p2_values)

    # Compute expected values for each position (remaining score each player will get)
    expected_p1_values = [game_data.final_p1_score - pos.p1_score for pos in game_data.positions]
    expected_p2_values = [game_data.final_p2_score - pos.p2_score for pos in game_data.positions]

    # Values in shards are shuffled, so check the set of values matches
    # (can't check order since prepare_training_set shuffles)
    np.testing.assert_allclose(
        sorted(all_p1_values), sorted(expected_p1_values), err_msg="Shard p1_values mismatch"
    )
    np.testing.assert_allclose(
        sorted(all_p2_values), sorted(expected_p2_values), err_msg="Shard p2_values mismatch"
    )

    # Policies should be valid distributions
    assert np.all(all_policies_p1 >= 0), "P1 policies should be non-negative"
    assert np.all(all_policies_p2 >= 0), "P2 policies should be non-negative"
    np.testing.assert_allclose(
        all_policies_p1.sum(axis=1), 1.0, rtol=1e-5, err_msg="P1 policies should sum to 1"
    )
    np.testing.assert_allclose(
        all_policies_p2.sum(axis=1), 1.0, rtol=1e-5, err_msg="P2 policies should sum to 1"
    )


# =============================================================================
# Integration Tests
# =============================================================================


class TestPipelineIntegrity:
    """Verify data survives from game through to training shards."""

    def test_full_pipeline_all_positions(self, tmp_path: Path) -> None:
        """Play real game, verify invariants at each stage for all positions.

        This test:
        1. Plays a real game with real MCTS (30 sims for speed)
        2. Loops over ALL positions in the game
        3. Verifies semantic invariants at each pipeline stage
        4. Checks the final training shards for correctness
        """
        width, height = 7, 7

        # === Config: Real game, fast MCTS ===
        config = SamplingConfig(
            mcts=PythonMCTSConfig(simulations=30, c_puct=1.5),
            game=GameConfig(
                width=width,
                height=height,
                max_turns=50,
                cheese=CheeseConfig(count=5),
            ),
            sampling=SamplingParams(num_games=1, workers=1),
            group="test_group",
            experiments_dir=str(tmp_path),
        )

        # === Stage 1: Play and record ===
        batch_dir, _ = run_sampling(config, verbose=False)
        game_files = list((batch_dir / "games").glob("*.npz"))
        assert len(game_files) >= 1, f"Expected at least 1 game file, got {len(game_files)}"

        # === Stage 2: Load game data ===
        game_data = _load_first_game(game_files)

        # Verify game-level invariants
        assert game_data.width == width
        assert game_data.height == height
        assert game_data.maze.shape == (height, width, 4)
        _verify_maze_edges(game_data.maze, height, width)

        # Must have at least 1 position
        assert len(game_data.positions) >= 1, "Game should have at least 1 position"

        # === Stage 3-5: Verify each position ===
        builder = FlatObservationBuilder(width=width, height=height)

        for i, pos in enumerate(game_data.positions):
            # Stage 3: Extract ObservationInput
            obs_input = from_game_arrays(game_data, pos)
            _verify_obs_input_matches_position(obs_input, pos, game_data)

            # Stage 4: Build observation
            obs = builder.build(obs_input)
            assert obs.shape == builder.obs_shape, f"Obs shape mismatch at position {i}"
            assert obs.dtype == np.float32, f"Obs dtype should be float32 at position {i}"
            _verify_observation_encoding(obs, obs_input, width, height)

            # Stage 5: Build targets
            targets = build_targets(game_data, pos)
            _verify_targets(targets, pos, game_data)

        # === Stage 6: Prepare training set ===
        training_output = tmp_path / "training"
        training_output.mkdir()

        training_dir = prepare_training_set(
            batch_dirs=[batch_dir],
            output_dir=training_output,
            builder=builder,
            seed=42,
        )

        # Verify shard contents
        _verify_training_shards(training_dir, game_data, builder)

    def test_maze_topology_preserved(self, tmp_path: Path) -> None:
        """Verify maze walls and mud survive the full pipeline."""
        width, height = 5, 5

        config = SamplingConfig(
            mcts=PythonMCTSConfig(simulations=20, c_puct=1.5),
            game=GameConfig(
                width=width,
                height=height,
                max_turns=30,
                cheese=CheeseConfig(count=3),
            ),
            sampling=SamplingParams(num_games=1, workers=1),
            group="test_topology",
            experiments_dir=str(tmp_path),
        )

        batch_dir, _ = run_sampling(config, verbose=False)
        game_files = list((batch_dir / "games").glob("*.npz"))
        game_data = _load_first_game(game_files)

        # Verify edges are correctly marked
        _verify_maze_edges(game_data.maze, height, width)

        # Verify maze is preserved in observation input
        pos = game_data.positions[0]
        obs_input = from_game_arrays(game_data, pos)
        np.testing.assert_array_equal(obs_input.maze, game_data.maze)

        # Verify maze encoding in flat observation
        builder = FlatObservationBuilder(width=width, height=height)
        obs = builder.build(obs_input)

        # Maze is first H*W*4 elements, normalized
        maze_section = obs[: height * width * 4].reshape(height, width, 4)

        # Walls should be -1, valid moves should be > 0 (normalized cost)
        # Check that edge walls are preserved (Y-up: y=0 is bottom, y=height-1 is top)
        assert np.all(maze_section[0, :, Direction.DOWN] == -1), "Bottom edge DOWN should be -1"
        assert np.all(maze_section[height - 1, :, Direction.UP] == -1), "Top edge UP should be -1"
        assert np.all(maze_section[:, 0, Direction.LEFT] == -1), "Left edge LEFT should be -1"
        assert np.all(maze_section[:, width - 1, Direction.RIGHT] == -1), (
            "Right edge RIGHT should be -1"
        )

    def test_policies_are_valid_distributions(self, tmp_path: Path) -> None:
        """Verify MCTS policies are valid probability distributions throughout."""
        width, height = 5, 5

        config = SamplingConfig(
            mcts=PythonMCTSConfig(simulations=25, c_puct=1.5),
            game=GameConfig(
                width=width,
                height=height,
                max_turns=25,
                cheese=CheeseConfig(count=3),
            ),
            sampling=SamplingParams(num_games=1, workers=1),
            group="test_policies",
            experiments_dir=str(tmp_path),
        )

        batch_dir, _ = run_sampling(config, verbose=False)
        game_files = list((batch_dir / "games").glob("*.npz"))
        game_data = _load_first_game(game_files)

        for i, pos in enumerate(game_data.positions):
            # Policies should be non-negative
            assert np.all(pos.policy_p1 >= 0), f"P1 policy negative at position {i}"
            assert np.all(pos.policy_p2 >= 0), f"P2 policy negative at position {i}"

            # Policies should sum to 1
            np.testing.assert_allclose(
                pos.policy_p1.sum(), 1.0, rtol=1e-5, err_msg=f"P1 policy doesn't sum to 1 at {i}"
            )
            np.testing.assert_allclose(
                pos.policy_p2.sum(), 1.0, rtol=1e-5, err_msg=f"P2 policy doesn't sum to 1 at {i}"
            )

            # Targets should preserve this
            targets = build_targets(game_data, pos)
            np.testing.assert_allclose(
                targets.policy_p1.sum(),
                1.0,
                rtol=1e-5,
                err_msg=f"Target P1 policy doesn't sum to 1 at {i}",
            )
            np.testing.assert_allclose(
                targets.policy_p2.sum(),
                1.0,
                rtol=1e-5,
                err_msg=f"Target P2 policy doesn't sum to 1 at {i}",
            )
