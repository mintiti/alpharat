"""Integration test: verify Rust-written bundles are readable by Python loader.

Runs the `write_test_bundle` binary to produce a bundle with known data,
then loads it with `load_game_bundle()` and checks every field.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pytest

from alpharat.data.loader import is_bundle_file, load_game_bundle
from alpharat.data.types import CheeseOutcome

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CARGO_WORKSPACE = Path(__file__).resolve().parents[2] / "crates" / "alpharat-sampling"


def _build_and_run_test_binary(output_path: Path) -> None:
    """Build and run the write_test_bundle binary."""
    # Build
    result = subprocess.run(
        ["cargo", "build", "--bin", "write_test_bundle", "-p", "alpharat-sampling"],
        capture_output=True,
        text=True,
        cwd=CARGO_WORKSPACE.parent.parent,
    )
    if result.returncode != 0:
        pytest.skip(f"cargo build failed:\n{result.stderr}")

    # Find the binary
    target_dir = CARGO_WORKSPACE.parent.parent / "target" / "debug"
    binary = target_dir / "write_test_bundle"
    if not binary.exists():
        pytest.skip(f"binary not found at {binary}")

    # Run
    result = subprocess.run(
        [str(binary), str(output_path)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"write_test_bundle failed:\n{result.stderr}"


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def bundle_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Produce a bundle file from the Rust binary."""
    tmp = tmp_path_factory.mktemp("rust_bundle")
    path = tmp / "test_bundle.npz"
    _build_and_run_test_binary(path)
    return path


class TestRustBundleParity:
    """Verify Python can load Rust-written bundles and all fields match."""

    def test_is_bundle(self, bundle_path: Path) -> None:
        assert is_bundle_file(bundle_path)

    def test_game_count(self, bundle_path: Path) -> None:
        games = load_game_bundle(bundle_path)
        assert len(games) == 2

    def test_game0_metadata(self, bundle_path: Path) -> None:
        games = load_game_bundle(bundle_path)
        g = games[0]
        assert g.width == 3
        assert g.height == 3
        assert g.max_turns == 30
        assert g.result == 0  # Draw
        assert g.final_p1_score == pytest.approx(0.5)
        assert g.final_p2_score == pytest.approx(0.5)

    def test_game0_maze(self, bundle_path: Path) -> None:
        games = load_game_bundle(bundle_path)
        g = games[0]
        assert g.maze.shape == (3, 3, 4)
        assert g.maze.dtype == np.int8

        # (0,0) → y=0, x=0: DOWN=-1, LEFT=-1
        assert g.maze[0, 0, 2] == -1  # DOWN
        assert g.maze[0, 0, 3] == -1  # LEFT
        assert g.maze[0, 0, 0] == 1  # UP open
        assert g.maze[0, 0, 1] == 1  # RIGHT open

    def test_game0_initial_cheese(self, bundle_path: Path) -> None:
        games = load_game_bundle(bundle_path)
        g = games[0]
        assert g.initial_cheese.shape == (3, 3)
        # Cheese at (1,1) → row=1, col=1
        assert g.initial_cheese[1, 1]
        # No cheese elsewhere
        assert g.initial_cheese.sum() == 1

    def test_game0_cheese_outcomes(self, bundle_path: Path) -> None:
        games = load_game_bundle(bundle_path)
        g = games[0]
        assert g.cheese_outcomes is not None
        assert g.cheese_outcomes.shape == (3, 3)
        # (1,1) simultaneous
        assert g.cheese_outcomes[1, 1] == CheeseOutcome.SIMULTANEOUS
        # Others uncollected
        assert g.cheese_outcomes[0, 0] == CheeseOutcome.UNCOLLECTED

    def test_game0_positions(self, bundle_path: Path) -> None:
        games = load_game_bundle(bundle_path)
        g = games[0]
        assert len(g.positions) == 2

        p0 = g.positions[0]
        assert p0.p1_pos == (0, 0)
        assert p0.p2_pos == (2, 2)
        assert p0.turn == 0
        assert p0.p1_score == pytest.approx(0.0)
        assert p0.p2_score == pytest.approx(0.0)
        assert p0.p1_mud == 0
        assert p0.p2_mud == 0

        # Value estimates
        assert p0.value_p1 == pytest.approx(0.75)
        assert p0.value_p2 == pytest.approx(0.25)

        # Actions
        assert p0.action_p1 == 0  # UP
        assert p0.action_p2 == 2  # DOWN

    def test_game0_policies(self, bundle_path: Path) -> None:
        games = load_game_bundle(bundle_path)
        p0 = games[0].positions[0]

        np.testing.assert_allclose(p0.policy_p1, [0.625, 0.3125, 0.0, 0.0, 0.0625], atol=1e-6)
        np.testing.assert_allclose(p0.policy_p2, [0.0, 0.0, 0.375, 0.5, 0.125], atol=1e-6)

    def test_game0_priors(self, bundle_path: Path) -> None:
        games = load_game_bundle(bundle_path)
        p0 = games[0].positions[0]

        np.testing.assert_allclose(p0.prior_p1, [0.3, 0.3, 0.1, 0.1, 0.2], atol=1e-6)
        np.testing.assert_allclose(p0.prior_p2, [0.1, 0.1, 0.3, 0.3, 0.2], atol=1e-6)

    def test_game0_visit_counts(self, bundle_path: Path) -> None:
        games = load_game_bundle(bundle_path)
        p0 = games[0].positions[0]

        np.testing.assert_allclose(p0.visit_counts_p1, [10.0, 5.0, 0.0, 0.0, 1.0], atol=1e-6)
        np.testing.assert_allclose(p0.visit_counts_p2, [0.0, 0.0, 6.0, 8.0, 2.0], atol=1e-6)

    def test_game0_position1(self, bundle_path: Path) -> None:
        """Second position of game 0."""
        games = load_game_bundle(bundle_path)
        p1 = games[0].positions[1]

        assert p1.p1_pos == (0, 1)
        assert p1.p2_pos == (2, 1)
        assert p1.turn == 1
        assert p1.action_p1 == 1  # RIGHT
        assert p1.action_p2 == 3  # LEFT
        assert p1.value_p1 == pytest.approx(0.8)
        assert p1.value_p2 == pytest.approx(0.2)

    def test_game0_cheese_mask(self, bundle_path: Path) -> None:
        """Cheese mask tracks correctly across positions."""
        games = load_game_bundle(bundle_path)
        # Position 0: cheese at (1,1)
        p0_cheese = games[0].positions[0].cheese_positions
        assert (1, 1) in p0_cheese

    def test_game1_metadata(self, bundle_path: Path) -> None:
        games = load_game_bundle(bundle_path)
        g = games[1]
        assert g.width == 3
        assert g.height == 3
        assert g.max_turns == 20
        assert g.result == 1  # P1Win
        assert g.final_p1_score == pytest.approx(1.0)
        assert g.final_p2_score == pytest.approx(0.0)

    def test_game1_single_position(self, bundle_path: Path) -> None:
        games = load_game_bundle(bundle_path)
        g = games[1]
        assert len(g.positions) == 1

        p0 = g.positions[0]
        assert p0.p1_pos == (1, 0)
        assert p0.p2_pos == (1, 2)
        assert p0.p2_mud == 3  # stuck in mud
        assert p0.action_p1 == 3  # LEFT
        assert p0.action_p2 == 4  # STAY

    def test_game1_mud_policy(self, bundle_path: Path) -> None:
        """P2 stuck in mud — all visits on STAY."""
        games = load_game_bundle(bundle_path)
        p0 = games[1].positions[0]

        np.testing.assert_allclose(p0.visit_counts_p2, [0.0, 0.0, 0.0, 0.0, 16.0], atol=1e-6)
        np.testing.assert_allclose(p0.policy_p2, [0.0, 0.0, 0.0, 0.0, 1.0], atol=1e-6)

    def test_game1_cheese_outcomes(self, bundle_path: Path) -> None:
        games = load_game_bundle(bundle_path)
        g = games[1]
        assert g.cheese_outcomes is not None
        # (0,0) collected by P1
        assert g.cheese_outcomes[0, 0] == CheeseOutcome.P1_WIN
        # (2,2) uncollected
        assert g.cheese_outcomes[2, 2] == CheeseOutcome.UNCOLLECTED

    def test_raw_dtypes(self, bundle_path: Path) -> None:
        """Verify the raw numpy dtypes match what Python writes."""
        data = np.load(bundle_path)

        assert data["game_lengths"].dtype == np.int32
        assert data["maze"].dtype == np.int8
        assert data["initial_cheese"].dtype == bool
        assert data["cheese_outcomes"].dtype == np.int8
        assert data["max_turns"].dtype == np.int16
        assert data["result"].dtype == np.int8
        assert data["final_p1_score"].dtype == np.float32
        assert data["final_p2_score"].dtype == np.float32

        assert data["p1_pos"].dtype == np.int8
        assert data["p2_pos"].dtype == np.int8
        assert data["p1_score"].dtype == np.float32
        assert data["p2_score"].dtype == np.float32
        assert data["p1_mud"].dtype == np.int8
        assert data["p2_mud"].dtype == np.int8
        assert data["cheese_mask"].dtype == bool
        assert data["turn"].dtype == np.int16
        assert data["value_p1"].dtype == np.float32
        assert data["value_p2"].dtype == np.float32
        assert data["visit_counts_p1"].dtype == np.float32
        assert data["visit_counts_p2"].dtype == np.float32
        assert data["prior_p1"].dtype == np.float32
        assert data["prior_p2"].dtype == np.float32
        assert data["policy_p1"].dtype == np.float32
        assert data["policy_p2"].dtype == np.float32
        assert data["action_p1"].dtype == np.int8
        assert data["action_p2"].dtype == np.int8

    def test_raw_shapes(self, bundle_path: Path) -> None:
        """Verify array shapes in the raw npz."""
        data = np.load(bundle_path)

        # 2 games, total 3 positions (2 + 1)
        assert data["game_lengths"].shape == (2,)
        assert data["maze"].shape == (2, 3, 3, 4)
        assert data["initial_cheese"].shape == (2, 3, 3)
        assert data["cheese_outcomes"].shape == (2, 3, 3)
        assert data["max_turns"].shape == (2,)
        assert data["result"].shape == (2,)

        assert data["p1_pos"].shape == (3, 2)
        assert data["p2_pos"].shape == (3, 2)
        assert data["p1_score"].shape == (3,)
        assert data["cheese_mask"].shape == (3, 3, 3)
        assert data["policy_p1"].shape == (3, 5)
        assert data["action_p1"].shape == (3,)

    def test_all_keys_present(self, bundle_path: Path) -> None:
        """Every key from GameFileKey (bundle variant) should be present."""
        data = np.load(bundle_path)
        expected_keys = {
            "game_lengths",
            "maze",
            "initial_cheese",
            "cheese_outcomes",
            "max_turns",
            "result",
            "final_p1_score",
            "final_p2_score",
            "p1_pos",
            "p2_pos",
            "p1_score",
            "p2_score",
            "p1_mud",
            "p2_mud",
            "cheese_mask",
            "turn",
            "value_p1",
            "value_p2",
            "visit_counts_p1",
            "visit_counts_p2",
            "prior_p1",
            "prior_p2",
            "policy_p1",
            "policy_p2",
            "action_p1",
            "action_p2",
        }
        actual_keys = set(data.files)
        assert actual_keys == expected_keys
