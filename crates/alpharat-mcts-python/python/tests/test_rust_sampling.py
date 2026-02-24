"""Tests for the Rust self-play sampling pipeline bindings."""

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING

import numpy as np
import pytest
from alpharat_sampling import SelfPlayProgress, SelfPlayStats, rust_self_play

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def output_dir(tmp_path: Path) -> Path:
    """Create a temporary output directory for game bundles."""
    d = tmp_path / "games"
    d.mkdir()
    return d


# ---------------------------------------------------------------------------
# Basic self-play (no NN)
# ---------------------------------------------------------------------------


class TestRustSelfPlayNoNN:
    def test_runs_and_returns_stats(self, output_dir: Path) -> None:
        """Smoke test: run 10 games with SmartUniform, verify stats."""
        stats = rust_self_play(
            width=5,
            height=5,
            cheese_count=5,
            max_turns=30,
            num_games=10,
            simulations=20,
            batch_size=1,
            num_threads=2,
            output_dir=str(output_dir),
            max_games_per_bundle=5,
        )

        assert isinstance(stats, SelfPlayStats)
        assert stats.total_games == 10
        assert stats.total_positions > 0
        assert stats.total_simulations > 0
        assert stats.elapsed_secs > 0
        assert stats.p1_wins + stats.p2_wins + stats.draws == 10

    def test_writes_npz_bundles(self, output_dir: Path) -> None:
        """Verify NPZ bundle files are created on disk."""
        rust_self_play(
            width=5,
            height=5,
            cheese_count=5,
            max_turns=30,
            num_games=10,
            simulations=10,
            batch_size=1,
            num_threads=2,
            output_dir=str(output_dir),
            max_games_per_bundle=5,
        )

        bundles = list(output_dir.glob("*.npz"))
        assert len(bundles) >= 2  # 10 games / 5 per bundle = 2 bundles

    def test_stats_derived_metrics(self, output_dir: Path) -> None:
        """Verify derived metrics are computed correctly."""
        stats = rust_self_play(
            width=5,
            height=5,
            cheese_count=5,
            max_turns=30,
            num_games=5,
            simulations=10,
            batch_size=1,
            num_threads=1,
            output_dir=str(output_dir),
        )

        assert stats.games_per_second > 0
        assert stats.positions_per_second > 0
        assert stats.simulations_per_second > 0
        assert stats.avg_turns > 0
        assert 0 <= stats.cheese_utilization <= 1.0
        assert 0 <= stats.draw_rate <= 1.0

    def test_repr(self, output_dir: Path) -> None:
        """SelfPlayStats has a readable repr."""
        stats = rust_self_play(
            width=5,
            height=5,
            cheese_count=5,
            max_turns=30,
            num_games=3,
            simulations=10,
            batch_size=1,
            num_threads=1,
            output_dir=str(output_dir),
        )
        r = repr(stats)
        assert "SelfPlayStats" in r
        assert "games=3" in r


# ---------------------------------------------------------------------------
# Bundle loading compatibility
# ---------------------------------------------------------------------------


EXPECTED_BUNDLE_KEYS = {
    # Game-level
    "game_lengths",
    "maze",
    "initial_cheese",
    "cheese_outcomes",
    "max_turns",
    "result",
    "final_p1_score",
    "final_p2_score",
    # Position-level
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


class TestBundleCompatibility:
    def test_bundle_has_expected_keys(self, output_dir: Path) -> None:
        """Verify NPZ bundles contain all expected arrays."""
        rust_self_play(
            width=5,
            height=5,
            cheese_count=5,
            max_turns=30,
            num_games=5,
            simulations=10,
            batch_size=1,
            num_threads=1,
            output_dir=str(output_dir),
            max_games_per_bundle=10,
        )

        bundles = list(output_dir.glob("*.npz"))
        assert len(bundles) >= 1

        with np.load(bundles[0]) as data:
            actual_keys = set(data.files)
            missing = EXPECTED_BUNDLE_KEYS - actual_keys
            assert not missing, f"Missing keys in NPZ: {missing}"

    def test_bundle_array_shapes(self, output_dir: Path) -> None:
        """Verify array shapes and dtypes in bundles."""
        num_games = 3
        rust_self_play(
            width=5,
            height=5,
            cheese_count=5,
            max_turns=30,
            num_games=num_games,
            simulations=10,
            batch_size=1,
            num_threads=1,
            output_dir=str(output_dir),
            max_games_per_bundle=10,
        )

        bundles = list(output_dir.glob("*.npz"))
        with np.load(bundles[0]) as data:
            k = len(data["game_lengths"])  # number of games in this bundle
            n = int(data["game_lengths"].sum())  # total positions

            # Game-level arrays
            assert data["game_lengths"].shape == (k,)
            assert data["maze"].shape == (k, 5, 5, 4)
            assert data["initial_cheese"].shape == (k, 5, 5)
            assert data["cheese_outcomes"].shape == (k, 5, 5)
            assert data["max_turns"].shape == (k,)
            assert data["result"].shape == (k,)
            assert data["final_p1_score"].shape == (k,)
            assert data["final_p2_score"].shape == (k,)

            # Position-level arrays
            assert data["p1_pos"].shape == (n, 2)
            assert data["p2_pos"].shape == (n, 2)
            assert data["cheese_mask"].shape == (n, 5, 5)
            assert data["policy_p1"].shape == (n, 5)
            assert data["policy_p2"].shape == (n, 5)
            assert data["visit_counts_p1"].shape == (n, 5)
            assert data["visit_counts_p2"].shape == (n, 5)
            assert data["prior_p1"].shape == (n, 5)
            assert data["prior_p2"].shape == (n, 5)
            assert data["action_p1"].shape == (n,)
            assert data["action_p2"].shape == (n,)

    def test_load_with_game_bundle(self, output_dir: Path) -> None:
        """Verify bundles load correctly with load_game_bundle()."""
        pytest.importorskip("torch")

        rust_self_play(
            width=5,
            height=5,
            cheese_count=5,
            max_turns=30,
            num_games=3,
            simulations=10,
            batch_size=1,
            num_threads=1,
            output_dir=str(output_dir),
            max_games_per_bundle=10,
        )

        from alpharat.data.loader import load_game_bundle

        bundles = list(output_dir.glob("*.npz"))
        games = load_game_bundle(bundles[0])

        assert len(games) == 3
        for game_data in games:
            assert game_data.maze.shape == (5, 5, 4)
            assert game_data.initial_cheese.shape == (5, 5)
            assert len(game_data.positions) > 0


# ---------------------------------------------------------------------------
# Progress polling
# ---------------------------------------------------------------------------


class TestProgressPolling:
    def test_progress_starts_at_zero(self) -> None:
        """New progress object has all counters at 0."""
        progress = SelfPlayProgress()
        assert progress.games_completed == 0
        assert progress.positions_completed == 0
        assert progress.simulations_completed == 0

    def test_progress_increments(self, output_dir: Path) -> None:
        """Progress counters increment during self-play."""
        progress = SelfPlayProgress()
        num_games = 20

        observed_progress = False

        def poll() -> None:
            nonlocal observed_progress
            for _ in range(100):
                if progress.games_completed > 0:
                    observed_progress = True
                    return
                time.sleep(0.05)

        poller = threading.Thread(target=poll, daemon=True)
        poller.start()

        rust_self_play(
            width=5,
            height=5,
            cheese_count=5,
            max_turns=30,
            num_games=num_games,
            simulations=50,
            batch_size=1,
            num_threads=2,
            output_dir=str(output_dir),
            progress=progress,
        )

        poller.join(timeout=5)

        # After completion, all games should be done
        assert progress.games_completed == num_games
        assert progress.positions_completed > 0
        assert progress.simulations_completed > 0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_single_game(self, output_dir: Path) -> None:
        """Single game works correctly."""
        stats = rust_self_play(
            width=5,
            height=5,
            cheese_count=5,
            max_turns=30,
            num_games=1,
            simulations=10,
            batch_size=1,
            num_threads=1,
            output_dir=str(output_dir),
        )
        assert stats.total_games == 1

    def test_single_thread(self, output_dir: Path) -> None:
        """Single-threaded mode works."""
        stats = rust_self_play(
            width=5,
            height=5,
            cheese_count=5,
            max_turns=30,
            num_games=5,
            simulations=10,
            batch_size=1,
            num_threads=1,
            output_dir=str(output_dir),
        )
        assert stats.total_games == 5

    def test_asymmetric_games(self, output_dir: Path) -> None:
        """Asymmetric game generation works."""
        stats = rust_self_play(
            width=5,
            height=5,
            cheese_count=5,
            max_turns=30,
            num_games=5,
            simulations=10,
            batch_size=1,
            num_threads=1,
            cheese_symmetric=False,
            output_dir=str(output_dir),
        )
        assert stats.total_games == 5
