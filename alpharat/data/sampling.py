"""Self-play sampling for training data generation."""

from __future__ import annotations

import copy
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING

import numpy as np
from pydantic import BaseModel, Field

from alpharat.data.batch import GameParams, create_batch, get_batch_stats
from alpharat.data.recorder import GameRecorder
from alpharat.eval.game import is_terminal
from alpharat.mcts import MCTSConfig  # noqa: TC001
from alpharat.mcts.nash import select_action_from_strategy
from alpharat.mcts.node import MCTSNode
from alpharat.mcts.tree import MCTSTree

if TYPE_CHECKING:
    from pathlib import Path

    from pyrat_engine.core.game import PyRat


# --- Config Models ---


class SamplingParams(BaseModel):
    """Parameters for the sampling process."""

    num_games: int
    workers: int = 4


class SamplingConfig(BaseModel):
    """Full configuration for a sampling run."""

    mcts: MCTSConfig = Field(discriminator="variant")
    game: GameParams
    sampling: SamplingParams
    output_dir: str
    checkpoint: str | None = None


# --- Helper Functions ---


def build_tree(game: PyRat, gamma: float) -> MCTSTree:
    """Build fresh MCTS tree for search.

    The tree will initialize the root with smart uniform priors that only
    assign probability to distinct effective actions (blocked moves get 0).

    Args:
        game: Current game state (will be deep-copied for simulation).
        gamma: Discount factor for value backup.

    Returns:
        MCTSTree ready for search.
    """
    simulator = copy.deepcopy(game)

    # Dummy priors - tree will overwrite with smart uniform via _init_root_priors()
    dummy = np.ones(5) / 5

    root = MCTSNode(
        game_state=None,
        prior_policy_p1=dummy,
        prior_policy_p2=dummy,
        nn_payout_prediction=np.zeros((5, 5)),
        parent=None,
        p1_mud_turns_remaining=simulator.player1_mud_turns,
        p2_mud_turns_remaining=simulator.player2_mud_turns,
    )

    return MCTSTree(game=simulator, root=root, gamma=gamma, predict_fn=None)


def create_game(params: GameParams, seed: int) -> PyRat:
    """Create a new game from parameters.

    Args:
        params: Game configuration.
        seed: Random seed for maze generation.

    Returns:
        Configured PyRat game instance.
    """
    from pyrat_engine.core.game import PyRat

    return PyRat(
        width=params.width,
        height=params.height,
        cheese_count=params.cheese_count,
        max_turns=params.max_turns,
        seed=seed,
    )


# --- Core Sampling ---


def play_and_record_game(
    config: SamplingConfig,
    games_dir: Path,
    seed: int,
) -> Path | None:
    """Play one game with MCTS and record all positions.

    Args:
        config: Sampling configuration.
        games_dir: Directory to save the game file.
        seed: Random seed for game generation.

    Returns:
        Path to saved npz file, or None if game had no positions.
    """
    game = create_game(config.game, seed)

    with GameRecorder(game, games_dir, config.game.width, config.game.height) as recorder:
        while not is_terminal(game):
            # Build tree and run search
            tree = build_tree(game, config.mcts.gamma)
            search = config.mcts.build(tree)
            result = search.search()

            # Record position before making move
            recorder.record_position(
                game=game,
                search_result=result,
                prior_p1=tree.root.prior_policy_p1,
                prior_p2=tree.root.prior_policy_p2,
                visit_counts=tree.root.action_visits,
            )

            # Sample actions from Nash equilibrium policies
            a1 = select_action_from_strategy(result.policy_p1)
            a2 = select_action_from_strategy(result.policy_p2)

            game.make_move(a1, a2)

    return recorder.saved_path


def run_sampling(config: SamplingConfig, *, verbose: bool = True) -> Path:
    """Run full sampling session.

    Creates a batch directory, plays games in parallel, and records
    all positions to npz files.

    Args:
        config: Sampling configuration.
        verbose: Print progress if True.

    Returns:
        Path to the created batch directory.
    """
    batch_dir = create_batch(
        parent_dir=config.output_dir,
        checkpoint_path=config.checkpoint,
        mcts_config=config.mcts,
        game_params=config.game,
    )
    games_dir = batch_dir / "games"

    if verbose:
        print(f"Sampling {config.sampling.num_games} games")
        print(f"  MCTS: {config.mcts.variant}, {config.mcts.simulations} sims")
        print(
            f"  Game: {config.game.width}x{config.game.height}, {config.game.cheese_count} cheese"
        )
        print(f"  Output: {batch_dir}")
        print()

    with ThreadPoolExecutor(max_workers=config.sampling.workers) as executor:
        futures = [
            executor.submit(play_and_record_game, config, games_dir, seed=seed)
            for seed in range(config.sampling.num_games)
        ]

        for completed, future in enumerate(as_completed(futures), start=1):
            future.result()  # propagate exceptions
            if verbose and completed % 10 == 0:
                print(f"Completed {completed}/{config.sampling.num_games} games")

    if verbose:
        stats = get_batch_stats(batch_dir)
        print()
        print(f"Done! {stats.game_count} games, {stats.total_positions} positions")

    return batch_dir
