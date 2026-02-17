"""Searcher interface — abstracts Python and Rust MCTS backends.

Both backends implement the same protocol: take a game, return a SearchResult.
Consumers never touch search internals.
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np

from alpharat.mcts.result import SearchResult

if TYPE_CHECKING:
    from collections.abc import Callable

    from pyrat_engine.core.game import PyRat


@runtime_checkable
class Searcher(Protocol):
    """Protocol for MCTS search backends.

    Both Python and Rust MCTS implement this interface.
    """

    def search(self, game: PyRat) -> SearchResult: ...


class PythonSearcher:
    """Wraps Python MCTS — builds tree, runs search, packages canonical result.

    Encapsulates the tree-building complexity that was previously inline in
    the sampling loop.

    Args:
        simulations: Number of MCTS simulations.
        gamma: Discount factor for value backup.
        c_puct: Exploration constant.
        force_k: Forced playout coefficient.
        fpu_reduction: First-play urgency penalty.
        nn_ctx: Optional NN context for guided search.
    """

    def __init__(
        self,
        simulations: int,
        gamma: float,
        c_puct: float,
        force_k: float,
        fpu_reduction: float,
        nn_ctx: Any | None = None,
    ) -> None:
        from alpharat.mcts.decoupled_puct import DecoupledPUCTConfig

        self._config = DecoupledPUCTConfig(
            simulations=simulations,
            gamma=gamma,
            c_puct=c_puct,
            force_k=force_k,
            fpu_reduction=fpu_reduction,
        )
        self._nn_ctx = nn_ctx

    def search(self, game: PyRat) -> SearchResult:
        """Run Python MCTS search on the given game state."""
        from alpharat.config.checkpoint import make_predict_fn
        from alpharat.mcts.node import MCTSNode
        from alpharat.mcts.tree import MCTSTree

        simulator = copy.deepcopy(game)

        predict_fn = None
        if self._nn_ctx is not None:
            predict_fn = make_predict_fn(
                self._nn_ctx.model,
                self._nn_ctx.builder,
                simulator,
                self._nn_ctx.width,
                self._nn_ctx.height,
                self._nn_ctx.device,
            )

        dummy = np.ones(5) / 5
        root = MCTSNode(
            game_state=None,
            prior_policy_p1=dummy,
            prior_policy_p2=dummy,
            nn_value_p1=0.0,
            nn_value_p2=0.0,
            parent=None,
            p1_mud_turns_remaining=simulator.player1_mud_turns,
            p2_mud_turns_remaining=simulator.player2_mud_turns,
        )

        tree = MCTSTree(
            game=simulator,
            root=root,
            gamma=self._config.gamma,
            predict_fn=predict_fn,
        )

        search = self._config.build(tree)
        return search.search()


class RustSearcher:
    """Wraps Rust MCTS — calls rust_mcts_search, packages canonical result.

    Args:
        simulations: Number of MCTS simulations.
        c_puct: Exploration constant.
        force_k: Forced playout coefficient.
        fpu_reduction: First-play urgency penalty.
        batch_size: Within-tree batching size.
        noise_epsilon: Dirichlet noise mixing weight (0 = disabled).
        noise_concentration: Total Dirichlet concentration (KataGo-style).
        predict_fn: Optional batched predict_fn for NN priors.
        seed: Optional RNG seed for deterministic search.
    """

    def __init__(
        self,
        simulations: int,
        c_puct: float,
        force_k: float,
        fpu_reduction: float,
        batch_size: int = 8,
        noise_epsilon: float = 0.0,
        noise_concentration: float = 10.83,
        predict_fn: Callable[..., Any] | None = None,
        seed: int | None = None,
    ) -> None:
        self._simulations = simulations
        self._c_puct = c_puct
        self._force_k = force_k
        self._fpu_reduction = fpu_reduction
        self._batch_size = batch_size
        self._noise_epsilon = noise_epsilon
        self._noise_concentration = noise_concentration
        self._predict_fn = predict_fn
        self._seed = seed

    def search(self, game: PyRat) -> SearchResult:
        """Run Rust MCTS search on the given game state."""
        from alpharat_mcts import rust_mcts_search

        rust_result = rust_mcts_search(
            game,
            predict_fn=self._predict_fn,
            simulations=self._simulations,
            batch_size=self._batch_size,
            c_puct=self._c_puct,
            fpu_reduction=self._fpu_reduction,
            force_k=self._force_k,
            noise_epsilon=self._noise_epsilon,
            noise_concentration=self._noise_concentration,
            seed=self._seed,
        )

        # Renormalize policies: Rust normalizes in f32, promoting to f64
        # can drift the sum away from 1.0 which numpy.random.choice rejects.
        policy_p1 = np.asarray(rust_result.policy_p1, dtype=np.float64)
        policy_p2 = np.asarray(rust_result.policy_p2, dtype=np.float64)
        s1, s2 = policy_p1.sum(), policy_p2.sum()
        if s1 > 0:
            policy_p1 /= s1
        if s2 > 0:
            policy_p2 /= s2

        return SearchResult(
            policy_p1=policy_p1,
            policy_p2=policy_p2,
            value_p1=float(rust_result.value_p1),
            value_p2=float(rust_result.value_p2),
            visit_counts_p1=np.asarray(rust_result.visit_counts_p1, dtype=np.float64),
            visit_counts_p2=np.asarray(rust_result.visit_counts_p2, dtype=np.float64),
            prior_p1=np.asarray(rust_result.prior_p1, dtype=np.float64),
            prior_p2=np.asarray(rust_result.prior_p2, dtype=np.float64),
            total_visits=int(rust_result.total_visits),
        )
