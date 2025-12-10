"""MCTS agent for PyRat games."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import numpy as np

from alpharat.ai.base import Agent
from alpharat.mcts import MCTSConfig  # noqa: TC001
from alpharat.mcts.nash import select_action_from_strategy
from alpharat.mcts.node import MCTSNode
from alpharat.mcts.tree import MCTSTree

if TYPE_CHECKING:
    from pyrat_engine.core.game import PyRat


class MCTSAgent(Agent):
    """Agent that uses MCTS to select actions.

    Creates a fresh search tree each turn (no tree reuse for now).
    Uses uniform priors (no neural network).

    Attributes:
        config: MCTS configuration (variant, simulations, gamma, etc.).
    """

    def __init__(self, config: MCTSConfig) -> None:
        """Initialize MCTS agent.

        Args:
            config: MCTS configuration specifying search variant and parameters.
        """
        self.config = config

    def get_move(self, game: PyRat, player: int) -> int:
        """Select action using MCTS search.

        Args:
            game: Current game state (will be deep-copied, not modified).
            player: Which player we are (1 = Rat, 2 = Python).

        Returns:
            Action index (0-4).
        """
        # Clone game for simulation
        simulator = copy.deepcopy(game)

        # Get current mud state
        p1_mud = simulator.player1_mud_turns
        p2_mud = simulator.player2_mud_turns

        # Dummy priors - tree will overwrite with smart uniform via _init_root_priors()
        dummy = np.ones(5) / 5
        root = MCTSNode(
            game_state=None,
            prior_policy_p1=dummy,
            prior_policy_p2=dummy,
            nn_payout_prediction=np.zeros((5, 5)),
            parent=None,
            p1_mud_turns_remaining=p1_mud,
            p2_mud_turns_remaining=p2_mud,
        )

        # Run search
        tree = MCTSTree(
            game=simulator,
            root=root,
            gamma=self.config.gamma,
            predict_fn=None,
        )
        search = self.config.build(tree)
        result = search.search()

        # Sample from Nash strategy for our player
        # MCTS is P1-centric: policy_p1 is for Rat, policy_p2 is for Python
        policy = result.policy_p1 if player == 1 else result.policy_p2
        return select_action_from_strategy(policy, temperature=1.0)

    @property
    def name(self) -> str:
        if self.config.variant == "prior_sampling":
            return f"PS({self.config.simulations})"
        else:
            return f"PUCT({self.config.simulations},c={self.config.c_puct})"
