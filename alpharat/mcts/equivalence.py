"""Action equivalence utilities for MCTS.

This module handles action equivalence — when multiple actions lead to the same
game state due to walls, edges, or mud. Key concepts:

- **Effective action mapping**: Maps each action to what actually happens.
  Blocked actions (walls/edges) map to STAY (action 4).
- **Equivalence class**: All actions with the same effective action (e.g., {0, 4} if UP is blocked).
- **Reduction**: Collapsing a matrix/strategy to only effective actions.
- **Expansion**: Restoring to full action space (effective gets value, others get 0).

The main use case is Nash equilibrium computation:
1. Reduce payout matrix to effective actions only
2. Compute Nash on smaller matrix (unique equilibrium)
3. Expand strategies back (blocked actions get probability 0)

This ensures the NN learns that blocked actions should have 0 probability,
creating an implicit auxiliary task: learning maze topology.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable


def get_effective_actions(effective_map: list[int]) -> list[int]:
    """Return sorted list of unique effective action indices.

    Args:
        effective_map: Maps each action index to its effective action.
                      E.g., [4, 1, 2, 3, 4] means actions 0 and 4 are equivalent.

    Returns:
        Sorted list of unique effective actions.
        E.g., [1, 2, 3, 4] for the example above.
    """
    return sorted(set(effective_map))


def reduce_matrix(
    matrix: np.ndarray,
    p1_effective: list[int],
    p2_effective: list[int],
) -> tuple[np.ndarray, list[int], list[int]]:
    """Reduce a payout matrix to effective actions only.

    Takes the submatrix containing only rows/columns for effective actions.
    Since equivalent actions have identical rows/columns (by the equivalence
    invariant), no information is lost.

    Args:
        matrix: Full payout matrix, either [num_actions_p1, num_actions_p2] or
               [2, num_actions_p1, num_actions_p2] for separate player payoffs.
        p1_effective: Effective action mapping for player 1.
        p2_effective: Effective action mapping for player 2.

    Returns:
        Tuple of:
        - reduced_matrix: Smaller matrix with only effective actions (same dims as input)
        - p1_actions: List of effective actions for P1 (row indices in reduced)
        - p2_actions: List of effective actions for P2 (column indices in reduced)
    """
    p1_actions = get_effective_actions(p1_effective)
    p2_actions = get_effective_actions(p2_effective)

    # Handle both 2D and 3D (bimatrix) formats
    if matrix.ndim == 3:
        # Shape (2, p1_actions, p2_actions) - reduce both player matrices
        reduced = matrix[:, np.ix_(p1_actions, p2_actions)[0], np.ix_(p1_actions, p2_actions)[1]]
    else:
        # Shape (p1_actions, p2_actions) - legacy 2D format
        reduced = matrix[np.ix_(p1_actions, p2_actions)]

    return reduced, p1_actions, p2_actions


def expand_strategy(
    reduced_strategy: np.ndarray,
    effective_actions: list[int],
    effective_map: list[int],
) -> np.ndarray:
    """Expand a reduced strategy to full action space.

    Effective actions get their computed probability.
    Non-effective actions get 0 (they're equivalent to their effective action).

    Args:
        reduced_strategy: Strategy over effective actions only.
        effective_actions: List of effective action indices (same order as reduced_strategy).
        effective_map: Full effective action mapping for this player.

    Returns:
        Full strategy array with 0 for non-effective actions.
    """
    num_actions = len(effective_map)
    full_strategy = np.zeros(num_actions)

    # Map reduced strategy back to effective action positions
    for i, action in enumerate(effective_actions):
        full_strategy[action] = reduced_strategy[i]

    return full_strategy


def reduce_and_expand_nash(
    compute_fn: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]],
    payout_matrix: np.ndarray,
    p1_effective: list[int],
    p2_effective: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Nash equilibrium with equivalence reduction.

    1. Reduce payout matrix to effective actions
    2. Call the provided Nash computation function
    3. Expand strategies back to full action space

    Args:
        compute_fn: Function that computes Nash equilibrium on a matrix.
                   Signature: (matrix) -> (strategy_p1, strategy_p2)
        payout_matrix: Full payout matrix.
        p1_effective: Effective action mapping for player 1.
        p2_effective: Effective action mapping for player 2.

    Returns:
        Tuple of (strategy_p1, strategy_p2) in full action space.
        Non-effective actions have probability 0.
    """
    # Reduce
    reduced_matrix, p1_actions, p2_actions = reduce_matrix(
        payout_matrix, p1_effective, p2_effective
    )

    # Compute Nash on reduced matrix
    reduced_p1, reduced_p2 = compute_fn(reduced_matrix)

    # Expand back
    full_p1 = expand_strategy(reduced_p1, p1_actions, p1_effective)
    full_p2 = expand_strategy(reduced_p2, p2_actions, p2_effective)

    return full_p1, full_p2
