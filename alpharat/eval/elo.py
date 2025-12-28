"""Elo rating computation using Bradley-Terry MLE.

This module computes Elo ratings from head-to-head game results using
maximum likelihood estimation. One player is anchored at a fixed rating
to provide a reference point.

Example:
    >>> from alpharat.eval.elo import compute_elo, HeadToHead
    >>> records = [
    ...     HeadToHead("alice", "bob", wins_a=7, wins_b=3),
    ...     HeadToHead("bob", "carol", wins_a=6, wins_b=4),
    ... ]
    >>> result = compute_elo(records, anchor="carol", anchor_elo=1000)
    >>> print(result.format_table())
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from alpharat.eval.tournament import TournamentResult


# Standard Elo scaling: 400 points = 10:1 odds
ELO_SCALE = 400.0


@dataclass
class HeadToHead:
    """Win/loss record between two players.

    This is the input format for Elo computation, decoupled from
    TournamentResult so the module works with any game result source.

    Attributes:
        player_a: First player's identifier.
        player_b: Second player's identifier.
        wins_a: Games won by player A.
        wins_b: Games won by player B.
        draws: Games that ended in a draw.
    """

    player_a: str
    player_b: str
    wins_a: int
    wins_b: int
    draws: int = 0


@dataclass
class EloRating:
    """Rating for a single player.

    Attributes:
        name: Player identifier.
        elo: Maximum likelihood Elo rating.
        stderr: Standard error (uncertainty) on the rating.
                None if uncertainty wasn't computed.
    """

    name: str
    elo: float
    stderr: float | None = None


@dataclass
class EloResult:
    """Complete Elo computation result.

    Attributes:
        ratings: List of player ratings, sorted by Elo (highest first).
        anchor: Name of the anchor player (fixed reference).
        anchor_elo: The Elo value the anchor was fixed to.
    """

    ratings: list[EloRating]
    anchor: str
    anchor_elo: float

    def get(self, name: str) -> EloRating | None:
        """Get rating for a specific player."""
        for r in self.ratings:
            if r.name == name:
                return r
        return None

    def elo_difference(self, player_a: str, player_b: str) -> float:
        """Elo difference between two players (A - B)."""
        a = self.get(player_a)
        b = self.get(player_b)
        if a is None:
            raise ValueError(f"Player not found: {player_a}")
        if b is None:
            raise ValueError(f"Player not found: {player_b}")
        return a.elo - b.elo

    def expected_score(self, player_a: str, player_b: str) -> float:
        """Expected score for player A against player B (0.0 to 1.0)."""
        diff = self.elo_difference(player_a, player_b)
        return win_expectancy(diff, 0.0)

    def format_table(self) -> str:
        """Format ratings as a human-readable table."""
        lines = ["Elo Ratings", "=" * 50]
        lines.append(f"{'Rank':<6}{'Player':<25}{'Elo':>10}{'StdErr':>10}")
        lines.append("-" * 50)
        for i, r in enumerate(self.ratings, 1):
            stderr_str = f"+/- {r.stderr:.1f}" if r.stderr else ""
            marker = " *" if r.name == self.anchor else ""
            lines.append(f"{i:<6}{r.name + marker:<25}{r.elo:>10.1f}{stderr_str:>10}")
        lines.append("")
        lines.append(f"* Anchor fixed at {self.anchor_elo}")
        return "\n".join(lines)


def win_expectancy(elo_a: float, elo_b: float) -> float:
    """Expected win probability for player A against player B.

    Uses the logistic formula: P(A wins) = 1 / (1 + 10^((B-A)/400))

    Args:
        elo_a: Elo rating of player A.
        elo_b: Elo rating of player B.

    Returns:
        Probability (0.0 to 1.0) that A beats B.
    """
    return float(1.0 / (1.0 + 10.0 ** ((elo_b - elo_a) / ELO_SCALE)))


def elo_from_winrate(winrate: float, opponent_elo: float) -> float:
    """Infer Elo from observed winrate against known opponent.

    Useful for quick screening: if you beat greedy (1000) at 75%,
    your estimated Elo is ~1190.

    Args:
        winrate: Observed win proportion (0.0 to 1.0, exclusive).
        opponent_elo: Elo of the opponent.

    Returns:
        Implied Elo rating.

    Raises:
        ValueError: If winrate is not in (0, 1).
    """
    if winrate <= 0.0 or winrate >= 1.0:
        raise ValueError(f"winrate must be in (0, 1), got {winrate}")
    # Invert: winrate = 1 / (1 + 10^((opp - self) / 400))
    # 10^((opp - self) / 400) = 1/winrate - 1
    # (opp - self) / 400 = log10(1/winrate - 1)
    # self = opp - 400 * log10(1/winrate - 1)
    return opponent_elo - ELO_SCALE * math.log10(1.0 / winrate - 1.0)


def from_tournament_result(result: TournamentResult) -> list[HeadToHead]:
    """Convert TournamentResult to HeadToHead records.

    Args:
        result: Tournament result from run_tournament().

    Returns:
        List of HeadToHead records, one per matchup.
    """
    records = []
    for matchup in result.matchups:
        records.append(
            HeadToHead(
                player_a=matchup.agent_a,
                player_b=matchup.agent_b,
                wins_a=matchup.wins_a,
                wins_b=matchup.wins_b,
                draws=matchup.draws,
            )
        )
    return records


def compute_elo(
    records: list[HeadToHead],
    anchor: str = "greedy",
    anchor_elo: float = 1000.0,
    *,
    compute_uncertainty: bool = False,
    draw_weight: float = 0.5,
    prior_games: float = 2.0,
    max_iterations: int = 1000,
    tolerance: float = 0.001,
) -> EloResult:
    """Compute Elo ratings from head-to-head records using Bradley-Terry MLE.

    Uses iterative optimization to find maximum likelihood ratings for all
    players simultaneously. The anchor player is fixed at anchor_elo to
    provide a reference point.

    Args:
        records: Win/loss records between player pairs.
        anchor: Player name to anchor at fixed Elo. Must appear in records.
        anchor_elo: Elo value to assign to anchor player.
        compute_uncertainty: If True, compute standard errors from Hessian.
        draw_weight: How to weight draws (0.5 = half win each, default).
        prior_games: Bayesian prior strength. Adds virtual games at 50% winrate
                     toward anchor_elo to regularize extreme ratings.
        max_iterations: Maximum optimization iterations.
        tolerance: Convergence tolerance in Elo units.

    Returns:
        EloResult with ratings for all players.

    Raises:
        ValueError: If anchor not found, records are empty, or graph is disconnected.
    """
    if not records:
        raise ValueError("No game records provided")

    # Extract all players
    players_set: set[str] = set()
    for r in records:
        players_set.add(r.player_a)
        players_set.add(r.player_b)

    players = sorted(players_set)
    n = len(players)

    if n < 2:
        raise ValueError("Need at least 2 players")

    if anchor not in players_set:
        raise ValueError(f"Anchor '{anchor}' not found in records")

    player_idx = {name: i for i, name in enumerate(players)}
    anchor_idx = player_idx[anchor]

    # Build game and win matrices
    # game_matrix[i, j] = total games between i and j
    # win_matrix[i, j] = wins by i against j (draws weighted)
    game_matrix = np.zeros((n, n), dtype=np.float64)
    win_matrix = np.zeros((n, n), dtype=np.float64)

    for r in records:
        i = player_idx[r.player_a]
        j = player_idx[r.player_b]
        total_games = r.wins_a + r.wins_b + r.draws

        game_matrix[i, j] = total_games
        game_matrix[j, i] = total_games

        # Wins with draw weighting
        win_matrix[i, j] = r.wins_a + draw_weight * r.draws
        win_matrix[j, i] = r.wins_b + draw_weight * r.draws

    # Add prior: virtual games at 50% against anchor
    if prior_games > 0:
        for i in range(n):
            if i != anchor_idx:
                game_matrix[i, anchor_idx] += prior_games
                game_matrix[anchor_idx, i] += prior_games
                win_matrix[i, anchor_idx] += prior_games * 0.5
                win_matrix[anchor_idx, i] += prior_games * 0.5

    # Check connectivity
    if not _check_connected(game_matrix):
        raise ValueError("Player graph is disconnected - cannot compute relative ratings")

    # Check each player has games
    games_per_player = game_matrix.sum(axis=1)
    for i, name in enumerate(players):
        if games_per_player[i] == 0:
            raise ValueError(f"Player '{name}' has no games")

    # Optimize ratings
    ratings = _optimize_ratings(
        game_matrix=game_matrix,
        win_matrix=win_matrix,
        anchor_idx=anchor_idx,
        anchor_elo=anchor_elo,
        max_iterations=max_iterations,
        tolerance=tolerance,
    )

    # Compute uncertainty if requested
    stderrs: np.ndarray | None = None
    if compute_uncertainty:
        stderrs = _compute_hessian_uncertainty(
            ratings=ratings,
            game_matrix=game_matrix,
            anchor_idx=anchor_idx,
        )

    # Build result
    elo_ratings = []
    for i, name in enumerate(players):
        elo_ratings.append(
            EloRating(
                name=name,
                elo=ratings[i],
                stderr=stderrs[i] if stderrs is not None else None,
            )
        )

    # Sort by Elo descending
    elo_ratings.sort(key=lambda r: r.elo, reverse=True)

    return EloResult(
        ratings=elo_ratings,
        anchor=anchor,
        anchor_elo=anchor_elo,
    )


def _check_connected(game_matrix: np.ndarray) -> bool:
    """Check if all players are transitively connected via games."""
    n = game_matrix.shape[0]
    if n <= 1:
        return True

    has_games = game_matrix > 0

    # BFS from player 0
    visited = {0}
    queue = [0]
    while queue:
        i = queue.pop(0)
        for j in range(n):
            if j not in visited and has_games[i, j]:
                visited.add(j)
                queue.append(j)

    return len(visited) == int(n)


def _optimize_ratings(
    game_matrix: np.ndarray,
    win_matrix: np.ndarray,
    anchor_idx: int,
    anchor_elo: float,
    max_iterations: int,
    tolerance: float,
) -> np.ndarray:
    """Find MLE ratings via iterative optimization.

    Uses Bradley-Terry update: adjust each player's rating to reduce
    the difference between expected and observed wins.
    """
    n = game_matrix.shape[0]
    ratings = np.full(n, anchor_elo, dtype=np.float64)

    # Scaling factor for gradient step
    # Larger = faster but less stable
    step_scale = 50.0

    for _iteration in range(max_iterations):
        max_change = 0.0

        for i in range(n):
            if i == anchor_idx:
                continue

            # Compute expected wins for player i
            expected_wins = 0.0
            total_games = 0.0

            for j in range(n):
                if i == j or game_matrix[i, j] == 0:
                    continue
                p_win = win_expectancy(ratings[i], ratings[j])
                expected_wins += game_matrix[i, j] * p_win
                total_games += game_matrix[i, j]

            if total_games == 0:
                continue

            # Observed wins
            observed_wins = win_matrix[i, :].sum()

            # Gradient: difference between observed and expected
            diff = (observed_wins - expected_wins) / total_games

            # Adjustment scaled by step_scale
            # The derivative of log-likelihood suggests this scaling
            adjustment = diff * step_scale

            # Apply with damping for stability
            ratings[i] += adjustment
            max_change = max(max_change, abs(adjustment))

        # Re-anchor: shift all ratings so anchor stays at anchor_elo
        shift = anchor_elo - ratings[anchor_idx]
        ratings += shift

        # Check convergence
        if max_change < tolerance:
            break

    return ratings


def _compute_hessian_uncertainty(
    ratings: np.ndarray,
    game_matrix: np.ndarray,
    anchor_idx: int,
) -> np.ndarray:
    """Compute standard errors from Fisher information (Hessian).

    The negative Hessian of log-likelihood is the Fisher information matrix.
    Standard errors are sqrt of diagonal of inverse Fisher.
    """
    n = len(ratings)

    # Build Fisher information matrix
    # For Bradley-Terry: d2L/dR_i dR_j involves p_ij * (1 - p_ij) * games_ij
    fisher = np.zeros((n, n), dtype=np.float64)

    for i in range(n):
        for j in range(n):
            if i == j or game_matrix[i, j] == 0:
                continue

            p_ij = win_expectancy(ratings[i], ratings[j])
            # Second derivative contribution
            # The Hessian of log-likelihood for logistic model
            info = game_matrix[i, j] * p_ij * (1 - p_ij) / (ELO_SCALE * math.log(10) / 400) ** 2

            fisher[i, i] += info
            fisher[j, j] += info
            fisher[i, j] -= info
            fisher[j, i] -= info

    # Anchor is fixed, so remove its row/col for inversion
    # Create reduced matrix without anchor
    mask = np.ones(n, dtype=bool)
    mask[anchor_idx] = False
    reduced_fisher = fisher[mask][:, mask]

    # Invert to get covariance
    try:
        reduced_cov = np.linalg.inv(reduced_fisher)
    except np.linalg.LinAlgError:
        # Singular matrix - return large uncertainty
        return np.full(n, 1000.0)

    # Expand back to full size
    stderrs = np.zeros(n, dtype=np.float64)
    reduced_idx = 0
    for i in range(n):
        if i == anchor_idx:
            stderrs[i] = 0.0  # Anchor has no uncertainty (it's fixed)
        else:
            variance = reduced_cov[reduced_idx, reduced_idx]
            stderrs[i] = math.sqrt(max(0, variance))
            reduced_idx += 1

    return stderrs
