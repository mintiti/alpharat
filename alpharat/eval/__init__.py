"""Evaluation utilities for PyRat agents."""

from alpharat.eval.game import GameResult, play_game
from alpharat.eval.runner import evaluate
from alpharat.eval.tournament import (
    MatchupResult,
    TournamentConfig,
    TournamentResult,
    run_tournament,
)

__all__ = [
    "GameResult",
    "MatchupResult",
    "TournamentConfig",
    "TournamentResult",
    "evaluate",
    "play_game",
    "run_tournament",
]
