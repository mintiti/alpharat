"""Evaluation utilities for PyRat agents."""

from alpharat.eval.benchmark import (
    BenchmarkConfig,
    build_benchmark_tournament,
    build_standard_agents,
    get_game_config_from_checkpoint,
    print_benchmark_results,
)
from alpharat.eval.elo import (
    EloRating,
    EloResult,
    HeadToHead,
    compute_elo,
    elo_from_winrate,
    from_tournament_result,
    win_expectancy,
)
from alpharat.eval.game import GameResult, play_game
from alpharat.eval.runner import evaluate
from alpharat.eval.tournament import (
    MatchupResult,
    TournamentConfig,
    TournamentResult,
    run_tournament,
)

__all__ = [
    # Benchmark utilities
    "BenchmarkConfig",
    "build_benchmark_tournament",
    "build_standard_agents",
    "get_game_config_from_checkpoint",
    "print_benchmark_results",
    # Elo rating
    "EloRating",
    "EloResult",
    "HeadToHead",
    "compute_elo",
    "elo_from_winrate",
    "from_tournament_result",
    "win_expectancy",
    # Game execution
    "GameResult",
    "MatchupResult",
    "TournamentConfig",
    "TournamentResult",
    "evaluate",
    "play_game",
    "run_tournament",
]
