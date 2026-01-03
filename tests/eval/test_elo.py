"""Tests for Elo rating computation."""

from __future__ import annotations

import pytest

from alpharat.eval.elo import (
    HeadToHead,
    compute_elo,
    elo_from_winrate,
    from_tournament_result,
    win_expectancy,
)
from alpharat.eval.tournament import MatchupResult, TournamentResult


class TestWinExpectancy:
    """Unit tests for win probability formula."""

    def test_equal_ratings_gives_50_percent(self) -> None:
        assert win_expectancy(1000, 1000) == pytest.approx(0.5)

    def test_400_point_advantage(self) -> None:
        # 400 Elo = 10:1 odds = 10/11 ≈ 90.9% win rate
        assert win_expectancy(1400, 1000) == pytest.approx(10.0 / 11.0, rel=0.01)

    def test_symmetric(self) -> None:
        p_a = win_expectancy(1200, 1000)
        p_b = win_expectancy(1000, 1200)
        assert p_a + p_b == pytest.approx(1.0)

    def test_200_point_advantage(self) -> None:
        # 200 Elo ≈ 76% win rate
        p = win_expectancy(1200, 1000)
        assert 0.75 < p < 0.77

    def test_large_difference(self) -> None:
        # Very large difference should approach 1.0
        p = win_expectancy(2000, 1000)
        assert p > 0.99


class TestEloFromWinrate:
    """Tests for winrate-to-Elo conversion."""

    def test_50_percent_equals_opponent(self) -> None:
        assert elo_from_winrate(0.5, 1000) == pytest.approx(1000)

    def test_75_percent_is_about_190_above(self) -> None:
        # 75% winrate ≈ 3:1 odds ≈ 190 Elo
        elo = elo_from_winrate(0.75, 1000)
        assert 1180 < elo < 1200

    def test_90_percent_is_about_380_above(self) -> None:
        # 90% ≈ 9:1 odds ≈ 380 Elo
        elo = elo_from_winrate(0.90, 1000)
        assert 1350 < elo < 1400

    def test_inverse_of_win_expectancy(self) -> None:
        # Round-trip: elo -> winrate -> elo
        original_elo = 1234
        opponent_elo = 1000
        winrate = win_expectancy(original_elo, opponent_elo)
        recovered_elo = elo_from_winrate(winrate, opponent_elo)
        assert recovered_elo == pytest.approx(original_elo, rel=0.001)

    def test_invalid_winrate_zero(self) -> None:
        with pytest.raises(ValueError, match="winrate must be in"):
            elo_from_winrate(0.0, 1000)

    def test_invalid_winrate_one(self) -> None:
        with pytest.raises(ValueError, match="winrate must be in"):
            elo_from_winrate(1.0, 1000)


class TestComputeElo:
    """Integration tests for full Elo computation."""

    def test_two_players_75_percent(self) -> None:
        """Player A beats B 75% -> A is higher rated."""
        records = [HeadToHead("A", "B", wins_a=75, wins_b=25)]
        result = compute_elo(records, anchor="B", anchor_elo=1000, prior_games=0)

        rating_b = result.get("B")
        rating_a = result.get("A")
        assert rating_b is not None
        assert rating_a is not None
        assert rating_b.elo == pytest.approx(1000, abs=1)
        assert rating_a.elo > rating_b.elo
        # Roughly 190 Elo difference for 75% winrate
        assert 1150 < rating_a.elo < 1250

    def test_two_players_equal(self) -> None:
        """50-50 split should give equal ratings."""
        records = [HeadToHead("A", "B", wins_a=50, wins_b=50)]
        result = compute_elo(records, anchor="B", anchor_elo=1000, prior_games=0)

        rating_a = result.get("A")
        assert rating_a is not None
        assert rating_a.elo == pytest.approx(1000, abs=10)

    def test_three_player_transitive(self) -> None:
        """A > B > C should produce ordered ratings."""
        records = [
            HeadToHead("A", "B", wins_a=70, wins_b=30),
            HeadToHead("B", "C", wins_a=70, wins_b=30),
            HeadToHead("A", "C", wins_a=85, wins_b=15),
        ]
        result = compute_elo(records, anchor="C", anchor_elo=800, prior_games=0)

        elo_a = result.get("A")
        elo_b = result.get("B")
        elo_c = result.get("C")

        assert elo_a is not None and elo_b is not None and elo_c is not None
        assert elo_a.elo > elo_b.elo > elo_c.elo

    def test_rock_paper_scissors_equal_ratings(self) -> None:
        """Circular dominance should give roughly equal ratings."""
        records = [
            HeadToHead("Rock", "Scissors", wins_a=100, wins_b=0),
            HeadToHead("Scissors", "Paper", wins_a=100, wins_b=0),
            HeadToHead("Paper", "Rock", wins_a=100, wins_b=0),
        ]
        result = compute_elo(records, anchor="Rock", anchor_elo=1000, prior_games=0)

        # All should be close to 1000 (circular dominance cancels out)
        for name in ["Rock", "Paper", "Scissors"]:
            rating = result.get(name)
            assert rating is not None
            assert abs(rating.elo - 1000) < 100

    def test_draws_handled(self) -> None:
        """Draws count as 0.5 wins each."""
        # All draws = equal strength
        records = [HeadToHead("A", "B", wins_a=0, wins_b=0, draws=100)]
        result = compute_elo(records, anchor="B", anchor_elo=1000, prior_games=0)

        rating_a = result.get("A")
        assert rating_a is not None
        assert rating_a.elo == pytest.approx(1000, abs=10)

    def test_draws_plus_wins(self) -> None:
        """Mix of draws and wins."""
        # A wins 60, draws 20, loses 20 = 70 points out of 100 = 70%
        records = [HeadToHead("A", "B", wins_a=60, wins_b=20, draws=20)]
        result = compute_elo(records, anchor="B", anchor_elo=1000, prior_games=0)

        # 70% winrate ≈ 150 Elo difference
        rating_a = result.get("A")
        assert rating_a is not None
        assert rating_a.elo > 1100

    def test_anchor_not_found_raises(self) -> None:
        records = [HeadToHead("A", "B", wins_a=50, wins_b=50)]
        with pytest.raises(ValueError, match="Anchor"):
            compute_elo(records, anchor="C")

    def test_empty_records_raises(self) -> None:
        with pytest.raises(ValueError, match="No game"):
            compute_elo([])

    def test_single_player_raises(self) -> None:
        # This would require a self-play record which doesn't make sense
        # The module should detect this via player count
        records = [HeadToHead("A", "A", wins_a=50, wins_b=50)]
        with pytest.raises(ValueError, match="at least 2"):
            compute_elo(records, anchor="A")

    def test_prior_games_regularization(self) -> None:
        """Prior games should pull extreme ratings toward anchor."""
        # A wins everything - without prior, rating goes very high
        records = [HeadToHead("A", "B", wins_a=10, wins_b=0)]

        result_no_prior = compute_elo(records, anchor="B", anchor_elo=1000, prior_games=0)
        result_with_prior = compute_elo(records, anchor="B", anchor_elo=1000, prior_games=5)

        # With prior, A's rating should be lower (pulled toward anchor)
        rating_with_prior = result_with_prior.get("A")
        rating_no_prior = result_no_prior.get("A")
        assert rating_with_prior is not None
        assert rating_no_prior is not None
        assert rating_with_prior.elo < rating_no_prior.elo

    def test_uncertainty_computation(self) -> None:
        """Standard errors should be computed when requested."""
        records = [HeadToHead("A", "B", wins_a=70, wins_b=30)]
        result = compute_elo(
            records, anchor="B", anchor_elo=1000, compute_uncertainty=True, prior_games=0
        )

        rating_a = result.get("A")
        rating_b = result.get("B")
        assert rating_a is not None
        assert rating_a.stderr is not None
        assert rating_a.stderr > 0

        # Anchor has 0 uncertainty (it's fixed)
        assert rating_b is not None
        assert rating_b.stderr == 0

    def test_more_games_less_uncertainty(self) -> None:
        """Standard errors should be smaller with more games."""
        few_games = [HeadToHead("A", "B", wins_a=7, wins_b=3)]
        many_games = [HeadToHead("A", "B", wins_a=70, wins_b=30)]

        result_few = compute_elo(
            few_games, anchor="B", anchor_elo=1000, compute_uncertainty=True, prior_games=0
        )
        result_many = compute_elo(
            many_games, anchor="B", anchor_elo=1000, compute_uncertainty=True, prior_games=0
        )

        rating_few = result_few.get("A")
        rating_many = result_many.get("A")
        assert rating_few is not None and rating_few.stderr is not None
        assert rating_many is not None and rating_many.stderr is not None
        assert rating_few.stderr > rating_many.stderr

    def test_uncertainty_magnitude_is_reasonable(self) -> None:
        """Standard error should match theoretical Fisher information.

        For 100 games at 50%, stderr ≈ 400 / (ln(10) * sqrt(n * p(1-p)))
        which is about 34.7 Elo. We allow some tolerance for matrix effects.
        """
        import math

        records = [HeadToHead("A", "B", wins_a=50, wins_b=50)]
        result = compute_elo(
            records, anchor="B", anchor_elo=1000, compute_uncertainty=True, prior_games=0
        )

        rating_a = result.get("A")
        assert rating_a is not None and rating_a.stderr is not None

        # Theoretical stderr for single matchup: 1 / sqrt(n * p(1-p) * (ln10/400)^2)
        n, p = 100, 0.5
        theoretical = 1 / math.sqrt(n * p * (1 - p) * (math.log(10) / 400) ** 2)

        # Should be within 50% of theoretical (matrix inversion can differ slightly)
        assert 0.5 * theoretical < rating_a.stderr < 1.5 * theoretical, (
            f"stderr {rating_a.stderr:.1f} not close to theoretical {theoretical:.1f}"
        )

    def test_ratings_sorted_descending(self) -> None:
        """Ratings should be sorted by Elo, highest first."""
        records = [
            HeadToHead("A", "B", wins_a=80, wins_b=20),
            HeadToHead("B", "C", wins_a=80, wins_b=20),
        ]
        result = compute_elo(records, anchor="C", anchor_elo=1000, prior_games=0)

        elos = [r.elo for r in result.ratings]
        assert elos == sorted(elos, reverse=True)


class TestEloResult:
    """Tests for EloResult utility methods."""

    def test_get_existing_player(self) -> None:
        records = [HeadToHead("A", "B", wins_a=50, wins_b=50)]
        result = compute_elo(records, anchor="B", anchor_elo=1000, prior_games=0)

        assert result.get("A") is not None
        assert result.get("B") is not None

    def test_get_nonexistent_player(self) -> None:
        records = [HeadToHead("A", "B", wins_a=50, wins_b=50)]
        result = compute_elo(records, anchor="B", anchor_elo=1000, prior_games=0)

        assert result.get("C") is None

    def test_elo_difference(self) -> None:
        records = [HeadToHead("A", "B", wins_a=75, wins_b=25)]
        result = compute_elo(records, anchor="B", anchor_elo=1000, prior_games=0)

        diff = result.elo_difference("A", "B")
        assert diff > 0  # A is stronger

        # Symmetric
        assert result.elo_difference("B", "A") == -diff

    def test_elo_difference_unknown_player(self) -> None:
        records = [HeadToHead("A", "B", wins_a=50, wins_b=50)]
        result = compute_elo(records, anchor="B", anchor_elo=1000, prior_games=0)

        with pytest.raises(ValueError, match="not found"):
            result.elo_difference("A", "C")

    def test_expected_score(self) -> None:
        records = [HeadToHead("A", "B", wins_a=75, wins_b=25)]
        result = compute_elo(records, anchor="B", anchor_elo=1000, prior_games=0)

        # A should have > 50% expected score against B
        assert result.expected_score("A", "B") > 0.5
        assert result.expected_score("B", "A") < 0.5

        # Symmetric
        assert result.expected_score("A", "B") + result.expected_score("B", "A") == pytest.approx(
            1.0
        )

    def test_format_table(self) -> None:
        records = [HeadToHead("Alice", "Bob", wins_a=60, wins_b=40)]
        result = compute_elo(records, anchor="Bob", anchor_elo=1000, prior_games=0)

        table = result.format_table()
        assert "Elo Ratings" in table
        assert "Alice" in table
        assert "Bob" in table
        assert "* Anchor" in table or "Bob *" in table


class TestFromTournamentResult:
    """Tests for TournamentResult conversion."""

    def test_converts_matchups(self) -> None:
        matchups = [
            MatchupResult(
                agent_a="A",
                agent_b="B",
                wins_a=5,
                draws=2,
                wins_b=3,
                avg_cheese_a=2.5,
                avg_cheese_b=2.0,
            ),
            MatchupResult(
                agent_a="A",
                agent_b="C",
                wins_a=7,
                draws=1,
                wins_b=2,
                avg_cheese_a=3.0,
                avg_cheese_b=1.5,
            ),
        ]
        tournament = TournamentResult(matchups=matchups, agent_names=["A", "B", "C"])

        records = from_tournament_result(tournament)

        assert len(records) == 2

        # Find A vs B record
        ab_record = next(r for r in records if r.player_a == "A" and r.player_b == "B")
        assert ab_record.wins_a == 5
        assert ab_record.wins_b == 3
        assert ab_record.draws == 2

    def test_empty_tournament(self) -> None:
        tournament = TournamentResult(matchups=[], agent_names=[])
        records = from_tournament_result(tournament)
        assert records == []

    def test_integration_with_compute_elo(self) -> None:
        """Full integration: tournament result -> elo ratings."""
        matchups = [
            MatchupResult("greedy", "random", 8, 1, 1, 3.0, 1.0),
            MatchupResult("mcts", "greedy", 7, 2, 1, 3.5, 2.0),
            MatchupResult("mcts", "random", 9, 1, 0, 4.0, 0.5),
        ]
        tournament = TournamentResult(matchups=matchups, agent_names=["greedy", "mcts", "random"])

        records = from_tournament_result(tournament)
        result = compute_elo(records, anchor="greedy", anchor_elo=1000)

        # mcts should be highest, random lowest
        rating_mcts = result.get("mcts")
        rating_greedy = result.get("greedy")
        rating_random = result.get("random")
        assert rating_mcts is not None
        assert rating_greedy is not None
        assert rating_random is not None
        assert rating_mcts.elo > rating_greedy.elo > rating_random.elo
