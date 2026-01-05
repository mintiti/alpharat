"""Game recording for MCTS training data generation."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from types import TracebackType

    from alpharat.mcts.search import SearchResult

import numpy as np

from alpharat.data.maze import build_maze_array
from alpharat.data.types import CheeseOutcome, GameData, PositionData


class GameRecorder:
    """Accumulates game data turn-by-turn for training data generation.

    Records game state and MCTS outputs at each position during gameplay,
    then saves as a single compressed npz file per game.

    Usage as context manager:
        with GameRecorder(game, output_dir, width=15, height=11) as recorder:
            while not is_terminal(game):
                result = mcts.search()
                recorder.record_position(
                    game=game,
                    search_result=result,
                    prior_p1=root.prior_policy_p1,
                    prior_p2=root.prior_policy_p2,
                    visit_counts=root.action_visits,
                )
                game.make_move(...)
        # auto-finalize and save on exit

    Attributes:
        width: Maze width.
        height: Maze height.
        data: Accumulated game data (None until context entered).
        saved_path: Path to saved file (set after successful exit).
    """

    def __init__(
        self,
        game: Any,
        output_dir: Path | str,
        width: int,
        height: int,
        *,
        compress: bool = True,
    ) -> None:
        """Initialize recorder.

        Args:
            game: PyRat game instance at turn 0.
            output_dir: Directory to save the npz file.
            width: Maze width.
            height: Maze height.
            compress: If True, use savez_compressed (smaller files).
        """
        self._game = game
        self._output_dir = Path(output_dir)
        self.width = width
        self.height = height
        self._compress = compress
        self.data: GameData | None = None
        self.saved_path: Path | None = None

    def __enter__(self) -> GameRecorder:
        """Enter context: capture initial game state."""
        if self._game.turn != 0:
            raise RuntimeError(
                f"GameRecorder must be entered at turn 0, but game is at turn {self._game.turn}"
            )

        if not self._output_dir.exists():
            raise ValueError(f"Output directory does not exist: {self._output_dir}")

        maze = build_maze_array(self._game, self.width, self.height)
        initial_cheese = self._cheese_to_mask([(c.x, c.y) for c in self._game.cheese_positions()])

        self.data = GameData(
            maze=maze,
            initial_cheese=initial_cheese,
            max_turns=self._game.max_turns,
            width=self.width,
            height=self.height,
        )
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit context: finalize and save if no exception occurred."""
        if exc_type is not None:
            # Exception occurred, don't save incomplete data
            return

        if self.data is None or not self.data.positions:
            # No data recorded, nothing to save
            return

        self._finalize()
        self.saved_path = self._save()

    def record_position(
        self,
        game: Any,
        search_result: SearchResult,
        prior_p1: np.ndarray,
        prior_p2: np.ndarray,
        visit_counts: np.ndarray,
        action_p1: int,
        action_p2: int,
    ) -> None:
        """Record data for the current position.

        Must be called BEFORE making the move. The game state should match
        the state that MCTS searched from.

        Args:
            game: Current game state.
            search_result: MCTS search output (payout_matrix, policy_p1, policy_p2).
            prior_p1: Neural network's policy prior for player 1.
            prior_p2: Neural network's policy prior for player 2.
            visit_counts: MCTS visit counts for action pairs.
            action_p1: Action taken by player 1 (0-4).
            action_p2: Action taken by player 2 (0-4).

        Raises:
            RuntimeError: If not inside context manager.
        """
        if self.data is None:
            raise RuntimeError("record_position must be called inside context manager")

        position = PositionData(
            p1_pos=(game.player1_position.x, game.player1_position.y),
            p2_pos=(game.player2_position.x, game.player2_position.y),
            p1_score=float(game.player1_score),
            p2_score=float(game.player2_score),
            p1_mud=int(game.player1_mud_turns),
            p2_mud=int(game.player2_mud_turns),
            cheese_positions=[(c.x, c.y) for c in game.cheese_positions()],
            turn=int(game.turn),
            payout_matrix=search_result.payout_matrix.copy(),
            visit_counts=visit_counts.copy(),
            prior_p1=prior_p1.copy(),
            prior_p2=prior_p2.copy(),
            policy_p1=search_result.policy_p1.copy(),
            policy_p2=search_result.policy_p2.copy(),
            action_p1=action_p1,
            action_p2=action_p2,
        )
        self.data.positions.append(position)

    def _finalize(self) -> None:
        """Mark game as complete and capture final scores."""
        assert self.data is not None

        p1_score = float(self._game.player1_score)
        p2_score = float(self._game.player2_score)

        if p1_score > p2_score:
            result = 1
        elif p2_score > p1_score:
            result = 2
        else:
            result = 0

        self.data.result = result
        self.data.final_p1_score = p1_score
        self.data.final_p2_score = p2_score
        self.data.cheese_outcomes = self._compute_cheese_outcomes()

    def _compute_cheese_outcomes(self) -> np.ndarray:
        """Compute per-cheese outcomes from position history.

        Tracks when each cheese disappears and determines who collected it
        by checking player positions after the move.

        Returns:
            int8[H, W] array with CheeseOutcome values.
        """
        assert self.data is not None
        outcomes = np.full((self.height, self.width), CheeseOutcome.UNCOLLECTED, dtype=np.int8)

        positions = self.data.positions
        n = len(positions)

        for i in range(n):
            current_cheese = set(positions[i].cheese_positions)

            if i + 1 < n:
                next_cheese = set(positions[i + 1].cheese_positions)
                next_p1_pos = positions[i + 1].p1_pos
                next_p2_pos = positions[i + 1].p2_pos
            else:
                # Last recorded position - use final game state
                final_cheese = {(c.x, c.y) for c in self._game.cheese_positions()}
                next_cheese = final_cheese
                next_p1_pos = (
                    self._game.player1_position.x,
                    self._game.player1_position.y,
                )
                next_p2_pos = (
                    self._game.player2_position.x,
                    self._game.player2_position.y,
                )

            collected = current_cheese - next_cheese

            for x, y in collected:
                p1_there = next_p1_pos == (x, y)
                p2_there = next_p2_pos == (x, y)

                if p1_there and p2_there:
                    outcomes[y, x] = CheeseOutcome.SIMULTANEOUS
                elif p1_there:
                    outcomes[y, x] = CheeseOutcome.P1_WIN
                elif p2_there:
                    outcomes[y, x] = CheeseOutcome.P2_WIN

        return outcomes

    def _save(self) -> Path:
        """Save recorded game to disk."""
        self._validate()

        filename = f"{uuid.uuid4()}.npz"
        path = self._output_dir / filename

        arrays = self._build_arrays()

        if self._compress:
            np.savez_compressed(str(path), **arrays)  # type: ignore[arg-type]
        else:
            np.savez(str(path), **arrays)  # type: ignore[arg-type]

        return path

    def _validate(self) -> None:
        """Validate data consistency before saving.

        Raises:
            ValueError: If data is inconsistent or incomplete.
        """
        assert self.data is not None

        if self.data.result not in (0, 1, 2):
            raise ValueError(f"Invalid result: {self.data.result}")

        for i, pos in enumerate(self.data.positions):
            if pos.payout_matrix.shape != (2, 5, 5):
                raise ValueError(
                    f"Position {i}: invalid payout_matrix shape {pos.payout_matrix.shape}"
                )
            if pos.visit_counts.shape != (5, 5):
                raise ValueError(
                    f"Position {i}: invalid visit_counts shape {pos.visit_counts.shape}"
                )
            if pos.prior_p1.shape != (5,):
                raise ValueError(f"Position {i}: invalid prior_p1 shape {pos.prior_p1.shape}")
            if pos.prior_p2.shape != (5,):
                raise ValueError(f"Position {i}: invalid prior_p2 shape {pos.prior_p2.shape}")
            if pos.policy_p1.shape != (5,):
                raise ValueError(f"Position {i}: invalid policy_p1 shape {pos.policy_p1.shape}")
            if pos.policy_p2.shape != (5,):
                raise ValueError(f"Position {i}: invalid policy_p2 shape {pos.policy_p2.shape}")
            if not (0 <= pos.action_p1 <= 4):
                raise ValueError(f"Position {i}: invalid action_p1 {pos.action_p1}")
            if not (0 <= pos.action_p2 <= 4):
                raise ValueError(f"Position {i}: invalid action_p2 {pos.action_p2}")

    def _build_arrays(self) -> dict[str, np.ndarray]:
        """Convert accumulated data to numpy arrays for saving.

        Returns:
            Dictionary of arrays matching the data format spec.
        """
        assert self.data is not None
        assert self.data.cheese_outcomes is not None
        n = len(self.data.positions)
        h, w = self.height, self.width

        # Position-level arrays
        p1_pos = np.zeros((n, 2), dtype=np.int8)
        p2_pos = np.zeros((n, 2), dtype=np.int8)
        p1_score = np.zeros(n, dtype=np.float32)
        p2_score = np.zeros(n, dtype=np.float32)
        p1_mud = np.zeros(n, dtype=np.int8)
        p2_mud = np.zeros(n, dtype=np.int8)
        cheese_mask = np.zeros((n, h, w), dtype=bool)
        turn = np.zeros(n, dtype=np.int16)
        payout_matrix = np.zeros((n, 2, 5, 5), dtype=np.float32)
        visit_counts = np.zeros((n, 5, 5), dtype=np.int32)
        prior_p1 = np.zeros((n, 5), dtype=np.float32)
        prior_p2 = np.zeros((n, 5), dtype=np.float32)
        policy_p1 = np.zeros((n, 5), dtype=np.float32)
        policy_p2 = np.zeros((n, 5), dtype=np.float32)
        action_p1 = np.zeros(n, dtype=np.int8)
        action_p2 = np.zeros(n, dtype=np.int8)

        for i, pos in enumerate(self.data.positions):
            p1_pos[i] = pos.p1_pos
            p2_pos[i] = pos.p2_pos
            p1_score[i] = pos.p1_score
            p2_score[i] = pos.p2_score
            p1_mud[i] = pos.p1_mud
            p2_mud[i] = pos.p2_mud
            cheese_mask[i] = self._cheese_to_mask(pos.cheese_positions)
            turn[i] = pos.turn
            payout_matrix[i] = pos.payout_matrix.astype(np.float32)
            visit_counts[i] = pos.visit_counts.astype(np.int32)
            prior_p1[i] = pos.prior_p1.astype(np.float32)
            prior_p2[i] = pos.prior_p2.astype(np.float32)
            policy_p1[i] = pos.policy_p1.astype(np.float32)
            policy_p2[i] = pos.policy_p2.astype(np.float32)
            action_p1[i] = np.int8(pos.action_p1)
            action_p2[i] = np.int8(pos.action_p2)

        return {
            # Game-level
            "maze": self.data.maze,
            "initial_cheese": self.data.initial_cheese,
            "cheese_outcomes": self.data.cheese_outcomes,
            "max_turns": np.array(self.data.max_turns, dtype=np.int16),
            "result": np.array(self.data.result, dtype=np.int8),
            "final_p1_score": np.array(self.data.final_p1_score, dtype=np.float32),
            "final_p2_score": np.array(self.data.final_p2_score, dtype=np.float32),
            "num_positions": np.array(n, dtype=np.int32),
            # Position-level
            "p1_pos": p1_pos,
            "p2_pos": p2_pos,
            "p1_score": p1_score,
            "p2_score": p2_score,
            "p1_mud": p1_mud,
            "p2_mud": p2_mud,
            "cheese_mask": cheese_mask,
            "turn": turn,
            "payout_matrix": payout_matrix,
            "visit_counts": visit_counts,
            "prior_p1": prior_p1,
            "prior_p2": prior_p2,
            "policy_p1": policy_p1,
            "policy_p2": policy_p2,
            "action_p1": action_p1,
            "action_p2": action_p2,
        }

    def _cheese_to_mask(self, cheese_positions: list[tuple[int, int]]) -> np.ndarray:
        """Convert cheese position list to bool[H, W] mask.

        Args:
            cheese_positions: List of (x, y) cheese positions.

        Returns:
            Boolean array of shape (height, width).
        """
        mask = np.zeros((self.height, self.width), dtype=bool)
        for x, y in cheese_positions:
            mask[y, x] = True
        return mask
