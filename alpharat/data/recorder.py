"""Game recording for MCTS training data generation."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from types import TracebackType

    from alpharat.mcts.decoupled_puct import SearchResult

import numpy as np

from alpharat.data.maze import build_maze_array
from alpharat.data.types import CheeseOutcome, GameData, GameFileKey, PositionData


def _cheese_to_mask(cheese_positions: list[tuple[int, int]], height: int, width: int) -> np.ndarray:
    """Convert cheese position list to bool[H, W] mask.

    Args:
        cheese_positions: List of (x, y) cheese positions.
        height: Maze height.
        width: Maze width.

    Returns:
        Boolean array of shape (height, width).
    """
    mask = np.zeros((height, width), dtype=bool)
    for x, y in cheese_positions:
        mask[y, x] = True
    return mask


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
                    prior_p1=tree.root.prior_policy_p1,
                    prior_p2=tree.root.prior_policy_p2,
                    visit_counts_p1=result.visit_counts_p1,
                    visit_counts_p2=result.visit_counts_p2,
                    action_p1=a1,
                    action_p2=a2,
                )
                game.make_move(a1, a2)
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
        auto_save: bool = True,
    ) -> None:
        """Initialize recorder.

        Args:
            game: PyRat game instance at turn 0.
            output_dir: Directory to save the npz file.
            width: Maze width.
            height: Maze height.
            compress: If True, use savez_compressed (smaller files).
            auto_save: If True, save to disk on context exit. If False,
                finalize but don't save — access self.data after exit.
        """
        self._game = game
        self._output_dir = Path(output_dir)
        self.width = width
        self.height = height
        self._compress = compress
        self._auto_save = auto_save
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
        """Exit context: finalize and optionally save if no exception occurred."""
        if exc_type is not None:
            # Exception occurred, don't save incomplete data
            return

        if self.data is None or not self.data.positions:
            # No data recorded, nothing to save
            return

        self._finalize()

        if self._auto_save:
            self.saved_path = self._save()

    def record_position(
        self,
        game: Any,
        search_result: SearchResult,
        prior_p1: np.ndarray,
        prior_p2: np.ndarray,
        visit_counts_p1: np.ndarray,
        visit_counts_p2: np.ndarray,
        action_p1: int,
        action_p2: int,
    ) -> None:
        """Record data for the current position.

        Must be called BEFORE making the move. The game state should match
        the state that MCTS searched from.

        Args:
            game: Current game state.
            search_result: MCTS search output (policy_p1, policy_p2, value_p1, value_p2).
            prior_p1: Neural network's policy prior for player 1.
            prior_p2: Neural network's policy prior for player 2.
            visit_counts_p1: Marginal MCTS visit counts for P1 actions.
            visit_counts_p2: Marginal MCTS visit counts for P2 actions.
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
            value_p1=search_result.value_p1,
            value_p2=search_result.value_p2,
            visit_counts_p1=visit_counts_p1.copy(),
            visit_counts_p2=visit_counts_p2.copy(),
            prior_p1=prior_p1.copy(),
            prior_p2=prior_p2.copy(),
            # Save visit-proportional policies (for NN training)
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
            if pos.visit_counts_p1.shape != (5,):
                raise ValueError(
                    f"Position {i}: invalid visit_counts_p1 shape {pos.visit_counts_p1.shape}"
                )
            if pos.visit_counts_p2.shape != (5,):
                raise ValueError(
                    f"Position {i}: invalid visit_counts_p2 shape {pos.visit_counts_p2.shape}"
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
        value_p1 = np.zeros(n, dtype=np.float32)
        value_p2 = np.zeros(n, dtype=np.float32)
        visit_counts_p1 = np.zeros((n, 5), dtype=np.float32)
        visit_counts_p2 = np.zeros((n, 5), dtype=np.float32)
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
            value_p1[i] = pos.value_p1
            value_p2[i] = pos.value_p2
            visit_counts_p1[i] = pos.visit_counts_p1.astype(np.float32)
            visit_counts_p2[i] = pos.visit_counts_p2.astype(np.float32)
            prior_p1[i] = pos.prior_p1.astype(np.float32)
            prior_p2[i] = pos.prior_p2.astype(np.float32)
            policy_p1[i] = pos.policy_p1.astype(np.float32)
            policy_p2[i] = pos.policy_p2.astype(np.float32)
            action_p1[i] = np.int8(pos.action_p1)
            action_p2[i] = np.int8(pos.action_p2)

        k = GameFileKey
        return {
            # Game-level
            k.MAZE: self.data.maze,
            k.INITIAL_CHEESE: self.data.initial_cheese,
            k.CHEESE_OUTCOMES: self.data.cheese_outcomes,
            k.MAX_TURNS: np.array(self.data.max_turns, dtype=np.int16),
            k.RESULT: np.array(self.data.result, dtype=np.int8),
            k.FINAL_P1_SCORE: np.array(self.data.final_p1_score, dtype=np.float32),
            k.FINAL_P2_SCORE: np.array(self.data.final_p2_score, dtype=np.float32),
            k.NUM_POSITIONS: np.array(n, dtype=np.int32),
            # Position-level
            k.P1_POS: p1_pos,
            k.P2_POS: p2_pos,
            k.P1_SCORE: p1_score,
            k.P2_SCORE: p2_score,
            k.P1_MUD: p1_mud,
            k.P2_MUD: p2_mud,
            k.CHEESE_MASK: cheese_mask,
            k.TURN: turn,
            k.VALUE_P1: value_p1,
            k.VALUE_P2: value_p2,
            k.VISIT_COUNTS_P1: visit_counts_p1,
            k.VISIT_COUNTS_P2: visit_counts_p2,
            k.PRIOR_P1: prior_p1,
            k.PRIOR_P2: prior_p2,
            k.POLICY_P1: policy_p1,
            k.POLICY_P2: policy_p2,
            k.ACTION_P1: action_p1,
            k.ACTION_P2: action_p2,
        }

    def _cheese_to_mask(self, cheese_positions: list[tuple[int, int]]) -> np.ndarray:
        """Convert cheese position list to bool[H, W] mask."""
        return _cheese_to_mask(cheese_positions, self.height, self.width)


class GameBundler:
    """Buffers completed games and writes them as bundled .npz files.

    Reduces I/O overhead by writing multiple games to a single file.
    Each bundle contains:
        - game_lengths: int32[K] — positions per game (for slicing)
        - Game-level arrays stacked along first axis
        - Position-level arrays concatenated

    Usage:
        bundler = GameBundler(output_dir, width=15, height=11)
        for game in games:
            with GameRecorder(game, ...) as recorder:
                # ... play and record ...
            bundler.add_game(recorder.data)  # Add finalized game
        bundler.flush()  # Write remaining games

    Thread safety: NOT thread-safe. Use one bundler per worker.
    """

    # Default threshold: ~50MB estimated buffer size before flush
    DEFAULT_THRESHOLD_BYTES = 50 * 1024 * 1024

    def __init__(
        self,
        output_dir: Path | str,
        width: int,
        height: int,
        *,
        threshold_bytes: int = DEFAULT_THRESHOLD_BYTES,
        compress: bool = True,
    ) -> None:
        """Initialize bundler.

        Args:
            output_dir: Directory to write bundle files.
            width: Maze width (for validation).
            height: Maze height (for validation).
            threshold_bytes: Flush when estimated buffer size exceeds this.
            compress: If True, use savez_compressed.
        """
        self._output_dir = Path(output_dir)
        if not self._output_dir.exists():
            raise ValueError(f"Output directory does not exist: {self._output_dir}")

        self.width = width
        self.height = height
        self._threshold_bytes = threshold_bytes
        self._compress = compress

        self._buffer: list[GameData] = []
        self._buffer_bytes = 0
        self._saved_paths: list[Path] = []

    def add_game(self, game_data: GameData) -> Path | None:
        """Add a completed game to the buffer.

        Args:
            game_data: Finalized GameData (must have cheese_outcomes set).

        Returns:
            Path to written bundle if a flush occurred, None otherwise.

        Raises:
            ValueError: If game dimensions don't match or data is incomplete.
        """
        self._validate_game(game_data)

        self._buffer.append(game_data)
        self._buffer_bytes += self._estimate_game_bytes(game_data)

        if self._buffer_bytes >= self._threshold_bytes:
            return self.flush()

        return None

    def flush(self) -> Path | None:
        """Write buffered games to disk and clear buffer.

        Returns:
            Path to written bundle file, or None if buffer was empty.
        """
        if not self._buffer:
            return None

        path = self._write_bundle()
        self._saved_paths.append(path)
        self._buffer = []
        self._buffer_bytes = 0

        return path

    @property
    def saved_paths(self) -> list[Path]:
        """Paths to all bundle files written by this bundler."""
        return list(self._saved_paths)

    @property
    def buffered_games(self) -> int:
        """Number of games currently in buffer."""
        return len(self._buffer)

    def _validate_game(self, game_data: GameData) -> None:
        """Validate game data before adding to buffer."""
        if game_data.width != self.width or game_data.height != self.height:
            raise ValueError(
                f"Game dimensions ({game_data.width}, {game_data.height}) "
                f"don't match bundler ({self.width}, {self.height})"
            )

        if game_data.cheese_outcomes is None:
            raise ValueError("Game must be finalized (cheese_outcomes is None)")

        if not game_data.positions:
            raise ValueError("Game has no positions")

    def _estimate_game_bytes(self, game_data: GameData) -> int:
        """Estimate memory footprint of a game in bytes.

        This is a rough estimate for threshold checking, not exact.
        """
        n = len(game_data.positions)
        h, w = self.height, self.width

        # Game-level arrays
        game_bytes = (
            h * w * 4  # maze
            + h * w  # initial_cheese
            + h * w  # cheese_outcomes
            + 16  # scalars
        )

        # Position-level arrays (per position)
        pos_bytes_per = (
            2
            + 2  # p1_pos, p2_pos
            + 4
            + 4  # scores
            + 1
            + 1  # mud
            + h * w  # cheese_mask
            + 2  # turn
            + 4
            + 4  # value_p1, value_p2
            + 5 * 4
            + 5 * 4  # visit_counts_p1, visit_counts_p2
            + 5 * 4 * 4  # priors and policies
            + 1
            + 1  # actions
        )

        return game_bytes + n * pos_bytes_per

    def _write_bundle(self) -> Path:
        """Write buffered games to a single npz file."""
        filename = f"bundle_{uuid.uuid4()}.npz"
        path = self._output_dir / filename

        arrays = self._build_bundle_arrays()

        if self._compress:
            np.savez_compressed(str(path), **arrays)  # type: ignore[arg-type]
        else:
            np.savez(str(path), **arrays)  # type: ignore[arg-type]

        return path

    def _build_bundle_arrays(self) -> dict[str, np.ndarray]:
        """Build arrays for bundle from buffered games.

        Returns:
            Dictionary of arrays in bundle format.
        """
        k = len(self._buffer)  # Number of games
        h, w = self.height, self.width

        # Compute game lengths and total positions
        game_lengths = np.array([len(g.positions) for g in self._buffer], dtype=np.int32)
        n = int(game_lengths.sum())  # Total positions

        # Game-level arrays (stacked)
        maze = np.zeros((k, h, w, 4), dtype=np.int8)
        initial_cheese = np.zeros((k, h, w), dtype=bool)
        cheese_outcomes = np.zeros((k, h, w), dtype=np.int8)
        max_turns = np.zeros(k, dtype=np.int16)
        result = np.zeros(k, dtype=np.int8)
        final_p1_score = np.zeros(k, dtype=np.float32)
        final_p2_score = np.zeros(k, dtype=np.float32)

        # Position-level arrays (concatenated)
        p1_pos = np.zeros((n, 2), dtype=np.int8)
        p2_pos = np.zeros((n, 2), dtype=np.int8)
        p1_score = np.zeros(n, dtype=np.float32)
        p2_score = np.zeros(n, dtype=np.float32)
        p1_mud = np.zeros(n, dtype=np.int8)
        p2_mud = np.zeros(n, dtype=np.int8)
        cheese_mask = np.zeros((n, h, w), dtype=bool)
        turn = np.zeros(n, dtype=np.int16)
        value_p1 = np.zeros(n, dtype=np.float32)
        value_p2 = np.zeros(n, dtype=np.float32)
        visit_counts_p1 = np.zeros((n, 5), dtype=np.float32)
        visit_counts_p2 = np.zeros((n, 5), dtype=np.float32)
        prior_p1 = np.zeros((n, 5), dtype=np.float32)
        prior_p2 = np.zeros((n, 5), dtype=np.float32)
        policy_p1 = np.zeros((n, 5), dtype=np.float32)
        policy_p2 = np.zeros((n, 5), dtype=np.float32)
        action_p1 = np.zeros(n, dtype=np.int8)
        action_p2 = np.zeros(n, dtype=np.int8)

        pos_offset = 0

        for gi, game in enumerate(self._buffer):
            # Game-level
            maze[gi] = game.maze
            initial_cheese[gi] = game.initial_cheese
            assert game.cheese_outcomes is not None  # Validated in _validate_game
            cheese_outcomes[gi] = game.cheese_outcomes
            max_turns[gi] = game.max_turns
            result[gi] = game.result
            final_p1_score[gi] = game.final_p1_score
            final_p2_score[gi] = game.final_p2_score

            # Position-level
            for pos in game.positions:
                p1_pos[pos_offset] = pos.p1_pos
                p2_pos[pos_offset] = pos.p2_pos
                p1_score[pos_offset] = pos.p1_score
                p2_score[pos_offset] = pos.p2_score
                p1_mud[pos_offset] = pos.p1_mud
                p2_mud[pos_offset] = pos.p2_mud
                cheese_mask[pos_offset] = self._cheese_to_mask(pos.cheese_positions)
                turn[pos_offset] = pos.turn
                value_p1[pos_offset] = pos.value_p1
                value_p2[pos_offset] = pos.value_p2
                visit_counts_p1[pos_offset] = pos.visit_counts_p1.astype(np.float32)
                visit_counts_p2[pos_offset] = pos.visit_counts_p2.astype(np.float32)
                prior_p1[pos_offset] = pos.prior_p1.astype(np.float32)
                prior_p2[pos_offset] = pos.prior_p2.astype(np.float32)
                policy_p1[pos_offset] = pos.policy_p1.astype(np.float32)
                policy_p2[pos_offset] = pos.policy_p2.astype(np.float32)
                action_p1[pos_offset] = np.int8(pos.action_p1)
                action_p2[pos_offset] = np.int8(pos.action_p2)
                pos_offset += 1

        gk = GameFileKey
        return {
            # Bundle metadata
            gk.GAME_LENGTHS: game_lengths,
            # Game-level (stacked)
            gk.MAZE: maze,
            gk.INITIAL_CHEESE: initial_cheese,
            gk.CHEESE_OUTCOMES: cheese_outcomes,
            gk.MAX_TURNS: max_turns,
            gk.RESULT: result,
            gk.FINAL_P1_SCORE: final_p1_score,
            gk.FINAL_P2_SCORE: final_p2_score,
            # Position-level (concatenated)
            gk.P1_POS: p1_pos,
            gk.P2_POS: p2_pos,
            gk.P1_SCORE: p1_score,
            gk.P2_SCORE: p2_score,
            gk.P1_MUD: p1_mud,
            gk.P2_MUD: p2_mud,
            gk.CHEESE_MASK: cheese_mask,
            gk.TURN: turn,
            gk.VALUE_P1: value_p1,
            gk.VALUE_P2: value_p2,
            gk.VISIT_COUNTS_P1: visit_counts_p1,
            gk.VISIT_COUNTS_P2: visit_counts_p2,
            gk.PRIOR_P1: prior_p1,
            gk.PRIOR_P2: prior_p2,
            gk.POLICY_P1: policy_p1,
            gk.POLICY_P2: policy_p2,
            gk.ACTION_P1: action_p1,
            gk.ACTION_P2: action_p2,
        }

    def _cheese_to_mask(self, cheese_positions: list[tuple[int, int]]) -> np.ndarray:
        """Convert cheese position list to bool[H, W] mask."""
        return _cheese_to_mask(cheese_positions, self.height, self.width)
