use alpharat_mcts::{run_search, Backend, BackendError, MCTSTree, SearchConfig, SearchResult};
use pyrat::{Coordinates, Direction, GameState};
use rand::distributions::WeightedIndex;
use rand::prelude::Distribution;
use rand::rngs::SmallRng;
use rand::SeedableRng;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering::Relaxed};
use std::time::Instant;

use crate::recording::BundleWriter;

// ---------------------------------------------------------------------------
// SelfPlayError — wraps backend + I/O errors
// ---------------------------------------------------------------------------

/// Error from self-play: either a backend evaluation failure or an I/O error.
#[derive(Debug)]
pub enum SelfPlayError {
    Backend(BackendError),
    Io(io::Error),
}

impl std::fmt::Display for SelfPlayError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Backend(e) => write!(f, "backend error: {e}"),
            Self::Io(e) => write!(f, "I/O error: {e}"),
        }
    }
}

impl std::error::Error for SelfPlayError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Backend(e) => Some(e),
            Self::Io(e) => Some(e),
        }
    }
}

impl From<BackendError> for SelfPlayError {
    fn from(e: BackendError) -> Self {
        Self::Backend(e)
    }
}

impl From<io::Error> for SelfPlayError {
    fn from(e: io::Error) -> Self {
        Self::Io(e)
    }
}

// ---------------------------------------------------------------------------
// Data structures
// ---------------------------------------------------------------------------

/// Game result from P1's perspective.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum GameOutcome {
    Draw = 0,
    P1Win = 1,
    P2Win = 2,
}

/// Per-cheese outcome from P1's perspective.
///
/// Values match Python's `CheeseOutcome` IntEnum exactly.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum CheeseOutcome {
    P1Win = 0,
    Simultaneous = 1,
    Uncollected = 2,
    P2Win = 3,
}

/// One position snapshot from a game.
#[derive(Clone, Debug)]
pub struct PositionRecord {
    pub p1_pos: [u8; 2],
    pub p2_pos: [u8; 2],
    pub p1_score: f32,
    pub p2_score: f32,
    pub p1_mud: u8,
    pub p2_mud: u8,
    pub turn: u16,
    /// H*W, 0/1, y-major (idx = y*w + x).
    pub cheese_mask: Vec<u8>,
    pub value_p1: f32,
    pub value_p2: f32,
    pub visit_counts_p1: [f32; 5],
    pub visit_counts_p2: [f32; 5],
    pub prior_p1: [f32; 5],
    pub prior_p2: [f32; 5],
    pub policy_p1: [f32; 5],
    pub policy_p2: [f32; 5],
    pub action_p1: u8,
    pub action_p2: u8,
}

/// Complete record of one game.
#[derive(Clone, Debug)]
pub struct GameRecord {
    pub width: u8,
    pub height: u8,
    pub max_turns: u16,
    /// H*W*4, i8, matching Python's maze array format.
    pub maze: Vec<i8>,
    /// H*W, snapshotted at turn 0.
    pub initial_cheese: Vec<u8>,
    pub positions: Vec<PositionRecord>,
    pub final_p1_score: f32,
    pub final_p2_score: f32,
    pub result: GameOutcome,
    /// Sum of total_visits across all positions.
    pub total_simulations: u64,
    /// Initial cheese count.
    pub cheese_available: u16,
    /// Index of this game in the self-play batch (for identity, not seeding).
    pub game_index: u32,
    /// H*W, y-major, CheeseOutcome values — who collected each cheese.
    pub cheese_outcomes: Vec<u8>,
    /// Total NN evaluations across all search calls in this game.
    pub total_nn_evals: u64,
    /// Total terminal descents (game-over leaves — free, no NN call).
    pub total_terminals: u64,
    /// Total collision descents (wasted — no backup).
    pub total_collisions: u64,
}

/// Aggregate stats computed from a set of GameRecords.
#[derive(Clone, Debug)]
pub struct SelfPlayStats {
    pub total_games: u32,
    pub total_positions: u64,
    pub total_simulations: u64,
    pub elapsed_secs: f64,
    pub p1_wins: u32,
    pub p2_wins: u32,
    pub draws: u32,
    pub total_cheese_collected: f32,
    pub total_cheese_available: u32,
    pub min_turns: u32,
    pub max_turns: u32,
    /// Total NN evaluations (descents that hit the NN).
    pub total_nn_evals: u64,
    /// Total terminal descents (free — no NN call).
    pub total_terminals: u64,
    /// Total collision descents (wasted — no backup).
    pub total_collisions: u64,
}

impl SelfPlayStats {
    /// Create empty stats for incremental accumulation.
    pub fn new() -> Self {
        Self {
            total_games: 0,
            total_positions: 0,
            total_simulations: 0,
            elapsed_secs: 0.0,
            p1_wins: 0,
            p2_wins: 0,
            draws: 0,
            total_cheese_collected: 0.0,
            total_cheese_available: 0,
            min_turns: u32::MAX,
            max_turns: 0,
            total_nn_evals: 0,
            total_terminals: 0,
            total_collisions: 0,
        }
    }

    /// Incrementally add one game's data to the stats.
    pub fn add_game(&mut self, game: &GameRecord) {
        self.total_games += 1;
        let n = game.positions.len() as u64;
        self.total_positions += n;
        self.total_simulations += game.total_simulations;
        self.total_nn_evals += game.total_nn_evals;
        self.total_terminals += game.total_terminals;
        self.total_collisions += game.total_collisions;
        self.total_cheese_collected += game.final_p1_score + game.final_p2_score;
        self.total_cheese_available += game.cheese_available as u32;

        let turns = n as u32;
        self.min_turns = self.min_turns.min(turns);
        self.max_turns = self.max_turns.max(turns);

        match game.result {
            GameOutcome::P1Win => self.p1_wins += 1,
            GameOutcome::P2Win => self.p2_wins += 1,
            GameOutcome::Draw => self.draws += 1,
        }
    }

    pub fn from_games(games: &[GameRecord], elapsed_secs: f64) -> Self {
        let mut stats = Self::new();
        stats.elapsed_secs = elapsed_secs;
        for g in games {
            stats.add_game(g);
        }

        if games.is_empty() {
            stats.min_turns = 0;
        }

        stats
    }

    pub fn games_per_second(&self) -> f64 {
        if self.elapsed_secs > 0.0 {
            self.total_games as f64 / self.elapsed_secs
        } else {
            0.0
        }
    }

    pub fn positions_per_second(&self) -> f64 {
        if self.elapsed_secs > 0.0 {
            self.total_positions as f64 / self.elapsed_secs
        } else {
            0.0
        }
    }

    pub fn simulations_per_second(&self) -> f64 {
        if self.elapsed_secs > 0.0 {
            self.total_simulations as f64 / self.elapsed_secs
        } else {
            0.0
        }
    }

    pub fn cheese_utilization(&self) -> f64 {
        if self.total_cheese_available > 0 {
            self.total_cheese_collected as f64 / self.total_cheese_available as f64
        } else {
            0.0
        }
    }

    pub fn avg_turns(&self) -> f64 {
        if self.total_games > 0 {
            self.total_positions as f64 / self.total_games as f64
        } else {
            0.0
        }
    }

    pub fn draw_rate(&self) -> f64 {
        if self.total_games > 0 {
            self.draws as f64 / self.total_games as f64
        } else {
            0.0
        }
    }

    /// NN evaluations per second (lc0's EPS).
    pub fn nn_evals_per_second(&self) -> f64 {
        if self.elapsed_secs > 0.0 {
            self.total_nn_evals as f64 / self.elapsed_secs
        } else {
            0.0
        }
    }

    /// Fraction of total simulations that were NN evals.
    pub fn nn_eval_fraction(&self) -> f64 {
        if self.total_simulations > 0 {
            self.total_nn_evals as f64 / self.total_simulations as f64
        } else {
            0.0
        }
    }

    /// Fraction of total simulations that were terminal.
    pub fn terminal_fraction(&self) -> f64 {
        if self.total_simulations > 0 {
            self.total_terminals as f64 / self.total_simulations as f64
        } else {
            0.0
        }
    }

    /// Fraction of total descents that were collisions.
    /// Denominator includes collisions (nn_evals + terminals + collisions).
    pub fn collision_fraction(&self) -> f64 {
        let total_descents =
            self.total_nn_evals + self.total_terminals + self.total_collisions;
        if total_descents > 0 {
            self.total_collisions as f64 / total_descents as f64
        } else {
            0.0
        }
    }
}

/// Result of a self-play session.
#[derive(Clone, Debug)]
#[must_use]
pub struct SelfPlayResult {
    pub games: Vec<GameRecord>,
    pub stats: SelfPlayStats,
}

/// Parameters for self-play games.
#[derive(Clone, Debug)]
pub struct SelfPlayConfig {
    pub n_sims: u32,
    pub batch_size: u32,
    /// Number of worker threads (only used by `run_self_play`).
    pub num_threads: u32,
}

/// Atomic counters for tracking self-play progress from outside.
#[derive(Debug)]
pub struct SelfPlayProgress {
    pub games_completed: AtomicU32,
    pub positions_completed: AtomicU64,
    pub simulations_completed: AtomicU64,
    pub nn_evals_completed: AtomicU64,
}

impl SelfPlayProgress {
    pub fn new() -> Self {
        Self {
            games_completed: AtomicU32::new(0),
            positions_completed: AtomicU64::new(0),
            simulations_completed: AtomicU64::new(0),
            nn_evals_completed: AtomicU64::new(0),
        }
    }
}

impl Default for SelfPlayProgress {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

/// Build i8[H, W, 4] maze array in C-order (y-major).
///
/// Values: -1 = wall/boundary, 1 = normal passage, >=2 = mud cost.
pub fn build_maze_array(game: &GameState) -> Vec<i8> {
    let w = game.width as u8;
    let h = game.height as u8;
    let directions = [
        Direction::Up,
        Direction::Right,
        Direction::Down,
        Direction::Left,
    ];

    let mut maze = vec![0i8; h as usize * w as usize * 4];
    let mut i = 0;
    for y in 0..h {
        for x in 0..w {
            let pos = Coordinates::new(x, y);
            for &dir in &directions {
                if !game.move_table.is_move_valid(pos, dir) {
                    maze[i] = -1;
                } else {
                    let neighbor = dir.apply_to(pos);
                    match game.mud.get(pos, neighbor) {
                        Some(cost) => maze[i] = cost as i8,
                        None => maze[i] = 1,
                    }
                }
                i += 1;
            }
        }
    }
    maze
}

/// Build u8[H*W] cheese mask, y-major (idx = y*w + x).
pub fn build_cheese_mask(game: &GameState) -> Vec<u8> {
    let w = game.width as u8;
    let h = game.height as u8;
    let mut mask = vec![0u8; h as usize * w as usize];
    for y in 0..h {
        for x in 0..w {
            let pos = Coordinates::new(x, y);
            let idx = pos.to_index(w);
            if game.cheese.has_cheese(pos) {
                mask[idx] = 1;
            }
        }
    }
    mask
}

/// Compute per-cell cheese outcomes by diffing consecutive cheese masks.
///
/// For each position, compares current cheese mask to the next one (or final game state).
/// When a cheese disappears, checks which player(s) are at that cell to assign the outcome.
fn compute_cheese_outcomes(
    positions: &[PositionRecord],
    game: &GameState,
    w: u8,
    h: u8,
) -> Vec<u8> {
    let size = h as usize * w as usize;
    let mut outcomes = vec![CheeseOutcome::Uncollected as u8; size];

    let n = positions.len();
    for i in 0..n {
        let current_mask = &positions[i].cheese_mask;

        // Next state: either the next position's snapshot or the final game state.
        let (next_mask_owned, next_p1, next_p2);
        if i + 1 < n {
            next_mask_owned = None;
            let next = &positions[i + 1];
            next_p1 = next.p1_pos;
            next_p2 = next.p2_pos;
        } else {
            next_mask_owned = Some(build_cheese_mask(game));
            next_p1 = [game.player1.current_pos.x, game.player1.current_pos.y];
            next_p2 = [game.player2.current_pos.x, game.player2.current_pos.y];
        }
        let next_mask = match &next_mask_owned {
            Some(m) => m.as_slice(),
            None => positions[i + 1].cheese_mask.as_slice(),
        };

        // Find cells where cheese disappeared (1 → 0).
        for idx in 0..size {
            if current_mask[idx] == 1 && next_mask[idx] == 0 {
                let x = (idx % w as usize) as u8;
                let y = (idx / w as usize) as u8;
                let cell = [x, y];

                let p1_there = next_p1 == cell;
                let p2_there = next_p2 == cell;

                outcomes[idx] = if p1_there && p2_there {
                    CheeseOutcome::Simultaneous as u8
                } else if p1_there {
                    CheeseOutcome::P1Win as u8
                } else if p2_there {
                    CheeseOutcome::P2Win as u8
                } else {
                    // Cheese vanished but neither player is there — shouldn't happen
                    // in normal gameplay, but default to Uncollected.
                    CheeseOutcome::Uncollected as u8
                };
            }
        }
    }

    outcomes
}

/// Sample an action from a policy distribution. Falls back to STAY (4) if all zero.
pub fn sample_action(policy: &[f32; 5], rng: &mut impl rand::Rng) -> u8 {
    match WeightedIndex::new(policy) {
        Ok(dist) => dist.sample(rng) as u8,
        Err(_) => 4, // All zero → STAY
    }
}

// ---------------------------------------------------------------------------
// Per-game loop
// ---------------------------------------------------------------------------

/// Record a position snapshot from the current game state + search result.
fn record_position(
    game: &GameState,
    result: &SearchResult,
    action_p1: u8,
    action_p2: u8,
) -> PositionRecord {
    PositionRecord {
        p1_pos: [game.player1.current_pos.x, game.player1.current_pos.y],
        p2_pos: [game.player2.current_pos.x, game.player2.current_pos.y],
        p1_score: game.player1.score,
        p2_score: game.player2.score,
        p1_mud: game.player1.mud_timer as u8,
        p2_mud: game.player2.mud_timer as u8,
        turn: game.turn as u16,
        cheese_mask: build_cheese_mask(game),
        value_p1: result.value_p1,
        value_p2: result.value_p2,
        visit_counts_p1: result.visit_counts_p1,
        visit_counts_p2: result.visit_counts_p2,
        prior_p1: result.prior_p1,
        prior_p2: result.prior_p2,
        policy_p1: result.policy_p1,
        policy_p2: result.policy_p2,
        action_p1,
        action_p2,
    }
}

/// Play a single game to completion, returning a GameRecord.
pub fn play_game(
    mut game: GameState,
    backend: &dyn Backend,
    search_config: &SearchConfig,
    config: &SelfPlayConfig,
    rng: &mut impl rand::Rng,
    game_index: u32,
) -> Result<GameRecord, BackendError> {
    let w = game.width as u8;
    let h = game.height as u8;

    let maze = build_maze_array(&game);
    let initial_cheese = build_cheese_mask(&game);
    let cheese_available = game.cheese.remaining_cheese() as u16;

    let mut tree = MCTSTree::new(&game);
    let mut positions = Vec::with_capacity(game.max_turns as usize);
    let mut total_simulations: u64 = 0;
    let mut total_nn_evals: u64 = 0;
    let mut total_terminals: u64 = 0;
    let mut total_collisions: u64 = 0;

    while !game.check_game_over() {
        let result = run_search(
            &mut tree,
            &game,
            backend,
            search_config,
            config.n_sims,
            config.batch_size,
            rng,
        )?;
        total_simulations += result.total_visits as u64;
        total_nn_evals += result.nn_evals as u64;
        total_terminals += result.terminals as u64;
        total_collisions += result.collisions as u64;

        let a1 = sample_action(&result.policy_p1, rng);
        let a2 = sample_action(&result.policy_p2, rng);

        positions.push(record_position(&game, &result, a1, a2));

        let d1 = Direction::try_from(a1).expect("valid direction");
        let d2 = Direction::try_from(a2).expect("valid direction");
        let _undo = game.make_move(d1, d2);

        // Tree reuse: advance root to the child matching the played actions.
        // Falls back to fresh tree if the action pair wasn't explored.
        if !tree.advance_root(a1, a2) {
            tree.reinit(&game);
        }
    }

    let final_p1 = game.player1.score;
    let final_p2 = game.player2.score;
    let result = if final_p1 > final_p2 {
        GameOutcome::P1Win
    } else if final_p2 > final_p1 {
        GameOutcome::P2Win
    } else {
        GameOutcome::Draw
    };

    let cheese_outcomes = compute_cheese_outcomes(&positions, &game, w, h);

    Ok(GameRecord {
        width: w,
        height: h,
        max_turns: game.max_turns as u16,
        maze,
        initial_cheese,
        positions,
        final_p1_score: final_p1,
        final_p2_score: final_p2,
        result,
        total_simulations,
        cheese_available,
        game_index,
        cheese_outcomes,
        total_nn_evals,
        total_terminals,
        total_collisions,
    })
}

// ---------------------------------------------------------------------------
// Multi-game orchestration
// ---------------------------------------------------------------------------

/// Work-stealing game loop shared by `run_self_play` and `run_self_play_to_disk`.
///
/// Each call claims games via `next_game`, plays them, updates progress counters,
/// and passes completed records to `on_record`. Only the record destination differs
/// between callers (vec push vs channel send).
fn game_worker_loop<F>(
    games: &[GameState],
    backend: &dyn Backend,
    search_config: &SearchConfig,
    config: &SelfPlayConfig,
    next_game: &AtomicU32,
    progress: Option<&SelfPlayProgress>,
    mut on_record: F,
) -> Result<(), BackendError>
where
    F: FnMut(GameRecord),
{
    let n_games = games.len();
    let mut rng = SmallRng::from_entropy();
    loop {
        let idx = next_game.fetch_add(1, Relaxed) as usize;
        if idx >= n_games {
            break;
        }
        let record = play_game(
            games[idx].clone(),
            backend,
            search_config,
            config,
            &mut rng,
            idx as u32,
        )?;

        if let Some(p) = progress {
            p.positions_completed
                .fetch_add(record.positions.len() as u64, Relaxed);
            p.simulations_completed
                .fetch_add(record.total_simulations, Relaxed);
            p.nn_evals_completed
                .fetch_add(record.total_nn_evals, Relaxed);
            p.games_completed.fetch_add(1, Relaxed);
        }

        on_record(record);
    }
    Ok(())
}

/// Run self-play on a slice of games using `config.num_threads` worker threads.
///
/// Each thread claims games by atomic index, clones + plays them.
/// Each thread uses its own entropy-seeded RNG (no deterministic seeding).
/// Results are sorted by `game_index` before returning.
pub fn run_self_play(
    games: &[GameState],
    backend: &dyn Backend,
    search_config: &SearchConfig,
    config: &SelfPlayConfig,
    progress: Option<&SelfPlayProgress>,
) -> Result<SelfPlayResult, BackendError> {
    let start = Instant::now();
    let next_game = AtomicU32::new(0);

    let all_results: Vec<Result<Vec<GameRecord>, BackendError>> = std::thread::scope(|s| {
        let handles: Vec<_> = (0..config.num_threads)
            .map(|_| {
                s.spawn(|| {
                    let mut local = Vec::new();
                    game_worker_loop(
                        games,
                        backend,
                        search_config,
                        config,
                        &next_game,
                        progress,
                        |record| local.push(record),
                    )?;
                    Ok(local)
                })
            })
            .collect();

        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });

    // Fail fast: return first error
    let mut game_records: Vec<GameRecord> = Vec::new();
    for result in all_results {
        game_records.extend(result?);
    }
    game_records.sort_by_key(|g| g.game_index);

    let elapsed = start.elapsed().as_secs_f64();
    let stats = SelfPlayStats::from_games(&game_records, elapsed);

    Ok(SelfPlayResult {
        games: game_records,
        stats,
    })
}

// ---------------------------------------------------------------------------
// Self-play to disk (writer thread pattern)
// ---------------------------------------------------------------------------

/// Result of `run_self_play_to_disk`.
#[derive(Clone, Debug)]
pub struct SelfPlayToDiskResult {
    pub stats: SelfPlayStats,
    pub written_paths: Vec<PathBuf>,
}

/// Run self-play and stream completed games to disk via a writer thread.
///
/// Game threads play games and send records through a channel. A dedicated
/// writer thread accumulates records and flushes bundles when full.
/// The channel is unbounded — game threads never block on I/O.
pub fn run_self_play_to_disk(
    games: &[GameState],
    backend: &dyn Backend,
    search_config: &SearchConfig,
    config: &SelfPlayConfig,
    output_dir: &Path,
    max_games_per_bundle: usize,
    progress: Option<&SelfPlayProgress>,
) -> Result<SelfPlayToDiskResult, SelfPlayError> {
    if games.is_empty() {
        return Ok(SelfPlayToDiskResult {
            stats: SelfPlayStats::from_games(&[], 0.0),
            written_paths: Vec::new(),
        });
    }

    let start = Instant::now();
    let width = games[0].width as u8;
    let height = games[0].height as u8;

    // Channel for game results (unbounded).
    let (tx, rx) = std::sync::mpsc::channel::<GameRecord>();

    // Writer thread — owns rx and BundleWriter, no borrows.
    let output_dir_owned = output_dir.to_path_buf();
    let writer_handle = std::thread::spawn(move || -> io::Result<(Vec<PathBuf>, SelfPlayStats)> {
        let mut writer = BundleWriter::new(&output_dir_owned, width, height, max_games_per_bundle);
        let mut stats = SelfPlayStats::new();

        while let Ok(record) = rx.recv() {
            stats.add_game(&record);
            writer.add_game(record)?;
        }

        let paths = writer.finish()?;
        Ok((paths, stats))
    });

    // Game threads (scoped — borrow games, backend, etc.)
    // Collect thread results to propagate backend errors.
    let next_game = AtomicU32::new(0);

    let thread_results: Vec<Result<(), BackendError>> = std::thread::scope(|s| {
        let next_game = &next_game;
        let handles: Vec<_> = (0..config.num_threads)
            .map(|_| {
                let tx = tx.clone();
                s.spawn(move || {
                    game_worker_loop(
                        games,
                        backend,
                        search_config,
                        config,
                        next_game,
                        progress,
                        |record| {
                            // If writer errored and rx is dropped, we still finish playing
                            // but the record is lost. That's fine.
                            let _ = tx.send(record);
                        },
                    )
                })
            })
            .collect();

        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });

    // Drop the original sender to close the channel.
    drop(tx);

    // Check for backend errors from game threads.
    for result in thread_results {
        result?;
    }

    // Join writer thread.
    let (written_paths, mut stats) = writer_handle.join().unwrap()?;
    stats.elapsed_secs = start.elapsed().as_secs_f64();
    if stats.total_games == 0 {
        stats.min_turns = 0;
    }

    Ok(SelfPlayToDiskResult {
        stats,
        written_paths,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use alpharat_mcts::SmartUniformBackend;
    use pyrat::game::types::MudMap;
    use std::collections::HashMap;

    const BACKEND: SmartUniformBackend = SmartUniformBackend;

    fn open_5x5(p1: Coordinates, p2: Coordinates, cheese: &[Coordinates]) -> GameState {
        GameState::new_with_config(5, 5, HashMap::new(), MudMap::new(), cheese, p1, p2, 100)
    }

    fn standard_game() -> GameState {
        open_5x5(
            Coordinates::new(0, 0),
            Coordinates::new(4, 4),
            &[
                Coordinates::new(1, 0),
                Coordinates::new(2, 2),
                Coordinates::new(3, 4),
            ],
        )
    }

    fn short_game() -> GameState {
        GameState::new_with_config(
            5,
            5,
            HashMap::new(),
            MudMap::new(),
            &[Coordinates::new(1, 0)],
            Coordinates::new(0, 0),
            Coordinates::new(4, 4),
            5,
        )
    }

    // ---- sample_action ----

    #[test]
    fn sample_action_deterministic() {
        let mut rng = SmallRng::seed_from_u64(0);
        let policy = [1.0, 0.0, 0.0, 0.0, 0.0];
        for _ in 0..20 {
            assert_eq!(sample_action(&policy, &mut rng), 0);
        }
    }

    #[test]
    fn sample_action_zero_fallback() {
        let mut rng = SmallRng::seed_from_u64(0);
        let policy = [0.0, 0.0, 0.0, 0.0, 0.0];
        assert_eq!(sample_action(&policy, &mut rng), 4);
    }

    // ---- build_maze_array ----

    #[test]
    fn build_maze_array_open() {
        let game = open_5x5(
            Coordinates::new(2, 2),
            Coordinates::new(2, 2),
            &[Coordinates::new(0, 0)],
        );
        let maze = build_maze_array(&game);
        assert_eq!(maze.len(), 5 * 5 * 4);

        // Interior cell (2,2): all 4 directions should be passable (=1)
        let idx_base = (2 * 5 + 2) * 4; // y=2, x=2
        for d in 0..4 {
            assert_eq!(maze[idx_base + d], 1, "direction {d} at (2,2) should be 1");
        }

        // Corner (0,0): UP=1, RIGHT=1, DOWN=-1, LEFT=-1
        let corner_base = 0; // y=0, x=0
        assert_eq!(maze[corner_base + 0], 1, "UP at (0,0)"); // UP valid on 5x5
        assert_eq!(maze[corner_base + 1], 1, "RIGHT at (0,0)");
        assert_eq!(maze[corner_base + 2], -1, "DOWN at (0,0)"); // boundary
        assert_eq!(maze[corner_base + 3], -1, "LEFT at (0,0)"); // boundary
    }

    #[test]
    fn build_maze_array_walls() {
        let mut walls = HashMap::new();
        walls.insert(
            Coordinates::new(2, 2),
            vec![Coordinates::new(2, 3)],
        );
        walls.insert(
            Coordinates::new(2, 3),
            vec![Coordinates::new(2, 2)],
        );
        let game = GameState::new_with_config(
            5,
            5,
            walls,
            MudMap::new(),
            &[Coordinates::new(0, 0)],
            Coordinates::new(0, 0),
            Coordinates::new(4, 4),
            100,
        );
        let maze = build_maze_array(&game);

        // Wall between (2,2) and (2,3): UP from (2,2) blocked
        let idx_22 = (2 * 5 + 2) * 4;
        assert_eq!(maze[idx_22 + 0], -1, "UP from (2,2) should be blocked by wall");
        assert_eq!(maze[idx_22 + 1], 1, "RIGHT from (2,2) should be open");

        // DOWN from (2,3) should also be blocked
        let idx_23 = (3 * 5 + 2) * 4;
        assert_eq!(maze[idx_23 + 2], -1, "DOWN from (2,3) should be blocked by wall");
    }

    // ---- build_cheese_mask ----

    #[test]
    fn build_cheese_mask_positions() {
        let cheese = [Coordinates::new(0, 0), Coordinates::new(2, 3), Coordinates::new(4, 4)];
        let game = open_5x5(Coordinates::new(1, 1), Coordinates::new(3, 3), &cheese);
        let mask = build_cheese_mask(&game);
        assert_eq!(mask.len(), 25);

        // Check cheese positions are 1
        for &c in &cheese {
            let idx = c.to_index(5);
            assert_eq!(mask[idx], 1, "cheese at ({},{}) should be 1", c.x, c.y);
        }

        // Check non-cheese positions are 0
        let non_cheese = Coordinates::new(1, 1);
        assert_eq!(mask[non_cheese.to_index(5)], 0);
    }

    // ---- play_game ----

    #[test]
    fn play_game_completes() {
        let game = short_game();
        let search = SearchConfig::default();
        let sp = SelfPlayConfig { n_sims: 16, batch_size: 8, num_threads: 1 };
        let mut rng = SmallRng::seed_from_u64(42);
        let record = play_game(game, &BACKEND, &search, &sp, &mut rng, 0).unwrap();

        assert!(!record.positions.is_empty(), "should have at least one position");
        assert!(record.positions.len() <= 5, "max 5 turns");
        assert!(record.final_p1_score >= 0.0);
        assert!(record.final_p2_score >= 0.0);
        assert!(matches!(record.result, GameOutcome::P1Win | GameOutcome::P2Win | GameOutcome::Draw));
        assert_eq!(record.width, 5);
        assert_eq!(record.height, 5);
        assert_eq!(record.game_index, 0);
    }

    #[test]
    fn play_game_position_fields() {
        let game = short_game();
        let search = SearchConfig::default();
        let sp = SelfPlayConfig { n_sims: 16, batch_size: 8, num_threads: 1 };
        let mut rng = SmallRng::seed_from_u64(42);
        let record = play_game(game, &BACKEND, &search, &sp, &mut rng, 0).unwrap();

        let first = &record.positions[0];
        assert_eq!(first.p1_pos, [0, 0], "P1 starts at (0,0)");
        assert_eq!(first.p2_pos, [4, 4], "P2 starts at (4,4)");
        assert_eq!(first.turn, 0);
        assert_eq!(first.p1_score, 0.0);
        assert_eq!(first.p2_score, 0.0);

        // Policy should sum to ~1
        let sum_p1: f32 = first.policy_p1.iter().sum();
        let sum_p2: f32 = first.policy_p2.iter().sum();
        assert!((sum_p1 - 1.0).abs() < 0.01, "P1 policy sum: {sum_p1}");
        assert!((sum_p2 - 1.0).abs() < 0.01, "P2 policy sum: {sum_p2}");
    }

    #[test]
    fn play_game_simulations_tracked() {
        let game = short_game();
        let search = SearchConfig::default();
        let sp = SelfPlayConfig { n_sims: 32, batch_size: 8, num_threads: 1 };
        let mut rng = SmallRng::seed_from_u64(42);
        let record = play_game(game, &BACKEND, &search, &sp, &mut rng, 0).unwrap();

        assert!(record.total_simulations > 0);
        // Should be approximately n_sims * num_positions
        let expected_approx = sp.n_sims as u64 * record.positions.len() as u64;
        // Allow some variance (terminal nodes, etc.)
        assert!(
            record.total_simulations >= expected_approx / 2,
            "total_simulations={} expected ~{}",
            record.total_simulations,
            expected_approx
        );
    }

    #[test]
    fn play_game_maze_and_cheese_recorded() {
        let game = standard_game();
        let search = SearchConfig::default();
        let sp = SelfPlayConfig { n_sims: 8, batch_size: 8, num_threads: 1 };
        let mut rng = SmallRng::seed_from_u64(42);
        let record = play_game(game, &BACKEND, &search, &sp, &mut rng, 0).unwrap();

        assert_eq!(record.maze.len(), 5 * 5 * 4);
        assert_eq!(record.initial_cheese.len(), 25);
        assert_eq!(record.cheese_available, 3);
    }

    // ---- run_self_play ----

    #[test]
    fn run_self_play_game_count() {
        let games: Vec<GameState> = (0..10).map(|_| short_game()).collect();
        let search = SearchConfig::default();
        let sp = SelfPlayConfig { n_sims: 8, batch_size: 8, num_threads: 2 };

        let result = run_self_play(&games, &BACKEND, &search, &sp, None).unwrap();

        assert_eq!(result.games.len(), 10);
        assert_eq!(result.stats.total_games, 10);
    }

    #[test]
    fn run_self_play_game_index_order() {
        let games: Vec<GameState> = (0..8).map(|_| short_game()).collect();
        let search = SearchConfig::default();
        let sp = SelfPlayConfig { n_sims: 8, batch_size: 8, num_threads: 4 };

        let result = run_self_play(&games, &BACKEND, &search, &sp, None).unwrap();

        // Results should be sorted by game_index
        for (i, record) in result.games.iter().enumerate() {
            assert_eq!(record.game_index, i as u32, "game_index mismatch at position {i}");
        }
    }

    #[test]
    fn run_self_play_progress() {
        let games: Vec<GameState> = (0..4).map(|_| short_game()).collect();
        let search = SearchConfig::default();
        let sp = SelfPlayConfig { n_sims: 8, batch_size: 8, num_threads: 2 };
        let progress = SelfPlayProgress::new();

        let result = run_self_play(&games, &BACKEND, &search, &sp, Some(&progress)).unwrap();

        assert_eq!(progress.games_completed.load(Relaxed), 4);
        assert_eq!(
            progress.positions_completed.load(Relaxed),
            result.stats.total_positions
        );
        assert_eq!(
            progress.simulations_completed.load(Relaxed),
            result.stats.total_simulations
        );
    }

    // ---- stats ----

    #[test]
    fn stats_from_games() {
        // Construct known GameRecords
        let records = vec![
            GameRecord {
                width: 5,
                height: 5,
                max_turns: 10,
                maze: vec![],
                initial_cheese: vec![],
                positions: vec![],     // 0 turns for this one
                final_p1_score: 3.0,
                final_p2_score: 2.0,
                result: GameOutcome::P1Win,
                total_simulations: 100,
                cheese_available: 5,
                game_index: 0,
                cheese_outcomes: vec![],
                total_nn_evals: 0,
                total_terminals: 0,
                total_collisions: 0,
            },
            GameRecord {
                width: 5,
                height: 5,
                max_turns: 10,
                maze: vec![],
                initial_cheese: vec![],
                positions: vec![
                    // Dummy positions — we only care about count
                    PositionRecord {
                        p1_pos: [0, 0],
                        p2_pos: [4, 4],
                        p1_score: 0.0,
                        p2_score: 0.0,
                        p1_mud: 0,
                        p2_mud: 0,
                        turn: 0,
                        cheese_mask: vec![],
                        value_p1: 0.0,
                        value_p2: 0.0,
                        visit_counts_p1: [0.0; 5],
                        visit_counts_p2: [0.0; 5],
                        prior_p1: [0.0; 5],
                        prior_p2: [0.0; 5],
                        policy_p1: [0.0; 5],
                        policy_p2: [0.0; 5],
                        action_p1: 0,
                        action_p2: 0,
                    },
                    PositionRecord {
                        p1_pos: [0, 0],
                        p2_pos: [4, 4],
                        p1_score: 0.0,
                        p2_score: 0.0,
                        p1_mud: 0,
                        p2_mud: 0,
                        turn: 1,
                        cheese_mask: vec![],
                        value_p1: 0.0,
                        value_p2: 0.0,
                        visit_counts_p1: [0.0; 5],
                        visit_counts_p2: [0.0; 5],
                        prior_p1: [0.0; 5],
                        prior_p2: [0.0; 5],
                        policy_p1: [0.0; 5],
                        policy_p2: [0.0; 5],
                        action_p1: 0,
                        action_p2: 0,
                    },
                ],
                final_p1_score: 2.0,
                final_p2_score: 2.0,
                result: GameOutcome::Draw,
                total_simulations: 200,
                cheese_available: 5,
                game_index: 1,
                cheese_outcomes: vec![],
                total_nn_evals: 0,
                total_terminals: 0,
                total_collisions: 0,
            },
            GameRecord {
                width: 5,
                height: 5,
                max_turns: 10,
                maze: vec![],
                initial_cheese: vec![],
                positions: vec![
                    PositionRecord {
                        p1_pos: [0, 0],
                        p2_pos: [4, 4],
                        p1_score: 0.0,
                        p2_score: 0.0,
                        p1_mud: 0,
                        p2_mud: 0,
                        turn: 0,
                        cheese_mask: vec![],
                        value_p1: 0.0,
                        value_p2: 0.0,
                        visit_counts_p1: [0.0; 5],
                        visit_counts_p2: [0.0; 5],
                        prior_p1: [0.0; 5],
                        prior_p2: [0.0; 5],
                        policy_p1: [0.0; 5],
                        policy_p2: [0.0; 5],
                        action_p1: 0,
                        action_p2: 0,
                    },
                ],
                final_p1_score: 1.0,
                final_p2_score: 4.0,
                result: GameOutcome::P2Win,
                total_simulations: 50,
                cheese_available: 5,
                game_index: 2,
                cheese_outcomes: vec![],
                total_nn_evals: 0,
                total_terminals: 0,
                total_collisions: 0,
            },
        ];

        let stats = SelfPlayStats::from_games(&records, 1.0);

        assert_eq!(stats.total_games, 3);
        assert_eq!(stats.total_positions, 3); // 0 + 2 + 1
        assert_eq!(stats.total_simulations, 350); // 100 + 200 + 50
        assert_eq!(stats.p1_wins, 1);
        assert_eq!(stats.p2_wins, 1);
        assert_eq!(stats.draws, 1);
        assert_eq!(stats.total_cheese_available, 15); // 3 * 5
        assert!((stats.total_cheese_collected - 14.0).abs() < 0.01); // 5 + 4 + 5
        assert_eq!(stats.min_turns, 0);
        assert_eq!(stats.max_turns, 2);
        assert!((stats.avg_turns() - 1.0).abs() < 0.01); // 3 / 3
        assert!((stats.cheese_utilization() - 14.0 / 15.0).abs() < 0.01);
        assert!((stats.draw_rate() - 1.0 / 3.0).abs() < 0.01);
    }

    // ---- compute_cheese_outcomes ----

    /// Helper: build a 3x3 cheese mask with cheese at given positions.
    fn mask_3x3(cheese_positions: &[(u8, u8)]) -> Vec<u8> {
        let mut mask = vec![0u8; 9];
        for &(x, y) in cheese_positions {
            mask[y as usize * 3 + x as usize] = 1;
        }
        mask
    }

    /// Helper: build a minimal PositionRecord for cheese outcome tests.
    fn pos_record(p1: [u8; 2], p2: [u8; 2], cheese: &[(u8, u8)]) -> PositionRecord {
        PositionRecord {
            p1_pos: p1,
            p2_pos: p2,
            p1_score: 0.0,
            p2_score: 0.0,
            p1_mud: 0,
            p2_mud: 0,
            turn: 0,
            cheese_mask: mask_3x3(cheese),
            value_p1: 0.0,
            value_p2: 0.0,
            visit_counts_p1: [0.0; 5],
            visit_counts_p2: [0.0; 5],
            prior_p1: [0.0; 5],
            prior_p2: [0.0; 5],
            policy_p1: [0.0; 5],
            policy_p2: [0.0; 5],
            action_p1: 0,
            action_p2: 0,
        }
    }

    #[test]
    fn cheese_outcomes_p1_collects() {
        // 3x3 grid, cheese at (1,1). P1 moves to (1,1) and collects it.
        let positions = vec![
            pos_record([0, 0], [2, 2], &[(1, 1)]),
            pos_record([1, 1], [2, 2], &[]),       // cheese gone, P1 is there
        ];
        // Build a 3x3 game with no cheese left (final state).
        let game = GameState::new_with_config(
            3, 3, HashMap::new(), MudMap::new(), &[], // no cheese remaining
            Coordinates::new(1, 1), Coordinates::new(2, 2), 10,
        );
        let outcomes = compute_cheese_outcomes(&positions, &game, 3, 3);
        let idx = 1 * 3 + 1; // y=1, x=1
        assert_eq!(outcomes[idx], CheeseOutcome::P1Win as u8);
    }

    #[test]
    fn cheese_outcomes_p2_collects() {
        let positions = vec![
            pos_record([0, 0], [2, 2], &[(2, 2)]),
            pos_record([0, 0], [2, 2], &[]),       // cheese gone, P2 was already there
        ];
        let game = GameState::new_with_config(
            3, 3, HashMap::new(), MudMap::new(), &[],
            Coordinates::new(0, 0), Coordinates::new(2, 2), 10,
        );
        let outcomes = compute_cheese_outcomes(&positions, &game, 3, 3);
        let idx = 2 * 3 + 2; // y=2, x=2
        assert_eq!(outcomes[idx], CheeseOutcome::P2Win as u8);
    }

    #[test]
    fn cheese_outcomes_simultaneous() {
        // Both players land on the same cheese cell.
        let positions = vec![
            pos_record([0, 0], [2, 2], &[(1, 1)]),
            pos_record([1, 1], [1, 1], &[]),       // both at (1,1)
        ];
        let game = GameState::new_with_config(
            3, 3, HashMap::new(), MudMap::new(), &[],
            Coordinates::new(1, 1), Coordinates::new(1, 1), 10,
        );
        let outcomes = compute_cheese_outcomes(&positions, &game, 3, 3);
        let idx = 1 * 3 + 1;
        assert_eq!(outcomes[idx], CheeseOutcome::Simultaneous as u8);
    }

    #[test]
    fn cheese_outcomes_uncollected() {
        // Cheese at (1,1) never collected — still present in final state.
        let positions = vec![
            pos_record([0, 0], [2, 2], &[(1, 1)]),
        ];
        let game = GameState::new_with_config(
            3, 3, HashMap::new(), MudMap::new(),
            &[Coordinates::new(1, 1)], // cheese still there
            Coordinates::new(0, 0), Coordinates::new(2, 2), 10,
        );
        let outcomes = compute_cheese_outcomes(&positions, &game, 3, 3);
        let idx = 1 * 3 + 1;
        assert_eq!(outcomes[idx], CheeseOutcome::Uncollected as u8);
    }

    #[test]
    fn cheese_outcomes_non_cheese_cells() {
        // Cells that never had cheese should be Uncollected.
        let positions = vec![
            pos_record([0, 0], [2, 2], &[(1, 1)]),
            pos_record([1, 1], [2, 2], &[]),
        ];
        let game = GameState::new_with_config(
            3, 3, HashMap::new(), MudMap::new(), &[],
            Coordinates::new(1, 1), Coordinates::new(2, 2), 10,
        );
        let outcomes = compute_cheese_outcomes(&positions, &game, 3, 3);
        // (0,0) never had cheese
        assert_eq!(outcomes[0], CheeseOutcome::Uncollected as u8);
        // (1,1) was collected by P1
        assert_eq!(outcomes[1 * 3 + 1], CheeseOutcome::P1Win as u8);
    }

    #[test]
    fn cheese_outcomes_empty_positions() {
        // No positions recorded — all cheese is Uncollected.
        let game = GameState::new_with_config(
            3, 3, HashMap::new(), MudMap::new(),
            &[Coordinates::new(1, 1)],
            Coordinates::new(0, 0), Coordinates::new(2, 2), 10,
        );
        let outcomes = compute_cheese_outcomes(&[], &game, 3, 3);
        for &o in &outcomes {
            assert_eq!(o, CheeseOutcome::Uncollected as u8);
        }
    }

    #[test]
    fn cheese_outcomes_multi_turn() {
        // Turn 0: cheese at (0,0) and (2,2)
        // Turn 1: P1 collects (0,0), cheese at (2,2) remains
        // Turn 2: P2 collects (2,2)
        let positions = vec![
            pos_record([1, 0], [1, 2], &[(0, 0), (2, 2)]),
            pos_record([0, 0], [1, 2], &[(2, 2)]),        // P1 at (0,0), cheese gone
            pos_record([0, 0], [2, 2], &[]),               // P2 at (2,2), cheese gone
        ];
        let game = GameState::new_with_config(
            3, 3, HashMap::new(), MudMap::new(), &[],
            Coordinates::new(0, 0), Coordinates::new(2, 2), 10,
        );
        let outcomes = compute_cheese_outcomes(&positions, &game, 3, 3);
        assert_eq!(outcomes[0 * 3 + 0], CheeseOutcome::P1Win as u8);   // (0,0)
        assert_eq!(outcomes[2 * 3 + 2], CheeseOutcome::P2Win as u8);   // (2,2)
    }

    #[test]
    fn cheese_outcomes_last_turn_uses_final_state() {
        // Only one position, cheese collected on that move → uses final game state.
        // P1 starts at (0,0), moves to (1,0) where cheese is.
        let positions = vec![
            pos_record([0, 0], [2, 2], &[(1, 0)]),
        ];
        // Final state: P1 at (1,0), no cheese left.
        let game = GameState::new_with_config(
            3, 3, HashMap::new(), MudMap::new(), &[],
            Coordinates::new(1, 0), Coordinates::new(2, 2), 10,
        );
        let outcomes = compute_cheese_outcomes(&positions, &game, 3, 3);
        assert_eq!(outcomes[0 * 3 + 1], CheeseOutcome::P1Win as u8); // (1,0) → idx=0*3+1=1
    }

    #[test]
    fn play_game_cheese_outcomes_populated() {
        let game = standard_game(); // 5x5, 3 cheese
        let search = SearchConfig::default();
        let sp = SelfPlayConfig { n_sims: 16, batch_size: 8, num_threads: 1 };
        let mut rng = SmallRng::seed_from_u64(42);
        let record = play_game(game, &BACKEND, &search, &sp, &mut rng, 0).unwrap();

        assert_eq!(record.cheese_outcomes.len(), 25, "should be H*W");

        // Non-cheese cells must be Uncollected.
        let initial_cheese_positions: Vec<usize> = record
            .initial_cheese
            .iter()
            .enumerate()
            .filter(|(_, &v)| v == 1)
            .map(|(i, _)| i)
            .collect();
        for idx in 0..25 {
            if !initial_cheese_positions.contains(&idx) {
                assert_eq!(
                    record.cheese_outcomes[idx],
                    CheeseOutcome::Uncollected as u8,
                    "non-cheese cell {idx} should be Uncollected"
                );
            }
        }
    }

    // ---- new test coverage ----

    #[test]
    fn build_maze_array_mud() {
        // Mud between (2,2) and (2,3) with cost 3.
        let mut mud = MudMap::new();
        mud.insert(Coordinates::new(2, 2), Coordinates::new(2, 3), 3);

        let game = GameState::new_with_config(
            5,
            5,
            HashMap::new(),
            mud,
            &[Coordinates::new(0, 0)],
            Coordinates::new(0, 0),
            Coordinates::new(4, 4),
            100,
        );
        let maze = build_maze_array(&game);

        // UP from (2,2) → (2,3) should be mud cost (3)
        let idx_22 = (2 * 5 + 2) * 4;
        assert_eq!(maze[idx_22 + 0], 3, "UP from (2,2) through mud should be 3");
        // RIGHT from (2,2) → (3,2) should be normal passage
        assert_eq!(maze[idx_22 + 1], 1, "RIGHT from (2,2) should be 1");

        // DOWN from (2,3) → (2,2) should also be mud cost
        let idx_23 = (3 * 5 + 2) * 4;
        assert_eq!(maze[idx_23 + 2], 3, "DOWN from (2,3) through mud should be 3");
    }

    #[test]
    fn advance_root_fallback_reinit() {
        // With n_sims=1 the chosen action pair likely wasn't explored,
        // forcing the reinit fallback path in play_game.
        let game = short_game();
        let search = SearchConfig::default();
        let sp = SelfPlayConfig { n_sims: 1, batch_size: 1, num_threads: 1 };
        let mut rng = SmallRng::seed_from_u64(42);
        let record = play_game(game, &BACKEND, &search, &sp, &mut rng, 0).unwrap();

        // Should still complete without panicking.
        assert!(!record.positions.is_empty());
        assert!(record.total_simulations > 0);
    }

    #[test]
    fn play_game_result_matches_scores() {
        let game = standard_game();
        let search = SearchConfig::default();
        let sp = SelfPlayConfig { n_sims: 16, batch_size: 8, num_threads: 1 };
        let mut rng = SmallRng::seed_from_u64(42);
        let record = play_game(game, &BACKEND, &search, &sp, &mut rng, 0).unwrap();

        match record.result {
            GameOutcome::P1Win => assert!(record.final_p1_score > record.final_p2_score),
            GameOutcome::P2Win => assert!(record.final_p2_score > record.final_p1_score),
            GameOutcome::Draw => {
                assert!((record.final_p1_score - record.final_p2_score).abs() < f32::EPSILON)
            }
        }
    }

    #[test]
    fn run_self_play_zero_games() {
        let games: Vec<GameState> = vec![];
        let search = SearchConfig::default();
        let sp = SelfPlayConfig { n_sims: 8, batch_size: 8, num_threads: 2 };

        let result = run_self_play(&games, &BACKEND, &search, &sp, None).unwrap();

        assert!(result.games.is_empty());
        assert_eq!(result.stats.total_games, 0);
        assert_eq!(result.stats.total_positions, 0);
        assert_eq!(result.stats.min_turns, 0);
    }

    #[test]
    fn run_self_play_more_threads_than_games() {
        let games: Vec<GameState> = (0..2).map(|_| short_game()).collect();
        let search = SearchConfig::default();
        let sp = SelfPlayConfig { n_sims: 8, batch_size: 8, num_threads: 8 };

        let result = run_self_play(&games, &BACKEND, &search, &sp, None).unwrap();

        assert_eq!(result.games.len(), 2);
        assert_eq!(result.stats.total_games, 2);
        for (i, record) in result.games.iter().enumerate() {
            assert_eq!(record.game_index, i as u32);
        }
    }

    // ---- run_self_play_to_disk ----

    #[test]
    fn run_self_play_to_disk_writes_bundles() {
        let dir = std::env::temp_dir().join("selfplay_to_disk_test");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();

        let games: Vec<GameState> = (0..4).map(|_| short_game()).collect();
        let search = SearchConfig::default();
        let sp = SelfPlayConfig { n_sims: 8, batch_size: 8, num_threads: 2 };

        let result =
            run_self_play_to_disk(&games, &BACKEND, &search, &sp, &dir, 2, None).unwrap();

        assert_eq!(result.stats.total_games, 4);
        // 4 games with max_per_bundle=2 → 2 bundle files
        assert_eq!(result.written_paths.len(), 2);
        for p in &result.written_paths {
            assert!(p.exists());
        }

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn run_self_play_to_disk_stats_match() {
        let dir = std::env::temp_dir().join("selfplay_to_disk_stats_test");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();

        let games: Vec<GameState> = (0..4).map(|_| short_game()).collect();
        let search = SearchConfig::default();
        let sp = SelfPlayConfig { n_sims: 8, batch_size: 8, num_threads: 2 };

        let disk_result =
            run_self_play_to_disk(&games, &BACKEND, &search, &sp, &dir, 100, None).unwrap();

        // Stats should be consistent
        assert_eq!(disk_result.stats.total_games, 4);
        assert!(disk_result.stats.total_positions > 0);
        assert!(disk_result.stats.total_simulations > 0);
        assert_eq!(
            disk_result.stats.p1_wins + disk_result.stats.p2_wins + disk_result.stats.draws,
            4
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn run_self_play_to_disk_zero_games() {
        let dir = std::env::temp_dir().join("selfplay_to_disk_zero_test");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();

        let games: Vec<GameState> = vec![];
        let search = SearchConfig::default();
        let sp = SelfPlayConfig { n_sims: 8, batch_size: 8, num_threads: 2 };

        let result =
            run_self_play_to_disk(&games, &BACKEND, &search, &sp, &dir, 100, None).unwrap();

        assert_eq!(result.stats.total_games, 0);
        assert!(result.written_paths.is_empty());

        let _ = std::fs::remove_dir_all(&dir);
    }

    // ---- stats incremental ----

    #[test]
    fn stats_add_game_matches_from_games() {
        let records = vec![
            GameRecord {
                width: 5,
                height: 5,
                max_turns: 10,
                maze: vec![],
                initial_cheese: vec![],
                positions: vec![
                    PositionRecord {
                        p1_pos: [0, 0],
                        p2_pos: [4, 4],
                        p1_score: 0.0,
                        p2_score: 0.0,
                        p1_mud: 0,
                        p2_mud: 0,
                        turn: 0,
                        cheese_mask: vec![],
                        value_p1: 0.0,
                        value_p2: 0.0,
                        visit_counts_p1: [0.0; 5],
                        visit_counts_p2: [0.0; 5],
                        prior_p1: [0.0; 5],
                        prior_p2: [0.0; 5],
                        policy_p1: [0.0; 5],
                        policy_p2: [0.0; 5],
                        action_p1: 0,
                        action_p2: 0,
                    },
                ],
                final_p1_score: 3.0,
                final_p2_score: 2.0,
                result: GameOutcome::P1Win,
                total_simulations: 100,
                cheese_available: 5,
                game_index: 0,
                cheese_outcomes: vec![],
                total_nn_evals: 0,
                total_terminals: 0,
                total_collisions: 0,
            },
            GameRecord {
                width: 5,
                height: 5,
                max_turns: 10,
                maze: vec![],
                initial_cheese: vec![],
                positions: vec![
                    PositionRecord {
                        p1_pos: [0, 0],
                        p2_pos: [4, 4],
                        p1_score: 0.0,
                        p2_score: 0.0,
                        p1_mud: 0,
                        p2_mud: 0,
                        turn: 0,
                        cheese_mask: vec![],
                        value_p1: 0.0,
                        value_p2: 0.0,
                        visit_counts_p1: [0.0; 5],
                        visit_counts_p2: [0.0; 5],
                        prior_p1: [0.0; 5],
                        prior_p2: [0.0; 5],
                        policy_p1: [0.0; 5],
                        policy_p2: [0.0; 5],
                        action_p1: 0,
                        action_p2: 0,
                    },
                    PositionRecord {
                        p1_pos: [1, 0],
                        p2_pos: [3, 4],
                        p1_score: 1.0,
                        p2_score: 0.0,
                        p1_mud: 0,
                        p2_mud: 0,
                        turn: 1,
                        cheese_mask: vec![],
                        value_p1: 0.0,
                        value_p2: 0.0,
                        visit_counts_p1: [0.0; 5],
                        visit_counts_p2: [0.0; 5],
                        prior_p1: [0.0; 5],
                        prior_p2: [0.0; 5],
                        policy_p1: [0.0; 5],
                        policy_p2: [0.0; 5],
                        action_p1: 0,
                        action_p2: 0,
                    },
                ],
                final_p1_score: 2.0,
                final_p2_score: 3.0,
                result: GameOutcome::P2Win,
                total_simulations: 200,
                cheese_available: 5,
                game_index: 1,
                cheese_outcomes: vec![],
                total_nn_evals: 0,
                total_terminals: 0,
                total_collisions: 0,
            },
        ];

        let batch = SelfPlayStats::from_games(&records, 1.0);

        let mut incremental = SelfPlayStats::new();
        for g in &records {
            incremental.add_game(g);
        }
        incremental.elapsed_secs = 1.0;

        assert_eq!(batch.total_games, incremental.total_games);
        assert_eq!(batch.total_positions, incremental.total_positions);
        assert_eq!(batch.total_simulations, incremental.total_simulations);
        assert_eq!(batch.p1_wins, incremental.p1_wins);
        assert_eq!(batch.p2_wins, incremental.p2_wins);
        assert_eq!(batch.draws, incremental.draws);
        assert_eq!(batch.min_turns, incremental.min_turns);
        assert_eq!(batch.max_turns, incremental.max_turns);
        assert!(
            (batch.total_cheese_collected - incremental.total_cheese_collected).abs() < f32::EPSILON
        );
    }
}
