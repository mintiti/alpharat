use alpharat_mcts::{run_search, Backend, MCTSTree, SearchConfig, SearchResult};
use pyrat::{Coordinates, Direction, GameState};
use rand::distributions::WeightedIndex;
use rand::prelude::Distribution;
use rand::rngs::SmallRng;
use rand::SeedableRng;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering::Relaxed};
use std::time::Instant;

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
}

/// Aggregate stats computed from a set of GameRecords.
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
    pub total_turns: u64,
    pub min_turns: u32,
    pub max_turns: u32,
}

impl SelfPlayStats {
    pub fn from_games(games: &[GameRecord], elapsed_secs: f64) -> Self {
        let mut stats = Self {
            total_games: games.len() as u32,
            total_positions: 0,
            total_simulations: 0,
            elapsed_secs,
            p1_wins: 0,
            p2_wins: 0,
            draws: 0,
            total_cheese_collected: 0.0,
            total_cheese_available: 0,
            total_turns: 0,
            min_turns: u32::MAX,
            max_turns: 0,
        };

        for g in games {
            let n = g.positions.len() as u64;
            stats.total_positions += n;
            stats.total_simulations += g.total_simulations;
            stats.total_cheese_collected += g.final_p1_score + g.final_p2_score;
            stats.total_cheese_available += g.cheese_available as u32;
            stats.total_turns += n;

            let turns = n as u32;
            stats.min_turns = stats.min_turns.min(turns);
            stats.max_turns = stats.max_turns.max(turns);

            match g.result {
                GameOutcome::P1Win => stats.p1_wins += 1,
                GameOutcome::P2Win => stats.p2_wins += 1,
                GameOutcome::Draw => stats.draws += 1,
            }
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
            self.total_turns as f64 / self.total_games as f64
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
}

/// Result of a self-play session.
pub struct SelfPlayResult {
    pub games: Vec<GameRecord>,
    pub stats: SelfPlayStats,
}

/// Atomic counters for tracking self-play progress from outside.
pub struct SelfPlayProgress {
    pub games_completed: AtomicU32,
    pub positions_completed: AtomicU64,
    pub simulations_completed: AtomicU64,
}

impl SelfPlayProgress {
    pub fn new() -> Self {
        Self {
            games_completed: AtomicU32::new(0),
            positions_completed: AtomicU64::new(0),
            simulations_completed: AtomicU64::new(0),
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
    n_sims: u32,
    batch_size: u32,
    rng: &mut impl rand::Rng,
    game_index: u32,
) -> GameRecord {
    let w = game.width as u8;
    let h = game.height as u8;

    let maze = build_maze_array(&game);
    let initial_cheese = build_cheese_mask(&game);
    let cheese_available = game.cheese.remaining_cheese() as u16;

    let mut tree = MCTSTree::new(&game);
    let mut positions = Vec::new();
    let mut total_simulations: u64 = 0;

    while !game.check_game_over() {
        let result = run_search(&mut tree, &game, backend, search_config, n_sims, batch_size, rng);
        total_simulations += result.total_visits as u64;

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

    GameRecord {
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
    }
}

// ---------------------------------------------------------------------------
// Multi-game orchestration
// ---------------------------------------------------------------------------

/// Run self-play on a slice of games using `num_threads` worker threads.
///
/// Each thread claims games by atomic index, clones + plays them.
/// Each thread uses its own entropy-seeded RNG (no deterministic seeding).
/// Results are sorted by `game_index` before returning.
pub fn run_self_play(
    games: &[GameState],
    backend: &dyn Backend,
    search_config: &SearchConfig,
    n_sims: u32,
    batch_size: u32,
    num_threads: u32,
    progress: Option<&SelfPlayProgress>,
) -> SelfPlayResult {
    let start = Instant::now();
    let next_game = AtomicU32::new(0);
    let n_games = games.len();

    let all_results: Vec<Vec<GameRecord>> = std::thread::scope(|s| {
        let handles: Vec<_> = (0..num_threads)
            .map(|_| {
                s.spawn(|| {
                    let mut rng = SmallRng::from_entropy();
                    let mut local = Vec::new();
                    loop {
                        let idx = next_game.fetch_add(1, Relaxed) as usize;
                        if idx >= n_games {
                            break;
                        }
                        let record = play_game(
                            games[idx].clone(),
                            backend,
                            search_config,
                            n_sims,
                            batch_size,
                            &mut rng,
                            idx as u32,
                        );

                        if let Some(p) = progress {
                            p.positions_completed
                                .fetch_add(record.positions.len() as u64, Relaxed);
                            p.simulations_completed
                                .fetch_add(record.total_simulations, Relaxed);
                            p.games_completed.fetch_add(1, Relaxed);
                        }

                        local.push(record);
                    }
                    local
                })
            })
            .collect();

        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });

    let mut game_records: Vec<GameRecord> = all_results.into_iter().flatten().collect();
    game_records.sort_by_key(|g| g.game_index);

    let elapsed = start.elapsed().as_secs_f64();
    let stats = SelfPlayStats::from_games(&game_records, elapsed);

    SelfPlayResult {
        games: game_records,
        stats,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use alpharat_mcts::SmartUniformBackend;
    use std::collections::HashMap;

    const BACKEND: SmartUniformBackend = SmartUniformBackend;

    fn open_5x5_game(p1: Coordinates, p2: Coordinates, cheese: &[Coordinates]) -> GameState {
        GameState::new_with_config(5, 5, HashMap::new(), Default::default(), cheese, p1, p2, 100)
    }

    fn standard_game() -> GameState {
        open_5x5_game(
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
            Default::default(),
            &[Coordinates::new(1, 0)],
            Coordinates::new(0, 0),
            Coordinates::new(4, 4),
            5,
        )
    }

    // ---- sample_action ----

    #[test]
    fn test_sample_action_deterministic() {
        let mut rng = SmallRng::seed_from_u64(0);
        let policy = [1.0, 0.0, 0.0, 0.0, 0.0];
        for _ in 0..20 {
            assert_eq!(sample_action(&policy, &mut rng), 0);
        }
    }

    #[test]
    fn test_sample_action_zero_fallback() {
        let mut rng = SmallRng::seed_from_u64(0);
        let policy = [0.0, 0.0, 0.0, 0.0, 0.0];
        assert_eq!(sample_action(&policy, &mut rng), 4);
    }

    // ---- build_maze_array ----

    #[test]
    fn test_build_maze_array_open() {
        let game = open_5x5_game(
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
    fn test_build_maze_array_walls() {
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
            Default::default(),
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
    fn test_build_cheese_mask() {
        let cheese = [Coordinates::new(0, 0), Coordinates::new(2, 3), Coordinates::new(4, 4)];
        let game = open_5x5_game(Coordinates::new(1, 1), Coordinates::new(3, 3), &cheese);
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
    fn test_play_game_completes() {
        let game = short_game();
        let config = SearchConfig::default();
        let mut rng = SmallRng::seed_from_u64(42);
        let record = play_game(game, &BACKEND, &config, 16, 8, &mut rng, 0);

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
    fn test_play_game_position_fields() {
        let game = short_game();
        let config = SearchConfig::default();
        let mut rng = SmallRng::seed_from_u64(42);
        let record = play_game(game, &BACKEND, &config, 16, 8, &mut rng, 0);

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
    fn test_play_game_simulations_tracked() {
        let game = short_game();
        let config = SearchConfig::default();
        let n_sims = 32;
        let mut rng = SmallRng::seed_from_u64(42);
        let record = play_game(game, &BACKEND, &config, n_sims, 8, &mut rng, 0);

        assert!(record.total_simulations > 0);
        // Should be approximately n_sims * num_positions
        let expected_approx = n_sims as u64 * record.positions.len() as u64;
        // Allow some variance (terminal nodes, etc.)
        assert!(
            record.total_simulations >= expected_approx / 2,
            "total_simulations={} expected ~{}",
            record.total_simulations,
            expected_approx
        );
    }

    #[test]
    fn test_play_game_maze_and_cheese_recorded() {
        let game = standard_game();
        let config = SearchConfig::default();
        let mut rng = SmallRng::seed_from_u64(42);
        let record = play_game(game, &BACKEND, &config, 8, 8, &mut rng, 0);

        assert_eq!(record.maze.len(), 5 * 5 * 4);
        assert_eq!(record.initial_cheese.len(), 25);
        assert_eq!(record.cheese_available, 3);
    }

    // ---- run_self_play ----

    #[test]
    fn test_run_self_play_game_count() {
        let games: Vec<GameState> = (0..10).map(|_| short_game()).collect();
        let config = SearchConfig::default();

        let result = run_self_play(&games, &BACKEND, &config, 8, 8, 2, None);

        assert_eq!(result.games.len(), 10);
        assert_eq!(result.stats.total_games, 10);
    }

    #[test]
    fn test_run_self_play_game_index_order() {
        let games: Vec<GameState> = (0..8).map(|_| short_game()).collect();
        let config = SearchConfig::default();

        let result = run_self_play(&games, &BACKEND, &config, 8, 8, 4, None);

        // Results should be sorted by game_index
        for (i, record) in result.games.iter().enumerate() {
            assert_eq!(record.game_index, i as u32, "game_index mismatch at position {i}");
        }
    }

    #[test]
    fn test_run_self_play_progress() {
        let games: Vec<GameState> = (0..4).map(|_| short_game()).collect();
        let config = SearchConfig::default();
        let progress = SelfPlayProgress::new();

        let result = run_self_play(&games, &BACKEND, &config, 8, 8, 2, Some(&progress));

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
    fn test_stats_from_games() {
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
    fn test_cheese_outcomes_p1_collects() {
        // 3x3 grid, cheese at (1,1). P1 moves to (1,1) and collects it.
        let positions = vec![
            pos_record([0, 0], [2, 2], &[(1, 1)]),
            pos_record([1, 1], [2, 2], &[]),       // cheese gone, P1 is there
        ];
        // Build a 3x3 game with no cheese left (final state).
        let game = GameState::new_with_config(
            3, 3, HashMap::new(), Default::default(), &[], // no cheese remaining
            Coordinates::new(1, 1), Coordinates::new(2, 2), 10,
        );
        let outcomes = compute_cheese_outcomes(&positions, &game, 3, 3);
        let idx = 1 * 3 + 1; // y=1, x=1
        assert_eq!(outcomes[idx], CheeseOutcome::P1Win as u8);
    }

    #[test]
    fn test_cheese_outcomes_p2_collects() {
        let positions = vec![
            pos_record([0, 0], [2, 2], &[(2, 2)]),
            pos_record([0, 0], [2, 2], &[]),       // cheese gone, P2 was already there
        ];
        let game = GameState::new_with_config(
            3, 3, HashMap::new(), Default::default(), &[],
            Coordinates::new(0, 0), Coordinates::new(2, 2), 10,
        );
        let outcomes = compute_cheese_outcomes(&positions, &game, 3, 3);
        let idx = 2 * 3 + 2; // y=2, x=2
        assert_eq!(outcomes[idx], CheeseOutcome::P2Win as u8);
    }

    #[test]
    fn test_cheese_outcomes_simultaneous() {
        // Both players land on the same cheese cell.
        let positions = vec![
            pos_record([0, 0], [2, 2], &[(1, 1)]),
            pos_record([1, 1], [1, 1], &[]),       // both at (1,1)
        ];
        let game = GameState::new_with_config(
            3, 3, HashMap::new(), Default::default(), &[],
            Coordinates::new(1, 1), Coordinates::new(1, 1), 10,
        );
        let outcomes = compute_cheese_outcomes(&positions, &game, 3, 3);
        let idx = 1 * 3 + 1;
        assert_eq!(outcomes[idx], CheeseOutcome::Simultaneous as u8);
    }

    #[test]
    fn test_cheese_outcomes_uncollected() {
        // Cheese at (1,1) never collected — still present in final state.
        let positions = vec![
            pos_record([0, 0], [2, 2], &[(1, 1)]),
        ];
        let game = GameState::new_with_config(
            3, 3, HashMap::new(), Default::default(),
            &[Coordinates::new(1, 1)], // cheese still there
            Coordinates::new(0, 0), Coordinates::new(2, 2), 10,
        );
        let outcomes = compute_cheese_outcomes(&positions, &game, 3, 3);
        let idx = 1 * 3 + 1;
        assert_eq!(outcomes[idx], CheeseOutcome::Uncollected as u8);
    }

    #[test]
    fn test_cheese_outcomes_non_cheese_cells() {
        // Cells that never had cheese should be Uncollected.
        let positions = vec![
            pos_record([0, 0], [2, 2], &[(1, 1)]),
            pos_record([1, 1], [2, 2], &[]),
        ];
        let game = GameState::new_with_config(
            3, 3, HashMap::new(), Default::default(), &[],
            Coordinates::new(1, 1), Coordinates::new(2, 2), 10,
        );
        let outcomes = compute_cheese_outcomes(&positions, &game, 3, 3);
        // (0,0) never had cheese
        assert_eq!(outcomes[0], CheeseOutcome::Uncollected as u8);
        // (1,1) was collected by P1
        assert_eq!(outcomes[1 * 3 + 1], CheeseOutcome::P1Win as u8);
    }

    #[test]
    fn test_cheese_outcomes_empty_positions() {
        // No positions recorded — all cheese is Uncollected.
        let game = GameState::new_with_config(
            3, 3, HashMap::new(), Default::default(),
            &[Coordinates::new(1, 1)],
            Coordinates::new(0, 0), Coordinates::new(2, 2), 10,
        );
        let outcomes = compute_cheese_outcomes(&[], &game, 3, 3);
        for &o in &outcomes {
            assert_eq!(o, CheeseOutcome::Uncollected as u8);
        }
    }

    #[test]
    fn test_cheese_outcomes_multi_turn() {
        // Turn 0: cheese at (0,0) and (2,2)
        // Turn 1: P1 collects (0,0), cheese at (2,2) remains
        // Turn 2: P2 collects (2,2)
        let positions = vec![
            pos_record([1, 0], [1, 2], &[(0, 0), (2, 2)]),
            pos_record([0, 0], [1, 2], &[(2, 2)]),        // P1 at (0,0), cheese gone
            pos_record([0, 0], [2, 2], &[]),               // P2 at (2,2), cheese gone
        ];
        let game = GameState::new_with_config(
            3, 3, HashMap::new(), Default::default(), &[],
            Coordinates::new(0, 0), Coordinates::new(2, 2), 10,
        );
        let outcomes = compute_cheese_outcomes(&positions, &game, 3, 3);
        assert_eq!(outcomes[0 * 3 + 0], CheeseOutcome::P1Win as u8);   // (0,0)
        assert_eq!(outcomes[2 * 3 + 2], CheeseOutcome::P2Win as u8);   // (2,2)
    }

    #[test]
    fn test_cheese_outcomes_last_turn_uses_final_state() {
        // Only one position, cheese collected on that move → uses final game state.
        // P1 starts at (0,0), moves to (1,0) where cheese is.
        let positions = vec![
            pos_record([0, 0], [2, 2], &[(1, 0)]),
        ];
        // Final state: P1 at (1,0), no cheese left.
        let game = GameState::new_with_config(
            3, 3, HashMap::new(), Default::default(), &[],
            Coordinates::new(1, 0), Coordinates::new(2, 2), 10,
        );
        let outcomes = compute_cheese_outcomes(&positions, &game, 3, 3);
        assert_eq!(outcomes[0 * 3 + 1], CheeseOutcome::P1Win as u8); // (1,0) → idx=0*3+1=1
    }

    #[test]
    fn test_play_game_cheese_outcomes_populated() {
        let game = standard_game(); // 5x5, 3 cheese
        let config = SearchConfig::default();
        let mut rng = SmallRng::seed_from_u64(42);
        let record = play_game(game, &BACKEND, &config, 16, 8, &mut rng, 0);

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
}
