mod pv;

use std::time::Instant;

use alpharat_mcgs::{
    run_search, Backend, MCGSTree, SearchConfig, SearchPlayer, SmartUniformBackend,
};
use pyrat_sdk::{
    Bot, Context, DeriveOptions, Direction, GameResult, GameSim, GameState, InfoParams, Player,
};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

#[cfg(feature = "onnx")]
use alpharat_sampling::{CachedBackend, ExecutionProvider, FlatEncoder, OnnxBackend};

use pv::{best_p1_outcome, best_p2_outcome, extract_pvs};

const MAX_PV_LINES: usize = 3;

/// Minimum interval between info updates (milliseconds).
/// Matches lc0's cadence — updates also fire immediately when the best move changes.
const INFO_MIN_INTERVAL_MS: u128 = 5000;

#[derive(DeriveOptions)]
pub struct McgsBot {
    #[spin(default = 1500, min = 100, max = 10000)]
    c_puct_milli: i32,
    #[spin(default = 8, min = 1, max = 2000000000)]
    batch_size: i32,
    #[check(default = false)]
    noise: bool,
    #[check(default = true)]
    argmax: bool,

    #[cfg(feature = "onnx")]
    #[str_opt(default = "")]
    model: String,
    #[cfg(feature = "onnx")]
    #[combo(default = "cpu", choices = ["cpu", "coreml", "cuda"])]
    device: String,
    #[cfg(feature = "onnx")]
    #[spin(default = 8192, min = 0, max = 1000000)]
    cache_size: i32,

    // Internal state (unannotated — DeriveOptions ignores these).
    tree: Option<MCGSTree>,
    sim: Option<GameSim>,
    backend: Option<Box<dyn Backend>>,
    rng: SmallRng,
    is_player1: bool,
}

impl Default for McgsBot {
    fn default() -> Self {
        Self::new()
    }
}

impl McgsBot {
    pub fn new() -> Self {
        Self {
            c_puct_milli: 1500,
            batch_size: 8,
            noise: false,
            argmax: true,
            #[cfg(feature = "onnx")]
            model: String::new(),
            #[cfg(feature = "onnx")]
            device: "cpu".to_string(),
            #[cfg(feature = "onnx")]
            cache_size: 8192,
            tree: None,
            sim: None,
            backend: None,
            rng: SmallRng::from_entropy(),
            is_player1: true,
        }
    }

    /// Build the inference backend from the current options.
    fn build_backend(&self, state: &GameState) -> Box<dyn Backend> {
        #[cfg(feature = "onnx")]
        if !self.model.is_empty() {
            let encoder = FlatEncoder::new(state.width(), state.height());
            let provider = ExecutionProvider::try_from(self.device.as_str())
                .unwrap_or_else(|e| panic!("invalid device: {e}"));
            let onnx: Box<dyn Backend> =
                Box::new(OnnxBackend::with_provider(&self.model, encoder, provider)
                    .unwrap_or_else(|e| panic!("failed to load ONNX model '{}': {e}", self.model)));
            if self.cache_size > 0 {
                return Box::new(CachedBackend::new(onnx, self.cache_size as usize));
            }
            return onnx;
        }
        let _ = state;
        Box::new(SmartUniformBackend)
    }

    fn search_config(&self) -> SearchConfig {
        SearchConfig {
            c_puct: self.c_puct_milli as f32 / 1000.0,
            noise_epsilon: if self.noise { 0.25 } else { 0.0 },
            ..SearchConfig::default()
        }
    }

    /// Incremental search loop. Runs batches until `ctx.should_stop()`.
    /// Sends info on the lc0 cadence: when the best move changes, or at
    /// least every INFO_MIN_INTERVAL_MS.
    fn search_loop(&mut self, ctx: &Context, is_preprocess: bool) {
        let config = self.search_config();
        let batch_size = self.batch_size as u32;
        let is_player1 = self.is_player1;

        let mut nps_start: Option<Instant> = None;
        let mut last_info_time = Instant::now();
        let mut last_info_best: Option<u8> = None;

        // Guarantee at least one batch outside preprocess, so we always have
        // a valid result even if should_stop() fires immediately.
        let min_sims = if is_preprocess { 0 } else { batch_size };

        loop {
            // --- exit check + run one batch ---
            {
                let tree = self.tree.as_mut().expect("tree not initialized");
                let sim = self.sim.as_ref().expect("sim not initialized");
                let visits = tree.observe(|view| {
                    view.with_root(|root| root.stats().total_edge_visits)
                });
                if visits >= min_sims && ctx.should_stop() {
                    break;
                }
                let backend = self.backend.as_deref().expect("backend not initialized");
                let _ = run_search(
                    tree,
                    sim,
                    backend,
                    &config,
                    batch_size,
                    batch_size,
                    &mut self.rng,
                );
            }

            // NPS timer starts on first completed batch (lc0 pattern).
            if nps_start.is_none() {
                nps_start = Some(Instant::now());
            }

            // --- throttled info (all mutable borrows dropped) ---
            let now = Instant::now();
            let (total, current_best, best_direction) = {
                let tree = self.tree.as_ref().expect("tree not initialized");
                tree.observe(|view| view.with_root(|root| {
                    let best = if is_player1 {
                        best_p1_outcome(root)
                    } else {
                        best_p2_outcome(root)
                    };
                    let dir = Direction::try_from(best.action).unwrap_or(Direction::Stay);
                    (root.stats().total_edge_visits, best.index, dir)
                }))
            };

            let best_changed = last_info_best != Some(current_best);

            if best_changed {
                ctx.send_provisional(best_direction);
            }
            let interval_elapsed =
                now.duration_since(last_info_time).as_millis() >= INFO_MIN_INTERVAL_MS;

            if best_changed || interval_elapsed {
                let nps = compute_nps(total, nps_start, now);
                self.send_info(ctx, total, nps);
                last_info_time = now;
                last_info_best = Some(current_best);
            }
        }

        // Always send final info so the GUI shows the settled state.
        let now = Instant::now();
        let total = {
            let tree = self.tree.as_ref().expect("tree not initialized");
            tree.observe(|view| {
                view.with_root(|root| root.stats().total_edge_visits)
            })
        };
        let nps = compute_nps(total, nps_start, now);
        self.send_info(ctx, total, nps);
    }

    /// Send info for both players (multi-PV, lc0 style).
    /// Shared stats (nodes) across all lines; per-line score, depth, PV.
    fn send_info(&self, ctx: &Context, total_nodes: u32, nps: u64) {
        let tree = self.tree.as_ref().expect("tree not initialized");
        let sim = self.sim.as_ref().expect("sim not initialized");

        for &player in &[Player::Player1, Player::Player2] {
            let is_p1 = player == Player::Player1;
            let current_score = if is_p1 {
                sim.player1_score()
            } else {
                sim.player2_score()
            };
            let pvs = extract_pvs(tree, sim, is_p1, MAX_PV_LINES);

            for (rank, pv) in pvs.iter().enumerate() {
                let msg = if rank == 0 {
                    format!("{nps} nps")
                } else {
                    String::new()
                };

                ctx.send_info(&InfoParams {
                    player,
                    multipv: (rank + 1) as u16,
                    target: pv.target,
                    depth: pv.moves.len() as u16,
                    nodes: total_nodes,
                    score: Some(current_score + pv.score),
                    pv: &pv.moves,
                    message: &msg,
                });
            }
        }
    }

    /// Extract the best move from the current tree.
    fn pick_move(&mut self) -> Direction {
        let tree = self.tree.as_ref().expect("tree not initialized");
        let player = if self.is_player1 {
            SearchPlayer::Player1
        } else {
            SearchPlayer::Player2
        };
        let argmax = self.argmax;
        let rng = &mut self.rng;

        tree.observe(|view| {
            view.with_root(|root| {
                if argmax {
                    // Deterministic: most-visited outcome, tiebreak by Q then prior.
                    let best = if player == SearchPlayer::Player1 {
                        best_p1_outcome(root)
                    } else {
                        best_p2_outcome(root)
                    };
                    Direction::try_from(best.action)
                        .expect("invalid direction from outcome_action")
                } else {
                    // Stochastic: sample from visit-proportional distribution.
                    let Some(policy) = normalize_visit_policy(root.action_visits(player)) else {
                        return Direction::Stay;
                    };
                    let action = sample_from_policy(&policy, rng);
                    Direction::try_from(action as u8).expect("invalid direction")
                }
            })
        })
    }
}

impl Bot for McgsBot {
    fn preprocess(&mut self, state: &GameState, ctx: &Context) {
        self.is_player1 = state.my_player() == Player::Player1;
        self.backend = Some(self.build_backend(state));
        let sim = state.to_sim();
        self.tree = Some(MCGSTree::new(&sim));
        self.sim = Some(sim);
        alpharat_mcgs::gc::start();
        self.search_loop(ctx, true);
    }

    fn think(&mut self, state: &GameState, ctx: &Context) -> Direction {
        // Always rebuild sim from authoritative server state.
        let sim = state.to_sim();
        self.sim = Some(sim);

        if state.turn() == 0 {
            // Turn 0: keep tree from preprocess, or create if preprocess didn't run.
            if self.tree.is_none() {
                self.tree = Some(MCGSTree::new(self.sim.as_ref().unwrap()));
                alpharat_mcgs::gc::start();
            }
        } else if let Some(tree) = self.tree.as_mut() {
            // Tree reuse: advance to child matching last turn's moves.
            let p1 = state.player1_last_move() as u8;
            let p2 = state.player2_last_move() as u8;
            tree.advance_root(self.sim.as_ref().unwrap(), p1, p2);
        } else {
            // Defensive: no tree (shouldn't happen in normal flow).
            self.tree = Some(MCGSTree::new(self.sim.as_ref().unwrap()));
            alpharat_mcgs::gc::start();
        }

        self.search_loop(ctx, false);
        self.pick_move()
    }

    fn on_game_over(&mut self, _result: GameResult, _scores: (f32, f32)) {
        self.tree = None;
        self.sim = None;
        self.backend = None;
        alpharat_mcgs::gc::stop();
        alpharat_mcgs::gc::wait();
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn compute_nps(total_visits: u32, nps_start: Option<Instant>, now: Instant) -> u64 {
    nps_start.map_or(0, |t| {
        let ms = now.duration_since(t).as_millis() as u64;
        if ms > 0 {
            (total_visits as u64) * 1000 / ms
        } else {
            0
        }
    })
}

fn normalize_visit_policy(visits: [f32; 5]) -> Option<[f32; 5]> {
    let total: f32 = visits.iter().sum();
    if total == 0.0 {
        return None;
    }
    let mut policy = [0.0f32; 5];
    for (index, visits) in visits.into_iter().enumerate() {
        policy[index] = visits / total;
    }
    Some(policy)
}

/// Weighted random sample from a probability distribution.
fn sample_from_policy(policy: &[f32; 5], rng: &mut impl Rng) -> usize {
    let r: f32 = rng.gen();
    let mut cum = 0.0;
    for (i, &p) in policy.iter().enumerate() {
        cum += p;
        if r < cum {
            return i;
        }
    }
    // Fallback to last non-zero action.
    policy.iter().rposition(|&p| p > 0.0).unwrap_or(4)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn visit_policy_normalizes_reachable_actions_and_rejects_zero_visits() {
        let policy = normalize_visit_policy([3.0, 5.0, 0.0, 0.0, 7.0]).unwrap();
        assert_eq!(policy, [0.2, 1.0 / 3.0, 0.0, 0.0, 7.0 / 15.0]);
        assert_eq!(normalize_visit_policy([0.0; 5]), None);
    }

    #[test]
    fn sampling_never_selects_an_action_outside_policy_support() {
        let policy = [0.0, 0.25, 0.0, 0.75, 0.0];
        let mut rng = SmallRng::seed_from_u64(42);
        for _ in 0..1_000 {
            assert!(matches!(sample_from_policy(&policy, &mut rng), 1 | 3));
        }
    }
}
