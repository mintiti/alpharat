use pyrat::GameState;
use std::fmt;

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------

/// Uniform prior over unique effective actions only.
///
/// Each unique outcome action gets `1/n_unique`; all others get 0.
pub fn smart_uniform_prior(effective: &[u8; 5]) -> [f32; 5] {
    let mut seen = [false; 5];
    let mut count = 0u8;
    for &e in effective {
        if !seen[e as usize] {
            seen[e as usize] = true;
            count += 1;
        }
    }
    let p = 1.0 / count as f32;
    let mut prior = [0.0f32; 5];
    for &e in effective {
        prior[e as usize] = p;
    }
    prior
}

/// Deduplicate effective actions into sorted unique outcomes + reverse mapping.
///
/// Returns `(outcomes, n_outcomes, action_to_idx)` where:
/// - `outcomes[0..n]` are the sorted unique outcome actions
/// - `action_to_idx[a]` maps raw action `a` to its index in `outcomes`
pub fn compute_outcomes(effective: [u8; 5]) -> ([u8; 5], u8, [u8; 5]) {
    let mut unique = [0u8; 5];
    let mut n = 0u8;

    for &val in &effective {
        let pos = unique[..n as usize].partition_point(|&v| v < val);
        if pos < n as usize && unique[pos] == val {
            continue;
        }
        let mut i = n as usize;
        while i > pos {
            unique[i] = unique[i - 1];
            i -= 1;
        }
        unique[pos] = val;
        n += 1;
    }

    let mut action_to_idx = [0u8; 5];
    for action in 0..5 {
        let outcome = effective[action];
        let idx = unique[..n as usize].partition_point(|&v| v < outcome);
        debug_assert!(idx < n as usize && unique[idx] == outcome);
        action_to_idx[action] = idx as u8;
    }

    (unique, n, action_to_idx)
}

// ---------------------------------------------------------------------------
// BackendError — error type for backend evaluation failures
// ---------------------------------------------------------------------------

/// Error from backend evaluation (ONNX session, model mismatch, etc.).
///
/// Wraps `Box<dyn Error + Send + Sync>` so search crates stay decoupled
/// from ONNX-specific error types.
#[derive(Debug)]
pub struct BackendError(Box<dyn std::error::Error + Send + Sync>);

impl BackendError {
    /// Wrap any error into a BackendError.
    pub fn new(err: impl std::error::Error + Send + Sync + 'static) -> Self {
        Self(Box::new(err))
    }

    /// Create from a string message.
    pub fn msg(msg: impl Into<String>) -> Self {
        Self(msg.into().into())
    }
}

impl fmt::Display for BackendError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for BackendError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        self.0.source()
    }
}

impl From<String> for BackendError {
    fn from(s: String) -> Self {
        Self::msg(s)
    }
}

// ---------------------------------------------------------------------------
// EvalResult — return type from any backend
// ---------------------------------------------------------------------------

/// Evaluation output: policies in 5-action space + scalar values per player.
///
/// Policies are in raw 5-action space (not outcome-indexed). Node constructors
/// handle reduction to outcome-indexed space — the backend doesn't need to
/// know about action equivalence.
#[derive(Clone, Copy, Debug)]
#[must_use]
pub struct EvalResult {
    pub policy_p1: [f32; 5],
    pub policy_p2: [f32; 5],
    pub value_p1: f32,
    pub value_p2: f32,
}

// ---------------------------------------------------------------------------
// Backend trait
// ---------------------------------------------------------------------------

/// Clean boundary between search and evaluation.
///
/// Search calls `evaluate()` and gets back priors + values. It doesn't know
/// whether the backend is a neural network, uniform prior, or anything else.
///
/// `Send + Sync` because backends are shared across game threads via `Arc`
/// (lc0 pattern: single shared backend, per-thread computation).
pub trait Backend: Send + Sync {
    fn evaluate(&self, game: &GameState) -> Result<EvalResult, BackendError>;

    /// Batch evaluation — sequential fallback, real backends override for GPU batching.
    fn evaluate_batch(&self, games: &[&GameState]) -> Result<Vec<EvalResult>, BackendError> {
        games.iter().map(|g| self.evaluate(g)).collect()
    }
}

// ---------------------------------------------------------------------------
// SmartUniformBackend — no-NN mode
// ---------------------------------------------------------------------------

/// Uniform priors over unique effective actions, zero values.
///
/// This is the no-NN mode: assigns `1/n_unique` to each unique outcome action,
/// 0 to blocked/duplicate actions, and value 0 for both players.
pub struct SmartUniformBackend;

impl Backend for SmartUniformBackend {
    fn evaluate(&self, game: &GameState) -> Result<EvalResult, BackendError> {
        Ok(EvalResult {
            policy_p1: smart_uniform_prior(&game.effective_actions_p1()),
            policy_p2: smart_uniform_prior(&game.effective_actions_p2()),
            value_p1: 0.0,
            value_p2: 0.0,
        })
    }
}

// ---------------------------------------------------------------------------
// ConstantValueBackend — test-only, configurable constant values
// ---------------------------------------------------------------------------

/// Smart uniform priors + constant value outputs. Test-only.
///
/// Use this to test backup propagation with non-zero leaf values,
/// which SmartUniformBackend can't exercise (it always returns 0).
pub struct ConstantValueBackend {
    pub value_p1: f32,
    pub value_p2: f32,
}

impl Backend for ConstantValueBackend {
    fn evaluate(&self, game: &GameState) -> Result<EvalResult, BackendError> {
        Ok(EvalResult {
            policy_p1: smart_uniform_prior(&game.effective_actions_p1()),
            policy_p2: smart_uniform_prior(&game.effective_actions_p2()),
            value_p1: self.value_p1,
            value_p2: self.value_p2,
        })
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_util::{mud_game_both_stuck, mud_game_p1_stuck, open_5x5_game, wall_game};
    use pyrat::Coordinates;
    use std::collections::HashMap;

    const CHEESE: [Coordinates; 1] = [Coordinates { x: 0, y: 0 }];
    const BACKEND: SmartUniformBackend = SmartUniformBackend;

    // ---- SmartUniformBackend: topology coverage ----

    #[test]
    fn smart_uniform_open_center() {
        let game = open_5x5_game(Coordinates::new(2, 2), Coordinates::new(2, 2), &CHEESE);
        let result = BACKEND.evaluate(&game).unwrap();

        for &p in &result.policy_p1 {
            assert!((p - 0.2).abs() < 1e-6);
        }
        for &p in &result.policy_p2 {
            assert!((p - 0.2).abs() < 1e-6);
        }
    }

    #[test]
    fn smart_uniform_corner_bottom_left() {
        let game = open_5x5_game(Coordinates::new(0, 0), Coordinates::new(2, 2), &CHEESE);
        let result = BACKEND.evaluate(&game).unwrap();

        let third = 1.0 / 3.0;
        assert!((result.policy_p1[0] - third).abs() < 1e-6); // UP
        assert!((result.policy_p1[1] - third).abs() < 1e-6); // RIGHT
        assert_eq!(result.policy_p1[2], 0.0); // DOWN blocked
        assert_eq!(result.policy_p1[3], 0.0); // LEFT blocked
        assert!((result.policy_p1[4] - third).abs() < 1e-6); // STAY

        for &p in &result.policy_p2 {
            assert!((p - 0.2).abs() < 1e-6);
        }
    }

    #[test]
    fn smart_uniform_corner_top_right() {
        let game = open_5x5_game(Coordinates::new(2, 2), Coordinates::new(4, 4), &CHEESE);
        let result = BACKEND.evaluate(&game).unwrap();

        for &p in &result.policy_p1 {
            assert!((p - 0.2).abs() < 1e-6);
        }

        let third = 1.0 / 3.0;
        assert_eq!(result.policy_p2[0], 0.0); // UP blocked
        assert_eq!(result.policy_p2[1], 0.0); // RIGHT blocked
        assert!((result.policy_p2[2] - third).abs() < 1e-6); // DOWN
        assert!((result.policy_p2[3] - third).abs() < 1e-6); // LEFT
        assert!((result.policy_p2[4] - third).abs() < 1e-6); // STAY
    }

    #[test]
    fn smart_uniform_edge_bottom() {
        let game = open_5x5_game(Coordinates::new(2, 0), Coordinates::new(2, 2), &CHEESE);
        let result = BACKEND.evaluate(&game).unwrap();

        let quarter = 0.25;
        assert!((result.policy_p1[0] - quarter).abs() < 1e-6); // UP
        assert!((result.policy_p1[1] - quarter).abs() < 1e-6); // RIGHT
        assert_eq!(result.policy_p1[2], 0.0); // DOWN blocked
        assert!((result.policy_p1[3] - quarter).abs() < 1e-6); // LEFT
        assert!((result.policy_p1[4] - quarter).abs() < 1e-6); // STAY
    }

    #[test]
    fn smart_uniform_one_wall() {
        let mut walls = HashMap::new();
        walls.insert(Coordinates::new(2, 2), vec![Coordinates::new(2, 3)]);
        walls.insert(Coordinates::new(2, 3), vec![Coordinates::new(2, 2)]);

        let game = wall_game(
            Coordinates::new(2, 2),
            Coordinates::new(0, 0),
            walls,
            &CHEESE,
        );
        let result = BACKEND.evaluate(&game).unwrap();

        let quarter = 0.25;
        assert_eq!(result.policy_p1[0], 0.0); // UP blocked by wall
        assert!((result.policy_p1[1] - quarter).abs() < 1e-6);
        assert!((result.policy_p1[2] - quarter).abs() < 1e-6);
        assert!((result.policy_p1[3] - quarter).abs() < 1e-6);
        assert!((result.policy_p1[4] - quarter).abs() < 1e-6);
    }

    #[test]
    fn smart_uniform_mud_p1_stuck() {
        let game = mud_game_p1_stuck();
        let result = BACKEND.evaluate(&game).unwrap();

        for a in 0..4 {
            assert_eq!(result.policy_p1[a], 0.0);
        }
        assert!((result.policy_p1[4] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn smart_uniform_mud_both_stuck() {
        let game = mud_game_both_stuck();
        let result = BACKEND.evaluate(&game).unwrap();

        for a in 0..4 {
            assert_eq!(result.policy_p1[a], 0.0);
            assert_eq!(result.policy_p2[a], 0.0);
        }
        assert!((result.policy_p1[4] - 1.0).abs() < 1e-6);
        assert!((result.policy_p2[4] - 1.0).abs() < 1e-6);
    }

    // ---- Prior properties ----

    #[test]
    fn prior_sums_to_one() {
        let games = [
            open_5x5_game(Coordinates::new(2, 2), Coordinates::new(2, 2), &CHEESE),
            open_5x5_game(Coordinates::new(0, 0), Coordinates::new(4, 4), &CHEESE),
            open_5x5_game(Coordinates::new(2, 0), Coordinates::new(0, 4), &CHEESE),
            mud_game_p1_stuck(),
        ];

        for game in &games {
            let result = BACKEND.evaluate(game).unwrap();
            let sum_p1: f32 = result.policy_p1.iter().sum();
            let sum_p2: f32 = result.policy_p2.iter().sum();
            assert!(
                (sum_p1 - 1.0).abs() < 1e-6,
                "P1 prior doesn't sum to 1: {sum_p1}"
            );
            assert!(
                (sum_p2 - 1.0).abs() < 1e-6,
                "P2 prior doesn't sum to 1: {sum_p2}"
            );
        }
    }

    #[test]
    fn prior_uniform_nonzero() {
        let games = [
            open_5x5_game(Coordinates::new(2, 2), Coordinates::new(2, 2), &CHEESE),
            open_5x5_game(Coordinates::new(0, 0), Coordinates::new(4, 4), &CHEESE),
            mud_game_p1_stuck(),
        ];

        for game in &games {
            let result = BACKEND.evaluate(game).unwrap();
            for policy in [&result.policy_p1, &result.policy_p2] {
                let nonzero: Vec<f32> = policy.iter().copied().filter(|&p| p > 0.0).collect();
                if nonzero.len() > 1 {
                    let first = nonzero[0];
                    for &p in &nonzero[1..] {
                        assert!(
                            (p - first).abs() < 1e-6,
                            "Non-uniform nonzero values: {nonzero:?}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn zero_for_blocked_nonzero_for_canonical() {
        let test_cases: Vec<(GameState, &str)> = vec![
            (
                open_5x5_game(Coordinates::new(0, 0), Coordinates::new(2, 2), &CHEESE),
                "corner",
            ),
            (
                open_5x5_game(Coordinates::new(2, 0), Coordinates::new(2, 2), &CHEESE),
                "edge",
            ),
            (mud_game_p1_stuck(), "mud"),
        ];

        for (game, label) in &test_cases {
            let result = BACKEND.evaluate(game).unwrap();

            let eff_p1 = game.effective_actions_p1();
            let mut seen_p1 = [false; 5];
            for a in 0..5usize {
                let outcome = eff_p1[a] as usize;
                if outcome != a {
                    assert_eq!(
                        result.policy_p1[a], 0.0,
                        "{label}: P1 action {a} is blocked but has nonzero prior"
                    );
                }
                if !seen_p1[outcome] {
                    seen_p1[outcome] = true;
                    assert!(
                        result.policy_p1[outcome] > 0.0,
                        "{label}: P1 canonical action {outcome} has zero prior"
                    );
                }
            }

            let eff_p2 = game.effective_actions_p2();
            let mut seen_p2 = [false; 5];
            for a in 0..5usize {
                let outcome = eff_p2[a] as usize;
                if outcome != a {
                    assert_eq!(
                        result.policy_p2[a], 0.0,
                        "{label}: P2 action {a} is blocked but has nonzero prior"
                    );
                }
                if !seen_p2[outcome] {
                    seen_p2[outcome] = true;
                    assert!(
                        result.policy_p2[outcome] > 0.0,
                        "{label}: P2 canonical action {outcome} has zero prior"
                    );
                }
            }
        }
    }

    // ---- Values ----

    #[test]
    fn smart_uniform_values_zero() {
        let games = [
            open_5x5_game(Coordinates::new(2, 2), Coordinates::new(2, 2), &CHEESE),
            open_5x5_game(Coordinates::new(0, 0), Coordinates::new(4, 4), &CHEESE),
            mud_game_p1_stuck(),
            mud_game_both_stuck(),
        ];

        for game in &games {
            let result = BACKEND.evaluate(game).unwrap();
            assert_eq!(result.value_p1, 0.0);
            assert_eq!(result.value_p2, 0.0);
        }
    }

    // ---- compute_outcomes ----

    #[test]
    fn compute_outcomes_all_unique() {
        let (outcomes, n, a2i) = compute_outcomes([0, 1, 2, 3, 4]);
        assert_eq!(n, 5);
        assert_eq!(outcomes[..5], [0, 1, 2, 3, 4]);
        assert_eq!(a2i, [0, 1, 2, 3, 4]);
    }

    #[test]
    fn compute_outcomes_corner() {
        // (0,0): UP→STAY, RIGHT→RIGHT, DOWN→STAY, LEFT→STAY
        let (outcomes, n, a2i) = compute_outcomes([4, 1, 4, 4, 4]);
        assert_eq!(n, 2);
        assert_eq!(outcomes[..2], [1, 4]);
        // action 0→STAY→idx 1, action 1→RIGHT→idx 0, etc.
        assert_eq!(a2i[0], 1); // UP→STAY
        assert_eq!(a2i[1], 0); // RIGHT
        assert_eq!(a2i[2], 1); // DOWN→STAY
        assert_eq!(a2i[3], 1); // LEFT→STAY
        assert_eq!(a2i[4], 1); // STAY
    }

    #[test]
    fn compute_outcomes_all_same() {
        let (outcomes, n, a2i) = compute_outcomes([4, 4, 4, 4, 4]);
        assert_eq!(n, 1);
        assert_eq!(outcomes[0], 4);
        assert_eq!(a2i, [0, 0, 0, 0, 0]);
    }
}
