pub use alpharat_eval_core::{
    smart_uniform_prior, Backend, BackendError, EvalResult, SmartUniformBackend,
};

/// Smart uniform priors + constant value outputs. Test-only.
///
/// Use this to test backup propagation with non-zero leaf values,
/// which SmartUniformBackend can't exercise (it always returns 0).
#[cfg(test)]
pub(crate) struct ConstantValueBackend {
    pub value_p1: f32,
    pub value_p2: f32,
}

#[cfg(test)]
impl Backend for ConstantValueBackend {
    fn evaluate(&self, game: &pyrat::GameState) -> Result<EvalResult, BackendError> {
        Ok(EvalResult {
            policy_p1: smart_uniform_prior(&game.effective_actions_p1()),
            policy_p2: smart_uniform_prior(&game.effective_actions_p2()),
            value_p1: self.value_p1,
            value_p2: self.value_p2,
        })
    }
}

// ---------------------------------------------------------------------------
// Tests — Backend → HalfNode boundary (integration)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_util::{mud_game_p1_stuck, open_5x5_game, wall_game};
    use crate::HalfNode;
    use pyrat::Coordinates;
    use std::collections::HashMap;

    const CHEESE: [Coordinates; 1] = [Coordinates { x: 0, y: 0 }];
    const BACKEND: SmartUniformBackend = SmartUniformBackend;

    #[test]
    fn backend_to_halfnode_open() {
        let game = open_5x5_game(Coordinates::new(2, 2), Coordinates::new(2, 2), &CHEESE);
        let result = BACKEND.evaluate(&game).unwrap();
        let eff = game.effective_actions_p1();
        let half = HalfNode::new(result.policy_p1, eff);

        assert_eq!(half.n_outcomes(), 5);
        for i in 0..5 {
            assert!((half.prior(i) - 0.2).abs() < 1e-6);
        }
    }

    #[test]
    fn backend_to_halfnode_corner() {
        let game = open_5x5_game(Coordinates::new(0, 0), Coordinates::new(2, 2), &CHEESE);
        let result = BACKEND.evaluate(&game).unwrap();
        let eff = game.effective_actions_p1();
        let half = HalfNode::new(result.policy_p1, eff);

        assert_eq!(half.n_outcomes(), 3);
        let third = 1.0 / 3.0;
        for i in 0..3 {
            assert!(
                (half.prior(i) - third).abs() < 1e-6,
                "outcome {i}: expected {third}, got {}",
                half.prior(i)
            );
        }
    }

    #[test]
    fn backend_to_halfnode_mud() {
        let game = mud_game_p1_stuck();
        let result = BACKEND.evaluate(&game).unwrap();
        let eff = game.effective_actions_p1();
        let half = HalfNode::new(result.policy_p1, eff);

        assert_eq!(half.n_outcomes(), 1);
        assert!((half.prior(0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn backend_to_halfnode_wall() {
        let mut walls = HashMap::new();
        walls.insert(Coordinates::new(2, 2), vec![Coordinates::new(2, 3)]);
        walls.insert(Coordinates::new(2, 3), vec![Coordinates::new(2, 2)]);

        let game = wall_game(Coordinates::new(2, 2), Coordinates::new(0, 0), walls, &CHEESE);
        let result = BACKEND.evaluate(&game).unwrap();
        let eff = game.effective_actions_p1();
        let half = HalfNode::new(result.policy_p1, eff);

        assert_eq!(half.n_outcomes(), 4);
        for i in 0..4 {
            assert!(
                (half.prior(i) - 0.25).abs() < 1e-6,
                "outcome {i}: expected 0.25, got {}",
                half.prior(i)
            );
        }
    }

    #[test]
    fn backend_to_halfnode_asymmetric_both_players() {
        let game = open_5x5_game(Coordinates::new(0, 0), Coordinates::new(2, 2), &CHEESE);
        let result = BACKEND.evaluate(&game).unwrap();

        let half_p1 = HalfNode::new(result.policy_p1, game.effective_actions_p1());
        let half_p2 = HalfNode::new(result.policy_p2, game.effective_actions_p2());

        assert_eq!(half_p1.n_outcomes(), 3);
        let third = 1.0 / 3.0;
        for i in 0..3 {
            assert!(
                (half_p1.prior(i) - third).abs() < 1e-6,
                "P1 outcome {i}: expected {third}, got {}",
                half_p1.prior(i)
            );
        }

        assert_eq!(half_p2.n_outcomes(), 5);
        for i in 0..5 {
            assert!(
                (half_p2.prior(i) - 0.2).abs() < 1e-6,
                "P2 outcome {i}: expected 0.2, got {}",
                half_p2.prior(i)
            );
        }
    }
}
