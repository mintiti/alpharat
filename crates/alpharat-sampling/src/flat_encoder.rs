use crate::encoder::ObservationEncoder;
use pyrat::{Coordinates, Direction, GameState};

/// Normalization constants matching Python's `FlatObservationBuilder`.
const MAX_MUD_COST: f32 = 10.0;
const MAX_MUD_TURNS: f32 = 10.0;
const MAX_SCORE: f32 = 10.0;

/// Directions to iterate for the maze layer (excludes STAY).
const DIRECTIONS: [Direction; 4] = [
    Direction::Up,
    Direction::Right,
    Direction::Down,
    Direction::Left,
];

/// Flat observation encoder matching Python's `FlatObservationBuilder.build()`.
///
/// Output layout for an H×W maze (all arrays flattened C-order = y-major):
/// ```text
/// [maze H*W*4] [p1_onehot H*W] [p2_onehot H*W] [cheese H*W] [6 scalars]
/// ```
///
/// Maze values per cell per direction:
///   -1.0  = wall or boundary
///   0.1   = normal passage (cost 1 / 10)
///   c/10  = mud with cost c
///
/// Scalars (in order):
///   1. p1_score - p2_score (raw)
///   2. turn / max_turns
///   3. p1_mud_timer / 10
///   4. p2_mud_timer / 10
///   5. p1_score / 10
///   6. p2_score / 10
pub struct FlatEncoder {
    width: u8,
    height: u8,
    obs_dim: usize,
}

impl FlatEncoder {
    pub fn new(width: u8, height: u8) -> Self {
        let spatial = width as usize * height as usize;
        // maze(H*W*4) + p1_pos(H*W) + p2_pos(H*W) + cheese(H*W) + 6 scalars
        let obs_dim = spatial * 7 + 6;
        Self {
            width,
            height,
            obs_dim,
        }
    }
}

impl ObservationEncoder for FlatEncoder {
    fn obs_dim(&self) -> usize {
        self.obs_dim
    }

    fn encode_into(&self, game: &GameState, buf: &mut [f32], offset: usize) {
        let w = self.width;
        let h = self.height;
        let spatial = w as usize * h as usize;
        let out = &mut buf[offset..offset + self.obs_dim];

        // --- Maze layer: H*W*4 floats ---
        // Iterate y (row) then x (col) then direction — C-order for (H, W, 4)
        let mut i = 0;
        for y in 0..h {
            for x in 0..w {
                let pos = Coordinates::new(x, y);
                for &dir in &DIRECTIONS {
                    if !game.move_table.is_move_valid(pos, dir) {
                        // Wall or boundary
                        out[i] = -1.0;
                    } else {
                        let neighbor = dir.apply_to(pos);
                        match game.mud.get(pos, neighbor) {
                            Some(cost) => out[i] = cost as f32 / MAX_MUD_COST,
                            None => out[i] = 1.0 / MAX_MUD_COST, // Normal passage: cost 1
                        }
                    }
                    i += 1;
                }
            }
        }

        // --- P1 position one-hot: H*W floats ---
        let p1_base = spatial * 4;
        for j in 0..spatial {
            out[p1_base + j] = 0.0;
        }
        let p1_idx = game.player1.current_pos.to_index(w);
        out[p1_base + p1_idx] = 1.0;

        // --- P2 position one-hot: H*W floats ---
        let p2_base = spatial * 5;
        for j in 0..spatial {
            out[p2_base + j] = 0.0;
        }
        let p2_idx = game.player2.current_pos.to_index(w);
        out[p2_base + p2_idx] = 1.0;

        // --- Cheese mask: H*W floats ---
        let cheese_base = spatial * 6;
        for y in 0..h {
            for x in 0..w {
                let pos = Coordinates::new(x, y);
                let idx = pos.to_index(w);
                out[cheese_base + idx] = if game.cheese.has_cheese(pos) {
                    1.0
                } else {
                    0.0
                };
            }
        }

        // --- 6 scalars ---
        let scalar_base = spatial * 7;
        let p1_score = game.player1.score;
        let p2_score = game.player2.score;
        let turn = game.turn;
        let max_turns = game.max_turns;

        out[scalar_base] = p1_score - p2_score;
        out[scalar_base + 1] = if max_turns > 0 {
            turn as f32 / max_turns as f32
        } else {
            0.0
        };
        out[scalar_base + 2] = game.player1.mud_timer as f32 / MAX_MUD_TURNS;
        out[scalar_base + 3] = game.player2.mud_timer as f32 / MAX_MUD_TURNS;
        out[scalar_base + 4] = p1_score / MAX_SCORE;
        out[scalar_base + 5] = p2_score / MAX_SCORE;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pyrat::game::types::MudMap;
    use pyrat::GameBuilder;
    use std::collections::HashMap;

    fn open_5x5(p1: Coordinates, p2: Coordinates, cheese: &[Coordinates]) -> GameState {
        GameBuilder::new(5, 5)
            .with_open_maze()
            .with_custom_positions(p1, p2)
            .with_custom_cheese(cheese.to_vec())
            .with_max_turns(100)
            .build()
            .create(None)
            .unwrap()
    }

    #[test]
    fn obs_dim_5x5() {
        let enc = FlatEncoder::new(5, 5);
        // 25*7 + 6 = 181
        assert_eq!(enc.obs_dim(), 181);
    }

    #[test]
    fn obs_dim_7x7() {
        let enc = FlatEncoder::new(7, 7);
        // 49*7 + 6 = 349
        assert_eq!(enc.obs_dim(), 349);
    }

    #[test]
    fn open_maze_center_no_walls() {
        // Open 5x5, player at center (2,2) — all 4 directions valid
        let game = open_5x5(
            Coordinates::new(2, 2),
            Coordinates::new(4, 4),
            &[Coordinates::new(0, 0)],
        );
        let enc = FlatEncoder::new(5, 5);
        let mut buf = vec![0.0f32; enc.obs_dim()];
        enc.encode_into(&game, &mut buf, 0);

        // Cell (2,2) in maze layer: index = (2*5 + 2)*4 = 48
        // All 4 directions should be 0.1 (normal passage from center)
        for d in 0..4 {
            assert_eq!(buf[48 + d], 0.1, "direction {d} at (2,2) should be 0.1");
        }
    }

    #[test]
    fn boundary_walls() {
        // Open 5x5, check boundaries
        let game = open_5x5(
            Coordinates::new(0, 0),
            Coordinates::new(4, 4),
            &[Coordinates::new(2, 2)],
        );
        let enc = FlatEncoder::new(5, 5);
        let mut buf = vec![0.0f32; enc.obs_dim()];
        enc.encode_into(&game, &mut buf, 0);

        // Cell (0,0): UP=valid, RIGHT=valid, DOWN=blocked, LEFT=blocked
        // index = (0*5 + 0)*4 = 0
        assert_eq!(buf[0], 0.1, "UP from (0,0) should be valid");
        assert_eq!(buf[1], 0.1, "RIGHT from (0,0) should be valid");
        assert_eq!(buf[2], -1.0, "DOWN from (0,0) should be blocked");
        assert_eq!(buf[3], -1.0, "LEFT from (0,0) should be blocked");

        // Cell (4,4): UP=blocked, RIGHT=blocked, DOWN=valid, LEFT=valid
        // index = (4*5 + 4)*4 = 96
        assert_eq!(buf[96], -1.0, "UP from (4,4) should be blocked");
        assert_eq!(buf[97], -1.0, "RIGHT from (4,4) should be blocked");
        assert_eq!(buf[98], 0.1, "DOWN from (4,4) should be valid");
        assert_eq!(buf[99], 0.1, "LEFT from (4,4) should be valid");
    }

    #[test]
    fn wall_between_cells() {
        // Wall between (2,2) and (2,3)
        let mut walls = HashMap::new();
        walls.insert(Coordinates::new(2, 2), vec![Coordinates::new(2, 3)]);
        walls.insert(Coordinates::new(2, 3), vec![Coordinates::new(2, 2)]);

        let game = GameBuilder::new(5, 5)
            .with_custom_maze(walls, MudMap::new())
            .with_custom_positions(Coordinates::new(0, 0), Coordinates::new(4, 4))
            .with_custom_cheese(vec![Coordinates::new(0, 0)])
            .with_max_turns(100)
            .build()
            .create(None)
            .unwrap();
        let enc = FlatEncoder::new(5, 5);
        let mut buf = vec![0.0f32; enc.obs_dim()];
        enc.encode_into(&game, &mut buf, 0);

        // Cell (2,2): UP should be blocked (wall to (2,3))
        // index = (2*5 + 2)*4 = 48, UP=dir 0
        assert_eq!(buf[48], -1.0, "UP from (2,2) should be blocked by wall");
        // RIGHT, DOWN, LEFT still valid
        assert_eq!(buf[49], 0.1, "RIGHT from (2,2) still valid");
        assert_eq!(buf[50], 0.1, "DOWN from (2,2) still valid");
        assert_eq!(buf[51], 0.1, "LEFT from (2,2) still valid");

        // Cell (2,3): DOWN should be blocked (wall from (2,3) to (2,2))
        // index = (3*5 + 2)*4 = 68, DOWN=dir 2
        assert_eq!(buf[70], -1.0, "DOWN from (2,3) should be blocked by wall");
    }

    #[test]
    fn mud_cost_encoding() {
        let mut mud = MudMap::new();
        mud.insert(Coordinates::new(2, 2), Coordinates::new(2, 3), 3);

        let game = GameBuilder::new(5, 5)
            .with_custom_maze(HashMap::new(), mud)
            .with_custom_positions(Coordinates::new(0, 0), Coordinates::new(4, 4))
            .with_custom_cheese(vec![Coordinates::new(0, 0)])
            .with_max_turns(100)
            .build()
            .create(None)
            .unwrap();
        let enc = FlatEncoder::new(5, 5);
        let mut buf = vec![0.0f32; enc.obs_dim()];
        enc.encode_into(&game, &mut buf, 0);

        // Cell (2,2): UP direction has mud cost 3 → 3/10 = 0.3
        // index = (2*5 + 2)*4 = 48, UP=dir 0
        assert!(
            (buf[48] - 0.3).abs() < 1e-6,
            "UP from (2,2) should be 0.3 (mud cost 3), got {}",
            buf[48]
        );

        // Cell (2,3): DOWN direction has mud cost 3 → 0.3
        // index = (3*5 + 2)*4 = 68, DOWN=dir 2
        assert!(
            (buf[70] - 0.3).abs() < 1e-6,
            "DOWN from (2,3) should be 0.3 (mud cost 3), got {}",
            buf[70]
        );
    }

    #[test]
    fn position_onehots() {
        let game = open_5x5(
            Coordinates::new(1, 3),
            Coordinates::new(4, 0),
            &[Coordinates::new(2, 2)],
        );
        let enc = FlatEncoder::new(5, 5);
        let mut buf = vec![0.0f32; enc.obs_dim()];
        enc.encode_into(&game, &mut buf, 0);

        let spatial = 25;

        // P1 at (1,3): index = 3*5 + 1 = 16
        let p1_base = spatial * 4;
        for j in 0..spatial {
            let expected = if j == 16 { 1.0 } else { 0.0 };
            assert_eq!(buf[p1_base + j], expected, "P1 one-hot at index {j}");
        }

        // P2 at (4,0): index = 0*5 + 4 = 4
        let p2_base = spatial * 5;
        for j in 0..spatial {
            let expected = if j == 4 { 1.0 } else { 0.0 };
            assert_eq!(buf[p2_base + j], expected, "P2 one-hot at index {j}");
        }
    }

    #[test]
    fn cheese_mask() {
        let cheese = [
            Coordinates::new(0, 0),
            Coordinates::new(2, 3),
            Coordinates::new(4, 4),
        ];
        let game = open_5x5(Coordinates::new(1, 1), Coordinates::new(3, 3), &cheese);
        let enc = FlatEncoder::new(5, 5);
        let mut buf = vec![0.0f32; enc.obs_dim()];
        enc.encode_into(&game, &mut buf, 0);

        let spatial = 25;
        let cheese_base = spatial * 6;

        // Expected cheese positions: (0,0)→0, (2,3)→17, (4,4)→24
        let cheese_indices: Vec<usize> = cheese.iter().map(|c| c.to_index(5)).collect();
        for j in 0..spatial {
            let expected = if cheese_indices.contains(&j) {
                1.0
            } else {
                0.0
            };
            assert_eq!(buf[cheese_base + j], expected, "Cheese mask at index {j}");
        }
    }

    #[test]
    fn scalars_initial_state() {
        let game = open_5x5(
            Coordinates::new(0, 0),
            Coordinates::new(4, 4),
            &[Coordinates::new(2, 2)],
        );
        let enc = FlatEncoder::new(5, 5);
        let mut buf = vec![0.0f32; enc.obs_dim()];
        enc.encode_into(&game, &mut buf, 0);

        let scalar_base = 25 * 7;
        assert_eq!(buf[scalar_base], 0.0, "score diff should be 0");
        assert_eq!(buf[scalar_base + 1], 0.0, "progress should be 0 at turn 0");
        assert_eq!(buf[scalar_base + 2], 0.0, "p1 mud should be 0");
        assert_eq!(buf[scalar_base + 3], 0.0, "p2 mud should be 0");
        assert_eq!(buf[scalar_base + 4], 0.0, "p1 score / 10 should be 0");
        assert_eq!(buf[scalar_base + 5], 0.0, "p2 score / 10 should be 0");
    }

    #[test]
    fn scalars_mid_game() {
        // Create game and advance a few turns to get non-zero scalars
        let mut game = open_5x5(
            Coordinates::new(1, 0),
            Coordinates::new(3, 0),
            &[Coordinates::new(0, 0), Coordinates::new(4, 0)],
        );
        // P1 moves LEFT to (0,0), P2 moves RIGHT to (4,0) — both collect cheese
        let _undo = game.make_move(Direction::Left, Direction::Right);

        let enc = FlatEncoder::new(5, 5);
        let mut buf = vec![0.0f32; enc.obs_dim()];
        enc.encode_into(&game, &mut buf, 0);

        let scalar_base = 25 * 7;
        // Both scored 1.0, so diff = 0
        assert_eq!(buf[scalar_base], 0.0, "score diff");
        // Turn 1, max_turns 100 → progress = 1/100 = 0.01
        assert!((buf[scalar_base + 1] - 0.01).abs() < 1e-6, "progress");
        // p1_score/10 = 0.1, p2_score/10 = 0.1
        assert!(
            (buf[scalar_base + 4] - 0.1).abs() < 1e-6,
            "p1 score normalized"
        );
        assert!(
            (buf[scalar_base + 5] - 0.1).abs() < 1e-6,
            "p2 score normalized"
        );
    }

    #[test]
    fn mud_timer_in_scalars() {
        let mut mud = MudMap::new();
        mud.insert(Coordinates::new(2, 2), Coordinates::new(2, 3), 3);

        let mut game = GameBuilder::new(5, 5)
            .with_custom_maze(HashMap::new(), mud)
            .with_custom_positions(Coordinates::new(2, 2), Coordinates::new(4, 4))
            .with_custom_cheese(vec![Coordinates::new(0, 0)])
            .with_max_turns(100)
            .build()
            .create(None)
            .unwrap();
        // Move P1 into mud
        let _undo = game.make_move(Direction::Up, Direction::Stay);
        assert!(game.player1.mud_timer > 0);

        let enc = FlatEncoder::new(5, 5);
        let mut buf = vec![0.0f32; enc.obs_dim()];
        enc.encode_into(&game, &mut buf, 0);

        let scalar_base = 25 * 7;
        // P1 mud timer should be non-zero
        assert!(buf[scalar_base + 2] > 0.0, "p1 mud timer should be > 0");
        // P2 mud timer should still be 0
        assert_eq!(buf[scalar_base + 3], 0.0, "p2 mud timer should be 0");
    }

    #[test]
    fn offset_writes_correctly() {
        let game = open_5x5(
            Coordinates::new(0, 0),
            Coordinates::new(4, 4),
            &[Coordinates::new(2, 2)],
        );
        let enc = FlatEncoder::new(5, 5);
        let dim = enc.obs_dim();

        // Write two observations back to back
        let mut buf = vec![-999.0f32; dim * 2];
        enc.encode_into(&game, &mut buf, 0);
        enc.encode_into(&game, &mut buf, dim);

        // Both halves should be identical
        for i in 0..dim {
            assert_eq!(
                buf[i],
                buf[dim + i],
                "Mismatch at index {i}: {} vs {}",
                buf[i],
                buf[dim + i]
            );
        }
    }

    #[test]
    fn non_square_grid_7x5() {
        let enc = FlatEncoder::new(7, 5);
        // 35*7 + 6 = 251
        assert_eq!(enc.obs_dim(), 251);

        let game = GameBuilder::new(7, 5)
            .with_open_maze()
            .with_custom_positions(Coordinates::new(0, 0), Coordinates::new(6, 4))
            .with_custom_cheese(vec![Coordinates::new(3, 2)])
            .with_max_turns(100)
            .build()
            .create(None)
            .unwrap();
        let mut buf = vec![0.0f32; enc.obs_dim()];
        enc.encode_into(&game, &mut buf, 0);

        let spatial = 35;

        // P1 at (0,0): index = 0*7 + 0 = 0
        let p1_base = spatial * 4;
        assert_eq!(buf[p1_base], 1.0, "P1 one-hot at (0,0)");
        // Rest should be zero
        for j in 1..spatial {
            assert_eq!(buf[p1_base + j], 0.0, "P1 one-hot should be 0 at index {j}");
        }

        // P2 at (6,4): index = 4*7 + 6 = 34
        let p2_base = spatial * 5;
        assert_eq!(buf[p2_base + 34], 1.0, "P2 one-hot at (6,4)");

        // Cheese at (3,2): index = 2*7 + 3 = 17
        let cheese_base = spatial * 6;
        assert_eq!(buf[cheese_base + 17], 1.0, "Cheese at (3,2)");

        // Boundary checks for non-square: (6,4) is top-right corner
        // Cell (6,4): UP=blocked (top edge), RIGHT=blocked (right edge),
        //             DOWN=valid, LEFT=valid
        // maze index = (4*7 + 6)*4 = 136
        assert_eq!(buf[136], -1.0, "UP from (6,4) should be blocked (top edge)");
        assert_eq!(buf[137], -1.0, "RIGHT from (6,4) should be blocked (right edge)");
        assert_eq!(buf[138], 0.1, "DOWN from (6,4) should be valid");
        assert_eq!(buf[139], 0.1, "LEFT from (6,4) should be valid");

        // Cell (0,0): bottom-left corner
        // UP=valid, RIGHT=valid, DOWN=blocked, LEFT=blocked
        assert_eq!(buf[0], 0.1, "UP from (0,0) should be valid");
        assert_eq!(buf[1], 0.1, "RIGHT from (0,0) should be valid");
        assert_eq!(buf[2], -1.0, "DOWN from (0,0) should be blocked (bottom edge)");
        assert_eq!(buf[3], -1.0, "LEFT from (0,0) should be blocked (left edge)");
    }

    #[test]
    fn max_turns_zero_progress() {
        // Edge case: turn=0, max_turns=1 → progress = 0/1 = 0
        let game = GameBuilder::new(5, 5)
            .with_open_maze()
            .with_custom_positions(Coordinates::new(0, 0), Coordinates::new(4, 4))
            .with_custom_cheese(vec![Coordinates::new(2, 2)])
            .with_max_turns(1)
            .build()
            .create(None)
            .unwrap();
        let enc = FlatEncoder::new(5, 5);
        let mut buf = vec![0.0f32; enc.obs_dim()];
        enc.encode_into(&game, &mut buf, 0);

        let scalar_base = 25 * 7;
        assert_eq!(
            buf[scalar_base + 1],
            0.0,
            "progress should be 0 at turn 0"
        );
    }
}
