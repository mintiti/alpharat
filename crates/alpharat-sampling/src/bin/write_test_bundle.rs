//! Writes a test bundle with known synthetic data for Python integration testing.
//!
//! Usage: write_test_bundle <output_path>

use alpharat_sampling::selfplay::{
    CheeseOutcome, GameOutcome, GameRecord, PositionRecord,
};
use alpharat_sampling::recording::write_bundle;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 2 {
        eprintln!("Usage: write_test_bundle <output_path>");
        std::process::exit(1);
    }
    let output_path = std::path::Path::new(&args[1]);

    let games = vec![make_game_0(), make_game_1()];
    write_bundle(&games, output_path).expect("failed to write bundle");
}

/// Game 0: 3x3 grid, 2 positions.
/// P1 at (0,0), P2 at (2,2). One cheese at (1,1), collected by P1.
fn make_game_0() -> GameRecord {
    let w: u8 = 3;
    let h: u8 = 3;
    let hw = w as usize * h as usize;

    // Maze: all open except boundaries blocked for corner cells.
    // Fill with 1 (open), then set boundary blocks.
    let mut maze = vec![1i8; hw * 4];
    // (0,0): DOWN=-1, LEFT=-1
    maze[0 * 4 + 2] = -1;
    maze[0 * 4 + 3] = -1;
    // (2,0): DOWN=-1, RIGHT=-1
    maze[2 * 4 + 1] = -1;
    maze[2 * 4 + 2] = -1;
    // (0,2): UP=-1, LEFT=-1
    maze[(2 * 3 + 0) * 4 + 0] = -1;
    maze[(2 * 3 + 0) * 4 + 3] = -1;
    // (2,2): UP=-1, RIGHT=-1
    maze[(2 * 3 + 2) * 4 + 0] = -1;
    maze[(2 * 3 + 2) * 4 + 1] = -1;

    let mut initial_cheese = vec![0u8; hw];
    initial_cheese[1 * 3 + 1] = 1; // cheese at (1,1)

    let positions = vec![
        PositionRecord {
            p1_pos: [0, 0],
            p2_pos: [2, 2],
            p1_score: 0.0,
            p2_score: 0.0,
            p1_mud: 0,
            p2_mud: 0,
            turn: 0,
            cheese_mask: initial_cheese.clone(),
            value_p1: 0.75,
            value_p2: 0.25,
            visit_counts_p1: [10.0, 5.0, 0.0, 0.0, 1.0],
            visit_counts_p2: [0.0, 0.0, 6.0, 8.0, 2.0],
            prior_p1: [0.3, 0.3, 0.1, 0.1, 0.2],
            prior_p2: [0.1, 0.1, 0.3, 0.3, 0.2],
            policy_p1: [0.625, 0.3125, 0.0, 0.0, 0.0625],
            policy_p2: [0.0, 0.0, 0.375, 0.5, 0.125],
            action_p1: 0, // UP
            action_p2: 2, // DOWN
        },
        PositionRecord {
            p1_pos: [0, 1],
            p2_pos: [2, 1],
            p1_score: 0.0,
            p2_score: 0.0,
            p1_mud: 0,
            p2_mud: 0,
            turn: 1,
            cheese_mask: initial_cheese.clone(), // cheese still there
            value_p1: 0.8,
            value_p2: 0.2,
            visit_counts_p1: [4.0, 12.0, 0.0, 0.0, 0.0],
            visit_counts_p2: [0.0, 0.0, 0.0, 12.0, 4.0],
            prior_p1: [0.4, 0.4, 0.1, 0.05, 0.05],
            prior_p2: [0.05, 0.05, 0.1, 0.4, 0.4],
            policy_p1: [0.25, 0.75, 0.0, 0.0, 0.0],
            policy_p2: [0.0, 0.0, 0.0, 0.75, 0.25],
            action_p1: 1, // RIGHT → (1,1)
            action_p2: 3, // LEFT → (1,1)
        },
    ];

    let mut cheese_outcomes = vec![CheeseOutcome::Uncollected as u8; hw];
    cheese_outcomes[1 * 3 + 1] = CheeseOutcome::Simultaneous as u8; // both arrived at (1,1)

    GameRecord {
        width: w,
        height: h,
        max_turns: 30,
        maze,
        initial_cheese,
        positions,
        final_p1_score: 0.5,
        final_p2_score: 0.5,
        result: GameOutcome::Draw,
        total_simulations: 32,
        cheese_available: 1,
        game_index: 0,
        cheese_outcomes,
    }
}

/// Game 1: 3x3 grid, 1 position.
/// P1 at (1,0), P2 at (1,2). Two cheese at (0,0) and (2,2).
fn make_game_1() -> GameRecord {
    let w: u8 = 3;
    let h: u8 = 3;
    let hw = w as usize * h as usize;

    let maze = vec![1i8; hw * 4]; // all open (simplified)

    let mut initial_cheese = vec![0u8; hw];
    initial_cheese[0 * 3 + 0] = 1; // (0,0)
    initial_cheese[2 * 3 + 2] = 1; // (2,2)

    let positions = vec![PositionRecord {
        p1_pos: [1, 0],
        p2_pos: [1, 2],
        p1_score: 0.0,
        p2_score: 0.0,
        p1_mud: 0,
        p2_mud: 3, // P2 stuck in mud
        turn: 0,
        cheese_mask: initial_cheese.clone(),
        value_p1: 1.2,
        value_p2: 0.8,
        visit_counts_p1: [2.0, 2.0, 2.0, 6.0, 4.0],
        visit_counts_p2: [0.0, 0.0, 0.0, 0.0, 16.0], // all STAY (mud)
        prior_p1: [0.2, 0.2, 0.2, 0.2, 0.2],
        prior_p2: [0.0, 0.0, 0.0, 0.0, 1.0],
        policy_p1: [0.125, 0.125, 0.125, 0.375, 0.25],
        policy_p2: [0.0, 0.0, 0.0, 0.0, 1.0],
        action_p1: 3, // LEFT → (0,0)
        action_p2: 4, // STAY
    }];

    let mut cheese_outcomes = vec![CheeseOutcome::Uncollected as u8; hw];
    cheese_outcomes[0] = CheeseOutcome::P1Win as u8; // P1 collected (0,0)

    GameRecord {
        width: w,
        height: h,
        max_turns: 20,
        maze,
        initial_cheese,
        positions,
        final_p1_score: 1.0,
        final_p2_score: 0.0,
        result: GameOutcome::P1Win,
        total_simulations: 16,
        cheese_available: 2,
        game_index: 1,
        cheese_outcomes,
    }
}
