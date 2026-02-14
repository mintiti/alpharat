use pyrat::game::types::MudMap;
use pyrat::{Coordinates, Direction, GameState};
use std::collections::HashMap;

pub fn open_5x5_game(p1: Coordinates, p2: Coordinates, cheese: &[Coordinates]) -> GameState {
    GameState::new_with_config(5, 5, HashMap::new(), Default::default(), cheese, p1, p2, 100)
}

pub fn wall_game(
    p1: Coordinates,
    p2: Coordinates,
    walls: HashMap<Coordinates, Vec<Coordinates>>,
    cheese: &[Coordinates],
) -> GameState {
    GameState::new_with_config(5, 5, walls, Default::default(), cheese, p1, p2, 100)
}

pub fn mud_game_p1_stuck() -> GameState {
    let mut mud = MudMap::new();
    mud.insert(Coordinates::new(2, 2), Coordinates::new(2, 3), 3);

    let mut game = GameState::new_with_config(
        5,
        5,
        HashMap::new(),
        mud,
        &[Coordinates::new(0, 0)],
        Coordinates::new(2, 2), // P1 at mud start
        Coordinates::new(4, 4),
        100,
    );

    // Move P1 into mud passage to activate mud_timer
    let _undo = game.make_move(Direction::Up, Direction::Stay);
    assert!(game.player1.mud_timer > 0, "P1 should be stuck in mud");
    game
}

/// 5×5 game with 1 cheese at (1,0), P1 at (0,0), P2 at (4,4).
/// P1 moving RIGHT collects the cheese.
pub fn one_cheese_adjacent_game() -> GameState {
    GameState::new_with_config(
        5,
        5,
        HashMap::new(),
        Default::default(),
        &[Coordinates::new(1, 0)],
        Coordinates::new(0, 0),
        Coordinates::new(4, 4),
        100,
    )
}

/// 5×5 game with 1 cheese at (1,0), P1 at (0,0), P2 at (2,0).
/// Both can reach the cheese in one move (P1 RIGHT, P2 LEFT).
pub fn contested_cheese_game() -> GameState {
    GameState::new_with_config(
        5,
        5,
        HashMap::new(),
        Default::default(),
        &[Coordinates::new(1, 0)],
        Coordinates::new(0, 0),
        Coordinates::new(2, 0),
        100,
    )
}

/// Horizontal corridor: walls above and below row 0.
/// P1 at (0,0), P2 at (4,0), cheese at (2,0).
/// Only RIGHT and LEFT are real moves; UP and DOWN are blocked.
pub fn corridor_game() -> GameState {
    // Build walls: block UP from every cell in row 0
    let mut walls = HashMap::new();
    for x in 0..5 {
        walls
            .entry(Coordinates::new(x, 0))
            .or_insert_with(Vec::new)
            .push(Coordinates::new(x, 1));
        walls
            .entry(Coordinates::new(x, 1))
            .or_insert_with(Vec::new)
            .push(Coordinates::new(x, 0));
    }

    GameState::new_with_config(
        5,
        5,
        walls,
        Default::default(),
        &[Coordinates::new(2, 0)],
        Coordinates::new(0, 0),
        Coordinates::new(4, 0),
        100,
    )
}

/// Game that's already over: 1 cheese, max_turns=1, advance one turn so turn >= max_turns.
pub fn terminal_game() -> GameState {
    let mut game = GameState::new_with_config(
        5,
        5,
        HashMap::new(),
        Default::default(),
        &[Coordinates::new(4, 4)],
        Coordinates::new(0, 0),
        Coordinates::new(0, 1),
        1, // max_turns = 1
    );
    // Play one turn to reach max_turns
    let _undo = game.make_move(Direction::Stay, Direction::Stay);
    game
}

/// Game with few cheese and few turns — terminals appear within shallow search.
pub fn short_game() -> GameState {
    GameState::new_with_config(
        5,
        5,
        HashMap::new(),
        Default::default(),
        &[Coordinates::new(1, 0)],
        Coordinates::new(0, 0),
        Coordinates::new(2, 0),
        3, // Only 3 turns
    )
}

pub fn mud_game_both_stuck() -> GameState {
    let mut mud = MudMap::new();
    mud.insert(Coordinates::new(2, 2), Coordinates::new(2, 3), 3);
    mud.insert(Coordinates::new(3, 3), Coordinates::new(3, 4), 3);

    let mut game = GameState::new_with_config(
        5,
        5,
        HashMap::new(),
        mud,
        &[Coordinates::new(0, 0)],
        Coordinates::new(2, 2),
        Coordinates::new(3, 3),
        100,
    );

    let _undo = game.make_move(Direction::Up, Direction::Up);
    assert!(game.player1.mud_timer > 0, "P1 should be stuck in mud");
    assert!(game.player2.mud_timer > 0, "P2 should be stuck in mud");
    game
}
