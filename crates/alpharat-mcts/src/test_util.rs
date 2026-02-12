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
