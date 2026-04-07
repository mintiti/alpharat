use pyrat::MudMap;
use pyrat::{Coordinates, Direction, GameBuilder, GameState};
use std::collections::HashMap;

pub fn open_5x5_game(p1: Coordinates, p2: Coordinates, cheese: &[Coordinates]) -> GameState {
    GameBuilder::new(5, 5)
        .with_open_maze()
        .with_custom_positions(p1, p2)
        .with_custom_cheese(cheese.to_vec())
        .with_max_turns(100)
        .build()
        .create(None)
        .unwrap()
}

pub fn wall_game(
    p1: Coordinates,
    p2: Coordinates,
    walls: HashMap<Coordinates, Vec<Coordinates>>,
    cheese: &[Coordinates],
) -> GameState {
    GameBuilder::new(5, 5)
        .with_custom_maze(walls, Default::default())
        .with_custom_positions(p1, p2)
        .with_custom_cheese(cheese.to_vec())
        .with_max_turns(100)
        .build()
        .create(None)
        .unwrap()
}

pub fn mud_game_p1_stuck() -> GameState {
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

    let _undo = game.make_move(Direction::Up, Direction::Stay);
    assert!(game.player1.mud_timer > 0, "P1 should be stuck in mud");
    game
}

pub fn mud_game_both_stuck() -> GameState {
    let mut mud = MudMap::new();
    mud.insert(Coordinates::new(2, 2), Coordinates::new(2, 3), 3);
    mud.insert(Coordinates::new(3, 3), Coordinates::new(3, 4), 3);

    let mut game = GameBuilder::new(5, 5)
        .with_custom_maze(HashMap::new(), mud)
        .with_custom_positions(Coordinates::new(2, 2), Coordinates::new(3, 3))
        .with_custom_cheese(vec![Coordinates::new(0, 0)])
        .with_max_turns(100)
        .build()
        .create(None)
        .unwrap();

    let _undo = game.make_move(Direction::Up, Direction::Up);
    assert!(game.player1.mud_timer > 0, "P1 should be stuck in mud");
    assert!(game.player2.mud_timer > 0, "P2 should be stuck in mud");
    game
}
