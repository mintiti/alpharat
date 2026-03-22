use pyrat::{Coordinates, GameBuilder, GameState};

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
