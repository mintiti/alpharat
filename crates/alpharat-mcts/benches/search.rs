use std::time::Duration;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

use alpharat_mcts::{run_search, MCTSTree, SearchConfig, SmartUniformBackend};
use pyrat::{Coordinates, GameBuilder, GameState};
use rand::rngs::SmallRng;
use rand::SeedableRng;

fn open_game(width: u8, height: u8, cheese: &[Coordinates]) -> GameState {
    GameBuilder::new(width, height)
        .with_open_maze()
        .with_custom_positions(Coordinates::new(0, 0), Coordinates::new(width - 1, height - 1))
        .with_custom_cheese(cheese.to_vec())
        .with_max_turns(100)
        .build()
        .create(None)
        .unwrap()
}

fn open_5x5_game() -> GameState {
    let cheese: Vec<_> = [
        (1, 1),
        (2, 3),
        (3, 0),
        (0, 4),
        (4, 2),
    ]
    .iter()
    .map(|&(x, y)| Coordinates::new(x, y))
    .collect();
    open_game(5, 5, &cheese)
}

fn open_7x7_game() -> GameState {
    let cheese: Vec<_> = [
        (1, 1),
        (2, 5),
        (3, 0),
        (0, 6),
        (5, 2),
        (6, 4),
        (4, 3),
        (1, 5),
        (3, 6),
        (5, 0),
    ]
    .iter()
    .map(|&(x, y)| Coordinates::new(x, y))
    .collect();
    open_game(7, 7, &cheese)
}

fn bench_search(c: &mut Criterion) {
    let backend = SmartUniformBackend;
    let config = SearchConfig::default();
    let batch_size = 64u32;

    let fixtures: Vec<(&str, GameState)> = vec![
        ("5x5", open_5x5_game()),
        ("7x7", open_7x7_game()),
    ];

    for &sims in &[500u32, 2_000, 8_000, 50_000, 200_000] {
        let mut group = c.benchmark_group(format!("search/{sims}_sims"));
        group.measurement_time(Duration::from_secs(5));
        group.sample_size(if sims >= 50_000 { 20 } else { 50 });

        for (name, game) in &fixtures {
            group.throughput(Throughput::Elements(sims as u64));
            group.bench_with_input(BenchmarkId::from_parameter(name), game, |b, game| {
                b.iter(|| {
                    let mut tree = MCTSTree::new(game);
                    let mut rng = SmallRng::seed_from_u64(42);
                    run_search(&mut tree, game, &backend, &config, sims, batch_size, &mut rng)
                        .unwrap()
                })
            });
        }

        group.finish();
    }
}

criterion_group!(benches, bench_search);
criterion_main!(benches);
