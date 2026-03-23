use std::time::Duration;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

use alpharat_mcts::{run_search, MCTSTree, SearchConfig, SmartUniformBackend};
use pyrat::{Coordinates, Direction, GameBuilder, GameState};
use rand::rngs::SmallRng;
use rand::SeedableRng;

fn open_game(width: u8, height: u8, n_cheese: u8, max_turns: u16) -> GameState {
    // Scatter cheese deterministically across the grid.
    let mut cheese = Vec::new();
    let mut placed = 0u8;
    'outer: for y in 0..height {
        for x in 0..width {
            if (x + y) % 2 == 1 && (x, y) != (0, 0) && (x, y) != (width - 1, height - 1) {
                cheese.push(Coordinates::new(x, y));
                placed += 1;
                if placed >= n_cheese {
                    break 'outer;
                }
            }
        }
    }

    GameBuilder::new(width, height)
        .with_open_maze()
        .with_custom_positions(Coordinates::new(0, 0), Coordinates::new(width - 1, height - 1))
        .with_custom_cheese(cheese)
        .with_max_turns(max_turns)
        .build()
        .create(None)
        .unwrap()
}

fn bench_search(c: &mut Criterion) {
    let backend = SmartUniformBackend;
    let config = SearchConfig::default();
    let batch_size = 64u32;

    // (label, width, height, cheese count, max turns)
    let specs: Vec<(&str, u8, u8, u8, u16)> = vec![
        ("5x5", 5, 5, 5, 30),
        ("7x7", 7, 7, 10, 50),
        ("11x11", 11, 11, 24, 80),
        ("15x15", 15, 15, 45, 120),
    ];

    let fixtures: Vec<(&str, GameState)> = specs
        .iter()
        .map(|&(label, w, h, nc, mt)| (label, open_game(w, h, nc, mt)))
        .collect();

    for &sims in &[500u32, 2_000, 8_000, 50_000, 200_000] {
        let mut group = c.benchmark_group(format!("search/{sims}_sims"));
        group.measurement_time(Duration::from_secs(5));
        group.sample_size(if sims >= 50_000 { 20 } else { 50 });

        for (name, game) in &fixtures {
            // Skip small grids at high sim counts (saturated, not interesting).
            if sims >= 50_000 && (*name == "5x5") {
                continue;
            }

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

fn best_action(visits: &[f32; 5]) -> u8 {
    visits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .unwrap()
        .0 as u8
}

fn bench_search_reuse(c: &mut Criterion) {
    let backend = SmartUniformBackend;
    let config = SearchConfig::default();
    let batch_size = 64u32;

    let specs: Vec<(&str, u8, u8, u8, u16)> = vec![("5x5", 5, 5, 5, 30), ("7x7", 7, 7, 10, 50)];

    let fixtures: Vec<(&str, GameState)> = specs
        .iter()
        .map(|&(label, w, h, nc, mt)| (label, open_game(w, h, nc, mt)))
        .collect();

    for &sims in &[500u32, 2_000, 8_000] {
        let mut group = c.benchmark_group(format!("search_reuse/{sims}_sims"));
        group.measurement_time(Duration::from_secs(5));
        group.sample_size(50);

        for (name, game) in &fixtures {
            group.throughput(Throughput::Elements(sims as u64));
            group.bench_with_input(BenchmarkId::from_parameter(name), game, |b, game| {
                b.iter_batched(
                    || {
                        // Setup: search N sims, then advance one move.
                        let mut tree = MCTSTree::new(game);
                        let mut rng = SmallRng::seed_from_u64(42);
                        let result = run_search(
                            &mut tree, game, &backend, &config, sims, batch_size, &mut rng,
                        )
                        .unwrap();

                        let a1 = best_action(&result.visit_counts_p1);
                        let a2 = best_action(&result.visit_counts_p2);

                        let mut game = game.clone();
                        let d1 = Direction::try_from(a1).expect("valid direction");
                        let d2 = Direction::try_from(a2).expect("valid direction");
                        game.make_move(d1, d2);

                        if !tree.advance_root(a1, a2) {
                            tree.reinit(&game);
                        }

                        (tree, game, SmallRng::seed_from_u64(123))
                    },
                    |(mut tree, game, mut rng)| {
                        run_search(
                            &mut tree, &game, &backend, &config, sims, batch_size, &mut rng,
                        )
                        .unwrap()
                    },
                    criterion::BatchSize::LargeInput,
                );
            });
        }

        group.finish();
    }
}

criterion_group!(benches, bench_search, bench_search_reuse);
criterion_main!(benches);
