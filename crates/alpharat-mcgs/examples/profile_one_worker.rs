//! Print a phase profile for the gated one-worker MCGS protocol.
//!
//! Run with:
//! `cargo run --release -p alpharat-mcgs --features bench-internals --example profile_one_worker -- 10`

use std::time::{Duration, Instant};

use alpharat_mcgs::{
    run_search_one_worker_profiled, MCGSTree, SearchConfig, SearchTimings, SmartUniformBackend,
};
use pyrat::{Coordinates, GameBuilder, GameState};
use rand::rngs::SmallRng;
use rand::SeedableRng;

fn benchmark_game() -> GameState {
    let mut cheese = Vec::new();
    'outer: for y in 0..7 {
        for x in 0..7 {
            if (x + y) % 2 == 1 && (x, y) != (0, 0) && (x, y) != (6, 6) {
                cheese.push(Coordinates::new(x, y));
                if cheese.len() == 10 {
                    break 'outer;
                }
            }
        }
    }

    GameBuilder::new(7, 7)
        .with_open_maze()
        .with_custom_positions(Coordinates::new(0, 0), Coordinates::new(6, 6))
        .with_custom_cheese(cheese)
        .with_max_turns(50)
        .build()
        .create(None)
        .unwrap()
}

fn add_timings(total: &mut SearchTimings, run: SearchTimings) {
    total.batches += run.batches;
    total.gather_wait += run.gather_wait;
    total.gather_hold += run.gather_hold;
    total.inference += run.inference;
    total.settle_wait += run.settle_wait;
    total.settle_hold += run.settle_hold;
    total.cleanup_wait += run.cleanup_wait;
    total.cleanup_hold += run.cleanup_hold;
    total.extract_wait += run.extract_wait;
    total.extract_hold += run.extract_hold;
}

fn print_phase(label: &str, duration: Duration, runs: u32, total: Duration) {
    let average = duration.div_f64(f64::from(runs));
    let percent = duration.as_secs_f64() / total.as_secs_f64() * 100.0;
    println!(
        "{label:<16} {:>8.3} ms/run  {:>6.2}%",
        average.as_secs_f64() * 1_000.0,
        percent
    );
}

fn main() {
    let runs: u32 = std::env::args()
        .nth(1)
        .map(|value| value.parse().expect("iteration count must be a positive integer"))
        .unwrap_or(10);
    assert!(runs > 0, "iteration count must be positive");

    let game = benchmark_game();
    let backend = SmartUniformBackend;
    let config = SearchConfig::default();
    let mut total_timings = SearchTimings::default();
    let mut total_wall = Duration::ZERO;
    let mut last_summary = None;

    for _ in 0..runs {
        let mut tree = MCGSTree::new(&game);
        let mut rng = SmallRng::seed_from_u64(42);
        let mut timings = SearchTimings::default();
        let started = Instant::now();
        let profile = run_search_one_worker_profiled(
            &mut tree,
            &game,
            &backend,
            &config,
            8_000,
            64,
            &mut rng,
            &mut timings,
        )
        .unwrap();
        total_wall += started.elapsed();
        add_timings(&mut total_timings, timings);
        assert_eq!(profile.ledger.outstanding(), 0);

        let summary = (
            profile.result.nn_evals,
            profile.result.terminals,
            profile.result.tt_stop_hits,
            profile.result.collisions,
            profile.ledger.reserved,
            profile.ledger.committed,
            profile.ledger.cancelled,
        );
        if let Some(previous) = last_summary {
            assert_eq!(summary, previous, "fixed-seed profile changed between runs");
        }
        last_summary = Some(summary);
    }

    let (nn_evals, terminals, tt_stops, collisions, reserved, committed, cancelled) =
        last_summary.unwrap();
    println!("gated one-worker: 7x7, 8000 simulations, batch 64, {runs} runs");
    println!(
        "productive: nn={nn_evals}, terminal={terminals}, tt-stop={tt_stops}; collisions={collisions}"
    );
    println!(
        "ledger: reserved={reserved}, committed={committed}, cancelled={cancelled}, outstanding=0"
    );
    println!(
        "wall             {:>8.3} ms/run",
        total_wall.div_f64(f64::from(runs)).as_secs_f64() * 1_000.0
    );
    print_phase("gather wait", total_timings.gather_wait, runs, total_wall);
    print_phase("gather hold", total_timings.gather_hold, runs, total_wall);
    print_phase("inference", total_timings.inference, runs, total_wall);
    print_phase("settle wait", total_timings.settle_wait, runs, total_wall);
    print_phase("settle hold", total_timings.settle_hold, runs, total_wall);
    print_phase("cleanup wait", total_timings.cleanup_wait, runs, total_wall);
    print_phase("cleanup hold", total_timings.cleanup_hold, runs, total_wall);
    print_phase("extract wait", total_timings.extract_wait, runs, total_wall);
    print_phase("extract hold", total_timings.extract_hold, runs, total_wall);
    println!(
        "batches          {:>8.1} /run",
        f64::from(total_timings.batches) / f64::from(runs)
    );
}
