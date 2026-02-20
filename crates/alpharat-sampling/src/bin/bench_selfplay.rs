//! Benchmark run_self_play throughput: threads × mux_max × within-tree batch_size.
//!
//! Game count scales automatically: 16 games per thread (enough work to amortize startup).
//!
//! Without ONNX (SmartUniform only):
//!   cargo run --release -p alpharat-sampling --bin bench_selfplay
//!
//! With ONNX backend:
//!   cargo run --release -p alpharat-sampling --features onnx-cuda --bin bench_selfplay <model.onnx>

use alpharat_mcts::{SearchConfig, SmartUniformBackend};
use alpharat_sampling::selfplay::{run_self_play, SelfPlayConfig, SelfPlayStats};
use alpharat_sampling::MuxStats;
use pyrat::{CheeseConfig, GameState, MazeConfig};
use std::sync::Arc;

#[cfg(feature = "onnx")]
use alpharat_mcts::Backend;
#[cfg(feature = "onnx")]
use alpharat_sampling::mux_backend::{MuxBackend, MuxConfig};
#[cfg(feature = "onnx")]
use alpharat_sampling::FlatEncoder;
#[cfg(feature = "onnx")]
use alpharat_sampling::OnnxBackend;

fn make_games(n: usize, width: u8, height: u8, cheese: u16, max_turns: u16) -> Vec<GameState> {
    (0..n)
        .map(|i| {
            let maze_config = MazeConfig {
                width,
                height,
                target_density: 0.0,
                connected: true,
                symmetry: true,
                mud_density: 0.0,
                mud_range: 0,
                seed: Some(i as u64),
            };
            let cheese_config = CheeseConfig {
                count: cheese,
                symmetry: true,
            };
            let mut game = GameState::new_random(width, height, maze_config, cheese_config);
            game.max_turns = max_turns;
            game
        })
        .collect()
}

fn print_header(label: &str) {
    println!("\n{}", label);
    println!("{:-<140}", "");
    println!(
        "{:>8} {:>6} {:>7} {:>6} {:>11} {:>11} {:>5} {:>10} {:>9} {:>8} {:>9} {:>8} {:>7} {:>8} {:>8}",
        "threads", "batch", "mux_max", "games", "sims/s", "nn_ev/s", "nn%", "pos/s", "games/s", "elapsed",
        "avg_bat", "nn_util", "nn_s", "wait_s", "turns"
    );
    println!("{:-<140}", "");
}

fn print_row(
    num_threads: u32,
    batch_size: u32,
    mux_max: &str,
    num_games: usize,
    s: &SelfPlayStats,
    mux_stats: Option<&Arc<MuxStats>>,
) {
    let (avg_bat, nn_util, nn_s, wait_s) = match mux_stats {
        Some(ms) => (
            format!("{:.1}", ms.avg_batch_size()),
            format!("{:.0}%", ms.nn_utilization() * 100.0),
            format!("{:.1}", ms.nn_time_secs()),
            format!("{:.1}", ms.wait_time_secs()),
        ),
        None => ("-".into(), "-".into(), "-".into(), "-".into()),
    };

    println!(
        "{:>8} {:>6} {:>7} {:>6} {:>11.0} {:>11.0} {:>4.0}% {:>10.1} {:>9.1} {:>7.2}s {:>9} {:>8} {:>7} {:>8} {:>8.1}",
        num_threads,
        batch_size,
        mux_max,
        num_games,
        s.simulations_per_second(),
        s.nn_evals_per_second(),
        s.nn_eval_fraction() * 100.0,
        s.positions_per_second(),
        s.games_per_second(),
        s.elapsed_secs,
        avg_bat,
        nn_util,
        nn_s,
        wait_s,
        s.avg_turns(),
    );
}

fn run_bench_uniform(
    label: &str,
    search_config: &SearchConfig,
    n_sims: u32,
    thread_counts: &[u32],
    width: u8,
    height: u8,
) {
    let backend = SmartUniformBackend;

    // Pre-generate the max game set once — all configs use slices of the same games.
    let max_games = thread_counts
        .iter()
        .map(|&t| (t as usize) * GAMES_PER_THREAD)
        .max()
        .unwrap();
    let games = make_games(max_games, width, height, 10, 50);

    print_header(label);

    // SmartUniform doesn't batch NN calls, so batch_size is irrelevant — use fixed value.
    let batch_size = 16;

    for &num_threads in thread_counts {
        let num_games = (num_threads as usize) * GAMES_PER_THREAD;

        let config = SelfPlayConfig {
            n_sims,
            batch_size,
            num_threads,
        };
        let result = run_self_play(&games[..num_games], &backend, search_config, &config, None);
        print_row(num_threads, batch_size, "-", num_games, &result.stats, None);
    }
}

#[cfg(feature = "onnx")]
fn make_onnx_backend(model_path: &str, width: u8, height: u8) -> OnnxBackend<FlatEncoder> {
    let encoder = FlatEncoder::new(width, height);

    #[cfg(feature = "onnx-coreml")]
    let onnx_backend = OnnxBackend::with_coreml(model_path, encoder);
    #[cfg(all(feature = "onnx-cuda", not(feature = "onnx-coreml")))]
    let onnx_backend = OnnxBackend::with_cuda(model_path, encoder);
    #[cfg(all(
        feature = "onnx",
        not(feature = "onnx-coreml"),
        not(feature = "onnx-cuda")
    ))]
    let onnx_backend = OnnxBackend::new(model_path, encoder);

    onnx_backend
}

#[cfg(feature = "onnx")]
fn run_bench_onnx(
    label: &str,
    search_config: &SearchConfig,
    n_sims: u32,
    model_path: &str,
    width: u8,
    height: u8,
    thread_counts: &[u32],
    batch_sizes: &[u32],
    mux_batch_sizes: &[usize],
) {
    // Pre-generate the max game set once — all configs use slices of the same games.
    let max_games = thread_counts
        .iter()
        .map(|&t| (t as usize) * GAMES_PER_THREAD)
        .max()
        .unwrap();
    let games = make_games(max_games, width, height, 10, 50);

    // Warmup: one inference to initialize CUDA context / kernel compilation.
    // Without this, the first config in the sweep eats the cold-start cost.
    {
        let warmup_backend = make_onnx_backend(model_path, width, height);
        let _ = warmup_backend.evaluate(&games[0]);
    }

    print_header(label);

    for &batch_size in batch_sizes {
        for &mux_max in mux_batch_sizes {
            for &num_threads in thread_counts {
                let num_games = (num_threads as usize) * GAMES_PER_THREAD;

                let onnx_backend = make_onnx_backend(model_path, width, height);

                // Single-thread: use OnnxBackend directly (no mux overhead).
                // Multi-thread: wrap in MuxBackend for cross-game batching.
                let (backend, mux_stats): (Box<dyn Backend>, Option<Arc<MuxStats>>) =
                    if num_threads == 1 {
                        (Box::new(onnx_backend), None)
                    } else {
                        let mux = MuxBackend::new(
                            onnx_backend,
                            MuxConfig {
                                max_batch_size: mux_max,
                            },
                        );
                        let stats = mux.stats().clone();
                        (Box::new(mux), Some(stats))
                    };

                let config = SelfPlayConfig {
                    n_sims,
                    batch_size,
                    num_threads,
                };

                let result =
                    run_self_play(&games[..num_games], backend.as_ref(), search_config, &config, None);
                let mux_label = format!("{}", mux_max);
                print_row(
                    num_threads,
                    batch_size,
                    &mux_label,
                    num_games,
                    &result.stats,
                    mux_stats.as_ref(),
                );
            }
        }
    }
}

/// Games per thread — enough work to amortize thread startup and get stable throughput.
const GAMES_PER_THREAD: usize = 16;

fn main() {
    let args: Vec<String> = std::env::args().collect();

    #[cfg(feature = "onnx")]
    let onnx_model: Option<String> = args.get(1).cloned();
    #[cfg(not(feature = "onnx"))]
    let _onnx_model: Option<String> = args.get(1).cloned();

    // SmartUniform: quick sweep, CPU-bound so lower threads are fine
    let thread_counts_uniform = [1, 8, 32, 64];
    // ONNX: need many threads to generate enough pending requests for GPU batching
    #[cfg(feature = "onnx")]
    let thread_counts_onnx = [32, 64, 128, 256, 512];
    #[cfg(feature = "onnx")]
    let batch_sizes_onnx = [16, 32, 64];
    #[cfg(feature = "onnx")]
    let mux_sizes = [1024, 2048, 4096, 8192];

    let search_7x7 = SearchConfig {
        c_puct: 0.512,
        fpu_reduction: 0.459,
        force_k: 0.103,
        noise_epsilon: 0.25,
        noise_concentration: 10.83,
        max_collisions: 0,
    };

    // --- SmartUniform baseline (no NN) ---
    run_bench_uniform(
        &format!(
            "7x7 open, 1897 sims, {}g/thread — SmartUniform (no NN)",
            GAMES_PER_THREAD
        ),
        &search_7x7,
        1897,
        &thread_counts_uniform,
        7,
        7,
    );

    // --- ONNX backend: sweep batch_size × mux_max × threads ---
    #[cfg(feature = "onnx")]
    if let Some(model_path) = onnx_model {
        run_bench_onnx(
            &format!(
                "7x7 open, 1897 sims, {}g/thread — ONNX (batch × mux × threads)",
                GAMES_PER_THREAD
            ),
            &search_7x7,
            1897,
            &model_path,
            7,
            7,
            &thread_counts_onnx,
            &batch_sizes_onnx,
            &mux_sizes,
        );
    }
}
