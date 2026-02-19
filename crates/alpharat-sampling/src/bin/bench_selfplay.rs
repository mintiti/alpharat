//! Benchmark run_self_play throughput: threads × mux_max × within-tree batch_size.
//!
//! Without ONNX (SmartUniform only):
//!   cargo run --release -p alpharat-sampling --bin bench_selfplay [num_games]
//!
//! With ONNX backend:
//!   cargo run --release -p alpharat-sampling --features onnx --bin bench_selfplay \
//!     [num_games] [onnx_model_path]

use alpharat_mcts::{Backend, SearchConfig, SmartUniformBackend};
use alpharat_sampling::selfplay::{run_self_play, SelfPlayConfig, SelfPlayStats};
use alpharat_sampling::MuxStats;
use pyrat::{CheeseConfig, GameState, MazeConfig};
use std::sync::Arc;

#[cfg(feature = "onnx")]
use alpharat_sampling::mux_backend::{MuxBackend, MuxConfig};
#[cfg(feature = "onnx")]
use alpharat_sampling::FlatEncoder;
#[cfg(feature = "onnx")]
use alpharat_sampling::OnnxBackend;

fn make_games(n: usize, width: u8, height: u8, cheese: u16, max_turns: u16) -> Vec<GameState> {
    (0..n)
        .map(|_| {
            let maze_config = MazeConfig {
                width,
                height,
                target_density: 0.0,
                connected: true,
                symmetry: true,
                mud_density: 0.0,
                mud_range: 0,
                seed: None,
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
    println!("{:-<134}", "");
    println!(
        "{:>8} {:>6} {:>7} {:>11} {:>11} {:>5} {:>10} {:>9} {:>8} {:>9} {:>8} {:>7} {:>8} {:>8}",
        "threads", "batch", "mux_max", "sims/s", "nn_ev/s", "nn%", "pos/s", "games/s", "elapsed",
        "avg_bat", "nn_util", "nn_s", "wait_s", "turns"
    );
    println!("{:-<134}", "");
}

fn print_row(
    num_threads: u32,
    batch_size: u32,
    mux_max: &str,
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
        "{:>8} {:>6} {:>7} {:>11.0} {:>11.0} {:>4.0}% {:>10.1} {:>9.1} {:>7.2}s {:>9} {:>8} {:>7} {:>8} {:>8.1}",
        num_threads,
        batch_size,
        mux_max,
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
    games: &[GameState],
    search_config: &SearchConfig,
    n_sims: u32,
    thread_counts: &[u32],
    batch_sizes: &[u32],
) {
    let backend = SmartUniformBackend;
    print_header(label);

    for &batch_size in batch_sizes {
        for &num_threads in thread_counts {
            let config = SelfPlayConfig {
                n_sims,
                batch_size,
                num_threads,
            };
            let result = run_self_play(games, &backend, search_config, &config, None);
            print_row(num_threads, batch_size, "-", &result.stats, None);
        }
    }
}

#[cfg(feature = "onnx")]
fn run_bench_onnx(
    label: &str,
    games: &[GameState],
    search_config: &SearchConfig,
    n_sims: u32,
    model_path: &str,
    width: u8,
    height: u8,
    thread_counts: &[u32],
    batch_sizes: &[u32],
    mux_batch_sizes: &[usize],
) {
    print_header(label);

    for &batch_size in batch_sizes {
        for &mux_max in mux_batch_sizes {
            for &num_threads in thread_counts {
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
                    run_self_play(games, backend.as_ref(), search_config, &config, None);
                let mux_label = format!("{}", mux_max);
                print_row(
                    num_threads,
                    batch_size,
                    &mux_label,
                    &result.stats,
                    mux_stats.as_ref(),
                );
            }
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let num_games: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(200);

    #[cfg(feature = "onnx")]
    let onnx_model: Option<String> = args.get(2).cloned();
    #[cfg(not(feature = "onnx"))]
    let _onnx_model: Option<String> = args.get(2).cloned();

    let thread_counts = [1, 4, 8, 16, 32, 64];
    let batch_sizes_uniform = [16];
    let batch_sizes_onnx = [8, 16, 32];
    let mux_sizes = [128, 256, 512, 1024];

    let search_7x7 = SearchConfig {
        c_puct: 0.512,
        fpu_reduction: 0.459,
        force_k: 0.103,
        noise_epsilon: 0.25,
        noise_concentration: 10.83,
        max_collisions: 0,
    };

    // --- SmartUniform baseline (no NN) ---
    let games_7x7 = make_games(num_games, 7, 7, 10, 50);
    run_bench_uniform(
        &format!(
            "7x7 open, 1897 sims, {} games — SmartUniform (no NN)",
            num_games
        ),
        &games_7x7,
        &search_7x7,
        1897,
        &thread_counts,
        &batch_sizes_uniform,
    );

    // --- ONNX backend: sweep batch_size × mux_max × threads ---
    #[cfg(feature = "onnx")]
    if let Some(model_path) = onnx_model {
        let games_7x7 = make_games(num_games, 7, 7, 10, 50);
        run_bench_onnx(
            &format!(
                "7x7 open, 1897 sims, {} games — ONNX (batch × mux × threads)",
                num_games
            ),
            &games_7x7,
            &search_7x7,
            1897,
            &model_path,
            7,
            7,
            &thread_counts,
            &batch_sizes_onnx,
            &mux_sizes,
        );
    }
}
