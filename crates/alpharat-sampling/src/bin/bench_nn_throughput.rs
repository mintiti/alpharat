//! Isolated ONNX inference throughput benchmark.
//!
//! Measures the NN ceiling: encoding + inference without MCTS, mux, or game threads.
//!
//! Usage:
//!   cargo run --release -p alpharat-sampling --features onnx --bin bench_nn_throughput -- <model.onnx>
//!   cargo run --release -p alpharat-sampling --features onnx --bin bench_nn_throughput -- <model.onnx> --width 5 --height 5
//!   cargo run --release -p alpharat-sampling --features onnx-coreml --bin bench_nn_throughput -- <model.onnx>

#[cfg(feature = "onnx")]
fn main() {
    use alpharat_sampling::encoder::ObservationEncoder;
    use alpharat_sampling::FlatEncoder;
    use ort::session::Session;
    use ort::value::Tensor;
    use pyrat::{CheeseConfig, GameState, MazeConfig};
    use std::time::Instant;

    // --- Parse args ---
    let args: Vec<String> = std::env::args().collect();
    let model_path = args.get(1).expect("usage: bench_nn_throughput <model.onnx> [--width W] [--height H]");

    let mut width: u8 = 7;
    let mut height: u8 = 7;
    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--width" => {
                width = args[i + 1].parse().expect("invalid width");
                i += 2;
            }
            "--height" => {
                height = args[i + 1].parse().expect("invalid height");
                i += 2;
            }
            other => panic!("unknown arg: {other}"),
        }
    }

    let encoder = FlatEncoder::new(width, height);
    let obs_dim = encoder.obs_dim();
    let cheese_count = ((width as u16 * height as u16) as f64 * 0.2).round() as u16;
    let max_turns: u16 = if width <= 5 { 30 } else { 50 };

    // --- Generate random games ---
    let num_games: usize = 4096;
    let games: Vec<GameState> = (0..num_games)
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
                count: cheese_count,
                symmetry: true,
            };
            let mut g = GameState::new_random(width, height, maze_config, cheese_config);
            g.max_turns = max_turns;
            g
        })
        .collect();

    // --- Pre-encode all games ---
    let mut encoded_buf = vec![0.0f32; num_games * obs_dim];
    let encode_start = Instant::now();
    for (i, game) in games.iter().enumerate() {
        encoder.encode_into(game, &mut encoded_buf, i * obs_dim);
    }
    let encode_all_us = encode_start.elapsed().as_micros();
    let encode_per_pos_ns = (encode_all_us as f64 * 1000.0) / num_games as f64;

    println!("ONNX Inference Throughput — {}x{}, obs_dim={}", width, height, obs_dim);
    println!("Pre-encoded {} games in {:.1}ms ({:.0}ns/pos)\n", num_games, encode_all_us as f64 / 1000.0, encode_per_pos_ns);

    let batch_sizes: Vec<usize> = vec![1, 8, 16, 32, 64, 128, 256, 512, 1024, 2048];
    let intra_thread_counts = [1, 2, 4];
    let warmup_iters = 50;
    let bench_iters = 500;

    // --- Build sessions for each EP × intra_threads combo ---
    struct EpConfig {
        name: &'static str,
        intra_threads: usize,
    }

    let mut ep_configs: Vec<EpConfig> = Vec::new();
    for &threads in &intra_thread_counts {
        ep_configs.push(EpConfig {
            name: "CPU",
            intra_threads: threads,
        });
    }

    // CoreML only makes sense with 1 intra thread (it manages its own parallelism)
    #[cfg(feature = "onnx-coreml")]
    ep_configs.push(EpConfig {
        name: "CoreML",
        intra_threads: 1,
    });

    #[cfg(feature = "onnx-cuda")]
    ep_configs.push(EpConfig {
        name: "CUDA",
        intra_threads: 1,
    });

    for ep_config in &ep_configs {
        // Build session
        let builder = Session::builder()
            .expect("failed to create session builder")
            .with_intra_threads(ep_config.intra_threads)
            .expect("failed to set intra threads");

        let mut session = match ep_config.name {
            #[cfg(feature = "onnx-coreml")]
            "CoreML" => builder
                .with_execution_providers([
                    ort::execution_providers::CoreMLExecutionProvider::default().build(),
                ])
                .expect("failed to register CoreML EP")
                .commit_from_file(model_path)
                .expect("failed to load model"),
            #[cfg(feature = "onnx-cuda")]
            "CUDA" => builder
                .with_execution_providers([
                    ort::execution_providers::CUDAExecutionProvider::default().build(),
                ])
                .expect("failed to register CUDA EP")
                .commit_from_file(model_path)
                .expect("failed to load model"),
            _ => builder
                .commit_from_file(model_path)
                .expect("failed to load model"),
        };

        println!(
            "EP: {} | intra_threads: {} | warmup: {} | iters: {}",
            ep_config.name, ep_config.intra_threads, warmup_iters, bench_iters
        );
        println!(
            "  {:>6} {:>10} {:>10} {:>10} {:>14} {:>14}",
            "batch", "infer_µs", "encode_µs", "total_µs", "pos/s(infer)", "pos/s(total)"
        );
        println!("  {:-<80}", "");

        let mut peak_infer_pos_s: f64 = 0.0;
        let mut peak_total_pos_s: f64 = 0.0;

        for &batch_size in &batch_sizes {
            if batch_size > num_games {
                break;
            }

            // --- Inference-only benchmark (from pre-encoded buffer) ---
            let batch_data = &encoded_buf[..batch_size * obs_dim];

            // Warmup
            for _ in 0..warmup_iters {
                let input =
                    Tensor::from_array(([batch_size, obs_dim], batch_data.to_vec()))
                        .expect("failed to create tensor");
                let _outputs = session
                    .run(ort::inputs!["observation" => input])
                    .expect("inference failed");
            }

            // Measure inference only
            let start = Instant::now();
            for _ in 0..bench_iters {
                let input =
                    Tensor::from_array(([batch_size, obs_dim], batch_data.to_vec()))
                        .expect("failed to create tensor");
                let _outputs = session
                    .run(ort::inputs!["observation" => input])
                    .expect("inference failed");
            }
            let infer_total_us = start.elapsed().as_micros() as f64;
            let infer_per_batch_us = infer_total_us / bench_iters as f64;

            // --- Full pipeline benchmark (encode + infer) ---
            let game_refs: Vec<&GameState> = games[..batch_size].iter().collect();

            // Warmup
            for _ in 0..warmup_iters {
                let mut buf = vec![0.0f32; batch_size * obs_dim];
                for (j, game) in game_refs.iter().enumerate() {
                    encoder.encode_into(game, &mut buf, j * obs_dim);
                }
                let input =
                    Tensor::from_array(([batch_size, obs_dim], buf))
                        .expect("failed to create tensor");
                let _outputs = session
                    .run(ort::inputs!["observation" => input])
                    .expect("inference failed");
            }

            // Measure encode + infer
            let start = Instant::now();
            for _ in 0..bench_iters {
                let mut buf = vec![0.0f32; batch_size * obs_dim];
                for (j, game) in game_refs.iter().enumerate() {
                    encoder.encode_into(game, &mut buf, j * obs_dim);
                }
                let input =
                    Tensor::from_array(([batch_size, obs_dim], buf))
                        .expect("failed to create tensor");
                let _outputs = session
                    .run(ort::inputs!["observation" => input])
                    .expect("inference failed");
            }
            let total_total_us = start.elapsed().as_micros() as f64;
            let total_per_batch_us = total_total_us / bench_iters as f64;
            let encode_per_batch_us = total_per_batch_us - infer_per_batch_us;

            let infer_pos_s = batch_size as f64 / (infer_per_batch_us / 1_000_000.0);
            let total_pos_s = batch_size as f64 / (total_per_batch_us / 1_000_000.0);

            if infer_pos_s > peak_infer_pos_s {
                peak_infer_pos_s = infer_pos_s;
            }
            if total_pos_s > peak_total_pos_s {
                peak_total_pos_s = total_pos_s;
            }

            println!(
                "  {:>6} {:>9.1} {:>9.1} {:>9.1} {:>14} {:>14}",
                batch_size,
                infer_per_batch_us,
                encode_per_batch_us,
                total_per_batch_us,
                format_throughput(infer_pos_s),
                format_throughput(total_pos_s),
            );
        }

        println!(
            "\n  Peak: {} pos/s (inference only), {} pos/s (with encoding)\n",
            format_throughput(peak_infer_pos_s),
            format_throughput(peak_total_pos_s),
        );
    }
}

#[cfg(feature = "onnx")]
fn format_throughput(pos_per_sec: f64) -> String {
    if pos_per_sec >= 1_000_000.0 {
        format!("{:.1}M", pos_per_sec / 1_000_000.0)
    } else if pos_per_sec >= 1_000.0 {
        format!("{:.0}k", pos_per_sec / 1_000.0)
    } else {
        format!("{:.0}", pos_per_sec)
    }
}

#[cfg(not(feature = "onnx"))]
fn main() {
    eprintln!("Build with --features onnx (or onnx-coreml / onnx-cuda)");
}
