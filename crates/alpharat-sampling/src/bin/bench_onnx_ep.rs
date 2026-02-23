//! Quick comparison: ONNX CPU vs CoreML EP on a single batch.
//!
//! cargo run --release -p alpharat-sampling --features onnx-coreml --bin bench_onnx_ep <model_path>

#[cfg(feature = "onnx")]
fn main() {
    use alpharat_mcts::Backend;
    use alpharat_sampling::{FlatEncoder, OnnxBackend};
    use pyrat::{GameBuilder, GameState, MazeParams};
    use std::time::Instant;

    let args: Vec<String> = std::env::args().collect();
    let model_path = args.get(1).expect("usage: bench_onnx_ep <model.onnx>");

    let _encoder = FlatEncoder::new(7, 7);

    // Create some test games
    let games: Vec<GameState> = (0..64)
        .map(|i| {
            let config = GameBuilder::new(7, 7)
                .with_max_turns(50)
                .with_random_maze(MazeParams {
                    wall_density: 0.0,
                    mud_density: 0.0,
                    mud_range: 2,
                    connected: true,
                    symmetric: true,
                })
                .with_corner_positions()
                .with_random_cheese(10, true)
                .build();
            config.create(Some(i as u64)).unwrap()
        })
        .collect();

    let game_refs: Vec<&GameState> = games.iter().collect();

    for batch_size in [1, 8, 16, 32, 64] {
        let batch = &game_refs[..batch_size];

        // CPU backend
        let cpu_backend = OnnxBackend::new(model_path, FlatEncoder::new(7, 7))
            .expect("failed to create CPU ONNX backend");
        // Warmup
        for _ in 0..5 { cpu_backend.evaluate_batch(batch).unwrap(); }
        let start = Instant::now();
        let iters = 200;
        for _ in 0..iters { cpu_backend.evaluate_batch(batch).unwrap(); }
        let cpu_us = start.elapsed().as_micros() as f64 / iters as f64;

        // CoreML backend
        #[cfg(feature = "onnx-coreml")]
        {
            let coreml_backend = OnnxBackend::with_coreml(model_path, FlatEncoder::new(7, 7))
                .expect("failed to create CoreML ONNX backend");
            // Warmup
            for _ in 0..5 { coreml_backend.evaluate_batch(batch).unwrap(); }
            let start = Instant::now();
            for _ in 0..iters { coreml_backend.evaluate_batch(batch).unwrap(); }
            let coreml_us = start.elapsed().as_micros() as f64 / iters as f64;

            println!(
                "batch={:>3}  CPU: {:>8.0}us  CoreML: {:>8.0}us  ratio: {:.2}x",
                batch_size, cpu_us, coreml_us, cpu_us / coreml_us
            );
        }

        #[cfg(not(feature = "onnx-coreml"))]
        println!("batch={:>3}  CPU: {:>8.0}us  (CoreML not enabled)", batch_size, cpu_us);
    }
}

#[cfg(not(feature = "onnx"))]
fn main() {
    eprintln!("Build with --features onnx or onnx-coreml");
}
