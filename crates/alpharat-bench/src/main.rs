use std::time::Instant;

use alpharat_eval_core::SmartUniformBackend;
use pyrat::{Coordinates, GameBuilder, GameState, MazeParams};
use rand::rngs::SmallRng;
use rand::SeedableRng;

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

struct Args {
    grids: Vec<u8>,
    sims: Vec<u32>,
    maze: MazeType,
    iters: usize,
    batch_size: u32,
}

#[derive(Clone, Copy)]
enum MazeType {
    Open,
    Walled,
}

fn parse_args() -> Args {
    let mut grids: Option<Vec<u8>> = None;
    let mut sims: Option<Vec<u32>> = None;
    let mut maze = MazeType::Open;
    let mut iters = 10usize;
    let mut batch_size = 64u32;

    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--grids" => {
                i += 1;
                grids = Some(args[i].split(',').map(|s| s.trim().parse().unwrap()).collect());
            }
            "--sims" => {
                i += 1;
                sims = Some(args[i].split(',').map(|s| s.trim().parse().unwrap()).collect());
            }
            "--maze" => {
                i += 1;
                maze = match args[i].as_str() {
                    "open" => MazeType::Open,
                    "walled" => MazeType::Walled,
                    other => panic!("unknown maze type: {other} (expected: open, walled)"),
                };
            }
            "--iters" => {
                i += 1;
                iters = args[i].parse().unwrap();
            }
            "--batch-size" => {
                i += 1;
                batch_size = args[i].parse().unwrap();
            }
            "--help" | "-h" => {
                eprintln!("Usage: bench-compare [OPTIONS]");
                eprintln!();
                eprintln!("Options:");
                eprintln!("  --grids 7,11,15,21     Grid sizes (default: 7,11,15,21)");
                eprintln!("  --sims 10000,50000,...  Sim budgets (default: 10000,50000,200000)");
                eprintln!("  --maze open|walled      Maze type (default: open)");
                eprintln!("  --iters N               Iterations per config (default: 10)");
                eprintln!("  --batch-size N          Batch size (default: 64)");
                std::process::exit(0);
            }
            other => panic!("unknown flag: {other}"),
        }
        i += 1;
    }

    Args {
        grids: grids.unwrap_or_else(|| vec![7, 11, 15, 21]),
        sims: sims.unwrap_or_else(|| vec![10_000, 50_000, 200_000]),
        maze,
        iters,
        batch_size,
    }
}

// ---------------------------------------------------------------------------
// Game fixtures
// ---------------------------------------------------------------------------

/// Cheese count targeting ~20% density, matching the project convention.
fn cheese_count(size: u8) -> u8 {
    let cells = size as u16 * size as u16;
    (cells / 5).min(255) as u8
}

fn max_turns(size: u8) -> u16 {
    match size {
        s if s <= 5 => 30,
        s if s <= 7 => 50,
        s if s <= 11 => 80,
        s if s <= 15 => 120,
        s if s <= 21 => 180,
        s => (s as u16) * 10,
    }
}

/// Deterministic cheese placement on odd-parity cells (matches MCTS bench).
fn scatter_cheese(width: u8, height: u8, n: u8) -> Vec<Coordinates> {
    let mut cheese = Vec::new();
    let mut placed = 0u8;
    'outer: for y in 0..height {
        for x in 0..width {
            if (x + y) % 2 == 1 && (x, y) != (0, 0) && (x, y) != (width - 1, height - 1) {
                cheese.push(Coordinates::new(x, y));
                placed += 1;
                if placed >= n {
                    break 'outer;
                }
            }
        }
    }
    cheese
}

fn make_game(size: u8, maze: MazeType, seed: u64) -> GameState {
    let nc = cheese_count(size);
    let mt = max_turns(size);
    let cheese = scatter_cheese(size, size, nc);

    let base = GameBuilder::new(size, size);
    let with_maze = match maze {
        MazeType::Open => base.with_open_maze(),
        MazeType::Walled => base.with_random_maze(MazeParams {
            wall_density: 0.5,
            mud_density: 0.0,
            mud_range: 2,
            connected: true,
            symmetric: true,
        }),
    };

    with_maze
        .with_custom_positions(Coordinates::new(0, 0), Coordinates::new(size - 1, size - 1))
        .with_custom_cheese(cheese)
        .with_max_turns(mt)
        .build()
        .create(Some(seed))
        .unwrap()
}

// ---------------------------------------------------------------------------
// Timing harness
// ---------------------------------------------------------------------------

struct RunStats {
    median_secs: f64,
    nn_evals: u32,
    terminals: u32,
    collisions: u32,
    tt_entries: Option<usize>,
    tt_live: Option<usize>,
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = values.len();
    if n % 2 == 0 {
        (values[n / 2 - 1] + values[n / 2]) / 2.0
    } else {
        values[n / 2]
    }
}

fn run_mcts(
    game: &GameState,
    n_sims: u32,
    batch_size: u32,
    iters: usize,
) -> RunStats {
    let backend = SmartUniformBackend;
    let config = alpharat_mcts::SearchConfig::default();

    // Warmup
    {
        let mut tree = alpharat_mcts::MCTSTree::new(game);
        let mut rng = SmallRng::seed_from_u64(0);
        let _ = alpharat_mcts::run_search(&mut tree, game, &backend, &config, n_sims, batch_size, &mut rng);
    }

    let mut timings = Vec::with_capacity(iters);
    let mut last_result = None;

    for i in 0..iters {
        let mut tree = alpharat_mcts::MCTSTree::new(game);
        let mut rng = SmallRng::seed_from_u64(42 + i as u64);

        let start = Instant::now();
        let result = alpharat_mcts::run_search(
            &mut tree, game, &backend, &config, n_sims, batch_size, &mut rng,
        )
        .unwrap();
        timings.push(start.elapsed().as_secs_f64());
        last_result = Some(result);
    }

    let r = last_result.unwrap();
    RunStats {
        median_secs: median(&mut timings),
        nn_evals: r.nn_evals,
        terminals: r.terminals,
        collisions: r.collisions,
        tt_entries: None,
        tt_live: None,
    }
}

fn run_mcgs(
    game: &GameState,
    n_sims: u32,
    batch_size: u32,
    iters: usize,
) -> RunStats {
    let backend = SmartUniformBackend;
    let config = alpharat_mcgs::SearchConfig::default();

    // Warmup
    {
        let mut tree = alpharat_mcgs::MCGSTree::new(game);
        let mut rng = SmallRng::seed_from_u64(0);
        let _ = alpharat_mcgs::run_search(&mut tree, game, &backend, &config, n_sims, batch_size, &mut rng);
    }

    let mut timings = Vec::with_capacity(iters);
    let mut last_result = None;
    let mut last_tt_entries = 0;
    let mut last_tt_live = 0;

    for i in 0..iters {
        let mut tree = alpharat_mcgs::MCGSTree::new(game);
        let mut rng = SmallRng::seed_from_u64(42 + i as u64);

        let start = Instant::now();
        let result = alpharat_mcgs::run_search(
            &mut tree, game, &backend, &config, n_sims, batch_size, &mut rng,
        )
        .unwrap();
        timings.push(start.elapsed().as_secs_f64());

        last_tt_entries = tree.tt().len();
        last_tt_live = tree.tt().live_count();
        last_result = Some(result);
    }

    let r = last_result.unwrap();
    RunStats {
        median_secs: median(&mut timings),
        nn_evals: r.nn_evals,
        terminals: r.terminals,
        collisions: r.collisions,
        tt_entries: Some(last_tt_entries),
        tt_live: Some(last_tt_live),
    }
}

// ---------------------------------------------------------------------------
// Output
// ---------------------------------------------------------------------------

fn format_sims_per_sec(sims: u32, secs: f64) -> String {
    let rate = sims as f64 / secs;
    if rate >= 1_000_000.0 {
        format!("{:.1}M", rate / 1_000_000.0)
    } else if rate >= 1_000.0 {
        format!("{:.1}K", rate / 1_000.0)
    } else {
        format!("{:.0}", rate)
    }
}

fn pct(num: u32, denom: u32) -> String {
    if denom == 0 {
        return "-".to_string();
    }
    format!("{:.1}%", num as f64 / denom as f64 * 100.0)
}

fn format_count(n: Option<usize>) -> String {
    match n {
        Some(v) => {
            if v >= 1_000_000 {
                format!("{:.1}M", v as f64 / 1_000_000.0)
            } else if v >= 1_000 {
                format!("{:.1}K", v as f64 / 1_000.0)
            } else {
                format!("{v}")
            }
        }
        None => "-".to_string(),
    }
}

fn print_header() {
    println!(
        "{:<8} {:>8} {:>6} {:>10} {:>8} {:>8} {:>8} {:>8} {:>12} {:>12}",
        "Grid", "Sims", "Engine", "Sims/s", "Time", "NN%", "Term%", "Coll%", "TT Entries", "TT Live"
    );
    println!("{}", "-".repeat(110));
}

fn print_row(grid: &str, sims: u32, engine: &str, stats: &RunStats) {
    let total = stats.nn_evals + stats.terminals + stats.collisions;
    println!(
        "{:<8} {:>8} {:>6} {:>10} {:>7.1}ms {:>8} {:>8} {:>8} {:>12} {:>12}",
        grid,
        sims,
        engine,
        format_sims_per_sec(sims, stats.median_secs),
        stats.median_secs * 1000.0,
        pct(stats.nn_evals, total),
        pct(stats.terminals, total),
        pct(stats.collisions, total),
        format_count(stats.tt_entries),
        format_count(stats.tt_live),
    );
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    let args = parse_args();

    let maze_label = match args.maze {
        MazeType::Open => "open",
        MazeType::Walled => "walled",
    };
    println!(
        "MCGS vs MCTS comparison  |  maze: {maze_label}  |  batch: {}  |  iters: {}",
        args.batch_size, args.iters
    );
    println!();
    print_header();

    for &size in &args.grids {
        let label = format!("{size}x{size}");
        let game = make_game(size, args.maze, 42);

        for &sims in &args.sims {
            let mcts = run_mcts(&game, sims, args.batch_size, args.iters);
            print_row(&label, sims, "MCTS", &mcts);

            let mcgs = run_mcgs(&game, sims, args.batch_size, args.iters);
            print_row(&label, sims, "MCGS", &mcgs);

            // Blank line between sim budgets for readability
            println!();
        }
    }
}
