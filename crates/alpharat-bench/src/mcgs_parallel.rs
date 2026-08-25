use std::collections::BTreeSet;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;

use alpharat_bench::calibration::{BackendRequest, OnnxProvider, SearchRequest, TrialStatus};
use alpharat_bench::production::{build_search_backend, ProductionBackendConfig};
use alpharat_bench::search::{
    default_search_workload, load_search_workload, run_search_case, write_search_trials,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Engine {
    SmartUniform,
    Onnx,
    TensorRt,
}

#[derive(Clone, Copy, Debug)]
enum InFlightRegime {
    FixedTotal(u32),
    PerWorker(u32),
}

impl InFlightRegime {
    fn total(self, workers: u32) -> Result<u32, String> {
        match self {
            Self::FixedTotal(total) => {
                if !total.is_multiple_of(workers) {
                    return Err(format!(
                        "fixed total in-flight capacity {total} is not divisible by {workers} workers"
                    ));
                }
                Ok(total)
            }
            Self::PerWorker(worker_batch) => worker_batch.checked_mul(workers).ok_or_else(|| {
                format!("per-worker batch {worker_batch} times {workers} workers exceeds u32")
            }),
        }
    }

    fn label(self) -> String {
        match self {
            Self::FixedTotal(total) => format!("fixed{total}"),
            Self::PerWorker(worker_batch) => format!("per-worker{worker_batch}"),
        }
    }
}

struct Args {
    engine: Engine,
    provider: OnnxProvider,
    model: Option<PathBuf>,
    tensorrt_cache: Option<PathBuf>,
    workers: Vec<u32>,
    regimes: Vec<InFlightRegime>,
    mux_max_batches: Vec<Option<u32>>,
    productive_work: u32,
    measured_trials: u32,
    warmup_trials: u32,
    workload: Option<PathBuf>,
    case_prefix: String,
    output: Option<PathBuf>,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("bench-mcgs-parallel: {error}");
        std::process::exit(2);
    }
}

fn run() -> Result<(), String> {
    let Some(args) = parse_args()? else {
        return Ok(());
    };
    let workload = match &args.workload {
        Some(path) => load_search_workload(path).map_err(|error| error.to_string())?,
        None => default_search_workload().map_err(|error| error.to_string())?,
    };
    let requested_backend = requested_backend(args.engine, args.provider);
    let backend_config = ProductionBackendConfig {
        model: args.model.clone(),
        tensorrt_cache: args.tensorrt_cache.clone(),
    };
    let mut trials = Vec::new();
    let mut failed_trials = 0usize;

    for &mux_max_batch in &args.mux_max_batches {
        let cases = planned_cases(&args, &requested_backend, mux_max_batch)?;
        let max_worker_batch = cases
            .iter()
            .map(|case| case.request.total_in_flight / case.request.workers)
            .max()
            .expect("validated non-empty plan");
        let (backend, resolved_backend) = build_search_backend(
            &requested_backend,
            &backend_config,
            mux_max_batch,
            max_worker_batch.max(mux_max_batch.unwrap_or(0)),
            &workload,
        )
        .map_err(|error| error.to_string())?;

        for case in cases {
            eprintln!(
                "measuring {}: {} warmup + {} measured trial(s), worker_batch={}",
                case.id,
                case.request.warmup_trials,
                case.request.measured_trials,
                case.request.total_in_flight / case.request.workers
            );
            let result = run_search_case(
                &case.id,
                &case.request,
                resolved_backend.clone(),
                &backend,
                &workload,
            )
            .map_err(|error| error.to_string())?;
            failed_trials += result
                .trials
                .iter()
                .filter(|trial| trial.status == TrialStatus::Failed)
                .count();
            trials.extend(result.trials);
        }
    }

    match &args.output {
        Some(path) => {
            let file = File::create(path)
                .map_err(|error| format!("failed to create '{}': {error}", path.display()))?;
            let mut output = BufWriter::new(file);
            write_search_trials(&mut output, &trials)
                .map_err(|error| format!("failed to write '{}': {error}", path.display()))?;
            output
                .flush()
                .map_err(|error| format!("failed to flush '{}': {error}", path.display()))?;
        }
        None => {
            let stdout = std::io::stdout();
            let mut output = BufWriter::new(stdout.lock());
            write_search_trials(&mut output, &trials)
                .map_err(|error| format!("failed to write search trials: {error}"))?;
            output
                .flush()
                .map_err(|error| format!("failed to flush search trials: {error}"))?;
        }
    }

    if failed_trials > 0 {
        return Err(format!(
            "{failed_trials} trial(s) failed; their rows were preserved in the output"
        ));
    }
    Ok(())
}

#[derive(Debug)]
struct PlannedCase {
    id: String,
    request: SearchRequest,
}

fn planned_cases(
    args: &Args,
    backend: &BackendRequest,
    mux_max_batch: Option<u32>,
) -> Result<Vec<PlannedCase>, String> {
    let mut cases = Vec::new();
    let mut coordinates = BTreeSet::new();
    for &workers in &args.workers {
        for &regime in &args.regimes {
            let total_in_flight = regime.total(workers)?;
            if !coordinates.insert((workers, total_in_flight)) {
                return Err(format!(
                    "two in-flight regimes resolve to the same coordinate: workers={workers}, total_in_flight={total_in_flight}"
                ));
            }
            let mux_label = mux_max_batch.map_or_else(
                || "direct".to_owned(),
                |max_batch| format!("mux{max_batch}"),
            );
            cases.push(PlannedCase {
                id: format!(
                    "{}-{}-{}-w{workers}-{mux_label}",
                    args.case_prefix,
                    backend_label(backend),
                    regime.label(),
                ),
                request: SearchRequest {
                    backend: backend.clone(),
                    workers,
                    total_in_flight,
                    mux_max_batch,
                    productive_work: args.productive_work,
                    warmup_trials: args.warmup_trials,
                    measured_trials: args.measured_trials,
                },
            });
        }
    }
    Ok(cases)
}

fn parse_args() -> Result<Option<Args>, String> {
    let mut engine = Engine::SmartUniform;
    let mut engine_explicit = false;
    let mut provider = OnnxProvider::Cpu;
    let mut model = None;
    let mut tensorrt_cache = None;
    let mut workers = vec![1, 2, 4];
    let mut fixed_totals = Vec::new();
    let mut worker_batches = Vec::new();
    let mut mux_max_batches = vec![None];
    let mut productive_work = 8_000;
    let mut measured_trials = 10;
    let mut warmup_trials = 2;
    let mut workload = None;
    let mut case_prefix = "search".to_owned();
    let mut output = None;

    let raw = std::env::args().collect::<Vec<_>>();
    let mut index = 1;
    while index < raw.len() {
        let flag = raw[index].as_str();
        let value = |index: &mut usize| -> Result<&str, String> {
            *index += 1;
            raw.get(*index)
                .map(String::as_str)
                .ok_or_else(|| format!("{flag} requires a value"))
        };
        match flag {
            "--backend" => {
                engine = match value(&mut index)? {
                    "smart-uniform" | "fake" => Engine::SmartUniform,
                    "onnx" => Engine::Onnx,
                    "tensorrt" => Engine::TensorRt,
                    other => {
                        return Err(format!(
                            "unknown backend '{other}'; expected smart-uniform, onnx, or tensorrt"
                        ))
                    }
                };
                engine_explicit = true;
            }
            "--provider" | "--device" => {
                provider = match value(&mut index)? {
                    "cpu" => OnnxProvider::Cpu,
                    "coreml" => OnnxProvider::Coreml,
                    "cuda" => OnnxProvider::Cuda,
                    other => {
                        return Err(format!(
                            "unknown ONNX provider '{other}'; expected cpu, coreml, or cuda"
                        ))
                    }
                };
            }
            "--model" => model = Some(PathBuf::from(value(&mut index)?)),
            "--tensorrt-cache" => {
                tensorrt_cache = Some(PathBuf::from(value(&mut index)?));
            }
            "--workers" => workers = parse_csv(value(&mut index)?, "worker count")?,
            "--total-in-flight" => {
                fixed_totals = parse_csv(value(&mut index)?, "total in-flight capacity")?;
            }
            "--worker-batches" | "--worker-batch" => {
                worker_batches = parse_csv(value(&mut index)?, "per-worker batch")?;
            }
            "--mux-max-batch" => {
                mux_max_batches = vec![Some(parse(value(&mut index)?, "mux max batch")?)];
            }
            "--mux-max-batches" => {
                mux_max_batches = parse_mux_batches(value(&mut index)?)?;
            }
            "--sims" | "--productive-work" => {
                productive_work = parse(value(&mut index)?, "productive work")?;
            }
            "--iters" | "--trials" => {
                measured_trials = parse(value(&mut index)?, "measured trial count")?;
            }
            "--warmups" => warmup_trials = parse(value(&mut index)?, "warmup count")?,
            "--workload" => workload = Some(PathBuf::from(value(&mut index)?)),
            "--case-prefix" => case_prefix = value(&mut index)?.to_owned(),
            "--output" => output = Some(PathBuf::from(value(&mut index)?)),
            "--help" | "-h" => {
                print_help();
                return Ok(None);
            }
            other => return Err(format!("unknown option '{other}'; use --help for usage")),
        }
        index += 1;
    }

    if model.is_some() && !engine_explicit {
        engine = Engine::Onnx;
    }
    require_positive("worker counts", &workers)?;
    require_positive("fixed total in-flight capacities", &fixed_totals)?;
    require_positive("per-worker batches", &worker_batches)?;
    if fixed_totals.is_empty() && worker_batches.is_empty() {
        fixed_totals.push(256);
    }
    if productive_work == 0 || measured_trials == 0 {
        return Err("productive work and measured trials must be positive".to_owned());
    }
    if case_prefix.trim().is_empty() || case_prefix.contains(',') {
        return Err("case prefix must be non-empty and cannot contain a comma".to_owned());
    }
    if matches!(engine, Engine::Onnx | Engine::TensorRt) && model.is_none() {
        return Err("--model is required for ONNX and TensorRT backends".to_owned());
    }
    if matches!(engine, Engine::SmartUniform) && model.is_some() {
        return Err("--model requires --backend onnx or --backend tensorrt".to_owned());
    }
    if !matches!(engine, Engine::TensorRt) && tensorrt_cache.is_some() {
        return Err("--tensorrt-cache requires --backend tensorrt".to_owned());
    }

    let regimes = fixed_totals
        .into_iter()
        .map(InFlightRegime::FixedTotal)
        .chain(worker_batches.into_iter().map(InFlightRegime::PerWorker))
        .collect();
    Ok(Some(Args {
        engine,
        provider,
        model,
        tensorrt_cache,
        workers,
        regimes,
        mux_max_batches,
        productive_work,
        measured_trials,
        warmup_trials,
        workload,
        case_prefix,
        output,
    }))
}

fn parse<T>(value: &str, label: &str) -> Result<T, String>
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    value
        .parse()
        .map_err(|error| format!("invalid {label} '{value}': {error}"))
}

fn parse_csv<T>(value: &str, label: &str) -> Result<Vec<T>, String>
where
    T: std::str::FromStr + Ord + Copy,
    T::Err: std::fmt::Display,
{
    let values = value
        .split(',')
        .map(|part| parse(part.trim(), label))
        .collect::<Result<Vec<_>, _>>()?;
    if values.is_empty() {
        return Err(format!("at least one {label} is required"));
    }
    if values.iter().copied().collect::<BTreeSet<_>>().len() != values.len() {
        return Err(format!("duplicate {label} values are not allowed"));
    }
    Ok(values)
}

fn parse_mux_batches(value: &str) -> Result<Vec<Option<u32>>, String> {
    let values = value
        .split(',')
        .map(|part| match part.trim() {
            "none" | "direct" => Ok(None),
            value => parse(value, "mux max batch").map(Some),
        })
        .collect::<Result<Vec<_>, _>>()?;
    if values.is_empty() {
        return Err("at least one mux setting is required".to_owned());
    }
    if values.contains(&Some(0)) {
        return Err("mux max batches must be positive".to_owned());
    }
    if values.iter().copied().collect::<BTreeSet<_>>().len() != values.len() {
        return Err("duplicate mux settings are not allowed".to_owned());
    }
    Ok(values)
}

fn require_positive(label: &str, values: &[u32]) -> Result<(), String> {
    if values.contains(&0) {
        Err(format!("{label} must contain positive values"))
    } else if values.iter().copied().collect::<BTreeSet<_>>().len() != values.len() {
        Err(format!("duplicate {label} are not allowed"))
    } else {
        Ok(())
    }
}

fn print_help() {
    eprintln!("Usage: bench-mcgs-parallel [OPTIONS]");
    eprintln!();
    eprintln!("Runs fixed-work MCGS cases and writes calibration-v1 search CSV rows.");
    eprintln!();
    eprintln!("  --backend smart-uniform|onnx|tensorrt  Backend (default: smart-uniform)");
    eprintln!("  --provider cpu|coreml|cuda              ONNX provider (default: cpu)");
    eprintln!("  --model PATH                            ONNX model for ONNX/TensorRT");
    eprintln!("  --tensorrt-cache DIR                    TensorRT engine cache directory");
    eprintln!("  --workers 1,2,4                         Worker sweep (default: 1,2,4)");
    eprintln!("  --total-in-flight 256,512               Fixed aggregate capacity cases");
    eprintln!("  --worker-batches 512,1024               Scaled per-worker capacity cases");
    eprintln!("  --mux-max-batches none,512,1024         Direct/mux setting sweep");
    eprintln!(
        "  --productive-work N                     Productive units per trial (default: 8000)"
    );
    eprintln!("  --trials N                              Measured fresh-tree trials (default: 10)");
    eprintln!("  --warmups N                             Warmup trials per case (default: 2)");
    eprintln!("  --workload PATH                         Fixture/config/seed JSON artifact");
    eprintln!("  --case-prefix TEXT                      Case-id prefix (default: search)");
    eprintln!("  --output PATH                           CSV output (default: stdout)");
    eprintln!();
    eprintln!("Legacy aliases: --device, --sims, --iters, --worker-batch, --mux-max-batch");
    eprintln!("Feature examples: --features mcgs-profile,onnx-coreml or mcgs-profile,tensorrt");
}

fn requested_backend(engine: Engine, provider: OnnxProvider) -> BackendRequest {
    match engine {
        Engine::SmartUniform => BackendRequest::SmartUniform,
        Engine::Onnx => BackendRequest::Onnx { provider },
        Engine::TensorRt => BackendRequest::TensorRt,
    }
}

fn backend_label(backend: &BackendRequest) -> &'static str {
    match backend {
        BackendRequest::SmartUniform => "smart-uniform",
        BackendRequest::Onnx {
            provider: OnnxProvider::Cpu,
        } => "onnx-cpu",
        BackendRequest::Onnx {
            provider: OnnxProvider::Coreml,
        } => "onnx-coreml",
        BackendRequest::Onnx {
            provider: OnnxProvider::Cuda,
        } => "onnx-cuda",
        BackendRequest::TensorRt => "tensorrt",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(regimes: Vec<InFlightRegime>) -> Args {
        Args {
            engine: Engine::SmartUniform,
            provider: OnnxProvider::Cpu,
            model: None,
            tensorrt_cache: None,
            workers: vec![1, 4],
            regimes,
            mux_max_batches: vec![None],
            productive_work: 100,
            measured_trials: 2,
            warmup_trials: 1,
            workload: None,
            case_prefix: "fixture".to_owned(),
            output: None,
        }
    }

    #[test]
    fn plan_keeps_fixed_total_and_per_worker_regimes_explicit() {
        let args = args(vec![
            InFlightRegime::FixedTotal(256),
            InFlightRegime::PerWorker(1_024),
        ]);
        let cases = planned_cases(&args, &BackendRequest::SmartUniform, Some(512)).unwrap();

        assert_eq!(cases.len(), 4);
        assert_eq!(cases[0].request.total_in_flight, 256);
        assert_eq!(cases[1].request.total_in_flight, 1_024);
        assert_eq!(cases[2].request.total_in_flight, 256);
        assert_eq!(cases[3].request.total_in_flight, 4_096);
        assert!(cases[2].id.contains("fixed256-w4-mux512"));
        assert!(cases[3].id.contains("per-worker1024-w4-mux512"));
        assert!(cases
            .iter()
            .all(|case| case.request.mux_max_batch == Some(512)));
    }

    #[test]
    fn plan_rejects_two_labels_for_the_same_runtime_coordinate() {
        let args = args(vec![
            InFlightRegime::FixedTotal(1_024),
            InFlightRegime::PerWorker(1_024),
        ]);

        let error = planned_cases(&args, &BackendRequest::SmartUniform, None).unwrap_err();

        assert!(error.contains("same coordinate"));
    }

    #[test]
    fn mux_sweep_parses_direct_and_bounded_cases() {
        assert_eq!(
            parse_mux_batches("none,512,1024").unwrap(),
            vec![None, Some(512), Some(1_024)]
        );
        assert!(parse_mux_batches("direct,none").is_err());
        assert!(parse_mux_batches("0").is_err());
    }
}
