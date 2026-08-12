//! Plan execution and self-validating run-folder materialization.

use std::collections::BTreeSet;
use std::ffi::OsStr;
use std::fs;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicBool, Ordering};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

use crate::calibration::{
    load_run_folder, render_run_comparison, summarize_run, validate_run_plan, AcceleratorIdentity,
    ArtifactIdentity, BuildIdentity, CapacityRequest, CapacityResolved, CapacityTrial, CaseOutcome,
    CasePlan, CaseRecord, ComparisonAxis, FailureStage, HardwareIdentity, LoadedRun, ProtocolError,
    RunContext, RunPlan, RunRecord, RuntimeIdentity, SearchRequest, SearchResolved, SearchTrial,
    SoftwareVersion, SourceIdentity, SourceState, TrialFile, TrialFiles, TrialPhase, TrialStatus,
    CAPACITY_TRIALS_FILE, CAPACITY_WORKLOAD_FILE, COMPARISON_FILE, PROTOCOL_VERSION,
    RUN_RECORD_FILE, SEARCH_TRIALS_FILE, SEARCH_WORKLOAD_FILE, SUMMARY_FILE,
};
use crate::capacity::{
    build_capacity_workload, run_capacity_case, standard_capacity_workload, write_capacity_trials,
    CapacityWorkload,
};
use crate::production::{
    build_capacity_backend, build_search_backend, BackendBuildError, ProductionBackendConfig,
};
use crate::search::{
    default_search_workload, load_search_workload, run_search_case, write_search_trials,
    SearchWorkload, DEFAULT_SEARCH_WORKLOAD_JSON,
};

const SOURCE_PATCH_FILE: &str = "source.patch";
const BUILD_SOURCE_PATCH: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/alpharat-build-source.patch"));

static INTERRUPTED: AtomicBool = AtomicBool::new(false);

/// Small executable input: paths and fixture recipes are resolved into protocol identities.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ExecutionPlan {
    pub protocol_version: u32,
    pub run_id: String,
    pub model: FileArtifact,
    pub comparison_axes: BTreeSet<ComparisonAxis>,
    pub cases: Vec<ExecutionCase>,
    pub capacity_workload: Option<CapacityWorkloadSource>,
    pub search_workload: Option<SearchWorkloadSource>,
    pub tensorrt_cache: Option<PathBuf>,
    #[serde(default)]
    pub accelerators: Vec<AcceleratorIdentity>,
    #[serde(default)]
    pub runtime_software: Vec<SoftwareVersion>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct FileArtifact {
    pub label: String,
    pub path: PathBuf,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields, rename_all = "snake_case", tag = "source")]
pub enum CapacityWorkloadSource {
    Standard {
        label: String,
        width: u8,
        height: u8,
    },
    File {
        label: String,
        path: PathBuf,
    },
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields, rename_all = "snake_case", tag = "source")]
pub enum SearchWorkloadSource {
    BuiltIn { label: String },
    File { label: String, path: PathBuf },
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields, rename_all = "snake_case", tag = "benchmark")]
pub enum ExecutionCase {
    BackendCapacity {
        id: String,
        requested: CapacityRequest,
    },
    Search {
        id: String,
        requested: SearchRequest,
    },
}

#[derive(Debug, Error)]
pub enum RunnerError {
    #[error("invalid calibration execution plan: {0}")]
    Invalid(String),
    #[error("failed to read {path}: {source}")]
    Read {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("invalid JSON in {path}: {source}")]
    Json {
        path: PathBuf,
        #[source]
        source: serde_json::Error,
    },
    #[error("failed to create or write {path}: {source}")]
    Write {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to serialize {label}: {source}")]
    Serialize {
        label: &'static str,
        #[source]
        source: serde_json::Error,
    },
    #[error(transparent)]
    Protocol(#[from] ProtocolError),
}

/// Execute exactly the cases in `plan_path` and leave a self-validating run folder.
pub fn execute_plan_file(
    plan_path: impl AsRef<Path>,
    output_folder: impl AsRef<Path>,
) -> Result<LoadedRun, RunnerError> {
    INTERRUPTED.store(false, Ordering::Release);
    install_interrupt_handler();

    let plan_path = plan_path.as_ref();
    let plan_bytes = read(plan_path)?;
    let execution: ExecutionPlan =
        serde_json::from_slice(&plan_bytes).map_err(|source| RunnerError::Json {
            path: plan_path.to_path_buf(),
            source,
        })?;
    validate_execution_plan(&execution)?;

    let base = plan_path.parent().unwrap_or_else(|| Path::new("."));
    let model_path = resolve_path(base, &execution.model.path);
    let model_bytes = read(&model_path)?;
    let model = artifact_identity(
        &execution.model.label,
        &model_bytes,
        Some(execution.model.path.to_string_lossy().into_owned()),
    )?;
    let source = capture_source_identity();
    let resolved_workloads = resolve_workloads(base, &execution)?;
    let run_plan = resolve_run_plan(&execution, &resolved_workloads)?;
    validate_run_plan(&run_plan)?;

    let output_folder = output_folder.as_ref();
    create_output_folder(output_folder)?;
    write_resolved_workloads(output_folder, &resolved_workloads)?;
    let source = materialize_source_identity(output_folder, source)?;
    let context = gather_context(&execution, model, source)?;

    let backend_config = ProductionBackendConfig {
        model: Some(model_path),
        tensorrt_cache: execution
            .tensorrt_cache
            .as_ref()
            .map(|path| resolve_path(base, path)),
    };
    let (cases, capacity_trials, search_trials) = execute_cases(
        &run_plan,
        &backend_config,
        resolved_workloads
            .capacity
            .as_ref()
            .map(|resolved| &resolved.value),
        resolved_workloads
            .search
            .as_ref()
            .map(|resolved| &resolved.value),
    );

    let trial_files =
        write_trial_files(output_folder, &run_plan, &capacity_trials, &search_trials)?;
    let record = RunRecord {
        protocol_version: PROTOCOL_VERSION,
        run_id: execution.run_id,
        created_at: utc_timestamp(),
        context,
        plan: run_plan,
        cases,
        trial_files,
    };
    write_json(output_folder.join(RUN_RECORD_FILE), &record, "run record")?;

    // The generated record must satisfy the same loader as a future comparison consumer.
    let loaded = load_run_folder(output_folder)?;
    let summary = summarize_run(&loaded);
    write_json(output_folder.join(SUMMARY_FILE), &summary, "run summary")?;
    write(
        output_folder.join(COMPARISON_FILE),
        render_run_comparison(&summary).as_bytes(),
    )?;
    Ok(loaded)
}

fn validate_execution_plan(plan: &ExecutionPlan) -> Result<(), RunnerError> {
    if plan.protocol_version != PROTOCOL_VERSION {
        return Err(RunnerError::Invalid(format!(
            "unsupported protocol version {}; expected {PROTOCOL_VERSION}",
            plan.protocol_version
        )));
    }
    if plan.run_id.trim().is_empty() {
        return Err(RunnerError::Invalid("run id cannot be empty".to_owned()));
    }
    if plan.model.label.trim().is_empty() || plan.model.path.as_os_str().is_empty() {
        return Err(RunnerError::Invalid(
            "model label and path must both be present".to_owned(),
        ));
    }
    if plan.cases.is_empty() {
        return Err(RunnerError::Invalid(
            "the execution plan has no cases".to_owned(),
        ));
    }
    let has_capacity = plan
        .cases
        .iter()
        .any(|case| matches!(case, ExecutionCase::BackendCapacity { .. }));
    let has_search = plan
        .cases
        .iter()
        .any(|case| matches!(case, ExecutionCase::Search { .. }));
    if has_capacity != plan.capacity_workload.is_some() {
        return Err(RunnerError::Invalid(
            "capacity workload presence must match the requested capacity cases".to_owned(),
        ));
    }
    if has_search != plan.search_workload.is_some() {
        return Err(RunnerError::Invalid(
            "search workload presence must match the requested search cases".to_owned(),
        ));
    }
    let mut software = BTreeSet::new();
    software.insert("alpharat-bench");
    for version in &plan.runtime_software {
        if version.name.trim().is_empty()
            || version.version.trim().is_empty()
            || !software.insert(version.name.as_str())
        {
            return Err(RunnerError::Invalid(format!(
                "runtime software entry '{}' is empty or duplicated",
                version.name
            )));
        }
    }
    Ok(())
}

struct ResolvedArtifact<T> {
    identity: ArtifactIdentity,
    bytes: Vec<u8>,
    value: T,
}

struct ResolvedWorkloads {
    capacity: Option<ResolvedArtifact<CapacityWorkload>>,
    search: Option<ResolvedArtifact<SearchWorkload>>,
}

fn resolve_workloads(base: &Path, plan: &ExecutionPlan) -> Result<ResolvedWorkloads, RunnerError> {
    let capacity = match &plan.capacity_workload {
        Some(CapacityWorkloadSource::Standard {
            label,
            width,
            height,
        }) => {
            let value = standard_capacity_workload(*width, *height)
                .map_err(|error| RunnerError::Invalid(error.to_string()))?;
            let bytes = pretty_json_bytes(&value, "capacity workload")?;
            Some(ResolvedArtifact {
                identity: artifact_identity(
                    label,
                    &bytes,
                    Some(CAPACITY_WORKLOAD_FILE.to_owned()),
                )?,
                bytes,
                value,
            })
        }
        Some(CapacityWorkloadSource::File { label, path }) => {
            let path = resolve_path(base, path);
            let bytes = read(&path)?;
            let value: CapacityWorkload =
                serde_json::from_slice(&bytes).map_err(|source| RunnerError::Json {
                    path: path.clone(),
                    source,
                })?;
            build_capacity_workload(&value)
                .map_err(|error| RunnerError::Invalid(error.to_string()))?;
            Some(ResolvedArtifact {
                identity: artifact_identity(
                    label,
                    &bytes,
                    Some(CAPACITY_WORKLOAD_FILE.to_owned()),
                )?,
                bytes,
                value,
            })
        }
        None => None,
    };
    let search = match &plan.search_workload {
        Some(SearchWorkloadSource::BuiltIn { label }) => {
            let bytes = DEFAULT_SEARCH_WORKLOAD_JSON.as_bytes().to_vec();
            let value = default_search_workload()
                .map_err(|error| RunnerError::Invalid(error.to_string()))?;
            Some(ResolvedArtifact {
                identity: artifact_identity(label, &bytes, Some(SEARCH_WORKLOAD_FILE.to_owned()))?,
                bytes,
                value,
            })
        }
        Some(SearchWorkloadSource::File { label, path }) => {
            let path = resolve_path(base, path);
            let bytes = read(&path)?;
            let value = load_search_workload(&path)
                .map_err(|error| RunnerError::Invalid(error.to_string()))?;
            Some(ResolvedArtifact {
                identity: artifact_identity(label, &bytes, Some(SEARCH_WORKLOAD_FILE.to_owned()))?,
                bytes,
                value,
            })
        }
        None => None,
    };
    Ok(ResolvedWorkloads { capacity, search })
}

fn resolve_run_plan(
    execution: &ExecutionPlan,
    workloads: &ResolvedWorkloads,
) -> Result<RunPlan, RunnerError> {
    let mut cases = Vec::with_capacity(execution.cases.len());
    for case in &execution.cases {
        cases.push(match case {
            ExecutionCase::BackendCapacity { id, requested } => CasePlan::BackendCapacity {
                id: id.clone(),
                workload: workloads
                    .capacity
                    .as_ref()
                    .ok_or_else(|| RunnerError::Invalid("missing capacity workload".to_owned()))?
                    .identity
                    .clone(),
                requested: requested.clone(),
            },
            ExecutionCase::Search { id, requested } => CasePlan::Search {
                id: id.clone(),
                workload: workloads
                    .search
                    .as_ref()
                    .ok_or_else(|| RunnerError::Invalid("missing search workload".to_owned()))?
                    .identity
                    .clone(),
                requested: requested.clone(),
            },
        });
    }
    Ok(RunPlan {
        comparison_axes: execution.comparison_axes.clone(),
        cases,
    })
}

fn write_resolved_workloads(
    folder: &Path,
    workloads: &ResolvedWorkloads,
) -> Result<(), RunnerError> {
    if let Some(workload) = &workloads.capacity {
        write(folder.join(CAPACITY_WORKLOAD_FILE), &workload.bytes)?;
    }
    if let Some(workload) = &workloads.search {
        write(folder.join(SEARCH_WORKLOAD_FILE), &workload.bytes)?;
    }
    Ok(())
}

fn execute_cases(
    plan: &RunPlan,
    backend_config: &ProductionBackendConfig,
    capacity_workload: Option<&CapacityWorkload>,
    search_workload: Option<&SearchWorkload>,
) -> (Vec<CaseRecord>, Vec<CapacityTrial>, Vec<SearchTrial>) {
    let capacity_games =
        capacity_workload.and_then(|workload| build_capacity_workload(workload).ok());
    let mut records = Vec::with_capacity(plan.cases.len());
    let mut capacity_trials = Vec::new();
    let mut search_trials = Vec::new();

    for case in &plan.cases {
        if INTERRUPTED.load(Ordering::Acquire) {
            records.push(interrupted_record(case));
            continue;
        }
        match case {
            CasePlan::BackendCapacity { id, requested, .. } => {
                let (outcome, mut trials) = execute_capacity_case(
                    id,
                    requested,
                    backend_config,
                    capacity_workload,
                    capacity_games.as_deref(),
                );
                let outcome = record_interrupt_after_case(outcome);
                capacity_trials.append(&mut trials);
                records.push(CaseRecord::BackendCapacity {
                    id: id.clone(),
                    outcome,
                });
            }
            CasePlan::Search { id, requested, .. } => {
                let (outcome, mut trials) =
                    execute_search_case(id, requested, backend_config, search_workload);
                let outcome = record_interrupt_after_case(outcome);
                search_trials.append(&mut trials);
                records.push(CaseRecord::Search {
                    id: id.clone(),
                    outcome,
                });
            }
        }
    }
    (records, capacity_trials, search_trials)
}

fn execute_capacity_case(
    case_id: &str,
    request: &CapacityRequest,
    config: &ProductionBackendConfig,
    workload: Option<&CapacityWorkload>,
    games: Option<&[pyrat::GameState]>,
) -> (CaseOutcome<CapacityResolved>, Vec<CapacityTrial>) {
    let Some(workload) = workload else {
        return (
            failed_setup("capacity workload was not resolved"),
            Vec::new(),
        );
    };
    let Some(games) = games else {
        return (
            failed_setup("capacity workload could not be built"),
            Vec::new(),
        );
    };
    let result = catch_unwind(AssertUnwindSafe(|| {
        let (backend, resolved) = build_capacity_backend(
            &request.backend,
            config,
            workload.width,
            workload.height,
            request.batch_size,
        )?;
        run_capacity_case(case_id, request, resolved, backend.as_ref(), games)
            .map_err(|error| BackendBuildError::Failed(error.to_string()))
    }));
    match result {
        Ok(Ok(result)) => {
            let outcome = classify_capacity_result(result.resolved, &result.trials);
            (outcome, result.trials)
        }
        Ok(Err(BackendBuildError::Unsupported(reason))) => {
            (CaseOutcome::Unsupported { reason }, Vec::new())
        }
        Ok(Err(BackendBuildError::Failed(message))) => (failed_setup(message), Vec::new()),
        Err(payload) => (
            failed_setup(format!("panic: {}", panic_message(payload))),
            Vec::new(),
        ),
    }
}

fn execute_search_case(
    case_id: &str,
    request: &SearchRequest,
    config: &ProductionBackendConfig,
    workload: Option<&SearchWorkload>,
) -> (CaseOutcome<SearchResolved>, Vec<SearchTrial>) {
    let Some(workload) = workload else {
        return (failed_setup("search workload was not resolved"), Vec::new());
    };
    let result = catch_unwind(AssertUnwindSafe(|| {
        let worker_batch = request.total_in_flight / request.workers;
        let max_device_batch = worker_batch.max(request.mux_max_batch.unwrap_or(0));
        let (backend, resolved) = build_search_backend(
            &request.backend,
            config,
            request.mux_max_batch,
            max_device_batch,
            workload,
        )?;
        run_search_case(case_id, request, resolved, &backend, workload)
            .map_err(|error| BackendBuildError::Failed(error.to_string()))
    }));
    match result {
        Ok(Ok(result)) => {
            let outcome = classify_search_result(result.resolved, &result.trials);
            (outcome, result.trials)
        }
        Ok(Err(BackendBuildError::Unsupported(reason))) => {
            (CaseOutcome::Unsupported { reason }, Vec::new())
        }
        Ok(Err(BackendBuildError::Failed(message))) => (failed_setup(message), Vec::new()),
        Err(payload) => (
            failed_setup(format!("panic: {}", panic_message(payload))),
            Vec::new(),
        ),
    }
}

fn classify_capacity_result(
    resolved: CapacityResolved,
    trials: &[CapacityTrial],
) -> CaseOutcome<CapacityResolved> {
    classify_trials(
        resolved,
        trials
            .iter()
            .map(|trial| (trial.phase, trial.status, trial.error.as_deref())),
    )
}

fn classify_search_result(
    resolved: SearchResolved,
    trials: &[SearchTrial],
) -> CaseOutcome<SearchResolved> {
    classify_trials(
        resolved,
        trials
            .iter()
            .map(|trial| (trial.phase, trial.status, trial.error.as_deref())),
    )
}

fn classify_trials<'a, T>(
    resolved: T,
    trials: impl Iterator<Item = (TrialPhase, TrialStatus, Option<&'a str>)>,
) -> CaseOutcome<T> {
    let mut first_failure = None;
    let mut first_interruption = None;
    for (phase, status, error) in trials {
        match status {
            TrialStatus::Completed => {}
            TrialStatus::Failed if first_failure.is_none() => {
                first_failure = Some((phase, error.unwrap_or("trial failed").to_owned()));
            }
            TrialStatus::Interrupted if first_interruption.is_none() => {
                first_interruption = Some(error.unwrap_or("trial interrupted").to_owned());
            }
            TrialStatus::Failed | TrialStatus::Interrupted => {}
        }
    }
    if let Some(reason) = first_interruption {
        CaseOutcome::Interrupted { reason }
    } else if let Some((phase, message)) = first_failure {
        CaseOutcome::Failed {
            stage: match phase {
                TrialPhase::Warmup => FailureStage::Warmup,
                TrialPhase::Measured => FailureStage::Measurement,
            },
            message,
        }
    } else {
        CaseOutcome::Completed { resolved }
    }
}

fn failed_setup<T>(message: impl Into<String>) -> CaseOutcome<T> {
    CaseOutcome::Failed {
        stage: FailureStage::Setup,
        message: message.into(),
    }
}

fn record_interrupt_after_case<T>(outcome: CaseOutcome<T>) -> CaseOutcome<T> {
    if INTERRUPTED.load(Ordering::Acquire) && matches!(&outcome, CaseOutcome::Completed { .. }) {
        CaseOutcome::Interrupted {
            reason: "interrupt requested while this case was running; its admitted trials finished"
                .to_owned(),
        }
    } else {
        outcome
    }
}

fn interrupted_record(case: &CasePlan) -> CaseRecord {
    let reason = "run interrupted before this case started".to_owned();
    match case {
        CasePlan::BackendCapacity { id, .. } => CaseRecord::BackendCapacity {
            id: id.clone(),
            outcome: CaseOutcome::Interrupted { reason },
        },
        CasePlan::Search { id, .. } => CaseRecord::Search {
            id: id.clone(),
            outcome: CaseOutcome::Interrupted { reason },
        },
    }
}

fn write_trial_files(
    folder: &Path,
    plan: &RunPlan,
    capacity_trials: &[CapacityTrial],
    search_trials: &[SearchTrial],
) -> Result<TrialFiles, RunnerError> {
    let has_capacity = plan
        .cases
        .iter()
        .any(|case| matches!(case, CasePlan::BackendCapacity { .. }));
    let has_search = plan
        .cases
        .iter()
        .any(|case| matches!(case, CasePlan::Search { .. }));
    let backend_capacity = if has_capacity {
        let mut bytes = Vec::new();
        write_capacity_trials(&mut bytes, capacity_trials).map_err(|error| {
            RunnerError::Invalid(format!("failed to encode capacity trials: {error}"))
        })?;
        write(folder.join(CAPACITY_TRIALS_FILE), &bytes)?;
        Some(trial_file(
            CAPACITY_TRIALS_FILE,
            &bytes,
            capacity_trials.len(),
        )?)
    } else {
        None
    };
    let search = if has_search {
        let mut bytes = Vec::new();
        write_search_trials(&mut bytes, search_trials).map_err(|error| {
            RunnerError::Invalid(format!("failed to encode search trials: {error}"))
        })?;
        write(folder.join(SEARCH_TRIALS_FILE), &bytes)?;
        Some(trial_file(SEARCH_TRIALS_FILE, &bytes, search_trials.len())?)
    } else {
        None
    };
    Ok(TrialFiles {
        backend_capacity,
        search,
    })
}

fn trial_file(path: &str, bytes: &[u8], rows: usize) -> Result<TrialFile, RunnerError> {
    Ok(TrialFile {
        path: path.to_owned(),
        sha256: sha256(bytes),
        rows: u32::try_from(rows)
            .map_err(|_| RunnerError::Invalid("trial row count exceeds u32".to_owned()))?,
    })
}

fn gather_context(
    plan: &ExecutionPlan,
    model: ArtifactIdentity,
    source: SourceIdentity,
) -> Result<RunContext, RunnerError> {
    let current_exe = std::env::current_exe().map_err(|source| RunnerError::Read {
        path: PathBuf::from("<current executable>"),
        source,
    })?;
    let binary = read(&current_exe)?;
    let mut features = BTreeSet::new();
    for (enabled, name) in [
        (cfg!(feature = "mcgs-profile"), "mcgs-profile"),
        (cfg!(feature = "onnx"), "onnx"),
        (cfg!(feature = "onnx-cuda"), "onnx-cuda"),
        (cfg!(feature = "onnx-coreml"), "onnx-coreml"),
        (cfg!(feature = "tensorrt"), "tensorrt"),
    ] {
        if enabled {
            features.insert(name.to_owned());
        }
    }
    let mut software = vec![SoftwareVersion {
        name: "alpharat-bench".to_owned(),
        version: env!("CARGO_PKG_VERSION").to_owned(),
    }];
    software.extend(plan.runtime_software.clone());
    Ok(RunContext {
        source,
        build: BuildIdentity {
            profile: env!("ALPHARAT_BUILD_PROFILE").to_owned(),
            target: env!("ALPHARAT_BUILD_TARGET").to_owned(),
            rustc_version: env!("ALPHARAT_BUILD_RUSTC_VERSION").to_owned(),
            features,
            binary_sha256: sha256(&binary),
            command: std::env::args_os()
                .map(|part| part.to_string_lossy().into_owned())
                .collect(),
        },
        hardware: gather_hardware(plan.accelerators.clone())?,
        runtime: RuntimeIdentity { software },
        model,
    })
}

fn gather_hardware(
    accelerators: Vec<AcceleratorIdentity>,
) -> Result<HardwareIdentity, RunnerError> {
    let operating_system = command_output("sw_vers", ["-productVersion"])
        .map(|version| format!("macOS {version}"))
        .or_else(|_| command_output("uname", ["-sr"]))
        .unwrap_or_else(|_| std::env::consts::OS.to_owned());
    let mac_hardware = mac_hardware();
    let cpu = command_output("sysctl", ["-n", "machdep.cpu.brand_string"])
        .or_else(|_| {
            mac_hardware
                .as_ref()
                .map(|hardware| hardware.0.clone())
                .ok_or_else(|| "macOS hardware profile was unavailable".to_owned())
        })
        .or_else(|_| linux_cpu_name())
        .map_err(|error| {
            RunnerError::Invalid(format!(
                "could not identify the CPU for the run record: {error}"
            ))
        })?;
    let memory_bytes = command_output("sysctl", ["-n", "hw.memsize"])
        .ok()
        .and_then(|value| value.parse().ok())
        .or_else(|| mac_hardware.as_ref().map(|hardware| hardware.1))
        .or_else(linux_memory_bytes)
        .ok_or_else(|| {
            RunnerError::Invalid("could not identify physical memory for the run record".to_owned())
        })?;
    let logical_cores = std::thread::available_parallelism()
        .ok()
        .and_then(|cores| u32::try_from(cores.get()).ok())
        .unwrap_or(1);
    Ok(HardwareIdentity {
        operating_system,
        architecture: std::env::consts::ARCH.to_owned(),
        cpu,
        logical_cores,
        memory_bytes,
        accelerators,
    })
}

fn mac_hardware() -> Option<(String, u64)> {
    let output = command_output("system_profiler", ["SPHardwareDataType", "-json"]).ok()?;
    let value: serde_json::Value = serde_json::from_str(&output).ok()?;
    let hardware = value.get("SPHardwareDataType")?.as_array()?.first()?;
    let cpu = hardware.get("chip_type")?.as_str()?.to_owned();
    let memory = parse_human_memory(hardware.get("physical_memory")?.as_str()?)?;
    Some((cpu, memory))
}

fn parse_human_memory(value: &str) -> Option<u64> {
    let mut parts = value.split_whitespace();
    let amount = parts.next()?.parse::<u64>().ok()?;
    let multiplier = match parts.next()?.to_ascii_lowercase().as_str() {
        "kb" | "kib" => 1024_u64,
        "mb" | "mib" => 1024_u64.pow(2),
        "gb" | "gib" => 1024_u64.pow(3),
        "tb" | "tib" => 1024_u64.pow(4),
        _ => return None,
    };
    amount.checked_mul(multiplier)
}

enum CapturedSource {
    Clean { revision: String },
    Patch { revision: String, bytes: Vec<u8> },
    NonReproducible { revision: String, reason: String },
}

fn capture_source_identity() -> CapturedSource {
    let revision = env!("ALPHARAT_BUILD_SOURCE_REVISION").to_owned();
    match env!("ALPHARAT_BUILD_SOURCE_STATE") {
        "clean" => CapturedSource::Clean { revision },
        "patched" => CapturedSource::Patch {
            revision,
            bytes: BUILD_SOURCE_PATCH.to_vec(),
        },
        "non_reproducible" => CapturedSource::NonReproducible {
            revision,
            reason: env!("ALPHARAT_BUILD_SOURCE_REASON").to_owned(),
        },
        state => panic!("build script emitted unknown source state '{state}'"),
    }
}

fn materialize_source_identity(
    folder: &Path,
    source: CapturedSource,
) -> Result<SourceIdentity, RunnerError> {
    Ok(match source {
        CapturedSource::Clean { revision } => SourceIdentity {
            revision,
            state: SourceState::Clean,
        },
        CapturedSource::Patch { revision, bytes } => {
            write(folder.join(SOURCE_PATCH_FILE), &bytes)?;
            SourceIdentity {
                revision,
                state: SourceState::Patched {
                    patch_file: SOURCE_PATCH_FILE.to_owned(),
                    patch_sha256: sha256(&bytes),
                },
            }
        }
        CapturedSource::NonReproducible { revision, reason } => SourceIdentity {
            revision,
            state: SourceState::NonReproducible { reason },
        },
    })
}

fn artifact_identity(
    label: &str,
    bytes: &[u8],
    path_hint: Option<String>,
) -> Result<ArtifactIdentity, RunnerError> {
    if label.trim().is_empty() {
        return Err(RunnerError::Invalid(
            "artifact label cannot be empty".to_owned(),
        ));
    }
    Ok(ArtifactIdentity {
        label: label.to_owned(),
        sha256: sha256(bytes),
        bytes: u64::try_from(bytes.len())
            .map_err(|_| RunnerError::Invalid("artifact size exceeds u64".to_owned()))?,
        path_hint,
    })
}

fn create_output_folder(folder: &Path) -> Result<(), RunnerError> {
    if folder.exists() {
        return Err(RunnerError::Invalid(format!(
            "output folder '{}' already exists; calibration never overwrites a record",
            folder.display()
        )));
    }
    if let Some(parent) = folder
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        fs::create_dir_all(parent).map_err(|source| RunnerError::Write {
            path: parent.to_path_buf(),
            source,
        })?;
    }
    fs::create_dir(folder).map_err(|source| RunnerError::Write {
        path: folder.to_path_buf(),
        source,
    })
}

fn resolve_path(base: &Path, path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        base.join(path)
    }
}

fn pretty_json_bytes(value: &impl Serialize, label: &'static str) -> Result<Vec<u8>, RunnerError> {
    let mut bytes = serde_json::to_vec_pretty(value)
        .map_err(|source| RunnerError::Serialize { label, source })?;
    bytes.push(b'\n');
    Ok(bytes)
}

fn write_json(
    path: PathBuf,
    value: &impl Serialize,
    label: &'static str,
) -> Result<(), RunnerError> {
    write(path, &pretty_json_bytes(value, label)?)
}

fn read(path: &Path) -> Result<Vec<u8>, RunnerError> {
    fs::read(path).map_err(|source| RunnerError::Read {
        path: path.to_path_buf(),
        source,
    })
}

fn write(path: PathBuf, bytes: &[u8]) -> Result<(), RunnerError> {
    fs::write(&path, bytes).map_err(|source| RunnerError::Write { path, source })
}

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn utc_timestamp() -> String {
    command_output("date", ["-u", "+%Y-%m-%dT%H:%M:%SZ"])
        .unwrap_or_else(|_| format!("unix-{}", unix_timestamp()))
}

fn unix_timestamp() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn command_output<I, S>(program: &str, args: I) -> Result<String, String>
where
    I: IntoIterator<Item = S>,
    S: AsRef<OsStr>,
{
    let output = Command::new(program)
        .args(args)
        .output()
        .map_err(|error| format!("failed to run {program}: {error}"))?;
    if !output.status.success() {
        return Err(format!(
            "{program} exited with {}: {}",
            output.status,
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    Ok(String::from_utf8_lossy(&output.stdout).trim().to_owned())
}

fn linux_cpu_name() -> Result<String, String> {
    let cpuinfo = fs::read_to_string("/proc/cpuinfo").map_err(|error| error.to_string())?;
    cpuinfo
        .lines()
        .find_map(|line| line.strip_prefix("model name\t: "))
        .map(str::to_owned)
        .ok_or_else(|| "Linux CPU model was not found".to_owned())
}

fn linux_memory_bytes() -> Option<u64> {
    let meminfo = fs::read_to_string("/proc/meminfo").ok()?;
    let kib = meminfo
        .lines()
        .find_map(|line| line.strip_prefix("MemTotal:"))?
        .split_whitespace()
        .next()?
        .parse::<u64>()
        .ok()?;
    kib.checked_mul(1024)
}

fn panic_message(payload: Box<dyn std::any::Any + Send>) -> String {
    if let Some(message) = payload.downcast_ref::<&str>() {
        (*message).to_owned()
    } else if let Some(message) = payload.downcast_ref::<String>() {
        message.clone()
    } else {
        "non-string panic payload".to_owned()
    }
}

#[cfg(unix)]
fn install_interrupt_handler() {
    extern "C" fn handle_interrupt(_signal: libc::c_int) {
        INTERRUPTED.store(true, Ordering::Release);
    }
    // The handler only performs one lock-free atomic store, which is signal-safe.
    unsafe {
        libc::signal(
            libc::SIGINT,
            handle_interrupt as *const () as libc::sighandler_t,
        );
    }
}

#[cfg(not(unix))]
fn install_interrupt_handler() {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classifies_failed_warmup_without_discarding_later_rows() {
        let outcome = classify_trials(
            7_u32,
            [
                (
                    TrialPhase::Warmup,
                    TrialStatus::Failed,
                    Some("warmup failed"),
                ),
                (TrialPhase::Measured, TrialStatus::Completed, None),
            ]
            .into_iter(),
        );

        assert!(matches!(
            outcome,
            CaseOutcome::Failed {
                stage: FailureStage::Warmup,
                message
            } if message == "warmup failed"
        ));
    }

    #[test]
    fn parses_system_profiler_memory_without_decimal_drift() {
        assert_eq!(parse_human_memory("24 GB"), Some(25_769_803_776));
        assert_eq!(parse_human_memory("16384 MB"), Some(17_179_869_184));
        assert_eq!(parse_human_memory("unknown"), None);
    }
}
