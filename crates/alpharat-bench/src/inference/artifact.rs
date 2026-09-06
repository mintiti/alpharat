use super::model::{Case, Mode, Plan, Variant, RUN_FORMAT, VERSION};
use super::Result;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FileIdentity {
    pub path: PathBuf,
    pub sha256: String,
    pub bytes: u64,
}
pub fn hash(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
pub fn identify(path: &Path) -> Result<FileIdentity> {
    use std::io::Read;
    let path = fs::canonicalize(path)?;
    let mut file = fs::File::open(&path)?;
    let mut digest = Sha256::new();
    let mut buffer = [0u8; 65536];
    let mut bytes = 0u64;
    loop {
        let n = file.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        digest.update(&buffer[..n]);
        bytes += n as u64;
    }
    Ok(FileIdentity {
        path,
        sha256: format!("{:x}", digest.finalize()),
        bytes,
    })
}

pub fn verify(identity: &FileIdentity) -> Result<()> {
    let found = identify(&identity.path)?;
    if found.sha256 != identity.sha256 || found.bytes != identity.bytes {
        return Err(format!("artifact changed: {}", identity.path.display()).into());
    }
    Ok(())
}
pub fn write_json(path: &Path, value: &impl Serialize) -> Result<()> {
    let temp = path.with_extension("tmp");
    let mut bytes = serde_json::to_vec_pretty(value)?;
    bytes.push(b'\n');
    use std::io::Write;
    let mut file = fs::File::create(&temp)?;
    file.write_all(&bytes)?;
    file.sync_all()?;
    fs::rename(temp, path)?;
    if let Some(parent) = path.parent() {
        fs::File::open(parent)?.sync_all()?;
    }
    Ok(())
}
pub fn read_json<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T> {
    Ok(serde_json::from_slice(&fs::read(path)?)?)
}
pub fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Identity {
    pub executable: FileIdentity,
    pub source_revision: String,
    pub source_state: String,
    pub source_patch_sha256: String,
    pub rustc: String,
    pub target: String,
    pub profile: String,
    pub features: String,
    pub rustflags: String,
    pub cargo_lock_sha256: String,
    pub libraries: Vec<FileIdentity>,
    pub hardware: String,
    pub runtime_environment: BTreeMap<String, String>,
    pub model_sha256: Option<String>,
    pub corpus_sha256: String,
    pub encoded_sha256: String,
    pub engine_sha256: Option<String>,
    pub physical_contexts: usize,
}
pub fn gpu_query(fields: &str) -> Option<String> {
    ["nvidia-smi", "/usr/lib/wsl/lib/nvidia-smi"]
        .iter()
        .find_map(|program| {
            Command::new(program)
                .args([
                    format!("--query-gpu={fields}"),
                    "--format=csv,noheader".into(),
                ])
                .output()
                .ok()
                .filter(|o| o.status.success())
                .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_owned())
        })
}
pub fn build_description() -> serde_json::Value {
    serde_json::json!({
        "format":"alpharat.inference.capabilities", "schema_version":VERSION,
        "worker_handshake":true, "mcgs_selfplay":true,
        "tensorrt":cfg!(feature="tensorrt"), "timeline":cfg!(feature="inference-trace"),
        "rustc":env!("ALPHARAT_BUILD_RUSTC_VERSION"),
        "source":env!("ALPHARAT_BUILD_SOURCE_REVISION")
    })
}
pub fn identity(
    corpus_hash: &str,
    encoded_hash: &str,
    model: Option<&FileIdentity>,
    engine: Option<String>,
    output: &Path,
) -> Result<Identity> {
    let patch = include_bytes!(concat!(env!("OUT_DIR"), "/alpharat-build-source.patch"));
    fs::write(output.join("source.patch"), patch)?;
    let paths = fs::read_to_string("/proc/self/maps")
        .unwrap_or_default()
        .lines()
        .filter_map(|line| line.split_whitespace().last())
        .filter(|s| {
            s.starts_with('/')
                && (s.contains("libtensorrt")
                    || s.contains("libcudart")
                    || s.contains("libcuda.so"))
        })
        .map(PathBuf::from)
        .collect::<BTreeSet<_>>();
    let libraries = paths
        .iter()
        .map(|p| identify(p))
        .collect::<Result<Vec<_>>>()?;
    let gpu = gpu_query("uuid,name,driver_version").unwrap_or_default();
    if engine.is_some() && gpu.is_empty() {
        return Err("cannot identify the active GPU/driver".into());
    }
    let cpu = fs::read_to_string("/proc/cpuinfo")
        .unwrap_or_default()
        .lines()
        .find(|line| line.starts_with("model name"))
        .unwrap_or("unknown cpu")
        .to_owned();
    Ok(Identity {
        executable: identify(&std::env::current_exe()?)?,
        source_revision: env!("ALPHARAT_BUILD_SOURCE_REVISION").into(),
        source_state: env!("ALPHARAT_BUILD_SOURCE_STATE").into(),
        source_patch_sha256: hash(patch),
        rustc: env!("ALPHARAT_BUILD_RUSTC_VERSION").into(),
        target: env!("ALPHARAT_BUILD_TARGET").into(),
        profile: env!("ALPHARAT_BUILD_PROFILE").into(),
        features: env!("ALPHARAT_INFER_FEATURES").into(),
        rustflags: env!("ALPHARAT_INFER_RUSTFLAGS").into(),
        cargo_lock_sha256: env!("ALPHARAT_INFER_LOCK_SHA256").into(),
        runtime_environment: [
            "CUDA_VISIBLE_DEVICES",
            "CUDA_MODULE_LOADING",
            "CUDA_LAUNCH_BLOCKING",
            "CUDA_DEVICE_MAX_CONNECTIONS",
            "OMP_NUM_THREADS",
            "RAYON_NUM_THREADS",
        ]
        .into_iter()
        .map(|key| {
            (
                key.into(),
                std::env::var(key).unwrap_or_else(|_| "<unset>".into()),
            )
        })
        .collect(),
        libraries,
        hardware: format!("{cpu}; {gpu}"),
        model_sha256: model.map(|m| m.sha256.clone()),
        corpus_sha256: corpus_hash.into(),
        encoded_sha256: encoded_hash.into(),
        physical_contexts: usize::from(engine.is_some()),
        engine_sha256: engine,
    })
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LatencySummary {
    pub samples: usize,
    pub dropped: u64,
    pub p50_ns: u64,
    pub p95_ns: u64,
    pub max_ns: u64,
}
pub fn latency_summary(mut times: Vec<u64>, dropped: u64) -> Option<LatencySummary> {
    if times.is_empty() {
        return None;
    }
    times.sort_unstable();
    let at = |percent: usize| times[(times.len() * percent).div_ceil(100).saturating_sub(1)];
    Some(LatencySummary {
        samples: times.len(),
        dropped,
        p50_ns: at(50),
        p95_ns: at(95),
        max_ns: *times.last().unwrap(),
    })
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Sample {
    pub trace_id: u64,
    pub worker: usize,
    pub request: usize,
    pub positions: usize,
    pub started_ns: u64,
    pub elapsed_ns: u64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MuxObservation {
    pub calls: u64,
    pub positions: u64,
    pub batch_histogram: Vec<(usize, u64)>,
    pub worker_backend_ns: u64,
    pub worker_wait_drain_ns: u64,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StageObservation {
    pub calls: u64,
    pub positions: u64,
    pub encode_ns: u64,
    pub input_stage_ns: u64,
    pub h2d_ns: u64,
    pub infer_ns: u64,
    pub d2h_ns: u64,
    pub output_alloc_ns: u64,
    pub parse_ns: u64,
    pub total_ns: u64,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SelfplayObservation {
    pub games: u32,
    pub positions: u64,
    pub simulations: u64,
    pub nn_evals: u64,
    pub terminals: u64,
    pub collisions: u64,
    #[serde(default, skip_serializing_if = "is_zero")]
    pub tt_stop_hits: u64,
    pub bundle_files: Vec<String>,
}
fn is_zero(value: &u64) -> bool {
    *value == 0
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Metrics {
    pub wall_ns: u64,
    pub completed_calls: Option<u64>,
    pub completed_evaluations: u64,
    pub latency: Option<LatencySummary>,
    pub mux: Option<MuxObservation>,
    pub stages: Option<StageObservation>,
    pub selfplay: Option<SelfplayObservation>,
}
impl Metrics {
    pub fn evaluations_per_second(&self) -> f64 {
        self.completed_evaluations as f64 * 1e9 / self.wall_ns as f64
    }
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TrialResult {
    pub schema_version: u32,
    pub case_key: String,
    pub variant_id: String,
    pub mode: Mode,
    pub identity: Identity,
    pub setup_ms: u64,
    pub warmup_ms: u64,
    pub warmup_passes: usize,
    pub warmed_shapes: Vec<usize>,
    pub metrics: Metrics,
    pub artifacts: BTreeMap<String, String>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TrialRequest {
    pub plan: Plan,
    pub case: Case,
    pub variant: Variant,
    pub corpus: FileIdentity,
    pub model: Option<FileIdentity>,
    pub executable: FileIdentity,
    pub expected_identity: Option<Identity>,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Status {
    Planned,
    Running,
    Completed,
    Failed,
    TimedOut,
    Interrupted,
    Unsupported,
    NotRun,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Attempt {
    pub id: String,
    pub case_key: String,
    pub variant_id: String,
    pub repetition: usize,
    pub status: Status,
    pub stage: String,
    pub error: Option<String>,
    pub elapsed_ms: u64,
    pub request_sha256: Option<String>,
    pub result_sha256: Option<String>,
    pub result: Option<TrialResult>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RunRecord {
    pub format: String,
    pub schema_version: u32,
    pub run_id: String,
    pub created_at: u64,
    pub status: Status,
    pub plan: Plan,
    pub plan_sha256: String,
    pub corpus: FileIdentity,
    pub model: Option<FileIdentity>,
    pub executables: BTreeMap<String, FileIdentity>,
    pub parent: Option<String>,
    pub attempts: Vec<Attempt>,
}
pub fn safe_artifact_path(folder: &Path, name: &str) -> Result<PathBuf> {
    use std::path::Component;
    if Path::new(name)
        .components()
        .any(|c| !matches!(c, Component::Normal(_)))
    {
        return Err("artifact path escapes run folder".into());
    }
    Ok(folder.join(name))
}
pub fn load_run(folder: &Path) -> Result<RunRecord> {
    let r: RunRecord = read_json(&folder.join("run.json"))?;
    if r.format != RUN_FORMAT || r.schema_version != VERSION {
        return Err("unsupported inference run".into());
    }
    r.plan.validate()?;
    if hash(&fs::read(folder.join("plan.json"))?) != r.plan_sha256
        || read_json::<Plan>(&folder.join("plan.json"))? != r.plan
    {
        return Err("saved plan identity differs".into());
    }
    if hash(&fs::read(folder.join("corpus.json"))?) != r.corpus.sha256 {
        return Err("saved corpus identity differs".into());
    }
    let expected_count =
        r.plan.cases.len() * r.plan.variants.len() * r.plan.measurement.repetitions;
    if r.attempts.len() != expected_count {
        return Err("attempt set differs from plan".into());
    }
    if r.status == Status::Completed && r.attempts.iter().any(|a| a.status != Status::Completed) {
        return Err("completed run contains incomplete attempts".into());
    }
    let mut keys = BTreeSet::new();
    let mut ids = BTreeSet::new();
    for a in &r.attempts {
        if !super::model::valid_id(&a.id) || !ids.insert(&a.id) {
            return Err("invalid/duplicate attempt id".into());
        }
        if a.repetition >= r.plan.measurement.repetitions
            || !r.plan.cases.iter().any(|c| c.key() == a.case_key)
            || !r.plan.variants.iter().any(|v| v.id == a.variant_id)
            || !keys.insert((&a.case_key, &a.variant_id, a.repetition))
        {
            return Err("invalid attempt schedule".into());
        }
        let dir = folder.join("attempts").join(&a.id);
        if let Some(expected) = &a.request_sha256 {
            if hash(&fs::read(dir.join("request.json"))?) != *expected {
                return Err("attempt request changed".into());
            }
        }
        if a.status == Status::Completed {
            let result = a.result.as_ref().ok_or("completed attempt has no result")?;
            let request: TrialRequest = read_json(&dir.join("request.json"))?;
            if a.request_sha256.is_none()
                || request.plan != r.plan
                || !r.plan.cases.contains(&request.case)
                || !r.plan.variants.contains(&request.variant)
                || request.case.key() != a.case_key
                || request.variant.id != a.variant_id
                || request.corpus.sha256 != r.corpus.sha256
                || request.model != r.model
                || Some(&request.executable) != r.executables.get(&a.variant_id)
                || result.identity.executable.sha256 != request.executable.sha256
                || result.identity.corpus_sha256 != r.corpus.sha256
                || result.identity.model_sha256 != r.model.as_ref().map(|m| m.sha256.clone())
                || result.artifacts.get("source.patch")
                    != Some(&result.identity.source_patch_sha256)
                || result.schema_version != VERSION
            {
                return Err("trial identity differs from saved request".into());
            }
            if result.case_key != a.case_key
                || result.variant_id != a.variant_id
                || result.mode != r.plan.measurement.mode
                || result.metrics.wall_ns == 0
            {
                return Err("inconsistent completed trial".into());
            }
            let bytes = fs::read(dir.join("result.json"))?;
            if Some(hash(&bytes)) != a.result_sha256
                || serde_json::from_slice::<TrialResult>(&bytes)? != *result
            {
                return Err("trial result identity differs".into());
            }
            for (name, expected) in &result.artifacts {
                if hash(&fs::read(safe_artifact_path(&dir, name)?)?) != *expected {
                    return Err(format!("trial artifact changed: {name}").into());
                }
            }
            let case = r
                .plan
                .cases
                .iter()
                .find(|c| c.key() == a.case_key)
                .ok_or("unknown trial case")?;
            if result.warmed_shapes != case.warm_shapes().into_iter().collect::<Vec<_>>()
                || result.warmup_passes < r.plan.measurement.warmup.passes
                || result.warmup_ms < r.plan.measurement.warmup.min_ms
            {
                return Err("warmup does not cover planned shapes and policy".into());
            }
            if let Case::Capacity { requests, .. } = case {
                let calls = (requests.callers() * r.plan.measurement.calls_per_caller) as u64;
                let positions = (0..r.plan.measurement.calls_per_caller)
                    .map(|i| requests.batch(i) as u64)
                    .sum::<u64>()
                    * requests.callers() as u64;
                if result.metrics.completed_calls != Some(calls)
                    || result.metrics.completed_evaluations != positions
                {
                    return Err("trial work differs from planned work".into());
                }
            }
            if let Some(m) = &result.metrics.mux {
                if m.calls != m.batch_histogram.iter().map(|(_, n)| n).sum::<u64>()
                    || m.positions
                        != m.batch_histogram
                            .iter()
                            .map(|(b, n)| *b as u64 * n)
                            .sum::<u64>()
                    || m.positions != result.metrics.completed_evaluations
                {
                    return Err("mux work accounting differs".into());
                }
            }
            if matches!(result.mode, Mode::Latency | Mode::Timeline) {
                if !result.artifacts.contains_key("requests.csv") {
                    return Err("latency samples missing".into());
                }
                let samples = csv::Reader::from_path(dir.join("requests.csv"))?
                    .deserialize::<Sample>()
                    .collect::<std::result::Result<Vec<_>, _>>()?;
                let summary = result
                    .metrics
                    .latency
                    .as_ref()
                    .ok_or("latency summary missing")?;
                let mut seen = BTreeSet::new();
                if samples
                    .iter()
                    .any(|s| s.positions == 0 || !seen.insert((s.worker, s.request)))
                    || result.metrics.completed_calls
                        != Some(samples.len() as u64 + summary.dropped)
                    || latency_summary(
                        samples.iter().map(|s| s.elapsed_ns).collect(),
                        summary.dropped,
                    ) != result.metrics.latency
                {
                    return Err("latency summary differs from retained requests".into());
                }
            }
            if let Case::Selfplay { config, .. } = case {
                let s = result
                    .metrics
                    .selfplay
                    .as_ref()
                    .ok_or("self-play counters missing")?;
                if s.games != config.games
                    || s.nn_evals != result.metrics.completed_evaluations
                    || s.bundle_files.is_empty()
                    || s.bundle_files
                        .iter()
                        .any(|f| !result.artifacts.contains_key(f))
                {
                    return Err("self-play completion accounting differs".into());
                }
            }
            if result.mode == Mode::Stages && result.metrics.stages.is_none() {
                return Err("stage observations missing".into());
            }
            if !matches!(result.mode, Mode::Latency | Mode::Timeline)
                && (result.metrics.latency.is_some()
                    || result.artifacts.contains_key("requests.csv")
                    || result.artifacts.contains_key("batch-links.json"))
            {
                return Err("unexpected request diagnostics".into());
            }
            if result.mode != Mode::Stages && result.metrics.stages.is_some() {
                return Err("unexpected stage diagnostics".into());
            }
            if let Some(stages) = &result.metrics.stages {
                if stages.calls == 0 || stages.positions != result.metrics.completed_evaluations {
                    return Err("stage work accounting differs".into());
                }
            }
            if result.mode == Mode::Clean
                && (result.metrics.latency.is_some() || result.metrics.stages.is_some())
            {
                return Err("clean result contains diagnostic metrics".into());
            }
        }
    }
    Ok(r)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn percentiles_use_request_samples_not_means() {
        let s = latency_summary(vec![100, 1, 3, 2], 0).unwrap();
        assert_eq!((s.samples, s.p50_ns, s.p95_ns), (4, 2, 100));
        assert!(latency_summary(vec![], 0).is_none());
    }
    #[test]
    fn artifacts_cannot_escape() {
        assert!(safe_artifact_path(Path::new("/tmp/run"), "../secret").is_err());
        assert!(safe_artifact_path(Path::new("/tmp/run"), "/absolute").is_err());
        assert!(safe_artifact_path(Path::new("/tmp/run"), "sub/file.csv").is_ok());
    }
    #[test]
    fn file_identity_streams_large_files_without_changing_digest() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("bytes");
        let bytes = (0..150001).map(|i| (i % 251) as u8).collect::<Vec<_>>();
        fs::write(&path, &bytes).unwrap();
        let id = identify(&path).unwrap();
        assert_eq!(id.sha256, hash(&bytes));
        assert_eq!(id.bytes, 150001);
        verify(&id).unwrap();
        fs::write(&path, b"changed").unwrap();
        assert!(verify(&id).is_err());
    }
}
