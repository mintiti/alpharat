use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

pub const PROTOCOL_VERSION: u32 = 1;
pub const RUN_RECORD_FILE: &str = "run.json";
pub const CAPACITY_TRIALS_FILE: &str = "capacity-trials.csv";
pub const SEARCH_TRIALS_FILE: &str = "search-trials.csv";

/// The durable record left by one execution of a calibration plan.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RunRecord {
    pub protocol_version: u32,
    pub run_id: String,
    pub created_at: String,
    pub context: RunContext,
    pub plan: RunPlan,
    pub cases: Vec<CaseRecord>,
    pub trial_files: TrialFiles,
}

/// Facts shared by every case in a run.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RunContext {
    pub source: SourceIdentity,
    pub build: BuildIdentity,
    pub hardware: HardwareIdentity,
    pub runtime: RuntimeIdentity,
    pub model: ArtifactIdentity,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ArtifactIdentity {
    /// A human-readable description, not comparison identity.
    pub label: String,
    /// The content identity used by comparison checks.
    pub sha256: String,
    pub bytes: u64,
    /// A recovery hint that may change when the same content moves.
    pub path_hint: Option<String>,
}

impl ArtifactIdentity {
    pub(crate) fn has_same_content_as(&self, other: &Self) -> bool {
        self.sha256 == other.sha256 && self.bytes == other.bytes
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SourceIdentity {
    pub revision: String,
    pub state: SourceState,
}

impl SourceIdentity {
    pub(crate) fn has_same_content_as(&self, other: &Self) -> bool {
        self.revision == other.revision && self.state.has_same_content_as(&other.state)
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields, rename_all = "snake_case", tag = "state")]
pub enum SourceState {
    Clean,
    Patched {
        patch_file: String,
        patch_sha256: String,
    },
    NonReproducible {
        reason: String,
    },
}

impl SourceState {
    pub fn is_reproducible(&self) -> bool {
        !matches!(self, Self::NonReproducible { .. })
    }

    fn has_same_content_as(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Clean, Self::Clean) => true,
            (
                Self::Patched {
                    patch_sha256: left, ..
                },
                Self::Patched {
                    patch_sha256: right,
                    ..
                },
            ) => left == right,
            (Self::NonReproducible { reason: left }, Self::NonReproducible { reason: right }) => {
                left == right
            }
            _ => false,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct BuildIdentity {
    pub profile: String,
    pub target: String,
    pub rustc_version: String,
    pub features: BTreeSet<String>,
    pub binary_sha256: String,
    pub command: Vec<String>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct HardwareIdentity {
    pub operating_system: String,
    pub architecture: String,
    pub cpu: String,
    pub logical_cores: u32,
    pub memory_bytes: u64,
    pub accelerators: Vec<AcceleratorIdentity>,
}

#[derive(Clone, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(deny_unknown_fields)]
pub struct AcceleratorIdentity {
    pub kind: String,
    pub name: String,
    pub memory_bytes: Option<u64>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimeIdentity {
    pub software: Vec<SoftwareVersion>,
}

impl RuntimeIdentity {
    pub(crate) fn has_same_versions_as(&self, other: &Self) -> bool {
        let mut left = self.software.iter().collect::<Vec<_>>();
        let mut right = other.software.iter().collect::<Vec<_>>();
        left.sort_unstable();
        right.sort_unstable();
        left == right
    }
}

#[derive(Clone, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SoftwareVersion {
    pub name: String,
    pub version: String,
}

/// The requested cases and the dimensions intentionally allowed to differ.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RunPlan {
    pub comparison_axes: BTreeSet<ComparisonAxis>,
    pub cases: Vec<CasePlan>,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ComparisonAxis {
    Source,
    Build,
    Hardware,
    Runtime,
    Backend,
    BatchSize,
    Callers,
    Workers,
    TotalInFlight,
    MuxMaxBatch,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields, rename_all = "snake_case", tag = "benchmark")]
pub enum CasePlan {
    BackendCapacity {
        id: String,
        workload: ArtifactIdentity,
        requested: CapacityRequest,
    },
    Search {
        id: String,
        workload: ArtifactIdentity,
        requested: SearchRequest,
    },
}

impl CasePlan {
    pub fn id(&self) -> &str {
        match self {
            Self::BackendCapacity { id, .. } | Self::Search { id, .. } => id,
        }
    }

    pub fn kind(&self) -> BenchmarkKind {
        match self {
            Self::BackendCapacity { .. } => BenchmarkKind::BackendCapacity,
            Self::Search { .. } => BenchmarkKind::Search,
        }
    }

    pub fn workload(&self) -> &ArtifactIdentity {
        match self {
            Self::BackendCapacity { workload, .. } | Self::Search { workload, .. } => workload,
        }
    }

    pub fn warmup_trials(&self) -> u32 {
        match self {
            Self::BackendCapacity { requested, .. } => requested.warmup_trials,
            Self::Search { requested, .. } => requested.warmup_trials,
        }
    }

    pub fn measured_trials(&self) -> u32 {
        match self {
            Self::BackendCapacity { requested, .. } => requested.measured_trials,
            Self::Search { requested, .. } => requested.measured_trials,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BenchmarkKind {
    BackendCapacity,
    Search,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CapacityRequest {
    pub backend: BackendRequest,
    pub batch_size: u32,
    pub callers: u32,
    pub calls_per_caller: u32,
    pub warmup_trials: u32,
    pub measured_trials: u32,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SearchRequest {
    pub backend: BackendRequest,
    pub workers: u32,
    pub total_in_flight: u32,
    pub mux_max_batch: Option<u32>,
    pub productive_work: u32,
    pub warmup_trials: u32,
    pub measured_trials: u32,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields, rename_all = "snake_case", tag = "engine")]
pub enum BackendRequest {
    SmartUniform,
    Onnx { provider: OnnxProvider },
    TensorRt,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum OnnxProvider {
    Cpu,
    Coreml,
    Cuda,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum BackendSerialization {
    None,
    SessionMutex,
    ContextMutex,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ResolvedBackend {
    pub backend: BackendRequest,
    pub serialization: BackendSerialization,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields, rename_all = "snake_case", tag = "benchmark")]
pub enum CaseRecord {
    BackendCapacity {
        id: String,
        outcome: CaseOutcome<CapacityResolved>,
    },
    Search {
        id: String,
        outcome: CaseOutcome<SearchResolved>,
    },
}

impl CaseRecord {
    pub fn id(&self) -> &str {
        match self {
            Self::BackendCapacity { id, .. } | Self::Search { id, .. } => id,
        }
    }

    pub fn kind(&self) -> BenchmarkKind {
        match self {
            Self::BackendCapacity { .. } => BenchmarkKind::BackendCapacity,
            Self::Search { .. } => BenchmarkKind::Search,
        }
    }

    pub fn state(&self) -> CaseState {
        match self {
            Self::BackendCapacity { outcome, .. } => outcome.state(),
            Self::Search { outcome, .. } => outcome.state(),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields, rename_all = "snake_case", tag = "state")]
pub enum CaseOutcome<T> {
    Completed {
        resolved: T,
    },
    Failed {
        stage: FailureStage,
        message: String,
    },
    Unsupported {
        reason: String,
    },
    Interrupted {
        reason: String,
    },
}

impl<T> CaseOutcome<T> {
    pub fn state(&self) -> CaseState {
        match self {
            Self::Completed { .. } => CaseState::Completed,
            Self::Failed { .. } => CaseState::Failed,
            Self::Unsupported { .. } => CaseState::Unsupported,
            Self::Interrupted { .. } => CaseState::Interrupted,
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CaseState {
    Completed,
    Failed,
    Unsupported,
    Interrupted,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum FailureStage {
    Setup,
    Warmup,
    Measurement,
    Finalization,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CapacityResolved {
    pub backend: ResolvedBackend,
    pub batch_size: u32,
    pub callers: u32,
    pub calls_per_caller: u32,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SearchResolved {
    pub backend: ResolvedBackend,
    pub workers: u32,
    pub worker_batch: u32,
    pub total_in_flight: u32,
    pub mux_max_batch: Option<u32>,
    pub productive_work: u32,
}

#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TrialFiles {
    pub backend_capacity: Option<TrialFile>,
    pub search: Option<TrialFile>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TrialFile {
    pub path: String,
    pub sha256: String,
    pub rows: u32,
}
