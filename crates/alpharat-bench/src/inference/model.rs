use super::Result;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::path::PathBuf;

pub const PLAN_FORMAT: &str = "alpharat.inference.plan";
pub const RUN_FORMAT: &str = "alpharat.inference.run";
pub const VERSION: u32 = 1;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Mode {
    #[default]
    Clean,
    Latency,
    Timeline,
    Stages,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FileRef {
    pub path: PathBuf,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Plan {
    pub format: String,
    pub schema_version: u32,
    pub plan_id: String,
    pub model: Option<FileRef>,
    pub corpus: FileRef,
    pub measurement: Measurement,
    pub limits: Limits,
    pub comparison: Option<Comparison>,
    pub variants: Vec<Variant>,
    pub cases: Vec<Case>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Measurement {
    #[serde(default)]
    pub mode: Mode,
    pub warmup: Warmup,
    pub calls_per_caller: usize,
    pub repetitions: usize,
    #[serde(default = "sample_limit")]
    pub max_samples_per_thread: usize,
}
fn sample_limit() -> usize {
    100_000
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Warmup {
    pub passes: usize,
    pub min_ms: u64,
    pub max_ms: u64,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Limits {
    pub setup_seconds: u64,
    pub trial_seconds: u64,
    pub total_seconds: u64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Comparison {
    pub baseline: String,
    pub candidate: String,
    pub vary: BTreeSet<String>,
    pub order: Vec<String>,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Variant {
    pub id: String,
    pub executable: PathBuf,
    pub backend: BackendSpec,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum BackendSpec {
    SmartUniform {},
    #[serde(rename = "tensorrt")]
    TensorRt {
        host_io: String,
        opt_batch: usize,
        max_batch: usize,
        #[serde(default, skip_serializing_if = "std::ops::Not::not")]
        pad_to_max: bool,
        #[serde(default, skip_serializing_if = "std::ops::Not::not")]
        cuda_graph: bool,
        cache_dir: PathBuf,
    },
}
impl BackendSpec {
    pub fn max_batch(&self) -> usize {
        match self {
            Self::SmartUniform {} => 4096,
            Self::TensorRt { max_batch, .. } => *max_batch,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum Topology {
    Direct {},
    EagerMux { max_batch: usize },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum Requests {
    Constant {
        batch_size: usize,
        callers: usize,
    },
    Sequence {
        batches: Vec<usize>,
        callers: usize,
        #[serde(default)]
        corpus_offset: usize,
    },
}
impl Requests {
    pub fn callers(&self) -> usize {
        match self {
            Self::Constant { callers, .. } | Self::Sequence { callers, .. } => *callers,
        }
    }
    pub fn sizes(&self) -> Vec<usize> {
        match self {
            Self::Constant { batch_size, .. } => vec![*batch_size],
            Self::Sequence { batches, .. } => batches.clone(),
        }
    }
    pub fn batch(&self, call: usize) -> usize {
        match self {
            Self::Constant { batch_size, .. } => *batch_size,
            Self::Sequence { batches, .. } => batches[call % batches.len()],
        }
    }
    pub fn offset(&self) -> usize {
        match self {
            Self::Sequence { corpus_offset, .. } => *corpus_offset,
            _ => 0,
        }
    }
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "driver", rename_all = "snake_case", deny_unknown_fields)]
pub enum Case {
    Capacity {
        key: String,
        topology: Topology,
        requests: Requests,
    },
    Selfplay {
        key: String,
        topology: Topology,
        config: Selfplay,
    },
}
impl Case {
    pub fn key(&self) -> &str {
        match self {
            Self::Capacity { key, .. } | Self::Selfplay { key, .. } => key,
        }
    }
    pub fn topology(&self) -> &Topology {
        match self {
            Self::Capacity { topology, .. } | Self::Selfplay { topology, .. } => topology,
        }
    }
    pub fn callers(&self) -> usize {
        match self {
            Self::Capacity { requests, .. } => requests.callers(),
            Self::Selfplay { config, .. } => config.workers as usize,
        }
    }
    pub fn sizes(&self) -> Vec<usize> {
        match self {
            Self::Capacity { requests, .. } => requests.sizes(),
            Self::Selfplay { config, .. } => (1..=config.batch_size as usize).collect(),
        }
    }
    pub fn warm_shapes(&self) -> BTreeSet<usize> {
        let sizes = self.sizes().into_iter().collect::<BTreeSet<_>>();
        match self.topology() {
            Topology::Direct {} => sizes.into_iter().collect(),
            Topology::EagerMux { max_batch } => {
                let mut reachable = BTreeSet::from([0]);
                for _ in 0..self.callers() {
                    let next = reachable
                        .iter()
                        .flat_map(|a| sizes.iter().map(move |b| a + b))
                        .filter(|n| *n <= *max_batch)
                        .collect::<Vec<_>>();
                    let before = reachable.len();
                    reachable.extend(next);
                    if reachable.len() == before {
                        break;
                    }
                }
                reachable.remove(&0);
                reachable
            }
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Selfplay {
    #[serde(default, skip_serializing_if = "SearchEngine::is_mcts")]
    pub engine: SearchEngine,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub search: Option<SearchParameters>,
    pub games: u32,
    pub workers: u32,
    pub simulations: u32,
    pub batch_size: u32,
    pub seed: u64,
    #[serde(default = "default_bundle")]
    pub games_per_bundle: usize,
}
fn default_bundle() -> usize {
    32
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SearchEngine {
    #[default]
    Mcts,
    Mcgs,
}
impl SearchEngine {
    fn is_mcts(&self) -> bool {
        matches!(self, Self::Mcts)
    }
}

/// Explicit search policy parameters. Independent-game workers are in Selfplay.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SearchParameters {
    pub c_puct: f32,
    pub fpu_reduction: f32,
    pub force_k: f32,
    pub noise_epsilon: f32,
    pub noise_concentration: f32,
    pub collision_limit_min: u32,
    pub collision_limit_max: u32,
    pub collision_scaling_start: u32,
    pub collision_scaling_end: u32,
    pub collision_scaling_power: f32,
}
impl Default for SearchParameters {
    fn default() -> Self {
        let c = alpharat_mcgs::SearchConfig::default();
        Self {
            c_puct: c.c_puct,
            fpu_reduction: c.fpu_reduction,
            force_k: c.force_k,
            noise_epsilon: c.noise_epsilon,
            noise_concentration: c.noise_concentration,
            collision_limit_min: c.collision_limit_min,
            collision_limit_max: c.collision_limit_max,
            collision_scaling_start: c.collision_scaling_start,
            collision_scaling_end: c.collision_scaling_end,
            collision_scaling_power: c.collision_scaling_power,
        }
    }
}
impl SearchParameters {
    pub fn mcgs(&self) -> alpharat_mcgs::SearchConfig {
        alpharat_mcgs::SearchConfig {
            c_puct: self.c_puct,
            fpu_reduction: self.fpu_reduction,
            force_k: self.force_k,
            noise_epsilon: self.noise_epsilon,
            noise_concentration: self.noise_concentration,
            collision_limit_min: self.collision_limit_min,
            collision_limit_max: self.collision_limit_max,
            collision_scaling_start: self.collision_scaling_start,
            collision_scaling_end: self.collision_scaling_end,
            collision_scaling_power: self.collision_scaling_power,
        }
    }
    pub fn mcts(&self) -> alpharat_mcts::SearchConfig {
        alpharat_mcts::SearchConfig {
            c_puct: self.c_puct,
            fpu_reduction: self.fpu_reduction,
            force_k: self.force_k,
            noise_epsilon: self.noise_epsilon,
            noise_concentration: self.noise_concentration,
            collision_limit_min: self.collision_limit_min,
            collision_limit_max: self.collision_limit_max,
            collision_scaling_start: self.collision_scaling_start,
            collision_scaling_end: self.collision_scaling_end,
            collision_scaling_power: self.collision_scaling_power,
        }
    }
    fn validate(&self) -> bool {
        [
            self.c_puct,
            self.fpu_reduction,
            self.force_k,
            self.noise_epsilon,
            self.noise_concentration,
            self.collision_scaling_power,
        ]
        .iter()
        .all(|x| x.is_finite() && *x >= 0.0)
            && self.noise_epsilon <= 1.0
            && self.noise_concentration > 0.0
            && self.collision_scaling_power > 0.0
            && self.collision_limit_min > 0
            && self.collision_limit_min <= self.collision_limit_max
            && self.collision_scaling_start < self.collision_scaling_end
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Corpus {
    pub schema_version: u32,
    pub label: String,
    pub positions: Vec<GameRecipe>,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GameRecipe {
    pub width: u8,
    pub height: u8,
    pub player_1: [u8; 2],
    pub player_2: [u8; 2],
    pub cheese: Vec<[u8; 2]>,
    pub max_turns: u16,
    pub creation_seed: u64,
    #[serde(default)]
    pub walls: Vec<[[u8; 2]; 2]>,
    #[serde(default)]
    pub mud: Vec<MudEdge>,
    #[serde(default)]
    pub actions: Vec<[u8; 2]>,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MudEdge {
    pub from: [u8; 2],
    pub to: [u8; 2],
    pub cost: u8,
}
pub fn valid_id(id: &str) -> bool {
    !id.is_empty()
        && id.len() <= 100
        && id
            .bytes()
            .all(|c| c.is_ascii_alphanumeric() || c == b'-' || c == b'_')
}

impl Plan {
    pub fn validate(&self) -> Result<()> {
        if self.format != PLAN_FORMAT || self.schema_version != VERSION {
            return Err("unsupported inference plan format/version".into());
        }
        if !valid_id(&self.plan_id) || self.variants.is_empty() || self.cases.is_empty() {
            return Err("plan needs a safe plan_id, variants, and cases".into());
        }
        if self.cases.len() > 1000
            || self.variants.len() > 100
            || self
                .cases
                .len()
                .saturating_mul(self.variants.len())
                .saturating_mul(self.measurement.repetitions)
                > 10000
        {
            return Err("plan exceeds 10000 attempts".into());
        }
        let m = &self.measurement;
        if m.calls_per_caller == 0
            || m.calls_per_caller > 10_000_000
            || m.repetitions == 0
            || m.repetitions > 1000
            || m.warmup.passes == 0
            || m.warmup.max_ms == 0
            || m.warmup.min_ms > m.warmup.max_ms
            || m.max_samples_per_thread == 0
            || m.max_samples_per_thread > 10_000_000
        {
            return Err("invalid or excessive measurement/warmup limits".into());
        }
        if [
            self.limits.setup_seconds,
            self.limits.trial_seconds,
            self.limits.total_seconds,
        ]
        .contains(&0)
        {
            return Err("all execution time limits must be positive".into());
        }
        let mut ids = BTreeSet::new();
        for v in &self.variants {
            if !valid_id(&v.id) || !ids.insert(&v.id) {
                return Err("invalid/duplicate variant id".into());
            }
            if let BackendSpec::TensorRt {
                host_io,
                opt_batch,
                max_batch,
                pad_to_max,
                ..
            } = &v.backend
            {
                if self.model.is_none()
                    || !["pinned", "pageable"].contains(&host_io.as_str())
                    || (*pad_to_max && host_io != "pinned")
                    || *opt_batch == 0
                    || opt_batch > max_batch
                    || *max_batch > 4096
                {
                    return Err("invalid TensorRT model, host_io or profile".into());
                }
            }
            if m.mode == Mode::Stages && matches!(v.backend, BackendSpec::SmartUniform {}) {
                return Err("stage timing requires TensorRT".into());
            }
        }

        let mut keys = BTreeSet::new();
        for c in &self.cases {
            if !valid_id(c.key()) || !keys.insert(c.key()) || c.callers() == 0 || c.callers() > 256
            {
                return Err("invalid/duplicate case key or caller count".into());
            }
            if c.sizes().is_empty() || c.sizes().len() > 4096 || c.sizes().contains(&0) {
                return Err("request batches must be nonempty and positive".into());
            }
            if let Case::Selfplay { config, .. } = c {
                if (config.engine == SearchEngine::Mcgs && config.search.is_none())
                    || config.search.as_ref().is_some_and(|c| !c.validate())
                {
                    return Err("MCGS requires explicit valid search parameters".into());
                }

                if config.games == 0
                    || config.simulations == 0
                    || config.batch_size == 0
                    || config.games_per_bundle == 0
                    || config.games > 100000
                    || config.batch_size > 4096
                {
                    return Err("invalid self-play work".into());
                }
            }
            for v in &self.variants {
                let bound = match c.topology() {
                    Topology::Direct {} => v.backend.max_batch(),
                    Topology::EagerMux { max_batch } => {
                        if *max_batch == 0 || *max_batch > v.backend.max_batch() {
                            return Err("mux bound exceeds backend profile".into());
                        }
                        *max_batch
                    }
                };
                if c.sizes().iter().any(|n| *n > bound) {
                    return Err("request exceeds batch bound".into());
                }
            }
        }
        if let Some(c) = &self.comparison {
            if self.variants.len() != 2
                || c.baseline == c.candidate
                || !ids.contains(&c.baseline)
                || !ids.contains(&c.candidate)
                || c.order.len() != m.repetitions
                || c.order.iter().any(|s| s != "AB" && s != "BA")
            {
                return Err(
                    "comparison requires two variants and one AB/BA order per repetition".into(),
                );
            }
            let allowed = [
                "host_io",
                "profile",
                "source",
                "build",
                "runtime",
                "hardware",
                "topology",
                "requests",
                "batch_shape",
                "cuda_graph",
            ];
            if c.vary.iter().any(|x| !allowed.contains(&x.as_str())) {
                return Err("unknown comparison axis".into());
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn reachable_mux_shapes_respect_caller_and_cap_bounds() {
        let c = Case::Capacity {
            key: "x".into(),
            topology: Topology::EagerMux { max_batch: 10 },
            requests: Requests::Sequence {
                batches: vec![3, 4],
                callers: 2,
                corpus_offset: 0,
            },
        };
        assert_eq!(c.warm_shapes(), BTreeSet::from([3, 4, 6, 7, 8]));
    }
    #[test]
    fn ids_cannot_escape_artifact_folders() {
        for value in ["", "../bad", "a/b", ".", "a b"] {
            assert!(!valid_id(value));
        }
        assert!(valid_id("direct-b32_c4"));
    }
    #[test]
    fn old_tensor_rt_plans_keep_exact_non_graph_execution() {
        let text = r#"{"kind":"tensorrt","host_io":"pinned","opt_batch":128,"max_batch":128,"cache_dir":"cache"}"#;
        let backend: BackendSpec = serde_json::from_str(text).unwrap();
        assert!(matches!(
            backend,
            BackendSpec::TensorRt {
                pad_to_max: false,
                cuda_graph: false,
                ..
            }
        ));
        let serialized = serde_json::to_value(backend).unwrap();
        assert!(serialized.get("pad_to_max").is_none());
        assert!(serialized.get("cuda_graph").is_none());
    }

    #[test]
    fn unknown_configuration_is_rejected() {
        assert!(serde_json::from_str::<Topology>(r#"{"kind":"direct","lanes":4}"#).is_err());
        assert!(
            serde_json::from_str::<BackendSpec>(r#"{"kind":"smart_uniform","fast":true}"#).is_err()
        );
    }
}
