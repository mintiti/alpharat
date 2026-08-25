use serde::{Deserialize, Serialize};

/// Exact protocol-v1 CSV header for backend-capacity trials.
pub const CAPACITY_HEADERS: &[&str] = &[
    "case_id",
    "phase",
    "trial",
    "status",
    "error",
    "wall_ms",
    "positions_per_s",
    "outer_calls",
    "outer_positions",
    "caller_time_ms",
    "caller_union_ms",
    "caller_peak",
    "device_calls",
    "device_positions",
    "device_avg_batch",
    "device_inference_ms",
];

/// Exact protocol-v1 CSV header for fixed-work search trials.
pub const SEARCH_HEADERS: &[&str] = &[
    "case_id",
    "phase",
    "trial",
    "status",
    "error",
    "wall_ms",
    "productive_per_s",
    "nn_evals",
    "terminals",
    "tt_stops",
    "collisions",
    "batches",
    "phase_occupancy_ms",
    "phase_occupancy_per_wall",
    "lease_wait_ms",
    "gate_wait_ms",
    "gate_hold_ms",
    "inference_caller_ms",
    "completion_wait_ms",
    "backend_calls",
    "backend_positions",
    "backend_caller_ms",
    "backend_union_ms",
    "backend_caller_per_union",
    "backend_peak_callers",
    "device_calls",
    "device_positions",
    "device_avg_batch",
    "device_inference_ms",
    "ledger_reserved",
    "ledger_committed",
    "ledger_cancelled",
    "policy_p1_0",
    "policy_p1_1",
    "policy_p1_2",
    "policy_p1_3",
    "policy_p1_4",
    "policy_p2_0",
    "policy_p2_1",
    "policy_p2_2",
    "policy_p2_3",
    "policy_p2_4",
];

/// Whether a trial is warming the runtime or contributing to the measured distribution.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TrialPhase {
    Warmup,
    Measured,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TrialStatus {
    Completed,
    Failed,
    Interrupted,
}

/// One attempt through the production backend boundary.
///
/// Metric names include units. Metrics are required for completed trials and may be absent when a
/// trial fails or is interrupted.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CapacityTrial {
    pub case_id: String,
    pub phase: TrialPhase,
    pub trial: u32,
    pub status: TrialStatus,
    pub error: Option<String>,
    pub wall_ms: Option<f64>,
    pub positions_per_s: Option<f64>,
    pub outer_calls: Option<u64>,
    pub outer_positions: Option<u64>,
    pub caller_time_ms: Option<f64>,
    pub caller_union_ms: Option<f64>,
    pub caller_peak: Option<u32>,
    pub device_calls: Option<u64>,
    pub device_positions: Option<u64>,
    pub device_avg_batch: Option<f64>,
    pub device_inference_ms: Option<f64>,
}

/// One fixed-work MCGS attempt.
///
/// This is deliberately the rich production-search surface already measured by
/// `bench-mcgs-parallel`, now with explicit trial phase and failure state.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SearchTrial {
    pub case_id: String,
    pub phase: TrialPhase,
    pub trial: u32,
    pub status: TrialStatus,
    pub error: Option<String>,
    pub wall_ms: Option<f64>,
    pub productive_per_s: Option<f64>,
    pub nn_evals: Option<u32>,
    pub terminals: Option<u32>,
    pub tt_stops: Option<u32>,
    pub collisions: Option<u32>,
    pub batches: Option<u32>,
    pub phase_occupancy_ms: Option<f64>,
    pub phase_occupancy_per_wall: Option<f64>,
    pub lease_wait_ms: Option<f64>,
    pub gate_wait_ms: Option<f64>,
    pub gate_hold_ms: Option<f64>,
    pub inference_caller_ms: Option<f64>,
    pub completion_wait_ms: Option<f64>,
    pub backend_calls: Option<u64>,
    pub backend_positions: Option<u64>,
    pub backend_caller_ms: Option<f64>,
    pub backend_union_ms: Option<f64>,
    pub backend_caller_per_union: Option<f64>,
    pub backend_peak_callers: Option<u32>,
    pub device_calls: Option<u64>,
    pub device_positions: Option<u64>,
    pub device_avg_batch: Option<f64>,
    pub device_inference_ms: Option<f64>,
    pub ledger_reserved: Option<u32>,
    pub ledger_committed: Option<u32>,
    pub ledger_cancelled: Option<u32>,
    pub policy_p1_0: Option<f32>,
    pub policy_p1_1: Option<f32>,
    pub policy_p1_2: Option<f32>,
    pub policy_p1_3: Option<f32>,
    pub policy_p1_4: Option<f32>,
    pub policy_p2_0: Option<f32>,
    pub policy_p2_1: Option<f32>,
    pub policy_p2_2: Option<f32>,
    pub policy_p2_3: Option<f32>,
    pub policy_p2_4: Option<f32>,
}
