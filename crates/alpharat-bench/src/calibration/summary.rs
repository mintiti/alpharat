use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use super::compare::{check_comparable, ComparisonIssue};
use super::folder::LoadedRun;
use super::model::{BenchmarkKind, CaseState, ComparisonAxis};
use super::trials::{CapacityTrial, SearchTrial, TrialPhase, TrialStatus};

/// A deterministic descriptive distribution derived from completed measured trials.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Distribution {
    pub count: u32,
    pub min: f64,
    pub p25: f64,
    pub median: f64,
    pub p75: f64,
    pub max: f64,
    pub mean: f64,
    pub standard_deviation: f64,
}

impl Distribution {
    fn from_values(mut values: Vec<f64>) -> Option<Self> {
        if values.is_empty() {
            return None;
        }
        values.sort_by(f64::total_cmp);
        let count = values.len();
        let mean = values.iter().sum::<f64>() / count as f64;
        let variance = values
            .iter()
            .map(|value| {
                let difference = value - mean;
                difference * difference
            })
            .sum::<f64>()
            / count as f64;
        Some(Self {
            count: u32::try_from(count).unwrap_or(u32::MAX),
            min: values[0],
            p25: quantile(&values, 0.25),
            median: quantile(&values, 0.5),
            p75: quantile(&values, 0.75),
            max: values[count - 1],
            mean,
            standard_deviation: variance.sqrt(),
        })
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SummaryBenchmark {
    BackendCapacity,
    Search,
}

impl From<BenchmarkKind> for SummaryBenchmark {
    fn from(value: BenchmarkKind) -> Self {
        match value {
            BenchmarkKind::BackendCapacity => Self::BackendCapacity,
            BenchmarkKind::Search => Self::Search,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TrialCounts {
    pub completed: u32,
    pub failed: u32,
    pub interrupted: u32,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CaseSummary {
    pub id: String,
    pub comparison_key: String,
    pub label: String,
    pub benchmark: SummaryBenchmark,
    pub state: CaseState,
    pub measured_trials: TrialCounts,
    /// Metric names retain their protocol-v1 units, for example `wall_ms`.
    pub metrics: BTreeMap<String, Distribution>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RunSummary {
    pub protocol_version: u32,
    pub plan_id: String,
    pub run_id: String,
    pub cases: Vec<CaseSummary>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct MetricDelta {
    pub left: Distribution,
    pub right: Distribution,
    pub median_delta: f64,
    pub median_delta_percent: Option<f64>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CaseComparison {
    pub comparison_key: String,
    pub left_id: String,
    pub right_id: String,
    pub left_label: String,
    pub right_label: String,
    pub benchmark: SummaryBenchmark,
    pub metrics: BTreeMap<String, MetricDelta>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RunComparison {
    pub protocol_version: u32,
    pub left_run_id: String,
    pub right_run_id: String,
    pub differing_axes: BTreeSet<ComparisonAxis>,
    pub cases: Vec<CaseComparison>,
    pub unavailable_cases: Vec<String>,
}

/// Derive the disposable machine-readable summary from a fully validated run.
pub fn summarize_run(run: &LoadedRun) -> RunSummary {
    let capacity_by_case = trials_by_case(&run.capacity_trials, |trial| &trial.case_id);
    let search_by_case = trials_by_case(&run.search_trials, |trial| &trial.case_id);
    let mut cases = Vec::with_capacity(run.record.plan.cases.len());

    for planned in &run.record.plan.cases {
        let state = run
            .record
            .cases
            .iter()
            .find(|record| record.id() == planned.id())
            .expect("loaded run has one outcome per planned case")
            .state();
        let summary = match planned.kind() {
            BenchmarkKind::BackendCapacity => {
                let trials = capacity_by_case
                    .get(planned.id())
                    .map(Vec::as_slice)
                    .unwrap_or_default();
                CaseSummary {
                    id: planned.id().to_owned(),
                    comparison_key: planned.comparison_key().to_owned(),
                    label: planned.label().to_owned(),
                    benchmark: SummaryBenchmark::BackendCapacity,
                    state,
                    measured_trials: capacity_counts(trials),
                    metrics: capacity_metrics(trials),
                }
            }
            BenchmarkKind::Search => {
                let trials = search_by_case
                    .get(planned.id())
                    .map(Vec::as_slice)
                    .unwrap_or_default();
                CaseSummary {
                    id: planned.id().to_owned(),
                    comparison_key: planned.comparison_key().to_owned(),
                    label: planned.label().to_owned(),
                    benchmark: SummaryBenchmark::Search,
                    state,
                    measured_trials: search_counts(trials),
                    metrics: search_metrics(trials),
                }
            }
        };
        cases.push(summary);
    }

    add_paired_policy_drift(&mut cases, &search_by_case);

    RunSummary {
        protocol_version: run.record.protocol_version,
        plan_id: run.record.plan_id.clone(),
        run_id: run.record.run_id.clone(),
        cases,
    }
}

/// Compare distributions from two records after enforcing the protocol's compatibility rules.
pub fn compare_runs(
    left: &LoadedRun,
    right: &LoadedRun,
) -> Result<RunComparison, Vec<ComparisonIssue>> {
    let check = check_comparable(left, right)?;
    let left_summary = summarize_run(left);
    let right_summary = summarize_run(right);
    let left_cases = left_summary
        .cases
        .iter()
        .map(|case| (case.comparison_key.as_str(), case))
        .collect::<BTreeMap<_, _>>();
    let right_cases = right_summary
        .cases
        .iter()
        .map(|case| (case.comparison_key.as_str(), case))
        .collect::<BTreeMap<_, _>>();
    let mut cases = Vec::with_capacity(check.comparable_cases.len());

    for comparison_key in &check.comparable_cases {
        let left_case = left_cases[comparison_key.as_str()];
        let right_case = right_cases[comparison_key.as_str()];
        let mut metrics = BTreeMap::new();
        for (name, left_distribution) in &left_case.metrics {
            let Some(right_distribution) = right_case.metrics.get(name) else {
                continue;
            };
            metrics.insert(
                name.clone(),
                metric_delta(left_distribution, right_distribution),
            );
        }
        cases.push(CaseComparison {
            comparison_key: comparison_key.clone(),
            left_id: left_case.id.clone(),
            right_id: right_case.id.clone(),
            left_label: left_case.label.clone(),
            right_label: right_case.label.clone(),
            benchmark: left_case.benchmark,
            metrics,
        });
    }

    Ok(RunComparison {
        protocol_version: left.record.protocol_version,
        left_run_id: left.record.run_id.clone(),
        right_run_id: right.record.run_id.clone(),
        differing_axes: check.differing_axes,
        cases,
        unavailable_cases: check.unavailable_cases,
    })
}

/// Render the compact within-record case comparison written beside every run record.
pub fn render_run_comparison(summary: &RunSummary) -> String {
    let mut output = format!(
        "# Calibration run: {}\n\nPlan: `{}`. Distributions are `median [p25, p75]` over completed measured trials. Trial counts are `completed/failed/interrupted`. Deltas are descriptive differences from the first completed case of the same benchmark; no ranking or recommendation is applied.\n",
        summary.run_id, summary.plan_id
    );
    render_case_table(
        &mut output,
        summary,
        SummaryBenchmark::BackendCapacity,
        "Backend capacity",
        "positions_per_s",
        &["wall_ms", "device_avg_batch", "caller_peak"],
    );
    render_case_table(
        &mut output,
        summary,
        SummaryBenchmark::Search,
        "Fixed-work search",
        "productive_per_s",
        &["wall_ms", "device_avg_batch", "gate_wait_ms", "nn_evals"],
    );
    render_search_behavior_table(&mut output, summary);
    output
}

/// Render the compact human surface for two compatible records.
pub fn render_record_comparison(comparison: &RunComparison) -> String {
    let axes = if comparison.differing_axes.is_empty() {
        "none".to_owned()
    } else {
        comparison
            .differing_axes
            .iter()
            .map(|axis| format!("`{}`", axis_name(*axis)))
            .collect::<Vec<_>>()
            .join(", ")
    };
    let mut output = format!(
        "# Calibration records: {} → {}\n\nDeclared axes that actually differ: {axes}. Values are `left → right (median delta)` over completed measured trials; no ranking or recommendation is applied.\n",
        comparison.left_run_id, comparison.right_run_id
    );
    render_record_table(
        &mut output,
        comparison,
        SummaryBenchmark::BackendCapacity,
        "Backend capacity",
        &[
            "positions_per_s",
            "wall_ms",
            "device_avg_batch",
            "caller_peak",
        ],
    );
    render_record_table(
        &mut output,
        comparison,
        SummaryBenchmark::Search,
        "Fixed-work search",
        &[
            "productive_per_s",
            "wall_ms",
            "device_avg_batch",
            "gate_wait_ms",
            "nn_evals",
        ],
    );
    render_record_table(
        &mut output,
        comparison,
        SummaryBenchmark::Search,
        "Search behavior",
        &[
            "nn_evals",
            "terminals",
            "tt_stops",
            "collisions",
            "policy_l1_vs_baseline",
        ],
    );
    if !comparison.unavailable_cases.is_empty() {
        output.push_str("\nUnavailable in at least one record: ");
        output.push_str(&comparison.unavailable_cases.join(", "));
        output.push_str(".\n");
    }
    output
}

fn quantile(sorted: &[f64], probability: f64) -> f64 {
    let position = (sorted.len() - 1) as f64 * probability;
    let lower = position.floor() as usize;
    let upper = position.ceil() as usize;
    if lower == upper {
        sorted[lower]
    } else {
        let fraction = position - lower as f64;
        sorted[lower] + (sorted[upper] - sorted[lower]) * fraction
    }
}

fn trials_by_case<'a, T>(
    trials: &'a [T],
    case_id: impl Fn(&'a T) -> &'a str,
) -> BTreeMap<&'a str, Vec<&'a T>> {
    let mut by_case = BTreeMap::<_, Vec<_>>::new();
    for trial in trials {
        by_case.entry(case_id(trial)).or_default().push(trial);
    }
    by_case
}

fn capacity_counts(trials: &[&CapacityTrial]) -> TrialCounts {
    trial_counts(
        trials
            .iter()
            .filter_map(|trial| (trial.phase == TrialPhase::Measured).then_some(trial.status)),
    )
}

fn search_counts(trials: &[&SearchTrial]) -> TrialCounts {
    trial_counts(
        trials
            .iter()
            .filter_map(|trial| (trial.phase == TrialPhase::Measured).then_some(trial.status)),
    )
}

fn trial_counts(statuses: impl Iterator<Item = TrialStatus>) -> TrialCounts {
    let mut counts = TrialCounts::default();
    for status in statuses {
        match status {
            TrialStatus::Completed => counts.completed += 1,
            TrialStatus::Failed => counts.failed += 1,
            TrialStatus::Interrupted => counts.interrupted += 1,
        }
    }
    counts
}

fn capacity_metrics(trials: &[&CapacityTrial]) -> BTreeMap<String, Distribution> {
    let trials = trials
        .iter()
        .copied()
        .filter(|trial| {
            trial.phase == TrialPhase::Measured && trial.status == TrialStatus::Completed
        })
        .collect::<Vec<_>>();
    let mut metrics = BTreeMap::new();
    insert_metric(
        &mut metrics,
        "wall_ms",
        trials.iter().filter_map(|t| t.wall_ms),
    );
    insert_metric(
        &mut metrics,
        "positions_per_s",
        trials.iter().filter_map(|t| t.positions_per_s),
    );
    insert_metric(
        &mut metrics,
        "outer_calls",
        trials
            .iter()
            .filter_map(|t| t.outer_calls.map(|v| v as f64)),
    );
    insert_metric(
        &mut metrics,
        "outer_positions",
        trials
            .iter()
            .filter_map(|t| t.outer_positions.map(|v| v as f64)),
    );
    insert_metric(
        &mut metrics,
        "caller_time_ms",
        trials.iter().filter_map(|t| t.caller_time_ms),
    );
    insert_metric(
        &mut metrics,
        "caller_union_ms",
        trials.iter().filter_map(|t| t.caller_union_ms),
    );
    insert_metric(
        &mut metrics,
        "caller_peak",
        trials.iter().filter_map(|t| t.caller_peak.map(f64::from)),
    );
    insert_metric(
        &mut metrics,
        "device_calls",
        trials
            .iter()
            .filter_map(|t| t.device_calls.map(|v| v as f64)),
    );
    insert_metric(
        &mut metrics,
        "device_positions",
        trials
            .iter()
            .filter_map(|t| t.device_positions.map(|v| v as f64)),
    );
    insert_metric(
        &mut metrics,
        "device_avg_batch",
        trials.iter().filter_map(|t| t.device_avg_batch),
    );
    insert_metric(
        &mut metrics,
        "device_inference_ms",
        trials.iter().filter_map(|t| t.device_inference_ms),
    );
    metrics
}

fn search_metrics(trials: &[&SearchTrial]) -> BTreeMap<String, Distribution> {
    let trials = trials
        .iter()
        .copied()
        .filter(|trial| {
            trial.phase == TrialPhase::Measured && trial.status == TrialStatus::Completed
        })
        .collect::<Vec<_>>();
    let mut metrics = BTreeMap::new();
    macro_rules! float_metric {
        ($field:ident) => {
            insert_metric(
                &mut metrics,
                stringify!($field),
                trials
                    .iter()
                    .filter_map(|trial| trial.$field.map(f64::from)),
            );
        };
    }
    macro_rules! number_metric {
        ($field:ident) => {
            insert_metric(
                &mut metrics,
                stringify!($field),
                trials
                    .iter()
                    .filter_map(|trial| trial.$field.map(|value| value as f64)),
            );
        };
    }
    let float_metrics: [(&str, Vec<f64>); 14] = [
        ("wall_ms", trials.iter().filter_map(|t| t.wall_ms).collect()),
        (
            "productive_per_s",
            trials.iter().filter_map(|t| t.productive_per_s).collect(),
        ),
        (
            "phase_occupancy_ms",
            trials.iter().filter_map(|t| t.phase_occupancy_ms).collect(),
        ),
        (
            "phase_occupancy_per_wall",
            trials
                .iter()
                .filter_map(|t| t.phase_occupancy_per_wall)
                .collect(),
        ),
        (
            "lease_wait_ms",
            trials.iter().filter_map(|t| t.lease_wait_ms).collect(),
        ),
        (
            "gate_wait_ms",
            trials.iter().filter_map(|t| t.gate_wait_ms).collect(),
        ),
        (
            "gate_hold_ms",
            trials.iter().filter_map(|t| t.gate_hold_ms).collect(),
        ),
        (
            "inference_caller_ms",
            trials
                .iter()
                .filter_map(|t| t.inference_caller_ms)
                .collect(),
        ),
        (
            "completion_wait_ms",
            trials.iter().filter_map(|t| t.completion_wait_ms).collect(),
        ),
        (
            "backend_caller_ms",
            trials.iter().filter_map(|t| t.backend_caller_ms).collect(),
        ),
        (
            "backend_union_ms",
            trials.iter().filter_map(|t| t.backend_union_ms).collect(),
        ),
        (
            "backend_caller_per_union",
            trials
                .iter()
                .filter_map(|t| t.backend_caller_per_union)
                .collect(),
        ),
        (
            "device_avg_batch",
            trials.iter().filter_map(|t| t.device_avg_batch).collect(),
        ),
        (
            "device_inference_ms",
            trials
                .iter()
                .filter_map(|t| t.device_inference_ms)
                .collect(),
        ),
    ];
    for (name, values) in float_metrics {
        insert_metric(&mut metrics, name, values.into_iter());
    }
    number_metric!(nn_evals);
    number_metric!(terminals);
    number_metric!(tt_stops);
    number_metric!(collisions);
    number_metric!(batches);
    number_metric!(backend_calls);
    number_metric!(backend_positions);
    number_metric!(backend_peak_callers);
    number_metric!(device_calls);
    number_metric!(device_positions);
    number_metric!(ledger_reserved);
    number_metric!(ledger_committed);
    number_metric!(ledger_cancelled);
    float_metric!(policy_p1_0);
    float_metric!(policy_p1_1);
    float_metric!(policy_p1_2);
    float_metric!(policy_p1_3);
    float_metric!(policy_p1_4);
    float_metric!(policy_p2_0);
    float_metric!(policy_p2_1);
    float_metric!(policy_p2_2);
    float_metric!(policy_p2_3);
    float_metric!(policy_p2_4);
    metrics
}

fn add_paired_policy_drift(
    cases: &mut [CaseSummary],
    search_by_case: &BTreeMap<&str, Vec<&SearchTrial>>,
) {
    let Some(baseline_id) = cases
        .iter()
        .find(|case| {
            case.benchmark == SummaryBenchmark::Search && case.state == CaseState::Completed
        })
        .map(|case| case.id.clone())
    else {
        return;
    };
    let Some(baseline_trials) = search_by_case.get(baseline_id.as_str()) else {
        return;
    };
    let baseline_by_trial = baseline_trials
        .iter()
        .copied()
        .filter(|trial| {
            trial.phase == TrialPhase::Measured && trial.status == TrialStatus::Completed
        })
        .map(|trial| (trial.trial, trial))
        .collect::<BTreeMap<_, _>>();

    for case in cases
        .iter_mut()
        .filter(|case| case.benchmark == SummaryBenchmark::Search)
    {
        let values = search_by_case
            .get(case.id.as_str())
            .into_iter()
            .flat_map(|trials| trials.iter().copied())
            .filter(|trial| {
                trial.phase == TrialPhase::Measured && trial.status == TrialStatus::Completed
            })
            .filter_map(|trial| {
                baseline_by_trial
                    .get(&trial.trial)
                    .map(|baseline| root_policy_l1(baseline, trial))
            });
        insert_metric(&mut case.metrics, "policy_l1_vs_baseline", values);
    }
}

fn root_policy_l1(left: &SearchTrial, right: &SearchTrial) -> f64 {
    let left = [
        left.policy_p1_0,
        left.policy_p1_1,
        left.policy_p1_2,
        left.policy_p1_3,
        left.policy_p1_4,
        left.policy_p2_0,
        left.policy_p2_1,
        left.policy_p2_2,
        left.policy_p2_3,
        left.policy_p2_4,
    ];
    let right = [
        right.policy_p1_0,
        right.policy_p1_1,
        right.policy_p1_2,
        right.policy_p1_3,
        right.policy_p1_4,
        right.policy_p2_0,
        right.policy_p2_1,
        right.policy_p2_2,
        right.policy_p2_3,
        right.policy_p2_4,
    ];
    left.into_iter()
        .zip(right)
        .map(|(left, right)| f64::from((left.unwrap() - right.unwrap()).abs()))
        .sum()
}

fn insert_metric(
    metrics: &mut BTreeMap<String, Distribution>,
    name: &str,
    values: impl Iterator<Item = f64>,
) {
    if let Some(distribution) = Distribution::from_values(values.collect()) {
        metrics.insert(name.to_owned(), distribution);
    }
}

fn metric_delta(left: &Distribution, right: &Distribution) -> MetricDelta {
    let median_delta = right.median - left.median;
    MetricDelta {
        left: left.clone(),
        right: right.clone(),
        median_delta,
        median_delta_percent: (left.median != 0.0).then_some(median_delta / left.median * 100.0),
    }
}

fn render_case_table(
    output: &mut String,
    summary: &RunSummary,
    benchmark: SummaryBenchmark,
    title: &str,
    headline: &str,
    supporting: &[&str],
) {
    let cases = summary
        .cases
        .iter()
        .filter(|case| case.benchmark == benchmark)
        .collect::<Vec<_>>();
    if cases.is_empty() {
        return;
    }
    let baseline = cases.iter().find_map(|case| {
        case.metrics
            .get(headline)
            .map(|metric| (case.id.as_str(), metric))
    });
    output.push_str(&format!("\n## {title}\n\n"));
    output.push_str("| Case | State | Trials | ");
    output.push_str(headline);
    output.push_str(" | Delta | ");
    output.push_str(&supporting.join(" | "));
    output.push_str(" |\n|---|---:|---:|---:|---:|");
    for _ in supporting {
        output.push_str("---:|");
    }
    output.push('\n');
    for case in cases {
        let headline_value = case
            .metrics
            .get(headline)
            .map(format_distribution)
            .unwrap_or_else(|| "—".to_owned());
        let delta = match (baseline, case.metrics.get(headline)) {
            (Some((baseline_id, baseline)), Some(value)) if baseline_id != case.id => {
                format_delta(value.median - baseline.median, baseline.median)
            }
            (Some((baseline_id, _)), Some(_)) if baseline_id == case.id => "baseline".to_owned(),
            _ => "—".to_owned(),
        };
        output.push_str(&format!(
            "| {} | {:?} | {}/{}/{} | {} | {} | ",
            case.label,
            case.state,
            case.measured_trials.completed,
            case.measured_trials.failed,
            case.measured_trials.interrupted,
            headline_value,
            delta
        ));
        for name in supporting {
            let value = case
                .metrics
                .get(*name)
                .map(format_distribution)
                .unwrap_or_else(|| "—".to_owned());
            output.push_str(&value);
            output.push_str(" | ");
        }
        output.push('\n');
    }
}

fn render_search_behavior_table(output: &mut String, summary: &RunSummary) {
    let cases = summary
        .cases
        .iter()
        .filter(|case| case.benchmark == SummaryBenchmark::Search)
        .collect::<Vec<_>>();
    if cases.is_empty() {
        return;
    }

    output.push_str(
        "\n### Search behavior\n\n`policy_l1_vs_baseline` is the same-trial P1+P2 root-policy distance from the first completed search case. It shows behavior change, not playing quality.\n\n",
    );
    output.push_str(
        "| Case | nn_evals | terminals | tt_stops | collisions | policy_l1_vs_baseline |\n|---|---:|---:|---:|---:|---:|\n",
    );
    for case in cases {
        output.push_str(&format!("| {} | ", case.label));
        for name in [
            "nn_evals",
            "terminals",
            "tt_stops",
            "collisions",
            "policy_l1_vs_baseline",
        ] {
            let value = case
                .metrics
                .get(name)
                .map(format_distribution)
                .unwrap_or_else(|| "—".to_owned());
            output.push_str(&value);
            output.push_str(" | ");
        }
        output.push('\n');
    }
}

fn render_record_table(
    output: &mut String,
    comparison: &RunComparison,
    benchmark: SummaryBenchmark,
    title: &str,
    metrics: &[&str],
) {
    let cases = comparison
        .cases
        .iter()
        .filter(|case| case.benchmark == benchmark)
        .collect::<Vec<_>>();
    if cases.is_empty() {
        return;
    }
    output.push_str(&format!("\n## {title}\n\n| Case | "));
    output.push_str(&metrics.join(" | "));
    output.push_str(" |\n|---|");
    for _ in metrics {
        output.push_str("---:|");
    }
    output.push('\n');
    for case in cases {
        let label = if case.left_label == case.right_label {
            case.left_label.clone()
        } else {
            format!("{} → {}", case.left_label, case.right_label)
        };
        output.push_str(&format!("| {label} | "));
        for name in metrics {
            let value = case
                .metrics
                .get(*name)
                .map(format_metric_delta)
                .unwrap_or_else(|| "—".to_owned());
            output.push_str(&value);
            output.push_str(" | ");
        }
        output.push('\n');
    }
}

fn format_distribution(value: &Distribution) -> String {
    format!(
        "{} [{}, {}]",
        format_number(value.median),
        format_number(value.p25),
        format_number(value.p75)
    )
}

fn format_metric_delta(value: &MetricDelta) -> String {
    format!(
        "{} → {} ({})",
        format_number(value.left.median),
        format_number(value.right.median),
        format_delta(value.median_delta, value.left.median)
    )
}

fn format_delta(delta: f64, baseline: f64) -> String {
    if baseline == 0.0 {
        let sign = if delta > 0.0 { "+" } else { "" };
        format!("{sign}{}", format_number(delta))
    } else {
        format!("{:+.2}%", delta / baseline * 100.0)
    }
}

fn format_number(value: f64) -> String {
    let magnitude = value.abs();
    if magnitude >= 100_000.0 {
        format!("{:.1}K", value / 1_000.0)
    } else if magnitude >= 1_000.0 {
        format!("{:.2}K", value / 1_000.0)
    } else if magnitude >= 10.0 {
        format!("{value:.2}")
    } else {
        format!("{value:.3}")
    }
}

fn axis_name(axis: ComparisonAxis) -> &'static str {
    match axis {
        ComparisonAxis::Source => "source",
        ComparisonAxis::Build => "build",
        ComparisonAxis::Hardware => "hardware",
        ComparisonAxis::Runtime => "runtime",
        ComparisonAxis::Backend => "backend",
        ComparisonAxis::BatchSize => "batch_size",
        ComparisonAxis::Callers => "callers",
        ComparisonAxis::Workers => "workers",
        ComparisonAxis::TotalInFlight => "total_in_flight",
        ComparisonAxis::MuxMaxBatch => "mux_max_batch",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn distribution_uses_interpolated_quartiles_and_population_noise() {
        let distribution = Distribution::from_values(vec![4.0, 1.0, 3.0, 2.0]).unwrap();

        assert_eq!(distribution.count, 4);
        assert_eq!(distribution.min, 1.0);
        assert_eq!(distribution.p25, 1.75);
        assert_eq!(distribution.median, 2.5);
        assert_eq!(distribution.p75, 3.25);
        assert_eq!(distribution.max, 4.0);
        assert_eq!(distribution.mean, 2.5);
        assert!((distribution.standard_deviation - 1.118_033_988_749_895).abs() < 1e-12);
    }
}
