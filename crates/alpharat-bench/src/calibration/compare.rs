use std::collections::{BTreeMap, BTreeSet};

use super::folder::{case_plan_differences, LoadedRun};
use super::model::{CaseState, ComparisonAxis};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ComparisonCheck {
    pub differing_axes: BTreeSet<ComparisonAxis>,
    pub comparable_cases: Vec<String>,
    pub unavailable_cases: Vec<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ComparisonIssue {
    ProtocolVersion {
        left: u32,
        right: u32,
    },
    NonReproducibleSource {
        run_id: String,
    },
    AxisDeclarations {
        left: BTreeSet<ComparisonAxis>,
        right: BTreeSet<ComparisonAxis>,
    },
    ModelIdentity,
    MissingCase {
        run_id: String,
        case_id: String,
    },
    BenchmarkKind {
        case_id: String,
    },
    WorkloadIdentity {
        case_id: String,
    },
    UndeclaredAxes {
        case_id: String,
        axes: BTreeSet<ComparisonAxis>,
    },
    FixedPlanDifference {
        case_id: String,
        message: String,
    },
}

/// Check whether two fully validated records can support a controlled comparison.
///
/// This does not calculate performance deltas. It only establishes that the records describe the
/// same model, workload, metric meanings, and fixed work while allowing the axes both plans named.
pub fn check_comparable(
    left: &LoadedRun,
    right: &LoadedRun,
) -> Result<ComparisonCheck, Vec<ComparisonIssue>> {
    let mut issues = Vec::new();
    let mut differing_axes = BTreeSet::new();

    if left.record.protocol_version != right.record.protocol_version {
        issues.push(ComparisonIssue::ProtocolVersion {
            left: left.record.protocol_version,
            right: right.record.protocol_version,
        });
    }
    for run in [left, right] {
        if !run.record.context.source.state.is_reproducible() {
            issues.push(ComparisonIssue::NonReproducibleSource {
                run_id: run.record.run_id.clone(),
            });
        }
    }
    if left.record.plan.comparison_axes != right.record.plan.comparison_axes {
        issues.push(ComparisonIssue::AxisDeclarations {
            left: left.record.plan.comparison_axes.clone(),
            right: right.record.plan.comparison_axes.clone(),
        });
    }
    if !left
        .record
        .context
        .model
        .has_same_content_as(&right.record.context.model)
    {
        issues.push(ComparisonIssue::ModelIdentity);
    }

    context_difference(
        !left
            .record
            .context
            .source
            .has_same_content_as(&right.record.context.source),
        ComparisonAxis::Source,
        left,
        right,
        &mut differing_axes,
        &mut issues,
    );
    context_difference(
        left.record.context.build != right.record.context.build,
        ComparisonAxis::Build,
        left,
        right,
        &mut differing_axes,
        &mut issues,
    );
    context_difference(
        !has_same_hardware(left, right),
        ComparisonAxis::Hardware,
        left,
        right,
        &mut differing_axes,
        &mut issues,
    );
    context_difference(
        !left
            .record
            .context
            .runtime
            .has_same_versions_as(&right.record.context.runtime),
        ComparisonAxis::Runtime,
        left,
        right,
        &mut differing_axes,
        &mut issues,
    );

    let left_plans: BTreeMap<_, _> = left
        .record
        .plan
        .cases
        .iter()
        .map(|case| (case.id(), case))
        .collect();
    let right_plans: BTreeMap<_, _> = right
        .record
        .plan
        .cases
        .iter()
        .map(|case| (case.id(), case))
        .collect();

    for case_id in left_plans.keys() {
        if !right_plans.contains_key(case_id) {
            issues.push(ComparisonIssue::MissingCase {
                run_id: right.record.run_id.clone(),
                case_id: (*case_id).to_owned(),
            });
        }
    }
    for case_id in right_plans.keys() {
        if !left_plans.contains_key(case_id) {
            issues.push(ComparisonIssue::MissingCase {
                run_id: left.record.run_id.clone(),
                case_id: (*case_id).to_owned(),
            });
        }
    }

    for (case_id, left_plan) in &left_plans {
        let Some(right_plan) = right_plans.get(case_id) else {
            continue;
        };
        if left_plan.kind() != right_plan.kind() {
            issues.push(ComparisonIssue::BenchmarkKind {
                case_id: (*case_id).to_owned(),
            });
            continue;
        }
        if !left_plan
            .workload()
            .has_same_content_as(right_plan.workload())
        {
            issues.push(ComparisonIssue::WorkloadIdentity {
                case_id: (*case_id).to_owned(),
            });
        }
        match case_plan_differences(left_plan, right_plan) {
            Ok(case_differences) => {
                let allowed = left
                    .record
                    .plan
                    .comparison_axes
                    .intersection(&right.record.plan.comparison_axes)
                    .copied()
                    .collect::<BTreeSet<_>>();
                let undeclared = case_differences
                    .difference(&allowed)
                    .copied()
                    .collect::<BTreeSet<_>>();
                if !undeclared.is_empty() {
                    issues.push(ComparisonIssue::UndeclaredAxes {
                        case_id: (*case_id).to_owned(),
                        axes: undeclared,
                    });
                }
                differing_axes.extend(case_differences);
            }
            Err(error) => issues.push(ComparisonIssue::FixedPlanDifference {
                case_id: (*case_id).to_owned(),
                message: error.to_string(),
            }),
        }
    }

    if !issues.is_empty() {
        return Err(issues);
    }

    let left_states: BTreeMap<_, _> = left
        .record
        .cases
        .iter()
        .map(|case| (case.id(), case.state()))
        .collect();
    let right_states: BTreeMap<_, _> = right
        .record
        .cases
        .iter()
        .map(|case| (case.id(), case.state()))
        .collect();
    let mut comparable_cases = Vec::new();
    let mut unavailable_cases = Vec::new();
    for case_id in left_plans.keys() {
        if left_states[case_id] == CaseState::Completed
            && right_states[case_id] == CaseState::Completed
        {
            comparable_cases.push((*case_id).to_owned());
        } else {
            unavailable_cases.push((*case_id).to_owned());
        }
    }

    Ok(ComparisonCheck {
        differing_axes,
        comparable_cases,
        unavailable_cases,
    })
}

fn has_same_hardware(left: &LoadedRun, right: &LoadedRun) -> bool {
    let left = &left.record.context.hardware;
    let right = &right.record.context.hardware;
    let mut left_accelerators = left.accelerators.iter().collect::<Vec<_>>();
    let mut right_accelerators = right.accelerators.iter().collect::<Vec<_>>();
    left_accelerators.sort_unstable();
    right_accelerators.sort_unstable();

    left.operating_system == right.operating_system
        && left.architecture == right.architecture
        && left.cpu == right.cpu
        && left.logical_cores == right.logical_cores
        && left.memory_bytes == right.memory_bytes
        && left_accelerators == right_accelerators
}

fn context_difference(
    differs: bool,
    axis: ComparisonAxis,
    left: &LoadedRun,
    right: &LoadedRun,
    differing_axes: &mut BTreeSet<ComparisonAxis>,
    issues: &mut Vec<ComparisonIssue>,
) {
    if !differs {
        return;
    }
    differing_axes.insert(axis);
    if !left.record.plan.comparison_axes.contains(&axis)
        || !right.record.plan.comparison_axes.contains(&axis)
    {
        issues.push(ComparisonIssue::UndeclaredAxes {
            case_id: "<run-context>".to_owned(),
            axes: BTreeSet::from([axis]),
        });
    }
}
