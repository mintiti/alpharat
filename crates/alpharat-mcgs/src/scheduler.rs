#![cfg_attr(loom, allow(dead_code))]

//! Search-call coordination independent of graph locking.
//!
//! Workers first reserve productive capacity here, then linearize admission
//! under a graph write epoch with [`ProvisionalLease::admit`]. Closing
//! admission rejects any still-provisional lease, while work admitted before
//! the close remains free to settle and charge the productive units it
//! actually committed.

use crate::access::WriteEpoch;
use crate::BackendError;

#[cfg(loom)]
use loom::sync::{Condvar, Mutex, MutexGuard};
#[cfg(not(loom))]
use std::sync::{Condvar, Mutex, MutexGuard};

/// Why this search call stopped accepting new graph work.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum AdmissionCloseReason {
    BudgetExhausted,
    StopRequested,
    Deadline,
    BackendFailure,
    WorkerPanicked,
    NoProgress,
}

/// Stable counters available after workers have joined.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct CoordinatorSnapshot {
    pub(crate) initial: u32,
    pub(crate) productive: u32,
    pub(crate) remaining: u32,
    pub(crate) provisional: u32,
    pub(crate) admitted: u32,
    pub(crate) outstanding: u32,
    pub(crate) close_reason: Option<AdmissionCloseReason>,
    pub(crate) first_failure_recorded: bool,
    pub(crate) leases_issued: u64,
    pub(crate) leases_admitted: u64,
    pub(crate) leases_rejected: u64,
    pub(crate) leases_completed: u64,
    pub(crate) leases_cancelled: u64,
    pub(crate) leases_abandoned: u64,
    pub(crate) admitted_leases_abandoned: u64,
    pub(crate) abandoned_units: u64,
    pub(crate) admitted_units_abandoned: u64,
    pub(crate) zero_progress_completions: u64,
    pub(crate) zero_progress_streak: u32,
}

impl CoordinatorSnapshot {
    pub(crate) fn charged(self) -> u32 {
        self.productive
    }
}

#[derive(Debug)]
struct CoordinatorState {
    initial: u32,
    productive: u32,
    remaining: u32,
    provisional: u32,
    admitted: u32,
    close_reason: Option<AdmissionCloseReason>,
    first_failure: Option<BackendError>,
    leases_issued: u64,
    leases_admitted: u64,
    leases_rejected: u64,
    leases_completed: u64,
    leases_cancelled: u64,
    leases_abandoned: u64,
    admitted_leases_abandoned: u64,
    abandoned_units: u64,
    admitted_units_abandoned: u64,
    zero_progress_completions: u64,
    zero_progress_streak: u32,
    drop_invariant_failed: bool,
}

impl CoordinatorState {
    fn outstanding(&self) -> u32 {
        self.provisional
            .checked_add(self.admitted)
            .expect("search coordinator outstanding capacity overflow")
    }

    fn settled_leases(&self) -> u64 {
        self.leases_completed
            .checked_add(self.leases_cancelled)
            .and_then(|total| total.checked_add(self.leases_abandoned))
            .and_then(|total| total.checked_add(self.leases_rejected))
            .expect("search coordinator settled-lease counter overflow")
    }

    fn assert_valid(&self) {
        assert!(
            !self.drop_invariant_failed,
            "productive lease Drop observed corrupt coordinator accounting"
        );
        assert_eq!(
            self.productive.checked_add(self.remaining),
            Some(self.initial),
            "search coordinator lost productive capacity"
        );
        assert!(
            self.outstanding() <= self.remaining,
            "search coordinator has more leased capacity than remaining work"
        );
        assert!(
            self.leases_admitted
                <= self
                    .leases_issued
                    .checked_sub(self.leases_rejected)
                    .expect("search coordinator rejected more leases than it issued"),
            "search coordinator admitted more leases than it issued"
        );
        assert!(
            self.settled_leases() <= self.leases_issued,
            "search coordinator settled more leases than it issued"
        );
        assert!(
            self.admitted_leases_abandoned <= self.leases_abandoned,
            "search coordinator lost an admitted abandoned lease"
        );
        assert!(
            self.admitted_units_abandoned <= self.abandoned_units,
            "search coordinator lost admitted abandoned capacity"
        );
        if self.close_reason == Some(AdmissionCloseReason::BudgetExhausted) {
            assert_eq!(
                self.remaining, 0,
                "budget exhaustion closed admission before all work was charged"
            );
        }
        if self.close_reason == Some(AdmissionCloseReason::BackendFailure) {
            assert!(
                self.first_failure.is_some(),
                "backend failure closed admission without retaining the error"
            );
        }
    }

    fn assert_quiescent(&self) {
        self.assert_valid();
        assert_eq!(
            self.outstanding(),
            0,
            "search workers joined with productive capacity still leased"
        );
        assert_eq!(
            self.settled_leases(),
            self.leases_issued,
            "search workers joined with an unsettled productive lease"
        );
        assert_eq!(
            self.admitted_leases_abandoned, 0,
            "an admitted lease was abandoned instead of explicitly cleaned up"
        );
    }

    fn close_if_open(&mut self, reason: AdmissionCloseReason) -> bool {
        if self.close_reason.is_some() {
            return false;
        }
        self.close_reason = Some(reason);
        true
    }

    fn snapshot(&self) -> CoordinatorSnapshot {
        CoordinatorSnapshot {
            initial: self.initial,
            productive: self.productive,
            remaining: self.remaining,
            provisional: self.provisional,
            admitted: self.admitted,
            outstanding: self.outstanding(),
            close_reason: self.close_reason,
            first_failure_recorded: self.first_failure.is_some(),
            leases_issued: self.leases_issued,
            leases_admitted: self.leases_admitted,
            leases_rejected: self.leases_rejected,
            leases_completed: self.leases_completed,
            leases_cancelled: self.leases_cancelled,
            leases_abandoned: self.leases_abandoned,
            admitted_leases_abandoned: self.admitted_leases_abandoned,
            abandoned_units: self.abandoned_units,
            admitted_units_abandoned: self.admitted_units_abandoned,
            zero_progress_completions: self.zero_progress_completions,
            zero_progress_streak: self.zero_progress_streak,
        }
    }
}

/// Search-scoped admission and productive-work budget.
///
/// The graph gate is deliberately absent. Capacity can be leased while
/// another worker owns a graph epoch, but [`ProvisionalLease::admit`] must be
/// called from inside the later write epoch before gather mutates the graph.
pub(crate) struct SearchCoordinator {
    state: Mutex<CoordinatorState>,
    capacity_changed: Condvar,
    max_zero_progress_streak: u32,
}

impl SearchCoordinator {
    pub(crate) fn new(initial: u32, max_zero_progress_streak: u32) -> Self {
        assert!(
            max_zero_progress_streak > 0,
            "zero-progress bound must be positive"
        );
        Self {
            state: Mutex::new(CoordinatorState {
                initial,
                productive: 0,
                remaining: initial,
                provisional: 0,
                admitted: 0,
                close_reason: (initial == 0).then_some(AdmissionCloseReason::BudgetExhausted),
                first_failure: None,
                leases_issued: 0,
                leases_admitted: 0,
                leases_rejected: 0,
                leases_completed: 0,
                leases_cancelled: 0,
                leases_abandoned: 0,
                admitted_leases_abandoned: 0,
                abandoned_units: 0,
                admitted_units_abandoned: 0,
                zero_progress_completions: 0,
                zero_progress_streak: 0,
                drop_invariant_failed: false,
            }),
            capacity_changed: Condvar::new(),
            max_zero_progress_streak,
        }
    }

    /// Wait for productive capacity or for admission to close.
    ///
    /// A returned lease is provisional. It does not authorize graph mutation
    /// until `admit` succeeds under the graph write epoch.
    pub(crate) fn lease(&self, max: u32) -> Option<ProvisionalLease<'_>> {
        assert!(max > 0, "productive lease size must be positive");
        let mut state = self.lock_state();

        loop {
            state.assert_valid();
            if state.close_reason.is_some() {
                return None;
            }

            let available = state.remaining - state.outstanding();
            if available > 0 {
                let units = available.min(max);
                let next_provisional = state
                    .provisional
                    .checked_add(units)
                    .expect("provisional-capacity counter overflow");
                let next_issued = state
                    .leases_issued
                    .checked_add(1)
                    .expect("issued-lease counter overflow");
                state.provisional = next_provisional;
                state.leases_issued = next_issued;
                state.assert_valid();
                return Some(ProvisionalLease {
                    core: Some(LeaseCore {
                        coordinator: self,
                        units,
                    }),
                });
            }

            state = self.wait_state(state);
        }
    }

    /// Close graph-work admission for an external stop, deadline, or panic.
    /// The first close reason remains the historical cause; already-admitted
    /// leases can still complete.
    ///
    /// Backend errors must use [`Self::record_first_failure`] so the close
    /// reason can never exist without its retained error payload.
    pub(crate) fn close(&self, reason: AdmissionCloseReason) -> bool {
        assert!(
            matches!(
                reason,
                AdmissionCloseReason::StopRequested
                    | AdmissionCloseReason::Deadline
                    | AdmissionCloseReason::WorkerPanicked
            ),
            "derived close reasons must use their coordinator transition"
        );
        let mut state = self.lock_state();
        let closed = state.close_if_open(reason);
        state.assert_valid();
        drop(state);
        if closed {
            self.capacity_changed.notify_all();
        }
        closed
    }

    /// Retain the first backend error independently of which event first
    /// closed admission.
    ///
    /// This lets a stop request win the admission race while a later backend
    /// error from already-admitted work is still returned after every worker
    /// joins. Later backend errors are discarded.
    pub(crate) fn record_first_failure(&self, error: BackendError) -> bool {
        let mut state = self.lock_state();
        let first = state.first_failure.is_none();
        if first {
            state.first_failure = Some(error);
        }
        let closed = state.close_if_open(AdmissionCloseReason::BackendFailure);
        state.assert_valid();
        drop(state);
        if closed {
            self.capacity_changed.notify_all();
        }
        first
    }

    pub(crate) fn snapshot(&self) -> CoordinatorSnapshot {
        let state = self.lock_state();
        state.assert_valid();
        state.snapshot()
    }

    pub(crate) fn assert_quiescent(&self) {
        self.lock_state().assert_quiescent();
    }

    /// Consume the coordinator after all scoped workers have joined.
    ///
    /// This is the join-before-result seam: no lease can still borrow the
    /// coordinator once it is consumed.
    pub(crate) fn finish(self) -> CoordinatorOutcome {
        let state = self
            .state
            .into_inner()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        state.assert_quiescent();
        assert!(
            state.close_reason.is_some(),
            "search coordinator finished before reaching a terminal close reason"
        );
        CoordinatorOutcome {
            snapshot: state.snapshot(),
            first_failure: state.first_failure,
        }
    }

    fn lock_state(&self) -> MutexGuard<'_, CoordinatorState> {
        self.state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    fn wait_state<'a>(
        &self,
        state: MutexGuard<'a, CoordinatorState>,
    ) -> MutexGuard<'a, CoordinatorState> {
        self.capacity_changed
            .wait(state)
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    fn settle_lease(
        &self,
        stage: LeaseStage,
        units: u32,
        produced: u32,
        settlement: LeaseSettlement,
    ) -> BudgetCompletion {
        let mut state = self.lock_state();
        state.assert_valid();
        assert!(
            produced <= units,
            "batch produced more work than its productive lease"
        );
        let stage_units = match stage {
            LeaseStage::Provisional => state.provisional,
            LeaseStage::Admitted => state.admitted,
        };
        assert!(
            units <= stage_units,
            "settled lease exceeds coordinator outstanding capacity"
        );
        assert!(
            produced <= state.remaining,
            "batch charged more work than the coordinator has remaining"
        );
        if settlement == LeaseSettlement::Completed {
            assert_eq!(
                stage,
                LeaseStage::Admitted,
                "provisional work cannot complete graph mutations"
            );
        }

        let next_productive = state
            .productive
            .checked_add(produced)
            .expect("productive-work counter overflow");
        let next_zero_progress = if settlement == LeaseSettlement::Completed && produced == 0 {
            Some(
                state
                    .zero_progress_completions
                    .checked_add(1)
                    .expect("zero-progress completion counter overflow"),
            )
        } else {
            None
        };
        let next_completed = match settlement {
            LeaseSettlement::Completed => Some(
                state
                    .leases_completed
                    .checked_add(1)
                    .expect("completed-lease counter overflow"),
            ),
            LeaseSettlement::Cancelled => None,
        };
        let next_cancelled = match settlement {
            LeaseSettlement::Completed => None,
            LeaseSettlement::Cancelled => Some(
                state
                    .leases_cancelled
                    .checked_add(1)
                    .expect("cancelled-lease counter overflow"),
            ),
        };

        match stage {
            LeaseStage::Provisional => state.provisional -= units,
            LeaseStage::Admitted => state.admitted -= units,
        }
        state.productive = next_productive;
        state.remaining -= produced;
        let mut wait_for_productive = false;

        match settlement {
            LeaseSettlement::Completed => {
                state.leases_completed = next_completed.expect("completed count was precomputed");
                if produced == 0 {
                    state.zero_progress_completions =
                        next_zero_progress.expect("zero-progress count was precomputed");
                    if state.outstanding() == 0 {
                        state.zero_progress_streak = state
                            .zero_progress_streak
                            .checked_add(1)
                            .expect("zero-progress streak overflow");
                        if state.zero_progress_streak >= self.max_zero_progress_streak {
                            state.close_if_open(AdmissionCloseReason::NoProgress);
                        }
                    } else if state.close_reason.is_none() {
                        // Another admitted/provisional batch may still make
                        // progress. Do not let collision-only retries race a
                        // slow inference to the global no-progress bound.
                        wait_for_productive = true;
                    }
                } else {
                    state.zero_progress_streak = 0;
                }
            }
            LeaseSettlement::Cancelled => {
                state.leases_cancelled = next_cancelled.expect("cancelled count was precomputed");
                if produced > 0 {
                    state.zero_progress_streak = 0;
                }
            }
        }

        if state.remaining == 0 {
            state.close_if_open(AdmissionCloseReason::BudgetExhausted);
        }
        state.assert_valid();
        drop(state);
        self.capacity_changed.notify_all();

        if wait_for_productive {
            let mut state = self.lock_state();
            while state.productive == next_productive
                && state.close_reason.is_none()
                && state.outstanding() > 0
            {
                state = self.wait_state(state);
            }
            state.assert_valid();
        }

        BudgetCompletion {
            reserved: units,
            charged: produced,
            returned: units - produced,
        }
    }

    /// Unwind-only capacity return. Graph cleanup remains explicit elsewhere.
    ///
    /// This path deliberately never asserts or panics: if a prior panic
    /// poisoned the mutex, it recovers the guard and records any accounting
    /// corruption for `finish` to surface after worker cleanup.
    fn abandon_lease(&self, stage: LeaseStage, units: u32) {
        let mut state = self.lock_state();
        let stage_units = match stage {
            LeaseStage::Provisional => &mut state.provisional,
            LeaseStage::Admitted => &mut state.admitted,
        };
        if units <= *stage_units {
            *stage_units -= units;
        } else {
            *stage_units = 0;
            state.drop_invariant_failed = true;
        }

        if let Some(next) = state.leases_abandoned.checked_add(1) {
            state.leases_abandoned = next;
        } else {
            state.drop_invariant_failed = true;
        }
        if let Some(next) = state.abandoned_units.checked_add(u64::from(units)) {
            state.abandoned_units = next;
        } else {
            state.drop_invariant_failed = true;
        }

        if stage == LeaseStage::Admitted {
            if let Some(next) = state.admitted_leases_abandoned.checked_add(1) {
                state.admitted_leases_abandoned = next;
            } else {
                state.drop_invariant_failed = true;
            }
            if let Some(next) = state.admitted_units_abandoned.checked_add(u64::from(units)) {
                state.admitted_units_abandoned = next;
            } else {
                state.drop_invariant_failed = true;
            }
        }

        drop(state);
        self.capacity_changed.notify_all();
    }
}

pub(crate) struct CoordinatorOutcome {
    pub(crate) snapshot: CoordinatorSnapshot,
    pub(crate) first_failure: Option<BackendError>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum LeaseStage {
    Provisional,
    Admitted,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum LeaseSettlement {
    Completed,
    Cancelled,
}

struct LeaseCore<'coordinator> {
    coordinator: &'coordinator SearchCoordinator,
    units: u32,
}

/// Capacity reserved by a prospective batch but not yet authorized to mutate
/// the graph.
#[must_use = "a provisional lease must be admitted, cancelled, or returned"]
pub(crate) struct ProvisionalLease<'coordinator> {
    core: Option<LeaseCore<'coordinator>>,
}

impl<'coordinator> ProvisionalLease<'coordinator> {
    pub(crate) fn units(&self) -> u32 {
        self.core
            .as_ref()
            .expect("provisional lease already settled")
            .units
    }

    /// Linearize graph-work admission under the caller's graph write epoch.
    ///
    /// If admission closed while this lease waited for the graph gate, the
    /// capacity is returned here and the caller must not gather.
    #[allow(dead_code)]
    pub(crate) fn admit(
        self,
        _epoch: &mut WriteEpoch<'_, '_, '_>,
    ) -> Result<AdmittedLease<'coordinator>, AdmissionCloseReason> {
        self.admit_inner()
    }

    #[cfg(test)]
    fn admit_for_test(self) -> Result<AdmittedLease<'coordinator>, AdmissionCloseReason> {
        self.admit_inner()
    }

    fn admit_inner(mut self) -> Result<AdmittedLease<'coordinator>, AdmissionCloseReason> {
        let coordinator = self
            .core
            .as_ref()
            .expect("provisional lease already settled")
            .coordinator;
        let units = self.units();
        let mut state = coordinator.lock_state();
        state.assert_valid();
        assert!(
            units <= state.provisional,
            "admitted lease exceeds provisional coordinator capacity"
        );
        if let Some(reason) = state.close_reason {
            let next_rejected = state
                .leases_rejected
                .checked_add(1)
                .expect("rejected-lease counter overflow");
            let _core = self.core.take().expect("provisional lease already settled");
            state.provisional -= units;
            state.leases_rejected = next_rejected;
            state.assert_valid();
            drop(state);
            coordinator.capacity_changed.notify_all();
            return Err(reason);
        }

        let next_admitted_units = state
            .admitted
            .checked_add(units)
            .expect("admitted-capacity counter overflow");
        let next_admitted_leases = state
            .leases_admitted
            .checked_add(1)
            .expect("admitted-lease counter overflow");
        state.provisional -= units;
        state.admitted = next_admitted_units;
        state.leases_admitted = next_admitted_leases;
        state.assert_valid();
        drop(state);
        Ok(AdmittedLease {
            core: Some(self.core.take().expect("provisional lease already settled")),
        })
    }

    /// Explicitly return a lease that never reached graph admission.
    pub(crate) fn cancel(mut self) -> BudgetCompletion {
        let completion = {
            let core = self
                .core
                .as_ref()
                .expect("provisional lease already settled");
            core.coordinator.settle_lease(
                LeaseStage::Provisional,
                core.units,
                0,
                LeaseSettlement::Cancelled,
            )
        };
        let _core = self.core.take().expect("provisional lease already settled");
        completion
    }
}

impl Drop for ProvisionalLease<'_> {
    fn drop(&mut self) {
        if let Some(core) = self.core.take() {
            core.coordinator
                .abandon_lease(LeaseStage::Provisional, core.units);
        }
    }
}

/// Capacity owned by a batch authorized to mutate the graph.
#[must_use = "an admitted lease must be completed or explicitly cancelled"]
pub(crate) struct AdmittedLease<'coordinator> {
    core: Option<LeaseCore<'coordinator>>,
}

impl AdmittedLease<'_> {
    pub(crate) fn units(&self) -> u32 {
        self.core
            .as_ref()
            .expect("admitted lease already settled")
            .units
    }

    /// Charge exactly the productive units committed by a successful batch.
    ///
    /// A zero-product completion may wait for another outstanding batch to
    /// advance productive work or close admission. Callers must therefore
    /// release the graph epoch before calling `complete`, even when they
    /// expect the batch to have made progress.
    pub(crate) fn complete(mut self, produced: u32) -> BudgetCompletion {
        assert!(
            produced <= self.units(),
            "batch produced more work than its productive lease"
        );
        let completion = {
            let core = self.core.as_ref().expect("admitted lease already settled");
            core.coordinator.settle_lease(
                LeaseStage::Admitted,
                core.units,
                produced,
                LeaseSettlement::Completed,
            )
        };
        let _core = self.core.take().expect("admitted lease already settled");
        completion
    }

    /// Cancel all remaining work after explicit graph cleanup.
    pub(crate) fn cancel(self) -> BudgetCompletion {
        self.cancel_with_committed(0)
    }

    /// Charge inline terminal/transposition commits, then cancel the batch's
    /// remaining reservations after an evaluation failure.
    pub(crate) fn cancel_with_committed(mut self, produced: u32) -> BudgetCompletion {
        assert!(
            produced <= self.units(),
            "batch produced more work than its productive lease"
        );
        let completion = {
            let core = self.core.as_ref().expect("admitted lease already settled");
            core.coordinator.settle_lease(
                LeaseStage::Admitted,
                core.units,
                produced,
                LeaseSettlement::Cancelled,
            )
        };
        let _core = self.core.take().expect("admitted lease already settled");
        completion
    }
}

impl Drop for AdmittedLease<'_> {
    fn drop(&mut self) {
        if let Some(core) = self.core.take() {
            core.coordinator
                .abandon_lease(LeaseStage::Admitted, core.units);
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct BudgetCompletion {
    pub(crate) reserved: u32,
    pub(crate) charged: u32,
    pub(crate) returned: u32,
}

#[cfg(all(test, not(loom)))]
mod tests {
    use super::{AdmissionCloseReason, BudgetCompletion, SearchCoordinator};
    use crate::BackendError;
    use std::panic::{catch_unwind, AssertUnwindSafe};
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::sync::{mpsc, Arc};
    use std::thread;
    use std::time::{Duration, Instant};

    #[test]
    fn partial_zero_and_drop_conserve_productive_capacity() {
        let coordinator = SearchCoordinator::new(7, 3);

        let first = coordinator.lease(4).unwrap().admit_for_test().unwrap();
        assert_eq!(first.units(), 4);
        assert_eq!(
            first.complete(2),
            BudgetCompletion {
                reserved: 4,
                charged: 2,
                returned: 2,
            }
        );

        let zero = coordinator.lease(5).unwrap().admit_for_test().unwrap();
        assert_eq!(
            zero.complete(0),
            BudgetCompletion {
                reserved: 5,
                charged: 0,
                returned: 5,
            }
        );

        let abandoned = coordinator.lease(3).unwrap();
        drop(abandoned);

        let middle = coordinator.snapshot();
        assert_eq!(middle.productive, 2);
        assert_eq!(middle.remaining, 5);
        assert_eq!(middle.outstanding, 0);
        assert_eq!(middle.zero_progress_completions, 1);
        assert_eq!(middle.zero_progress_streak, 1);
        assert_eq!(middle.leases_abandoned, 1);
        assert_eq!(middle.abandoned_units, 3);

        let final_lease = coordinator.lease(5).unwrap().admit_for_test().unwrap();
        assert_eq!(final_lease.complete(5).charged, 5);

        let outcome = coordinator.finish();
        assert_eq!(outcome.snapshot.initial, 7);
        assert_eq!(outcome.snapshot.productive, 7);
        assert_eq!(outcome.snapshot.remaining, 0);
        assert_eq!(outcome.snapshot.outstanding, 0);
        assert_eq!(outcome.snapshot.leases_completed, 3);
        assert!(outcome.first_failure.is_none());
    }

    #[test]
    fn close_rejects_a_provisional_lease_before_graph_admission() {
        let coordinator = SearchCoordinator::new(5, 2);
        let lease = coordinator.lease(5).unwrap();
        assert_eq!(lease.units(), 5);
        assert_eq!(coordinator.snapshot().provisional, 5);

        assert!(coordinator.close(AdmissionCloseReason::StopRequested));
        match lease.admit_for_test() {
            Err(reason) => assert_eq!(reason, AdmissionCloseReason::StopRequested),
            Ok(_) => panic!("closed admission accepted a provisional lease"),
        }
        assert!(coordinator.lease(1).is_none());

        let snapshot = coordinator.snapshot();
        assert_eq!(snapshot.remaining, 5);
        assert_eq!(snapshot.outstanding, 0);
        assert_eq!(snapshot.leases_admitted, 0);
        assert_eq!(snapshot.leases_rejected, 1);
        assert_eq!(
            snapshot.close_reason,
            Some(AdmissionCloseReason::StopRequested)
        );
        coordinator.assert_quiescent();
    }

    #[test]
    fn admitted_success_can_commit_after_stop_closes_admission() {
        let coordinator = SearchCoordinator::new(5, 2);
        let lease = coordinator.lease(5).unwrap().admit_for_test().unwrap();
        assert_eq!(coordinator.snapshot().admitted, 5);

        assert!(coordinator.close(AdmissionCloseReason::StopRequested));
        assert_eq!(lease.complete(3).charged, 3);

        let outcome = coordinator.finish();
        assert_eq!(outcome.snapshot.remaining, 2);
        assert_eq!(outcome.snapshot.charged(), 3);
        assert_eq!(
            outcome.snapshot.close_reason,
            Some(AdmissionCloseReason::StopRequested)
        );
    }

    #[test]
    fn stop_then_admitted_failure_retains_the_first_error() {
        let coordinator = SearchCoordinator::new(2, 2);
        let lease = coordinator.lease(2).unwrap().admit_for_test().unwrap();

        assert!(coordinator.close(AdmissionCloseReason::Deadline));
        assert!(coordinator.record_first_failure(BackendError::msg("late admitted failure")));
        assert!(!coordinator.record_first_failure(BackendError::msg("later failure")));
        assert_eq!(lease.cancel_with_committed(1).charged, 1);

        let outcome = coordinator.finish();
        assert!(outcome.snapshot.first_failure_recorded);
        assert_eq!(
            outcome.snapshot.close_reason,
            Some(AdmissionCloseReason::Deadline)
        );
        assert_eq!(
            outcome.first_failure.unwrap().to_string(),
            "late admitted failure"
        );
    }

    #[test]
    fn productive_completion_resets_the_no_progress_streak() {
        let coordinator = SearchCoordinator::new(4, 2);

        assert_eq!(
            coordinator
                .lease(1)
                .unwrap()
                .admit_for_test()
                .unwrap()
                .complete(0)
                .charged,
            0
        );
        assert_eq!(coordinator.snapshot().zero_progress_streak, 1);
        assert_eq!(
            coordinator
                .lease(1)
                .unwrap()
                .admit_for_test()
                .unwrap()
                .complete(1)
                .charged,
            1
        );
        assert_eq!(coordinator.snapshot().zero_progress_streak, 0);

        for expected_streak in [1, 2] {
            coordinator
                .lease(1)
                .unwrap()
                .admit_for_test()
                .unwrap()
                .complete(0);
            assert_eq!(coordinator.snapshot().zero_progress_streak, expected_streak);
        }

        let outcome = coordinator.finish();
        assert_eq!(outcome.snapshot.productive, 1);
        assert_eq!(outcome.snapshot.remaining, 3);
        assert_eq!(outcome.snapshot.zero_progress_completions, 3);
        assert_eq!(
            outcome.snapshot.close_reason,
            Some(AdmissionCloseReason::NoProgress)
        );
    }

    #[test]
    fn returned_capacity_wakes_a_waiting_worker() {
        let coordinator = SearchCoordinator::new(4, 4);
        let first = coordinator.lease(4).unwrap().admit_for_test().unwrap();
        let (started_tx, started_rx) = mpsc::channel();
        let (done_tx, done_rx) = mpsc::channel();

        thread::scope(|scope| {
            scope.spawn(|| {
                started_tx.send(()).unwrap();
                let second = coordinator.lease(4).unwrap().admit_for_test().unwrap();
                assert_eq!(second.units(), 2);
                done_tx.send(second.complete(2)).unwrap();
            });

            started_rx.recv().unwrap();
            assert!(done_rx.recv_timeout(Duration::from_millis(20)).is_err());
            assert_eq!(first.complete(2).charged, 2);
            assert_eq!(
                done_rx.recv_timeout(Duration::from_secs(1)).unwrap(),
                BudgetCompletion {
                    reserved: 2,
                    charged: 2,
                    returned: 0,
                }
            );
        });

        let outcome = coordinator.finish();
        assert_eq!(outcome.snapshot.remaining, 0);
        assert_eq!(
            outcome.snapshot.close_reason,
            Some(AdmissionCloseReason::BudgetExhausted)
        );
    }

    #[test]
    fn zero_progress_waits_for_an_outstanding_productive_batch() {
        let coordinator = SearchCoordinator::new(2, 2);
        let productive = coordinator.lease(1).unwrap().admit_for_test().unwrap();
        let collision = coordinator.lease(1).unwrap().admit_for_test().unwrap();
        let (started_tx, started_rx) = mpsc::channel();
        let (done_tx, done_rx) = mpsc::channel();

        thread::scope(|scope| {
            scope.spawn(|| {
                started_tx.send(()).unwrap();
                done_tx.send(collision.complete(0)).unwrap();
            });

            started_rx.recv().unwrap();
            let deadline = Instant::now() + Duration::from_secs(1);
            while Instant::now() < deadline {
                if coordinator.snapshot().zero_progress_completions == 1 {
                    break;
                }
                thread::yield_now();
            }
            let waiting = coordinator.snapshot();
            assert_eq!(waiting.zero_progress_completions, 1);
            assert_eq!(waiting.zero_progress_streak, 0);
            assert_eq!(waiting.close_reason, None);
            assert_eq!(waiting.admitted, 1);
            assert!(
                done_rx.recv_timeout(Duration::from_millis(20)).is_err(),
                "collision-only work must not race outstanding inference to no-progress closure"
            );
            assert_eq!(productive.complete(1).charged, 1);
            assert_eq!(
                done_rx.recv_timeout(Duration::from_secs(1)).unwrap(),
                BudgetCompletion {
                    reserved: 1,
                    charged: 0,
                    returned: 1,
                }
            );
        });

        let snapshot = coordinator.snapshot();
        assert_eq!(snapshot.productive, 1);
        assert_eq!(snapshot.remaining, 1);
        assert_eq!(snapshot.outstanding, 0);
        assert_eq!(snapshot.zero_progress_completions, 1);
        assert_eq!(snapshot.zero_progress_streak, 0);

        coordinator
            .lease(1)
            .unwrap()
            .admit_for_test()
            .unwrap()
            .complete(1);
        let outcome = coordinator.finish();
        assert_eq!(outcome.snapshot.productive, 2);
        assert_eq!(outcome.snapshot.outstanding, 0);
    }

    #[test]
    fn cancelling_the_last_outstanding_batch_releases_zero_progress_waiters() {
        let coordinator = SearchCoordinator::new(2, 2);
        let cancelled = coordinator.lease(1).unwrap().admit_for_test().unwrap();
        let collision = coordinator.lease(1).unwrap().admit_for_test().unwrap();
        let (done_tx, done_rx) = mpsc::channel();

        thread::scope(|scope| {
            scope.spawn(|| done_tx.send(collision.complete(0)).unwrap());

            let deadline = Instant::now() + Duration::from_secs(1);
            while Instant::now() < deadline {
                if coordinator.snapshot().zero_progress_completions == 1 {
                    break;
                }
                thread::yield_now();
            }
            assert_eq!(coordinator.snapshot().zero_progress_completions, 1);
            assert!(done_rx.recv_timeout(Duration::from_millis(20)).is_err());

            cancelled.cancel();
            assert_eq!(
                done_rx
                    .recv_timeout(Duration::from_secs(1))
                    .unwrap()
                    .charged,
                0
            );
        });

        let middle = coordinator.snapshot();
        assert_eq!(middle.productive, 0);
        assert_eq!(middle.remaining, 2);
        assert_eq!(middle.outstanding, 0);
        assert_eq!(middle.close_reason, None);

        coordinator
            .lease(2)
            .unwrap()
            .admit_for_test()
            .unwrap()
            .complete(2);
        let outcome = coordinator.finish();
        assert_eq!(outcome.snapshot.productive, 2);
    }

    #[test]
    fn close_wakes_waiters_but_preserves_in_flight_work() {
        let coordinator = SearchCoordinator::new(1, 2);
        let admitted = coordinator.lease(1).unwrap().admit_for_test().unwrap();
        let (started_tx, started_rx) = mpsc::channel();
        let (done_tx, done_rx) = mpsc::channel();

        thread::scope(|scope| {
            scope.spawn(|| {
                started_tx.send(()).unwrap();
                done_tx.send(coordinator.lease(1).is_none()).unwrap();
            });

            started_rx.recv().unwrap();
            assert!(coordinator.close(AdmissionCloseReason::StopRequested));
            assert!(!coordinator.close(AdmissionCloseReason::WorkerPanicked));
            assert!(done_rx.recv_timeout(Duration::from_secs(1)).unwrap());
        });

        admitted.complete(1);
        let outcome = coordinator.finish();
        assert_eq!(outcome.snapshot.productive, 1);
        assert_eq!(
            outcome.snapshot.close_reason,
            Some(AdmissionCloseReason::StopRequested)
        );
    }

    #[test]
    fn explicit_cancellation_does_not_count_as_zero_progress() {
        let coordinator = SearchCoordinator::new(3, 1);
        let provisional = coordinator.lease(1).unwrap();
        assert_eq!(provisional.cancel().returned, 1);
        let admitted = coordinator.lease(3).unwrap().admit_for_test().unwrap();
        assert_eq!(admitted.cancel().returned, 3);

        let snapshot = coordinator.snapshot();
        assert_eq!(snapshot.zero_progress_completions, 0);
        assert_eq!(snapshot.zero_progress_streak, 0);
        assert_eq!(snapshot.leases_cancelled, 2);
        assert_eq!(snapshot.remaining, 3);
        assert_eq!(snapshot.outstanding, 0);
        coordinator.assert_quiescent();
    }

    #[test]
    fn zero_target_is_already_quiescent_and_closed() {
        let outcome = SearchCoordinator::new(0, 1).finish();
        assert_eq!(outcome.snapshot.initial, 0);
        assert_eq!(outcome.snapshot.productive, 0);
        assert_eq!(outcome.snapshot.remaining, 0);
        assert_eq!(
            outcome.snapshot.close_reason,
            Some(AdmissionCloseReason::BudgetExhausted)
        );
    }

    #[test]
    #[should_panic(expected = "finished before reaching a terminal close reason")]
    fn finish_rejects_an_open_under_delivered_search() {
        SearchCoordinator::new(3, 1).finish();
    }

    #[test]
    fn external_close_cannot_forge_a_derived_reason() {
        let coordinator = SearchCoordinator::new(3, 1);

        for reason in [
            AdmissionCloseReason::BudgetExhausted,
            AdmissionCloseReason::BackendFailure,
            AdmissionCloseReason::NoProgress,
        ] {
            assert!(catch_unwind(AssertUnwindSafe(|| coordinator.close(reason))).is_err());
            assert_eq!(coordinator.snapshot().close_reason, None);
        }

        assert!(coordinator.close(AdmissionCloseReason::StopRequested));
        let outcome = coordinator.finish();
        assert_eq!(outcome.snapshot.productive, 0);
        assert_eq!(outcome.snapshot.remaining, 3);
    }

    #[test]
    fn invalid_completion_panics_without_leaking_admitted_capacity() {
        let coordinator = SearchCoordinator::new(2, 1);
        let lease = coordinator.lease(2).unwrap().admit_for_test().unwrap();

        assert!(catch_unwind(AssertUnwindSafe(|| lease.complete(3))).is_err());
        let snapshot = coordinator.snapshot();
        assert_eq!(snapshot.productive, 0);
        assert_eq!(snapshot.remaining, 2);
        assert_eq!(snapshot.provisional, 0);
        assert_eq!(snapshot.admitted, 0);
        assert_eq!(snapshot.outstanding, 0);
        assert_eq!(snapshot.leases_abandoned, 1);
        assert_eq!(snapshot.admitted_leases_abandoned, 1);
        assert_eq!(snapshot.admitted_units_abandoned, 2);
    }

    #[test]
    fn concurrent_leases_never_overshoot_productive_budget() {
        let coordinator = Arc::new(SearchCoordinator::new(1_000, 1_000));
        let charged = Arc::new(AtomicU32::new(0));
        let mut workers = Vec::new();

        for _ in 0..8 {
            let coordinator = Arc::clone(&coordinator);
            let charged = Arc::clone(&charged);
            workers.push(thread::spawn(move || {
                while let Some(lease) = coordinator.lease(7) {
                    let Ok(lease) = lease.admit_for_test() else {
                        break;
                    };
                    let produced = lease.units().saturating_sub(1).max(1);
                    let completion = lease.complete(produced);
                    charged.fetch_add(completion.charged, Ordering::Relaxed);
                }
            }));
        }

        for worker in workers {
            worker.join().unwrap();
        }

        assert_eq!(charged.load(Ordering::Relaxed), 1_000);
        let coordinator = Arc::try_unwrap(coordinator).ok().unwrap();
        let outcome = coordinator.finish();
        assert_eq!(outcome.snapshot.charged(), 1_000);
        assert_eq!(outcome.snapshot.remaining, 0);
        assert_eq!(outcome.snapshot.outstanding, 0);
    }
}

#[cfg(all(test, loom))]
mod loom_tests {
    use super::{AdmissionCloseReason, SearchCoordinator};
    use crate::BackendError;
    use loom::sync::atomic::{AtomicUsize, Ordering};
    use loom::sync::Arc;
    use loom::thread;

    #[test]
    fn loom_admit_races_close_linearly() {
        loom::model(|| {
            let coordinator = Arc::new(SearchCoordinator::new(1, 2));
            let worker = {
                let coordinator = Arc::clone(&coordinator);
                thread::spawn(move || {
                    let Some(lease) = coordinator.lease(1) else {
                        return false;
                    };
                    let Ok(lease) = lease.admit_for_test() else {
                        return false;
                    };
                    lease.complete(1);
                    true
                })
            };
            let stopper = {
                let coordinator = Arc::clone(&coordinator);
                thread::spawn(move || coordinator.close(AdmissionCloseReason::StopRequested))
            };

            let admitted = worker.join().unwrap();
            let stop_won = stopper.join().unwrap();
            assert!(coordinator.lease(1).is_none());
            let snapshot = coordinator.snapshot();
            assert_eq!(snapshot.outstanding, 0);
            if admitted {
                assert_eq!(snapshot.productive, 1);
                assert_eq!(snapshot.remaining, 0);
                assert_eq!(snapshot.leases_admitted, 1);
                assert_eq!(
                    snapshot.close_reason,
                    Some(if stop_won {
                        AdmissionCloseReason::StopRequested
                    } else {
                        AdmissionCloseReason::BudgetExhausted
                    })
                );
            } else {
                assert!(stop_won);
                assert_eq!(snapshot.productive, 0);
                assert_eq!(snapshot.remaining, 1);
                assert_eq!(snapshot.leases_admitted, 0);
                assert_eq!(
                    snapshot.close_reason,
                    Some(AdmissionCloseReason::StopRequested)
                );
            }
            coordinator.assert_quiescent();
        });
    }

    #[test]
    fn loom_temporary_exhaustion_wakes_after_zero_return() {
        loom::model(|| {
            let coordinator = Arc::new(SearchCoordinator::new(1, 2));
            let holder = coordinator.lease(1).unwrap().admit_for_test().unwrap();
            let waiter = {
                let coordinator = Arc::clone(&coordinator);
                thread::spawn(move || {
                    coordinator
                        .lease(1)
                        .unwrap()
                        .admit_for_test()
                        .unwrap()
                        .complete(1)
                })
            };

            assert_eq!(holder.complete(0).charged, 0);
            assert_eq!(waiter.join().unwrap().charged, 1);
            let snapshot = coordinator.snapshot();
            assert_eq!(snapshot.productive, 1);
            assert_eq!(snapshot.remaining, 0);
            assert_eq!(snapshot.outstanding, 0);
            coordinator.assert_quiescent();
        });
    }

    #[test]
    fn loom_zero_progress_and_productive_completion_cannot_miss_wakeup() {
        loom::model(|| {
            let coordinator = Arc::new(SearchCoordinator::new(2, 2));
            let admitted = Arc::new(AtomicUsize::new(0));
            let zero_worker = {
                let coordinator = Arc::clone(&coordinator);
                let admitted = Arc::clone(&admitted);
                thread::spawn(move || {
                    let zero = coordinator.lease(1).unwrap().admit_for_test().unwrap();
                    admitted.fetch_add(1, Ordering::SeqCst);
                    while admitted.load(Ordering::SeqCst) < 2 {
                        thread::yield_now();
                    }
                    zero.complete(0)
                })
            };
            let productive_worker = {
                let coordinator = Arc::clone(&coordinator);
                let admitted = Arc::clone(&admitted);
                thread::spawn(move || {
                    let productive = coordinator.lease(1).unwrap().admit_for_test().unwrap();
                    admitted.fetch_add(1, Ordering::SeqCst);
                    while admitted.load(Ordering::SeqCst) < 2 {
                        thread::yield_now();
                    }
                    productive.complete(1)
                })
            };

            assert_eq!(zero_worker.join().unwrap().charged, 0);
            assert_eq!(productive_worker.join().unwrap().charged, 1);
            let middle = coordinator.snapshot();
            assert_eq!(middle.productive, 1);
            assert_eq!(middle.remaining, 1);
            assert_eq!(middle.outstanding, 0);
            assert_eq!(middle.close_reason, None);

            coordinator
                .lease(1)
                .unwrap()
                .admit_for_test()
                .unwrap()
                .complete(1);
            let coordinator = Arc::try_unwrap(coordinator).ok().unwrap();
            let outcome = coordinator.finish();
            assert_eq!(outcome.snapshot.productive, 2);
            assert_eq!(outcome.snapshot.outstanding, 0);
        });
    }

    #[test]
    fn loom_first_close_reason_wins() {
        loom::model(|| {
            let coordinator = Arc::new(SearchCoordinator::new(1, 1));
            let stop = {
                let coordinator = Arc::clone(&coordinator);
                thread::spawn(move || {
                    (
                        coordinator.close(AdmissionCloseReason::StopRequested),
                        AdmissionCloseReason::StopRequested,
                    )
                })
            };
            let deadline = {
                let coordinator = Arc::clone(&coordinator);
                thread::spawn(move || {
                    (
                        coordinator.close(AdmissionCloseReason::Deadline),
                        AdmissionCloseReason::Deadline,
                    )
                })
            };

            let stop = stop.join().unwrap();
            let deadline = deadline.join().unwrap();
            assert_ne!(stop.0, deadline.0);
            let winner = if stop.0 { stop.1 } else { deadline.1 };
            assert_eq!(coordinator.snapshot().close_reason, Some(winner));
            assert!(coordinator.lease(1).is_none());
            coordinator.assert_quiescent();
        });
    }

    #[test]
    fn loom_first_failure_wins_and_retains_its_payload() {
        loom::model(|| {
            let coordinator = Arc::new(SearchCoordinator::new(1, 1));
            let first = {
                let coordinator = Arc::clone(&coordinator);
                thread::spawn(move || {
                    coordinator.record_first_failure(BackendError::msg("worker-a"))
                })
            };
            let second = {
                let coordinator = Arc::clone(&coordinator);
                thread::spawn(move || {
                    coordinator.record_first_failure(BackendError::msg("worker-b"))
                })
            };

            let first_won = first.join().unwrap();
            let second_won = second.join().unwrap();
            assert_ne!(first_won, second_won);
            assert!(coordinator.lease(1).is_none());
            let coordinator = Arc::try_unwrap(coordinator).ok().unwrap();
            let outcome = coordinator.finish();
            assert!(outcome.snapshot.first_failure_recorded);
            assert_eq!(
                outcome.snapshot.close_reason,
                Some(AdmissionCloseReason::BackendFailure)
            );
            assert_eq!(
                outcome.first_failure.unwrap().to_string(),
                if first_won { "worker-a" } else { "worker-b" }
            );
        });
    }
}
