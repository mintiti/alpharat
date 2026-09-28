//! Lane and context selection for the TensorRT backend. It holds no CUDA state,
//! so the routing that concurrent callers and warmup depend on is testable
//! without a GPU.

use alpharat_mcts::BackendError;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, MutexGuard, TryLockError};

/// Independently locked lanes of execution resources.
///
/// Ordinary calls rotate their starting lane and take the first idle one, which
/// spreads concurrent callers. A serial caller therefore visits lanes in a fixed
/// rotation: repeating one sequence of shapes can leave a lane/shape pair unused
/// forever. `lock` reaches a chosen lane instead.
pub(crate) struct Lanes<T> {
    lanes: Vec<Mutex<T>>,
    next: AtomicU64,
}

impl<T> Lanes<T> {
    pub(crate) fn new(lanes: Vec<T>) -> Self {
        Self {
            lanes: lanes.into_iter().map(Mutex::new).collect(),
            next: AtomicU64::new(0),
        }
    }

    pub(crate) fn len(&self) -> usize {
        self.lanes.len()
    }

    /// Take the first idle lane from a rotating start, or wait for the start lane.
    pub(crate) fn acquire(&self) -> Result<MutexGuard<'_, T>, BackendError> {
        let lane_count = self.lanes.len();
        let start = if lane_count == 1 {
            0
        } else {
            self.next.fetch_add(1, Ordering::Relaxed) as usize % lane_count
        };
        if lane_count > 1 {
            for offset in 0..lane_count {
                match self.lanes[(start + offset) % lane_count].try_lock() {
                    Ok(guard) => return Ok(guard),
                    Err(TryLockError::WouldBlock) => {}
                    Err(TryLockError::Poisoned(_)) => {
                        return Err(BackendError::msg(
                            "TensorRT lane lock poisoned after a panic",
                        ))
                    }
                }
            }
        }
        self.lock(start)
    }

    /// Wait for one specific lane. This does not advance the rotation.
    pub(crate) fn lock(&self, lane: usize) -> Result<MutexGuard<'_, T>, BackendError> {
        self.lanes
            .get(lane)
            .ok_or_else(|| BackendError::msg(format!("TensorRT lane {lane} does not exist")))?
            .lock()
            .map_err(|_| {
                BackendError::msg(
                    "TensorRT session lock poisoned after a panic; the session will not be reused",
                )
            })
    }
}

/// Context within a lane for a real batch of `n` rows: the smallest fitting
/// fixed size, or the lane's only context when there are no fixed sizes.
/// Callers validate `n` against the maximum batch first.
pub(crate) fn execution_slot(execution_sizes: &[usize], n: usize) -> usize {
    execution_sizes
        .iter()
        .position(|size| n <= *size)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::{execution_slot, Lanes};
    use std::collections::BTreeSet;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::mpsc;

    /// Route one call as TensorrtBackend does and record the lane, context and
    /// execution shape it reaches. Fixed sizes execute padded to their size;
    /// without them the single context runs the exact request shape.
    fn call(
        lanes: &Lanes<usize>,
        sizes: &[usize],
        lane: Option<usize>,
        n: usize,
        seen: &mut BTreeSet<(usize, usize, usize)>,
    ) {
        let lane = *match lane {
            Some(lane) => lanes.lock(lane),
            None => lanes.acquire(),
        }
        .unwrap();
        let slot = execution_slot(sizes, n);
        seen.insert((lane, slot, sizes.get(slot).copied().unwrap_or(n)));
    }

    fn every(lanes: usize, contexts: &[(usize, usize)]) -> BTreeSet<(usize, usize, usize)> {
        let lane_contexts = |lane| contexts.iter().map(move |&(slot, n)| (lane, slot, n));
        (0..lanes).flat_map(lane_contexts).collect()
    }

    #[test]
    fn serial_rotation_misses_bucket_lanes_that_explicit_lanes_reach() {
        let (sizes, shapes) = ([32, 64], [32, 64]);
        let lanes = Lanes::new(vec![0, 1]);
        let mut rotated = BTreeSet::new();
        for _ in 0..10 {
            for n in shapes {
                call(&lanes, &sizes, None, n, &mut rotated);
            }
        }
        // The reported example: lane 0/64 and lane 1/32 stay cold.
        assert_eq!(rotated, BTreeSet::from([(0, 0, 32), (1, 1, 64)]));

        let mut explicit = BTreeSet::new();
        for n in shapes {
            for lane in 0..lanes.len() {
                call(&lanes, &sizes, Some(lane), n, &mut explicit);
            }
        }
        assert_eq!(explicit, every(2, &[(0, 32), (1, 64)]));
    }

    #[test]
    fn serial_rotation_misses_exact_shapes_that_explicit_lanes_reach() {
        let shapes = 1..=8;
        let lanes = Lanes::new(vec![0, 1]);
        let exact = shapes.clone().map(|n| (0, n)).collect::<Vec<_>>();
        let mut rotated = BTreeSet::new();
        for _ in 0..5 {
            for n in shapes.clone() {
                call(&lanes, &[], None, n, &mut rotated);
            }
        }
        assert_eq!(rotated.len(), 8, "each shape reaches only one lane");

        let mut explicit = BTreeSet::new();
        for n in shapes {
            for lane in 0..lanes.len() {
                call(&lanes, &[], Some(lane), n, &mut explicit);
            }
        }
        assert_eq!(explicit, every(2, &exact));
    }

    #[test]
    fn fixed_sizes_select_the_smallest_fitting_context() {
        let sizes = [32, 64, 128];
        for (n, slot) in [(1, 0), (32, 0), (33, 1), (64, 1), (65, 2), (128, 2)] {
            assert_eq!(execution_slot(&sizes, n), slot, "n={n}");
        }
        assert_eq!(execution_slot(&[], 77), 0);
    }

    #[test]
    fn explicit_lanes_leave_the_rotation_unchanged() {
        let lanes = Lanes::new(vec![0, 1, 2]);
        assert_eq!(*lanes.acquire().unwrap(), 0);
        assert_eq!(*lanes.lock(2).unwrap(), 2);
        assert_eq!(*lanes.lock(0).unwrap(), 0);
        assert_eq!(*lanes.acquire().unwrap(), 1);
        let single = Lanes::new(vec![7]);
        assert_eq!(*single.acquire().unwrap(), 7);
        assert_eq!(*single.lock(0).unwrap(), 7);
    }

    #[test]
    fn explicit_lane_waits_for_a_busy_lane_instead_of_rerouting() {
        let lanes = Lanes::new(vec![0, 1]);
        let released = AtomicBool::new(false);
        let (held, release) = (mpsc::channel(), mpsc::channel::<()>());
        std::thread::scope(|scope| {
            let (lanes, released) = (&lanes, &released);
            scope.spawn(move || {
                let _guard = lanes.lock(1).unwrap();
                held.0.send(()).unwrap();
                release.1.recv().unwrap();
                released.store(true, Ordering::SeqCst);
            });
            held.1.recv().unwrap();
            // Rotation routes around the busy lane; an explicit lane does not.
            assert_eq!(*lanes.acquire().unwrap(), 0);
            assert_eq!(*lanes.acquire().unwrap(), 0);
            let waiter = scope.spawn(move || {
                let lane = *lanes.lock(1).unwrap();
                (lane, released.load(Ordering::SeqCst))
            });
            release.0.send(()).unwrap();
            assert_eq!(waiter.join().unwrap(), (1, true));
        });
    }

    #[test]
    fn missing_or_poisoned_lanes_fail_without_blocking_other_lanes() {
        let lanes = Lanes::new(vec![0, 1]);
        let error = lanes.lock(2).err().unwrap();
        assert!(error.to_string().contains("lane 2 does not exist"));
        std::thread::scope(|scope| {
            let lanes = &lanes;
            assert!(scope
                .spawn(move || {
                    let _guard = lanes.lock(1).unwrap();
                    panic!("injected panic while holding a lane");
                })
                .join()
                .is_err());
        });
        let error = lanes.lock(1).err().unwrap();
        assert!(error.to_string().contains("session lock poisoned"));
        assert_eq!(*lanes.lock(0).unwrap(), 0);
        // Rotation starts at lane 0 and then lane 1, as it did before the panic.
        assert_eq!(*lanes.acquire().unwrap(), 0);
        let error = lanes.acquire().err().unwrap();
        assert!(error.to_string().contains("lane lock poisoned"));
    }
}
