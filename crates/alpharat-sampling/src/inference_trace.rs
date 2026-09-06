//! Opt-in host ranges. Disabled by default; no CUDA events or synchronization.
use std::cell::Cell;
use std::ffi::CStr;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

static ENABLED: AtomicBool = AtomicBool::new(false);
static NEXT: AtomicU64 = AtomicU64::new(1);
static EPOCH: OnceLock<Instant> = OnceLock::new();
#[derive(Clone, Debug)]
pub struct Link {
    pub request_id: u64,
    pub batch_id: u64,
    pub positions: usize,
    pub queued_ns: u64,
    pub dequeued_ns: u64,
}
static LINKS: Mutex<Vec<Link>> = Mutex::new(Vec::new());
static DROPPED: AtomicU64 = AtomicU64::new(0);
const LINK_LIMIT: usize = 100_000;
thread_local! { static REQUEST: Cell<u64> = const { Cell::new(0) }; }

#[cfg(feature = "inference-trace")]
extern "C" {
    fn alpharat_nvtx_push(name: *const std::ffi::c_char, id: u64);
    fn alpharat_nvtx_pop();
}

pub fn available() -> bool {
    cfg!(feature = "inference-trace")
}
pub fn enable() -> Result<(), &'static str> {
    if !available() {
        return Err("NVTX support is not compiled");
    }
    EPOCH.get_or_init(Instant::now);
    LINKS
        .lock()
        .map_err(|_| "trace link lock poisoned")?
        .reserve(LINK_LIMIT);
    ENABLED.store(true, Ordering::Relaxed);
    Ok(())
}
#[inline]
pub fn enabled() -> bool {
    #[cfg(feature = "inference-trace")]
    {
        ENABLED.load(Ordering::Relaxed)
    }
    #[cfg(not(feature = "inference-trace"))]
    {
        false
    }
}
pub fn id() -> u64 {
    if enabled() {
        NEXT.fetch_add(1, Ordering::Relaxed)
    } else {
        0
    }
}
pub fn stamp() -> u64 {
    if enabled() {
        EPOCH.get().expect("trace epoch").elapsed().as_nanos() as u64
    } else {
        0
    }
}
pub fn request_id() -> u64 {
    if enabled() {
        REQUEST.get()
    } else {
        0
    }
}
pub fn links() -> (Vec<Link>, u64) {
    (
        LINKS.lock().expect("trace links").clone(),
        DROPPED.load(Ordering::Relaxed),
    )
}
pub fn link(request: u64, batch: u64, positions: usize, queued_ns: u64, dequeued_ns: u64) {
    if !enabled() {
        return;
    }
    let mut links = LINKS.lock().expect("trace links");
    if links.len() < LINK_LIMIT {
        links.push(Link {
            request_id: request,
            batch_id: batch,
            positions,
            queued_ns,
            dequeued_ns,
        });
    } else {
        DROPPED.fetch_add(1, Ordering::Relaxed);
    }
}
pub struct Range {
    active: bool,
    _thread_bound: std::marker::PhantomData<*mut ()>,
}
pub fn range(name: &'static CStr, id: u64) -> Range {
    let active = enabled();
    #[cfg(feature = "inference-trace")]
    if active {
        // SAFETY: static nul-terminated name; NVTX copies attributes within the call.
        unsafe {
            alpharat_nvtx_push(name.as_ptr(), id);
        }
    }
    #[cfg(not(feature = "inference-trace"))]
    let _ = (name, id);
    Range {
        active,
        _thread_bound: std::marker::PhantomData,
    }
}
impl Drop for Range {
    fn drop(&mut self) {
        #[cfg(feature = "inference-trace")]
        if self.active {
            // SAFETY: !Send range balances the push on the same thread.
            unsafe {
                alpharat_nvtx_pop();
            }
        }
        #[cfg(not(feature = "inference-trace"))]
        let _ = self.active;
    }
}
pub struct Request {
    previous: u64,
    _range: Range,
}
pub fn request() -> Request {
    scope(id(), c"inference.request")
}
pub fn batch(id: u64) -> Request {
    scope(id, c"inference.batch")
}
fn scope(id: u64, name: &'static CStr) -> Request {
    let previous = if enabled() { REQUEST.replace(id) } else { 0 };
    Request {
        previous,
        _range: range(name, id),
    }
}
impl Drop for Request {
    fn drop(&mut self) {
        if enabled() {
            REQUEST.set(self.previous);
        }
    }
}
#[cfg(test)]
mod tests {
    #[test]
    fn disabled_does_not_record() {
        assert!(!super::enabled());
        let _r = super::request();
        super::link(1, 2, 1, 0, 0);
        assert_eq!(super::id(), 0);
        assert!(super::links().0.is_empty());
    }
}
