//! Monte Carlo graph search with a scoped public observation surface.
//!
//! The raw DAG and transposition-table implementation types are intentionally
//! crate-private. These compiler checks protect that boundary:
//!
//! ```compile_fail
//! use alpharat_mcgs::SharedNode;
//! ```
//!
//! ```compile_fail
//! use alpharat_mcgs::LowNode;
//! ```
//!
//! ```compile_fail
//! use alpharat_mcgs::Edge;
//! ```
//!
//! ```compile_fail
//! use alpharat_mcgs::TranspositionTable;
//! ```
//!
//! ```compile_fail
//! use alpharat_mcgs::node;
//! ```
//!
//! ```compile_fail
//! let _ = alpharat_mcgs::MCGSTree::root;
//! ```
//!
//! ```compile_fail
//! let _ = alpharat_mcgs::MCGSTree::tt;
//! ```
//!
//! ```compile_fail
//! let _ = alpharat_mcgs::MCGSTree::tt_mut;
//! ```

#[cfg(feature = "python")]
pub mod bindings;
mod access;
pub mod gc;
#[cfg_attr(not(test), allow(dead_code))]
mod node;
mod observer;
mod search;
#[cfg(test)]
mod search_invariants;
mod tree;
#[cfg_attr(not(test), allow(dead_code))]
mod tt;

pub use alpharat_eval_core::{
    smart_uniform_prior, Backend, BackendError, ConstantValueBackend, EvalResult,
    SmartUniformBackend,
};
pub use observer::{
    ChildEdges, EdgeTransition, EdgeView, NodeHandle, NodeStats, NodeView, OutcomeStats, Outcomes,
    SearchPlayer, TranspositionEviction, TranspositionStats, TreeStats, TreeView,
};
pub use search::{run_search, SearchConfig, SearchResult};
#[cfg(feature = "bench-internals")]
pub use search::{
    run_search_one_worker_profiled, ProfiledSearchResult, SearchLedgerStats, SearchTimings,
};
pub use tree::MCGSTree;
