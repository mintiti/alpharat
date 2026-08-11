//! Versioned records for repeatable AlphaRat performance calibration.
//!
//! The protocol defines the shared language. A plan names the cases to run and the differences it
//! intends to test. A record preserves what was requested, what actually ran, and every warmup or
//! measured trial. A comparison check then decides whether two records describe the same fixed
//! work with only the planned differences. It deliberately does not calculate performance deltas.

mod compare;
mod folder;
mod model;
mod trials;

pub use compare::{check_comparable, ComparisonCheck, ComparisonIssue};
pub use folder::{load_run_folder, LoadedRun, ProtocolError};
pub use model::*;
pub use trials::*;
