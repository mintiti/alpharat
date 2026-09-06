//! Shared support for AlphaRat's benchmark and calibration binaries.

pub mod calibration;
pub mod capacity;
pub mod production;
#[cfg(feature = "mcgs-profile")]
pub mod runner;
#[cfg(feature = "mcgs-profile")]
pub mod search;

#[cfg(feature = "inference")]
pub mod inference;
