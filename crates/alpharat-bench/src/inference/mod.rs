//! Repeatable production-inference measurements and opt-in observations.
pub mod artifact;
pub mod driver;
pub mod model;
pub mod report;
pub mod supervisor;

pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;
