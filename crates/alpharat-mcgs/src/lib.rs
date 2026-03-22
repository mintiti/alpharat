pub mod node;

#[cfg(test)]
pub(crate) mod test_util;

pub use node::{smart_uniform_prior, Edge, HalfEdge, LowNode};
