pub mod node;

pub use alpharat_eval_core::{
    smart_uniform_prior, Backend, BackendError, ConstantValueBackend, EvalResult,
    SmartUniformBackend,
};
pub use node::{Edge, HalfEdge, LowNode};
