mod backend;

#[cfg(test)]
mod test_util;

pub use backend::{
    smart_uniform_prior, Backend, BackendError, ConstantValueBackend, EvalResult,
    SmartUniformBackend,
};
