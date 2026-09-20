//! A small, synchronous Rust/Metal tensor runtime for Apple Silicon.
#![deny(unsafe_op_in_unsafe_fn)]
#![deny(unsafe_code)]
#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
compile_error!("Ferrum Phase 1 requires Apple Silicon macOS");
pub mod error;
#[allow(unsafe_code)] // Audited FFI and shared-memory boundary.
pub mod metal;
pub mod ops;
pub mod reference;
pub mod tensor;
pub use error::{Error, Result};
pub use metal::MetalDevice;
pub use tensor::{DType, Tensor};

pub mod generation;
pub mod loader;
pub mod model;
pub mod nn;
pub mod sampling;
pub mod tokenizer;
