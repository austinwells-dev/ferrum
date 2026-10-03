//! LLM inference on Apple Silicon: a Metal tensor runtime with its own kernels,
//! safetensors/GGUF loaders, a general transformer engine (`model`) and a
//! Qwen3.5-family hybrid engine (`hybrid`) with speculative decoding.
#![deny(unsafe_op_in_unsafe_fn)]
#![deny(unsafe_code)]
#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
compile_error!("Ferrum requires macOS on Apple Silicon");
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
pub mod hybrid;
pub mod loader;
pub mod model;
pub mod nn;
pub mod quantization;
pub mod sampling;
pub mod tokenizer;
pub mod vision;
