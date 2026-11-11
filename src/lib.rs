//! LLM inference on Apple Silicon: a Metal tensor runtime with its own kernels,
//! safetensors/GGUF loaders and a Qwen3.5-family hybrid engine (`hybrid`, which
//! also runs Flint and Spark) with speculative decoding. The general transformer
//! engine (`model`, `ops`, `nn`, ...) is behind the `legacy-engine` feature.
#![deny(unsafe_op_in_unsafe_fn)]
#![deny(unsafe_code)]
#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
compile_error!("Ferrum requires macOS on Apple Silicon");
pub mod error;
#[allow(unsafe_code)] // Audited FFI and shared-memory boundary.
pub mod metal;
#[cfg(feature = "legacy-engine")]
pub mod ops;
#[cfg(feature = "legacy-engine")]
pub mod reference;
pub mod tensor;
pub use error::{Error, Result};
pub use metal::MetalDevice;
pub use tensor::{DType, Tensor};

pub mod diffusion;
#[cfg(feature = "legacy-engine")]
pub mod generation;
pub mod hybrid;
pub mod loader;
#[cfg(feature = "legacy-engine")]
pub mod model;
#[cfg(feature = "legacy-engine")]
pub mod nn;
#[cfg(feature = "legacy-engine")]
pub mod quantization;
#[cfg(feature = "legacy-engine")]
pub mod sampling;
pub mod tokenizer;
pub mod vision;
