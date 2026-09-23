use thiserror::Error;
#[derive(Debug, Error)]
pub enum Error {
    #[error("invalid model configuration: {0}")]
    Config(String),
    #[error("weight {name}: {message}")]
    Weight { name: String, message: String },
    #[error("invalid token ID {id}; vocabulary size {vocab}")]
    Token { id: u32, vocab: usize },
    #[error("KV cache: {0}")]
    Cache(String),
    #[error("safetensors: {0}")]
    Safetensors(String),
    #[error("GGUF: {0}")]
    Gguf(String),
    #[error("tokenizer: {0}")]
    Tokenizer(String),

    #[error("Metal initialization: {0}")]
    Initialization(String),
    #[error("Metal shader compilation: {0}")]
    Compilation(String),
    #[error("Metal function not found: {0}")]
    MissingKernel(String),
    #[error("Metal pipeline {name}: {message}")]
    Pipeline { name: String, message: String },
    #[error("invalid shape: {0}")]
    Shape(String),
    #[error("invalid reshape: {0}")]
    Reshape(String),
    #[error("dtype mismatch")]
    DType,
    #[error("allocation of {0} bytes failed or exceeds the device limit")]
    Allocation(usize),
    #[error("invalid buffer range: {0}")]
    Range(String),
    #[error("dispatch: {0}")]
    Dispatch(String),
    #[error("synchronization: {0}")]
    Synchronization(String),
    #[error("tensor belongs to a different MetalDevice context")]
    DeviceMismatch,
    #[error("invalid operation parameter: {0}")]
    Parameter(String),
    #[error("numerical validation: {0}")]
    Validation(String),
}
pub type Result<T> = std::result::Result<T, Error>;
