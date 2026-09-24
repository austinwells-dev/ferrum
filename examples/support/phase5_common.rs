//! Shared model adapter for the Phase 5 GGUF and MLX measurement programs.
use ferrum::{
    Error, MetalDevice, Result,
    model::{ModelConfig, Transformer, qwen_gguf, qwen_mlx},
    tokenizer::qwen::QwenTokenizer,
};
use std::path::Path;

// Each measurement example consumes a different subset of this shared record.
#[allow(dead_code)]
pub struct QuantizedModel {
    pub config: ModelConfig,
    pub model: Transformer,
    pub tokenizer: QwenTokenizer,
    pub format: &'static str,
    pub repository: &'static str,
    pub revision: &'static str,
    pub source_tensor_bytes: usize,
    pub quantized_tensor_bytes: usize,
    pub tensor_count: usize,
    pub parameter_count: usize,
}

pub fn load(device: &MetalDevice, path: &Path) -> Result<QuantizedModel> {
    if path.is_file() {
        let loaded = qwen_gguf::load(device, path)?;
        let (repository, revision) = if loaded.architecture == "qwen3" {
            (
                "Qwen/Qwen3-0.6B-GGUF",
                "23749fefcc72300e3a2ad315e1317431b06b590a",
            )
        } else {
            (
                "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
                "9217f5db79a29953eb74d5343926648285ec7e67",
            )
        };
        Ok(QuantizedModel {
            config: loaded.config,
            model: loaded.model,
            tokenizer: loaded.tokenizer,
            format: "GGUF",
            repository,
            revision,
            source_tensor_bytes: loaded.source_tensor_bytes,
            quantized_tensor_bytes: loaded.quantized_tensor_bytes,
            tensor_count: loaded.tensor_count,
            parameter_count: loaded.parameter_count,
        })
    } else if path.is_dir() {
        let loaded = qwen_mlx::load(device, path)?;
        Ok(QuantizedModel {
            config: loaded.config,
            model: loaded.model,
            tokenizer: loaded.tokenizer,
            format: "MLX affine Q4 group-64",
            repository: "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
            revision: "a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3",
            source_tensor_bytes: loaded.source_tensor_bytes,
            quantized_tensor_bytes: loaded.quantized_tensor_bytes,
            tensor_count: loaded.tensor_count,
            parameter_count: loaded.parameter_count,
        })
    } else {
        Err(Error::Config(format!(
            "quantized model path does not exist: {}",
            path.display()
        )))
    }
}
