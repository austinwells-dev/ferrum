//! Qwen3.5-family hybrid models (`qwen35`, `qwen35moe` GGUF): Gated DeltaNet
//! linear attention interleaved with gated full attention, on a dedicated
//! kernel library with preallocated state.
#![forbid(unsafe_code)]
pub mod bench;
pub mod config;
pub mod engine;
pub mod weights;

pub use config::{HybridConfig, Variant};
pub use engine::{HybridModel, HybridState, Output, Produced};

use crate::{
    Error, MetalDevice, Result,
    loader::gguf::{GgufFile, MetadataType, MetadataValue},
    tokenizer::Tokenizer,
};
use std::path::Path;

/// Qwen3.5 pre-tokenizer split (tokenizer.json of the Qwen3.5/3.6 family).
const QWEN35_PATTERN: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

pub struct LoadedHybrid {
    pub model: HybridModel,
    pub tokenizer: Tokenizer,
    pub eos_ids: Vec<u32>,
    pub chat_template: Option<String>,
    pub load_time: std::time::Duration,
}

/// Load a hybrid GGUF with `chunk` prompt rows per forward pass and
/// `logit_rows` rows of logits available to evaluation.
pub fn load(
    device: &MetalDevice,
    path: impl AsRef<Path>,
    chunk: usize,
    logit_rows: usize,
) -> Result<LoadedHybrid> {
    let start = std::time::Instant::now();
    let mut file = GgufFile::open(path)?;
    let config = HybridConfig::from_gguf(&file)?;
    let (tokenizer, eos_ids) = tokenizer(&file, config.vocab)?;
    let chat_template = match file.metadata_value("tokenizer.chat_template") {
        Some(MetadataValue::String(t)) => Some(t.clone()),
        _ => None,
    };
    let weights = weights::load(device, &mut file, &config)?;
    let model = HybridModel::new(device, config, weights, chunk, logit_rows)?;
    Ok(LoadedHybrid {
        model,
        tokenizer,
        eos_ids,
        chat_template,
        load_time: start.elapsed(),
    })
}

/// Build only the tokenizer of a hybrid GGUF (no weights are read).
pub fn load_tokenizer(path: impl AsRef<Path>) -> Result<Tokenizer> {
    let file = GgufFile::open(path)?;
    let config = HybridConfig::from_gguf(&file)?;
    Ok(tokenizer(&file, config.vocab)?.0)
}

fn tokenizer(file: &GgufFile, vocab: usize) -> Result<(Tokenizer, Vec<u32>)> {
    let md = file.metadata();
    let text = |key: &str| match md.get(key) {
        Some(MetadataValue::String(s)) => Ok(s.as_str()),
        _ => Err(Error::Tokenizer(format!("missing GGUF metadata {key}"))),
    };
    if text("tokenizer.ggml.model")? != "gpt2" || text("tokenizer.ggml.pre")? != "qwen35" {
        return Err(Error::Tokenizer(
            "hybrid GGUF must use the gpt2 model with the qwen35 pre-tokenizer".into(),
        ));
    }
    let strings = |key: &str| -> Result<Vec<String>> {
        match md.get(key) {
            Some(MetadataValue::Array {
                element_type: MetadataType::String,
                values,
            }) => values
                .iter()
                .map(|v| match v {
                    MetadataValue::String(s) => Ok(s.clone()),
                    _ => Err(Error::Tokenizer(format!("{key} holds a non-string"))),
                })
                .collect(),
            _ => Err(Error::Tokenizer(format!("missing GGUF metadata {key}"))),
        }
    };
    let tokens = strings("tokenizer.ggml.tokens")?;
    let merges = strings("tokenizer.ggml.merges")?;
    let types = match md.get("tokenizer.ggml.token_type") {
        Some(MetadataValue::Array { values, .. }) => values
            .iter()
            .map(|v| match v {
                MetadataValue::Int32(t) => Ok(*t),
                _ => Err(Error::Tokenizer("token_type holds a non-i32".into())),
            })
            .collect::<Result<Vec<_>>>()?,
        _ => return Err(Error::Tokenizer("missing tokenizer.ggml.token_type".into())),
    };
    if tokens.len() > vocab {
        return Err(Error::Tokenizer(
            "tokenizer vocabulary exceeds embedding rows".into(),
        ));
    }
    let tokenizer =
        Tokenizer::from_gguf_bpe_with_pattern(&tokens, &merges, &types, QWEN35_PATTERN)?;
    let eos = config::uint(md, "tokenizer.ggml.eos_token_id")? as u32;
    let mut eos_ids = vec![eos];
    // Chat turns end with <|im_end|>; raw completions may emit <|endoftext|>.
    for special in ["<|im_end|>", "<|endoftext|>"] {
        if let Some(id) = tokens.iter().position(|t| t == special)
            && !eos_ids.contains(&(id as u32))
        {
            eos_ids.push(id as u32);
        }
    }
    Ok((tokenizer, eos_ids))
}
