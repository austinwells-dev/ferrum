//! Qwen3.5-family hybrid models (`qwen35`, `qwen35moe` GGUF): Gated DeltaNet
//! linear attention interleaved with gated full attention, on a dedicated
//! kernel library with preallocated state. The same engine runs the
//! attention-only `flint` and `spark2_5` architectures (experimental), whose
//! sliding-window layers keep K/V rings.
#![forbid(unsafe_code)]
pub mod bench;
pub mod chat;
pub mod config;
pub mod draft;
pub mod engine;
#[cfg(test)]
mod iq_tests;
pub mod lookup;
pub mod mtp;
pub mod plan;
pub mod runtime;
pub mod session;
pub mod speculative;
pub mod weights;

pub use config::{AttnSpec, HybridConfig, KvCache, Variant};
pub use engine::{
    HybridModel, HybridState, Output, Penalties, Produced, RowOutput, RowsProduced, SpecGeometry,
};

use crate::{
    Error, MetalDevice, Result,
    loader::gguf::{GgufFile, MetadataType, MetadataValue},
    tokenizer::Tokenizer,
};
use std::path::Path;

/// Qwen3.5 pre-tokenizer split (tokenizer.json of the Qwen3.5/3.6 family).
const QWEN35_PATTERN: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

/// Flint pre-tokenizer (llama.cpp `LLAMA_VOCAB_PRE_TYPE_FLINT`): digit runs of
/// up to three, then CJK runs, then words, punctuation and whitespace. The
/// patterns apply in that order, each to the pieces the previous one made.
const FLINT_PATTERNS: [&str; 3] = [
    r"\p{N}{1,3}",
    r"[一-龥぀-ゟ゠-ヿ]+",
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?| ?[\p{P}\p{S}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
];

/// Spark pre-tokenizer (llama.cpp `LLAMA_VOCAB_PRE_TYPE_SPARK2_5`): digit runs
/// of up to three, CJK runs, words, punctuation and whitespace, then single
/// digits, applied in that order like `FLINT_PATTERNS`.
const SPARK_PATTERNS: [&str; 4] = [
    r"\p{N}{1,3}",
    r"[一-龥぀-ゟ゠-ヿ]+",
    r##"[!"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~][A-Za-z]+|[^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+| ?[\p{P}\p{S}]+|[\r\n]|\s+(?!\S)|\s+"##,
    r"\p{N}",
];

pub struct LoadedHybrid {
    pub model: HybridModel,
    pub tokenizer: Tokenizer,
    pub eos_ids: Vec<u32>,
    pub chat_template: Option<String>,
    /// The chat template's `bos_token` and `eos_token` strings.
    pub chat_special_tokens: (String, String),
    pub load_time: std::time::Duration,
    /// Memory plan computed from metadata before any weight was read.
    pub plan: plan::MemoryPlan,
    /// Device bytes the weights and scratch actually took (Metal accounting).
    pub loaded_bytes: usize,
}

/// Load a hybrid GGUF with `chunk` prompt rows per forward pass and
/// `logit_rows` rows of logits available to evaluation.
pub fn load(
    device: &MetalDevice,
    path: impl AsRef<Path>,
    chunk: usize,
    logit_rows: usize,
) -> Result<LoadedHybrid> {
    load_with(
        device,
        path,
        plan::PlanOptions {
            chunk,
            logit_rows,
            // Tools size their own state; only require that the model fits.
            context: Some(256),
            snapshots: 0,
            ..Default::default()
        },
    )
}

/// Plan memory from metadata (failing before any weight is read if the
/// request cannot fit), then load weights and scratch.
pub fn load_with(
    device: &MetalDevice,
    path: impl AsRef<Path>,
    options: plan::PlanOptions,
) -> Result<LoadedHybrid> {
    let start = std::time::Instant::now();
    let mut file = GgufFile::open(path)?;
    let mut config = HybridConfig::from_gguf(&file)?;
    config.kv_cache = options.kv_cache;
    let plan = plan::plan(
        &file,
        &config,
        device.recommended_max_working_set() as usize,
        options,
    )?;
    let (tokenizer, eos_ids) = tokenizer(&file, config.vocab)?;
    let chat_template = match file.metadata_value("tokenizer.chat_template") {
        Some(MetadataValue::String(t)) => Some(t.clone()),
        _ => None,
    };
    let chat_special_tokens = special_tokens(file.metadata());
    let before = device.allocated_bytes();
    let weights = weights::load(device, &mut file, &config)?;
    let model = HybridModel::new(device, config, weights, options.chunk, options.logit_rows)?;
    Ok(LoadedHybrid {
        model,
        tokenizer,
        eos_ids,
        chat_template,
        chat_special_tokens,
        load_time: start.elapsed(),
        plan,
        loaded_bytes: device.allocated_bytes().saturating_sub(before),
    })
}

/// The chat template embedded in a GGUF (no weights are read).
pub fn load_chat_template(path: impl AsRef<Path>) -> Result<Option<chat::ChatTemplate>> {
    let file = GgufFile::open(path)?;
    let (bos, eos) = special_tokens(file.metadata());
    let flint = matches!(
        file.metadata_value("general.architecture"),
        Some(MetadataValue::String(a)) if a == "flint"
    );
    match file.metadata_value("tokenizer.chat_template") {
        Some(MetadataValue::String(t)) => chat::ChatTemplate::with_special_tokens(t, &bos, &eos)
            .map(|t| Some(t.with_opens_thinking(flint))),
        _ => Ok(None),
    }
}

/// The vocabulary strings of `tokenizer.ggml.bos_token_id` and
/// `tokenizer.ggml.eos_token_id` (empty when absent): what a chat template
/// sees as `bos_token` and `eos_token`.
fn special_tokens(md: &std::collections::BTreeMap<String, MetadataValue>) -> (String, String) {
    let tokens: Vec<&str> = match md.get("tokenizer.ggml.tokens") {
        Some(MetadataValue::Array { values, .. }) => values
            .iter()
            .filter_map(|v| match v {
                MetadataValue::String(s) => Some(s.as_str()),
                _ => None,
            })
            .collect(),
        _ => Vec::new(),
    };
    let named = |key: &str| {
        config::uint(md, key)
            .ok()
            .and_then(|id| tokens.get(id).map(|t| (*t).to_owned()))
            .unwrap_or_default()
    };
    (
        named("tokenizer.ggml.bos_token_id"),
        named("tokenizer.ggml.eos_token_id"),
    )
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
    if text("tokenizer.ggml.model")? != "gpt2" {
        return Err(Error::Tokenizer(
            "hybrid GGUF must use the gpt2 model".into(),
        ));
    }
    // Qwen3.5 text is NFC-normalized before splitting; llama.cpp's Flint
    // tokenizer is not, so combining marks stay separate tokens.
    let (patterns, nfc): (&[&str], bool) = match text("tokenizer.ggml.pre")? {
        "qwen35" => (&[QWEN35_PATTERN], true),
        "flint" => (&FLINT_PATTERNS, false),
        "spark2_5" => (&SPARK_PATTERNS, false),
        other => {
            return Err(Error::Tokenizer(format!(
                "hybrid GGUF pre-tokenizer {other:?} is not supported (qwen35, flint, spark2_5)"
            )));
        }
    };
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
        Tokenizer::from_gguf_bpe_with_patterns(&tokens, &merges, &types, patterns, nfc)?;
    let eos = config::uint(md, "tokenizer.ggml.eos_token_id")? as u32;
    let mut eos_ids = vec![eos];
    // Architectures that list their end-of-generation tokens use that list.
    if let Some(MetadataValue::Array { values, .. }) = md.get("tokenizer.ggml.eog_token_ids") {
        for v in values {
            // Stored as i32 (GGUF array element type 5).
            let id = match v {
                MetadataValue::Int32(id) if *id >= 0 => *id as u32,
                MetadataValue::Uint32(id) => *id,
                _ => continue,
            };
            if !eos_ids.contains(&id) {
                eos_ids.push(id);
            }
        }
    }
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
