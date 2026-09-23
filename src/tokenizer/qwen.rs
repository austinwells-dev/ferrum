//! The official template's single system/user, no-tools branch.
use super::Tokenizer;
use crate::{Error, Result, model::qwen::QwenConfig};
use serde_json::Value;
use std::path::Path;
pub const DEFAULT_SYSTEM: &str = "You are a helpful assistant.";
pub const CHAT_TEMPLATE: &str = include_str!("qwen_chat_template.jinja");
pub fn chat_prompt(system: &str, user: &str) -> String {
    format!(
        "<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"
    )
}
pub struct QwenTokenizer {
    pub tokenizer: Tokenizer,
    pub eos_ids: Vec<u32>,
    pub pad_id: u32,
}
fn json(path: &Path) -> Result<Value> {
    let bytes =
        std::fs::read(path).map_err(|e| Error::Tokenizer(format!("{}: {e}", path.display())))?;
    serde_json::from_slice(&bytes).map_err(|e| Error::Tokenizer(format!("{}: {e}", path.display())))
}
impl QwenTokenizer {
    /// Official Qwen3 simple system/user chat branch. Its thinking-enabled
    /// generation prefix matches `chat_prompt`; tool calls are outside this API.
    pub fn load_qwen3(dir: &Path, vocab_size: usize, bos_id: u32, eos_id: u32) -> Result<Self> {
        let tc = json(&dir.join("tokenizer_config.json"))?;
        let gc = json(&dir.join("generation_config.json"))?;
        let bad = |message: &str| Error::Tokenizer(format!("Qwen3 tokenizer mismatch: {message}"));
        if tc["tokenizer_class"] != "Qwen2Tokenizer"
            || tc["add_bos_token"] != false
            || !tc["bos_token"].is_null()
            || tc["eos_token"] != "<|im_end|>"
            || tc["pad_token"] != "<|endoftext|>"
            || tc["clean_up_tokenization_spaces"] != false
            || !tc["chat_template"].as_str().is_some_and(|template| {
                template.contains("enable_thinking") && template.contains("<|im_start|>")
            })
        {
            return Err(bad("unsupported special-token or chat-template policy"));
        }
        if gc["bos_token_id"] != bos_id
            || gc["pad_token_id"] != bos_id
            || gc["eos_token_id"] != serde_json::json!([eos_id, bos_id])
        {
            return Err(bad("generation special-token IDs differ from model config"));
        }
        let tokenizer = Tokenizer::from_file(dir.join("tokenizer.json"))?;
        if tokenizer.vocab_size() > vocab_size {
            return Err(bad("tokenizer vocabulary exceeds embedding rows"));
        }
        for (token, id) in [
            ("<|endoftext|>", bos_id),
            ("<|im_start|>", 151644),
            ("<|im_end|>", eos_id),
        ] {
            if tokenizer.0.token_to_id(token) != Some(id) || tokenizer.encode(token)? != [id] {
                return Err(bad(&format!("invalid special token {token}")));
            }
        }
        Ok(Self {
            tokenizer,
            eos_ids: vec![eos_id, bos_id],
            pad_id: bos_id,
        })
    }

    pub(crate) fn from_gguf(
        tokens: &[String],
        merges: &[String],
        token_types: &[i32],
        vocab_size: usize,
        bos_id: u32,
        eos_id: u32,
        pad_id: u32,
    ) -> Result<Self> {
        let bad =
            |message: &str| Error::Tokenizer(format!("GGUF Qwen tokenizer mismatch: {message}"));
        if tokens.len() > vocab_size {
            return Err(bad("tokenizer vocabulary exceeds embedding rows"));
        }
        for id in [bos_id, eos_id, pad_id] {
            if id as usize >= tokens.len() {
                return Err(bad("BOS/EOS/PAD ID is outside tokenizer entries"));
            }
        }
        if tokens.get(bos_id as usize).map(String::as_str) != Some("<|endoftext|>")
            || tokens.get(eos_id as usize).map(String::as_str) != Some("<|im_end|>")
        {
            return Err(bad("unexpected Qwen BOS/EOS token strings"));
        }
        let tokenizer = Tokenizer::from_gguf_bpe(tokens, merges, token_types)?;
        for id in 0..tokens.len() as u32 {
            if tokenizer.0.id_to_token(id).is_none() {
                return Err(bad("tokenizer ID hole"));
            }
        }
        for (text, id) in [
            ("<|endoftext|>", bos_id),
            ("<|im_start|>", 151644),
            ("<|im_end|>", eos_id),
        ] {
            if tokenizer.0.token_to_id(text) != Some(id) || tokenizer.encode(text)? != [id] {
                return Err(bad(&format!(
                    "invalid special token {text}, expected ID {id}"
                )));
            }
        }
        Ok(Self {
            tokenizer,
            eos_ids: vec![eos_id, bos_id],
            pad_id,
        })
    }

    pub fn load(dir: &Path, config: &QwenConfig) -> Result<Self> {
        Self::load_inner(dir, config, false)
    }

    /// Load the MLX affine-Q4 Qwen repository variant. Some published MLX
    /// conversions omit generation_config.json; in that case the already
    /// validated Qwen config supplies the pinned BOS/EOS policy.
    pub(crate) fn load_mlx_affine4(dir: &Path, config: &QwenConfig) -> Result<Self> {
        Self::load_inner(dir, config, true)
    }

    fn load_inner(dir: &Path, config: &QwenConfig, allow_derived_generation: bool) -> Result<Self> {
        let tc = json(&dir.join("tokenizer_config.json"))?;
        let generation_path = dir.join("generation_config.json");
        let gc = if generation_path.is_file() {
            json(&generation_path)?
        } else if allow_derived_generation {
            serde_json::json!({
                "bos_token_id": config.bos_token_id,
                "pad_token_id": config.bos_token_id,
                "eos_token_id": [config.eos_token_id, config.bos_token_id]
            })
        } else {
            json(&generation_path)?
        };
        let tokenizer = Tokenizer::from_file(dir.join("tokenizer.json"))?;
        let bad = |s: &str| Error::Tokenizer(format!("Qwen tokenizer/config mismatch: {s}"));
        validate_metadata_inner(config, &tc, &gc, allow_derived_generation)?;
        // This checkpoint pads 151665 tokenizer IDs to 151936 embedding rows.
        if tokenizer.vocab_size() != 151665 || config.vocab_size != 151936 {
            return Err(bad(
                "expected 151665 tokenizer entries / 151936 embedding rows",
            ));
        }
        for id in 0..tokenizer.vocab_size() as u32 {
            if tokenizer.0.id_to_token(id).is_none() {
                return Err(bad("tokenizer ID hole"));
            }
        }
        for (text, id) in [
            ("<|endoftext|>", 151643),
            ("<|im_start|>", 151644),
            ("<|im_end|>", 151645),
        ] {
            if tokenizer.0.token_to_id(text) != Some(id) || tokenizer.encode(text)? != [id] {
                return Err(bad(&format!(
                    "invalid special token {text}, expected ID {id}"
                )));
            }
        }
        Ok(Self {
            tokenizer,
            eos_ids: vec![151645, 151643],
            pad_id: 151643,
        })
    }
    pub fn encode_prompt(
        &self,
        prompt: &str,
        system: &str,
        raw: bool,
    ) -> Result<(String, Vec<u32>)> {
        let text = if raw {
            prompt.to_owned()
        } else {
            chat_prompt(system, prompt)
        };
        let ids = self.tokenizer.encode(&text)?;
        Ok((text, ids))
    }
}

/// Validate the pinned text/special-token policy without loading vocabulary payloads.
pub fn validate_metadata(config: &QwenConfig, tc: &Value, gc: &Value) -> Result<()> {
    validate_metadata_inner(config, tc, gc, false)
}

fn validate_metadata_inner(
    config: &QwenConfig,
    tc: &Value,
    gc: &Value,
    allow_mlx_template: bool,
) -> Result<()> {
    let bad = |s: &str| Error::Tokenizer(format!("Qwen tokenizer/config mismatch: {s}"));
    let template = tc["chat_template"].as_str();
    let template_matches = chat_template_matches(template, allow_mlx_template);
    if !template_matches {
        return Err(bad(
            "unsupported chat_template; expected pinned official template",
        ));
    }
    if tc["add_bos_token"] != false
        || !tc["bos_token"].is_null()
        || tc["eos_token"] != "<|im_end|>"
        || tc["pad_token"] != "<|endoftext|>"
        || tc["clean_up_tokenization_spaces"] != false
        || tc["split_special_tokens"] != false
        || tc["add_prefix_space"] != false
        || tc["errors"] != "replace"
        || tc["tokenizer_class"] != "Qwen2Tokenizer"
    {
        return Err(bad("unsupported BOS/EOS/PAD or cleanup policy"));
    }
    if config.bos_token_id != 151643
        || config.eos_token_id != 151645
        || gc["bos_token_id"] != 151643
        || gc["pad_token_id"] != 151643
        || gc["eos_token_id"] != serde_json::json!([151645, 151643])
    {
        return Err(bad("invalid model/generation special token IDs"));
    }
    Ok(())
}

fn chat_template_matches(template: Option<&str>, allow_mlx_variant: bool) -> bool {
    template == Some(CHAT_TEMPLATE)
        || (allow_mlx_variant
            && template.is_some_and(|template| {
                template.replace(
                    "{{\\\"name\\\": <function-name>, \\\"arguments\\\": <args-json-object>}}",
                    "{\\\"name\\\": <function-name>, \\\"arguments\\\": <args-json-object>}",
                ) == CHAT_TEMPLATE
            }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mlx_chat_template_accepts_only_its_documentation_brace_variant() {
        let mlx_template = CHAT_TEMPLATE.replace(
            "{\\\"name\\\": <function-name>, \\\"arguments\\\": <args-json-object>}",
            "{{\\\"name\\\": <function-name>, \\\"arguments\\\": <args-json-object>}}",
        );
        assert!(chat_template_matches(Some(CHAT_TEMPLATE), false));
        assert!(!chat_template_matches(Some(&mlx_template), false));
        assert!(chat_template_matches(Some(&mlx_template), true));
        assert!(!chat_template_matches(Some("unrecognized template"), true));
    }
}
