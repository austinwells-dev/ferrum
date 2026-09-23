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
        let tc = json(&dir.join("tokenizer_config.json"))?;
        let gc = json(&dir.join("generation_config.json"))?;
        let tokenizer = Tokenizer::from_file(dir.join("tokenizer.json"))?;
        let bad = |s: &str| Error::Tokenizer(format!("Qwen tokenizer/config mismatch: {s}"));
        validate_metadata(config, &tc, &gc)?;
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
    let bad = |s: &str| Error::Tokenizer(format!("Qwen tokenizer/config mismatch: {s}"));
    if tc["chat_template"].as_str() != Some(CHAT_TEMPLATE) {
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
