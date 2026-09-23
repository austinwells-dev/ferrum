//! Local Hugging Face tokenizer.json and GGUF tokenizer metadata support.
#![forbid(unsafe_code)]
use crate::{Error, Result};
use std::collections::{BTreeMap, HashSet};
pub struct Tokenizer(tokenizers::Tokenizer, Option<HashSet<u32>>);
impl Tokenizer {
    pub fn from_file(path: impl AsRef<std::path::Path>) -> Result<Self> {
        tokenizers::Tokenizer::from_file(path)
            .map(|t| Self(t, None))
            .map_err(|e| Error::Tokenizer(e.to_string()))
    }
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        tokenizers::Tokenizer::from_bytes(bytes)
            .map(|t| Self(t, None))
            .map_err(|e| Error::Tokenizer(e.to_string()))
    }
    pub(crate) fn from_gguf_bpe(
        tokens: &[String],
        merges: &[String],
        token_types: &[i32],
    ) -> Result<Self> {
        const QWEN_REGEX: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
        if tokens.is_empty() || tokens.len() != token_types.len() {
            return Err(Error::Tokenizer(
                "GGUF Qwen token/type arrays are empty or differ in length".into(),
            ));
        }
        let mut vocab = BTreeMap::new();
        let mut valid_ids = HashSet::with_capacity(tokens.len());
        let mut special_tokens = Vec::new();
        for (index, (token, &kind)) in tokens.iter().zip(token_types).enumerate() {
            let id = u32::try_from(index)
                .map_err(|_| Error::Tokenizer("GGUF token ID exceeds u32".into()))?;
            if vocab.insert(token.clone(), id).is_some() {
                return Err(Error::Tokenizer(format!(
                    "duplicate GGUF vocabulary token at ID {id}: {token:?}"
                )));
            }
            match kind {
                1 | 2 | 3 | 4 | 6 => {
                    valid_ids.insert(id);
                }
                5 => {}
                _ => {
                    return Err(Error::Tokenizer(format!(
                        "unsupported GGUF token type {kind} at ID {id}"
                    )));
                }
            }
            if matches!(kind, 3 | 4) {
                special_tokens.push(serde_json::json!({
                    "id": id,
                    "content": token,
                    "single_word": false,
                    "lstrip": false,
                    "rstrip": false,
                    "normalized": false,
                    "special": true
                }));
            }
        }
        let mut merge_pairs = Vec::with_capacity(merges.len());
        for (index, merge) in merges.iter().enumerate() {
            let mut parts = merge.split_ascii_whitespace();
            let first = parts.next();
            let second = parts.next();
            if first.is_none() || second.is_none() || parts.next().is_some() {
                return Err(Error::Tokenizer(format!(
                    "malformed GGUF Qwen BPE merge at index {index}"
                )));
            }
            merge_pairs.push(serde_json::json!([first.unwrap(), second.unwrap()]));
        }

        // This mirrors Qwen2Tokenizer: NFC, its model-specific regex split,
        // byte-level mapping without a second regex pass, and ByteLevel decode.
        let tokenizer_config = serde_json::json!({
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": special_tokens,
            "normalizer": {"type": "NFC"},
            "pre_tokenizer": {
                "type": "Sequence",
                "pretokenizers": [
                    {
                        "type": "Split",
                        "pattern": {"Regex": QWEN_REGEX},
                        "behavior": "Isolated",
                        "invert": false
                    },
                    {
                        "type": "ByteLevel",
                        "add_prefix_space": false,
                        "trim_offsets": true,
                        "use_regex": false
                    }
                ]
            },
            "post_processor": null,
            "decoder": {
                "type": "ByteLevel",
                "add_prefix_space": true,
                "trim_offsets": true,
                "use_regex": true
            },
            "model": {
                "type": "BPE",
                "dropout": null,
                "unk_token": null,
                "continuing_subword_prefix": null,
                "end_of_word_suffix": null,
                "fuse_unk": false,
                "byte_fallback": false,
                "ignore_merges": false,
                "vocab": vocab,
                "merges": merge_pairs
            }
        });
        let bytes = serde_json::to_vec(&tokenizer_config)
            .map_err(|e| Error::Tokenizer(format!("serialize GGUF tokenizer: {e}")))?;
        let tokenizer = tokenizers::Tokenizer::from_bytes(&bytes)
            .map_err(|e| Error::Tokenizer(format!("build GGUF Qwen tokenizer: {e}")))?;
        Ok(Self(tokenizer, Some(valid_ids)))
    }
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        self.0
            .encode(text, false)
            .map(|e| e.get_ids().to_vec())
            .map_err(|e| Error::Tokenizer(e.to_string()))
    }
    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        for &id in ids {
            if self.0.id_to_token(id).is_none() {
                return Err(Error::Tokenizer(format!("unknown token ID {id}")));
            }
        }
        self.0
            .decode(ids, false)
            .map_err(|e| Error::Tokenizer(e.to_string()))
    }
    pub fn decode_stream(
        &self,
    ) -> tokenizers::tokenizer::DecodeStream<
        '_,
        tokenizers::ModelWrapper,
        tokenizers::NormalizerWrapper,
        tokenizers::PreTokenizerWrapper,
        tokenizers::PostProcessorWrapper,
        tokenizers::DecoderWrapper,
    > {
        self.0.decode_stream(false)
    }
    pub fn is_defined(&self, id: u32) -> bool {
        self.1
            .as_ref()
            .map_or_else(|| self.0.id_to_token(id).is_some(), |ids| ids.contains(&id))
    }
    pub fn vocab_size(&self) -> usize {
        self.0.get_vocab_size(true)
    }
}
pub mod qwen;
