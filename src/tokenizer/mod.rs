//! Local Hugging Face tokenizer.json support; no numerical runtime dependency.
#![forbid(unsafe_code)]
use crate::{Error, Result};
pub struct Tokenizer(tokenizers::Tokenizer);
impl Tokenizer {
    pub fn from_file(path: impl AsRef<std::path::Path>) -> Result<Self> {
        tokenizers::Tokenizer::from_file(path)
            .map(Self)
            .map_err(|e| Error::Tokenizer(e.to_string()))
    }
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        tokenizers::Tokenizer::from_bytes(bytes)
            .map(Self)
            .map_err(|e| Error::Tokenizer(e.to_string()))
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
    pub fn vocab_size(&self) -> usize {
        self.0.get_vocab_size(true)
    }
}
