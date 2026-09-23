//! Compact, model-agnostic GGUF architecture and tensor inventory.
use ferrum::{Result, loader::gguf::GgufFile};
use std::collections::BTreeMap;

fn main() -> Result<()> {
    let path = std::env::args()
        .nth(1)
        .expect("usage: gguf_inspect FILE.gguf");
    let file = GgufFile::open(path)?;
    println!(
        "GGUF v{}; {} tensors; {} bytes",
        file.version(),
        file.tensors().len(),
        file.file_len()
    );
    for (name, value) in file.metadata() {
        if name.starts_with("qwen")
            || name == "general.architecture"
            || name == "general.name"
            || name == "tokenizer.ggml.model"
            || name == "tokenizer.ggml.pre"
            || name.starts_with("tokenizer.ggml.") && name.ends_with("_token_id")
        {
            println!("{name}={value:?}");
        }
    }
    let mut types = BTreeMap::<u32, usize>::new();
    for info in file.tensors().values() {
        *types.entry(info.type_id).or_default() += 1;
    }
    println!("tensor type counts: {types:?}");
    for info in file.tensors().values().take(35) {
        println!("{} {:?} type {}", info.name, info.dimensions, info.type_id);
    }
    Ok(())
}
