//! Tokenizer checks for hybrid GGUFs.
//! usage: hybrid_tokenize MODEL.gguf FILE        (print IDs)
//!        hybrid_tokenize MODEL.gguf --decode ID... (print text)
fn main() -> ferrum::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let tokenizer = ferrum::hybrid::load_tokenizer(&args[1])?;
    if args[2] == "--decode" {
        let ids: Vec<u32> = args[3..]
            .iter()
            .map(|a| {
                a.trim_matches(|c| c == ',' || c == '[' || c == ']')
                    .parse()
                    .expect("token ID")
            })
            .collect();
        println!("{:?}", tokenizer.decode(&ids)?);
        return Ok(());
    }
    let text = std::fs::read_to_string(&args[2]).expect("read text");
    println!("{:?}", tokenizer.encode(&text)?);
    Ok(())
}
