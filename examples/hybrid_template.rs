//! Render chat cases with a GGUF's embedded template.
//! usage: hybrid_template MODEL.gguf CASES.json > rendered.json
//! CASES.json: [{"messages": [...], "tools": [...]?, "vars": {...}?}, ...]
//! `vars` are template variables (`enable_thinking`, `reasoning_effort`),
//! mapped onto the template the way the runtime maps them.
fn main() -> ferrum::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let template = ferrum::hybrid::load_chat_template(&args[1])?.expect("GGUF has a chat template");
    let cases: Vec<serde_json::Value> =
        serde_json::from_str(&std::fs::read_to_string(&args[2]).expect("read cases"))
            .expect("parse cases");
    let mut out = Vec::new();
    for case in cases {
        let messages = case["messages"].as_array().expect("messages").clone();
        let tools = case.get("tools").and_then(|t| t.as_array()).cloned();
        let mut vars = case
            .get("vars")
            .and_then(|v| v.as_object())
            .cloned()
            .unwrap_or_default();
        ferrum::hybrid::runtime::normalize_thinking(template.source(), &mut vars);
        out.push(template.render(&messages, tools.as_deref(), true, &vars)?);
    }
    println!("{}", serde_json::to_string(&out).expect("json"));
    Ok(())
}
