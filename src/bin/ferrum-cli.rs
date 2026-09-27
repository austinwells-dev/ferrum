//! Interactive terminal chat with a Qwen3.5-family GGUF.
//!
//! ferrum-cli --model FILE.gguf [--context N|auto] [--system TEXT]
//!            [--temperature T] [--top-p P] [--top-k K] [--min-p P]
//!            [--presence-penalty X] [--repetition-penalty X] [--seed N]
//!            [--max-tokens N] [--no-think] [--reasoning-effort LEVEL]
//!            [--reasoning-budget N] [--hide-thinking]
//!            [--draft mtp|DIR] [--draft-max N] [--draft-quant q8_0|q4_0]
//!            [--draft-context N] [--draft-p-min P]
//!
//! Commands: /reset, /think on|off, /stats, /exit. End a line with `\` to
//! continue the message on the next line.
use ferrum::{
    Error, Result,
    hybrid::{
        plan::PlanOptions,
        runtime::{ChatRequest, Delta, Runtime, SpecOptions},
    },
};
use serde_json::{Map, Value as Json, json};
use std::io::{BufRead, Write};

struct Options {
    model: String,
    context: Option<usize>,
    system: Option<String>,
    max_tokens: usize,
    think: bool,
    effort: Option<String>,
    show_thinking: bool,
    reasoning_budget: Option<usize>,
    overrides: Vec<(String, String)>,
    spec: Option<SpecOptions>,
}

fn parse() -> Result<Options> {
    let mut o = Options {
        model: String::new(),
        context: None,
        system: None,
        max_tokens: 32768,
        think: true,
        effort: None,
        show_thinking: true,
        reasoning_budget: None,
        overrides: Vec::new(),
        spec: None,
    };
    let mut args = std::env::args().skip(1);
    while let Some(key) = args.next() {
        match key.as_str() {
            "--no-think" => o.think = false,
            "--hide-thinking" => o.show_thinking = false,
            "-h" | "--help" => {
                println!(
                    "{}",
                    include_str!("ferrum-cli.rs")
                        .lines()
                        .take(14)
                        .map(|l| l.trim_start_matches("//!").trim_start_matches(' '))
                        .collect::<Vec<_>>()
                        .join("\n")
                );
                std::process::exit(0);
            }
            _ => {
                let value = args
                    .next()
                    .ok_or_else(|| Error::Parameter(format!("missing value for {key}")))?;
                let number = || {
                    value
                        .parse::<usize>()
                        .map_err(|_| Error::Parameter(format!("invalid {key}: {value}")))
                };
                if SpecOptions::apply_flag(&mut o.spec, &key, &value)? {
                    continue;
                }
                match key.as_str() {
                    "--model" | "-m" => o.model = value.clone(),
                    "--context" | "-c" => {
                        o.context = if value == "auto" {
                            None
                        } else {
                            Some(number()?)
                        }
                    }
                    "--system" => o.system = Some(value.clone()),
                    "--max-tokens" | "-n" => o.max_tokens = number()?,
                    "--reasoning-effort" => o.effort = Some(value.clone()),
                    "--reasoning-budget" => o.reasoning_budget = Some(number()?),
                    "--temperature"
                    | "--top-p"
                    | "--top-k"
                    | "--min-p"
                    | "--presence-penalty"
                    | "--frequency-penalty"
                    | "--repetition-penalty"
                    | "--seed" => o.overrides.push((key.clone(), value.clone())),
                    _ => {
                        return Err(Error::Parameter(format!(
                            "unknown option {key} (see --help)"
                        )));
                    }
                }
            }
        }
    }
    if o.model.is_empty() {
        return Err(Error::Parameter(
            "usage: ferrum-cli --model FILE.gguf (see --help)".into(),
        ));
    }
    Ok(o)
}

fn main() {
    if let Err(e) = run() {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    let o = parse()?;
    eprintln!("loading {} ...", o.model);
    let mut rt = Runtime::load_speculative(
        &o.model,
        PlanOptions {
            context: o.context,
            ..Default::default()
        },
        o.spec.as_ref(),
    )?;
    if let Some(drafter) = rt.session.drafter() {
        eprintln!(
            "speculative decoding: {} ({} drafts per step, {:.2} GiB)",
            drafter.name(),
            drafter.max_drafts(),
            rt.draft_loaded_bytes as f64 / (1u64 << 30) as f64
        );
    }
    rt.warmup()?;
    let mut sampling = rt.default_sampling.clone();
    for (key, value) in &o.overrides {
        let f = || {
            value
                .parse::<f32>()
                .map_err(|_| Error::Parameter(format!("invalid {key}: {value}")))
        };
        match key.as_str() {
            "--temperature" => sampling.temperature = f()?,
            "--top-p" => sampling.top_p = f()?,
            "--top-k" => sampling.top_k = f()? as usize,
            "--min-p" => sampling.min_p = f()?,
            "--presence-penalty" => sampling.presence_penalty = f()?,
            "--frequency-penalty" => sampling.frequency_penalty = f()?,
            "--repetition-penalty" => sampling.repetition_penalty = f()?,
            "--seed" => sampling.seed = f()? as u64,
            _ => {}
        }
    }
    sampling.validate()?;
    let plan = &rt.loaded.plan;
    eprintln!(
        "{}: context {} tokens (max {}), {:.1} GiB weights, loaded in {:.1} s",
        rt.model_name,
        plan.context,
        plan.max_context,
        plan.weights as f64 / (1u64 << 30) as f64,
        rt.loaded.load_time.as_secs_f64()
    );
    eprintln!(
        "sampling: temperature {} top-p {} top-k {} min-p {}; /reset /think on|off /stats /exit",
        sampling.temperature, sampling.top_p, sampling.top_k, sampling.min_p
    );
    let mut messages: Vec<Json> = Vec::new();
    if let Some(system) = &o.system {
        messages.push(json!({"role": "system", "content": system}));
    }
    let mut think = o.think;
    let stdin = std::io::stdin();
    let mut lines = stdin.lock().lines();
    let mut last_stats = String::from("no turns yet");
    loop {
        print!("\n\x1b[1m>\x1b[0m ");
        std::io::stdout().flush().ok();
        let mut input = String::new();
        loop {
            let Some(line) = lines.next() else {
                return Ok(());
            };
            let line = line.map_err(|e| Error::Parameter(e.to_string()))?;
            if let Some(stripped) = line.strip_suffix('\\') {
                input.push_str(stripped);
                input.push('\n');
                continue;
            }
            input.push_str(&line);
            break;
        }
        let input = input.trim();
        match input {
            "" => continue,
            "/exit" | "/quit" => return Ok(()),
            "/reset" => {
                messages.retain(|m| m["role"] == "system");
                rt.session.reset();
                eprintln!("(conversation cleared)");
                continue;
            }
            "/stats" => {
                eprintln!(
                    "{last_stats}; session {} / {} tokens",
                    rt.session.len(),
                    rt.session.capacity()
                );
                continue;
            }
            "/think on" | "/think off" => {
                think = input.ends_with("on");
                eprintln!("(thinking {})", if think { "on" } else { "off" });
                continue;
            }
            _ => {}
        }
        messages.push(json!({"role": "user", "content": input}));
        let mut vars = Map::new();
        vars.insert("enable_thinking".into(), Json::Bool(think));
        if let Some(effort) = &o.effort {
            vars.insert("reasoning_effort".into(), Json::String(effort.clone()));
        }
        let request = ChatRequest {
            messages: messages.clone(),
            tools: Vec::new(),
            max_tokens: o.max_tokens,
            sampling: sampling.clone(),
            template_vars: vars,
            stop: Vec::new(),
            reasoning_budget: o.reasoning_budget,
            raw_reasoning: false,
        };
        let mut in_reasoning = false;
        let show = o.show_thinking;
        let result = rt.chat(&request, |delta| {
            let mut out = std::io::stdout();
            match delta {
                Delta::Reasoning(t) if show => {
                    if !in_reasoning {
                        print!("\x1b[2m");
                        in_reasoning = true;
                    }
                    print!("{t}");
                }
                Delta::Reasoning(_) => {}
                Delta::Content(t) => {
                    if in_reasoning {
                        print!("\x1b[0m\n\n");
                        in_reasoning = false;
                    }
                    print!("{t}");
                }
            }
            out.flush().ok();
            Ok(())
        });
        print!("\x1b[0m");
        let result = match result {
            Ok(r) => r,
            Err(e) => {
                messages.pop();
                eprintln!("\nerror: {e}");
                continue;
            }
        };
        println!();
        let c = &result.completion;
        last_stats = format!(
            "prompt {} tokens ({} reused), prefill {:.0} tok/s; {} generated at {:.1} tok/s; stop {:?}",
            c.prompt_tokens,
            c.reused_tokens,
            (c.prompt_tokens - c.reused_tokens) as f64 / c.prefill.as_secs_f64().max(1e-9),
            c.tokens.len(),
            c.tokens.len() as f64 / c.decode.as_secs_f64().max(1e-9),
            c.stop
        );
        let spec = &rt.session.spec_stats;
        if spec.steps > 0 {
            last_stats += &format!(
                "; speculative acceptance {:.2} ({} of {} drafts)",
                spec.acceptance_length(),
                spec.accepted,
                spec.drafted
            );
        }
        eprintln!("\x1b[2m[{last_stats}]\x1b[0m");
        let mut reply = json!({"role": "assistant", "content": result.output.content});
        if let Some(reasoning) = result.output.reasoning {
            reply["reasoning_content"] = Json::String(reasoning);
        }
        messages.push(reply);
    }
}
