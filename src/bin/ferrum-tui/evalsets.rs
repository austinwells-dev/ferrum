//! Installing and removing the intelligence question sets, fetched with the
//! system `curl` from the Hugging Face datasets-server API and written as the
//! JSON lines [`eval`] reads. Each is a fixed, seeded sample:
//! - standard: GSM8K (MIT), MMLU (MIT), IFEval (Apache-2.0), CRUXEval (MIT)
//! - hard: MATH-500 levels 3-5 with plain-number answers (MIT), MMLU-Pro (MIT,
//!   ten options), the IFEval prompts with the most rules and the longest
//!   CRUXEval functions
use crate::eval::{self, CATEGORIES, Difficulty};
use crate::*;
use rand::SeedableRng;
use rand::seq::SliceRandom;
use rand_chacha::ChaCha8Rng;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
    mpsc::{Receiver, channel},
};
use std::thread::JoinHandle;

/// Questions kept per set.
pub const PER_SET: usize = 100;
/// The same seed always picks the same questions.
const SEED: u64 = 7;
const API: &str = "https://datasets-server.huggingface.co/rows";
/// Splits up to this size are fetched whole; larger ones in random windows.
const FETCH_ALL: usize = 2000;
/// Rows per random window of a large split: few requests, many subjects.
const WINDOW: usize = 4;
/// Pause between requests, so the anonymous API does not rate-limit us.
const PACE: Duration = Duration::from_millis(500);
/// Attempts per request; rate limits (HTTP 429) wait longer each time.
const ATTEMPTS: u64 = 6;

struct Dataset {
    cat: &'static str,
    name: &'static str,
    config: &'static str,
    split: &'static str,
    /// One datasets-server row (with `row_idx`) as an item, or None to skip it.
    convert: fn(&Json) -> Option<Json>,
    /// How hard a row is: when set, the hardest rows are kept. Only for
    /// splits fetched whole.
    rank: Option<fn(&Json) -> usize>,
}

const STANDARD: [Dataset; 5] = [
    Dataset {
        cat: "math",
        name: "openai/gsm8k",
        config: "main",
        split: "test",
        convert: gsm8k,
        rank: None,
    },
    Dataset {
        cat: "knowledge",
        name: "cais/mmlu",
        config: "all",
        split: "test",
        convert: mmlu,
        rank: None,
    },
    Dataset {
        cat: "instructions",
        name: "google/IFEval",
        config: "default",
        split: "train",
        convert: ifeval,
        rank: None,
    },
    Dataset {
        cat: "code",
        name: "cruxeval-org/cruxeval",
        config: "default",
        split: "test",
        convert: cruxeval,
        rank: None,
    },
    Dataset {
        cat: "coding",
        name: "evalplus/mbppplus",
        config: "default",
        split: "test",
        convert: mbpp,
        rank: None,
    },
];

const HARD: [Dataset; 5] = [
    Dataset {
        cat: "math",
        name: "HuggingFaceH4/MATH-500",
        config: "default",
        split: "test",
        convert: math500,
        rank: Some(math_level),
    },
    Dataset {
        cat: "knowledge",
        name: "TIGER-Lab/MMLU-Pro",
        config: "default",
        split: "test",
        convert: mmlu_pro,
        rank: None,
    },
    Dataset {
        cat: "instructions",
        name: "google/IFEval",
        config: "default",
        split: "train",
        convert: ifeval,
        rank: Some(rule_count),
    },
    Dataset {
        cat: "code",
        name: "cruxeval-org/cruxeval",
        config: "default",
        split: "test",
        convert: cruxeval,
        rank: Some(code_length),
    },
    Dataset {
        cat: "coding",
        name: "openai/openai_humaneval",
        config: "openai_humaneval",
        split: "test",
        convert: humaneval,
        rank: None,
    },
];

fn datasets(difficulty: Difficulty) -> &'static [Dataset; 5] {
    match difficulty {
        Difficulty::Standard => &STANDARD,
        Difficulty::Hard => &HARD,
    }
}

/// MATH problems of level 3-5 whose answer is a plain number (no LaTeX), so
/// the number check can grade them.
fn math500(r: &Json) -> Option<Json> {
    let row = &r["row"];
    if math_level(r) < 3 {
        return None;
    }
    let answer = row["answer"].as_str()?.trim().replace(',', "");
    let plain = !answer.is_empty()
        && answer
            .chars()
            .all(|c| c.is_ascii_digit() || c == '.' || c == '-')
        && answer.parse::<f64>().is_ok();
    plain.then(|| {
        json!({
            "id": format!("math500-{}", r["row_idx"]), "cat": "math", "check": "number",
            "answer": answer, "prompt": row["problem"].as_str().unwrap_or("").trim(),
            "max_tokens": 1024,
        })
    })
}

/// MATH levels come as strings ("4"); accept numbers too.
fn math_level(r: &Json) -> usize {
    let level = &r["row"]["level"];
    level
        .as_u64()
        .map(|n| n as usize)
        .or_else(|| level.as_str().and_then(|s| s.trim().parse().ok()))
        .unwrap_or(0)
}

fn mmlu_pro(r: &Json) -> Option<Json> {
    let row = &r["row"];
    let letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'];
    let options: Vec<String> = row["options"]
        .as_array()?
        .iter()
        .zip(letters)
        .map(|(c, l)| format!("{l}. {}", c.as_str().unwrap_or("")))
        .collect();
    Some(json!({
        "id": format!("mmlupro-{}", row["question_id"]), "cat": "knowledge", "check": "choice",
        "answer": row["answer"].as_str()?,
        "prompt": format!(
            "A question about {}.\n\n{}\n\n{}",
            row["category"].as_str()?,
            row["question"].as_str()?.trim(),
            options.join("\n")
        ),
        "max_tokens": 384,
    }))
}

fn rule_count(r: &Json) -> usize {
    r["row"]["instruction_id_list"]
        .as_array()
        .map_or(0, Vec::len)
}

fn code_length(r: &Json) -> usize {
    r["row"]["code"].as_str().map_or(0, str::len)
}

/// MBPP: a task and one of its asserts shown, all of them run.
fn mbpp(r: &Json) -> Option<Json> {
    let row = &r["row"];
    let tests: Vec<&str> = row["test_list"]
        .as_array()?
        .iter()
        .filter_map(Json::as_str)
        .collect();
    let imports: Vec<&str> = row["test_imports"]
        .as_array()
        .map(|a| a.iter().filter_map(Json::as_str).collect())
        .unwrap_or_default();
    Some(json!({
        "id": format!("mbpp-{}", row["task_id"]), "cat": "coding", "check": "program",
        "prompt": format!(
            "{}\n\nYour function must pass tests like this one:\n\n```python\n{}\n```",
            row["prompt"].as_str()?.trim(),
            tests.first()?
        ),
        "test": format!("{}\n{}", imports.join("\n"), tests.join("\n")),
        "max_tokens": 768,
    }))
}

/// HumanEval: complete the function; its `check` runs on the result.
fn humaneval(r: &Json) -> Option<Json> {
    let row = &r["row"];
    Some(json!({
        "id": format!("humaneval-{}", r["row_idx"]), "cat": "coding", "check": "program",
        "prompt": format!(
            "Complete this Python function:\n\n```python\n{}```",
            row["prompt"].as_str()?
        ),
        "test": format!("{}\n\ncheck({})\n", row["test"].as_str()?, row["entry_point"].as_str()?),
        "max_tokens": 1024,
    }))
}

// ---- tool calling: BFCL ----

const BFCL: &str = "https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard/resolve/main";

/// One call per request (choosing among several tools for `multiple`), or
/// several calls in one turn for the hard sets.
fn bfcl_sets(difficulty: Difficulty) -> [&'static str; 2] {
    match difficulty {
        Difficulty::Standard => ["simple", "multiple"],
        Difficulty::Hard => ["parallel", "parallel_multiple"],
    }
}

/// BFCL's Python-flavoured schema types as JSON Schema.
fn json_schema(v: &Json) -> Json {
    match v {
        Json::Object(o) => Json::Object(
            o.iter()
                .filter_map(|(k, v)| {
                    if k == "type" {
                        let t = match v.as_str()? {
                            "dict" => "object",
                            "float" => "number",
                            "tuple" => "array",
                            "any" => return None,
                            other => other,
                        };
                        Some((k.clone(), json!(t)))
                    } else {
                        Some((k.clone(), json_schema(v)))
                    }
                })
                .collect(),
        ),
        Json::Array(a) => Json::Array(a.iter().map(json_schema).collect()),
        other => other.clone(),
    }
}

/// A BFCL question and its possible answers as an item. Dotted names
/// become underscores, as for OpenAI models.
fn bfcl_item(q: &Json, truth: &Json) -> Option<Json> {
    let prompt: Vec<&str> = q["question"]
        .get(0)?
        .as_array()?
        .iter()
        .filter_map(|m| m["content"].as_str())
        .collect();
    let tools: Vec<Json> = q["function"]
        .as_array()?
        .iter()
        .map(|f| {
            json!({"type": "function", "function": {
                "name": f["name"].as_str().unwrap_or("").replace('.', "_"),
                "description": f["description"],
                "parameters": json_schema(&f["parameters"]),
            }})
        })
        .collect();
    let calls: Vec<Json> = truth
        .as_array()?
        .iter()
        .filter_map(|c| {
            let (name, args) = c.as_object()?.iter().next()?;
            Some(json!({"name": name.replace('.', "_"), "args": args}))
        })
        .collect();
    Some(json!({
        "id": format!("bfcl-{}", q["id"].as_str()?), "cat": "tools", "check": "calls",
        "prompt": prompt.join("\n"), "tools": tools, "calls": calls,
    }))
}

/// Half the questions from each of the difficulty's two BFCL sets.
fn sample_bfcl(
    difficulty: Difficulty,
    progress: &impl Fn(String),
    stop: &AtomicBool,
) -> Result<String, String> {
    let mut rng = ChaCha8Rng::seed_from_u64(SEED ^ eval_hash("tools"));
    let lines_of = |b: &[u8]| -> Vec<Json> {
        String::from_utf8_lossy(b)
            .lines()
            .filter_map(|l| serde_json::from_str(l).ok())
            .collect()
    };
    let mut lines = Vec::new();
    for set in bfcl_sets(difficulty) {
        progress(format!("tools · downloading BFCL {set}"));
        let questions = lines_of(&curl(
            &format!("{BFCL}/BFCL_v3_{set}.json"),
            "tools",
            progress,
            stop,
        )?);
        let answers = lines_of(&curl(
            &format!("{BFCL}/possible_answer/BFCL_v3_{set}.json"),
            "tools",
            progress,
            stop,
        )?);
        let mut items: Vec<String> = questions
            .iter()
            .filter_map(|q| {
                let truth = answers.iter().find(|a| a["id"] == q["id"])?;
                bfcl_item(q, &truth["ground_truth"]).and_then(checked_line)
            })
            .collect();
        items.shuffle(&mut rng);
        lines.extend(items.into_iter().take(PER_SET / 2));
    }
    if lines.is_empty() {
        return Err("BFCL: no usable questions".into());
    }
    lines.shuffle(&mut rng);
    Ok(lines.join("\n") + "\n")
}

// ---- long context: generated ----

const NAMES: [&str; 16] = [
    "Ada", "Bruno", "Chiara", "Dmitri", "Elena", "Farid", "Greta", "Hiro", "Ines", "Jonas",
    "Keiko", "Lars", "Mina", "Nadia", "Omar", "Priya",
];
const VERBS: [&str; 12] = [
    "moved",
    "inspected",
    "repaired",
    "labelled",
    "weighed",
    "counted",
    "cleaned",
    "photographed",
    "packed",
    "unloaded",
    "logged",
    "returned",
];
const THINGS: [&str; 16] = [
    "crates of glassware",
    "spare pumps",
    "the cold-storage racks",
    "two pallets of rice",
    "the backup generator",
    "a box of fuses",
    "the delivery van",
    "the loading ramp",
    "the inventory tablets",
    "three drums of oil",
    "the night shift's keys",
    "the fire doors",
    "a roll of copper wire",
    "the conveyor belt",
    "the scales",
    "the visitor badges",
];
const PLACES: [&str; 12] = [
    "dock four",
    "the north warehouse",
    "the mezzanine",
    "bay seven",
    "the yard",
    "the cold room",
    "the east gate",
    "the workshop",
    "aisle twelve",
    "the office",
    "the rail siding",
    "the roof",
];
const TIMES: [&str; 8] = [
    "before dawn",
    "at noon",
    "after the rain stopped",
    "during the inspection",
    "late in the evening",
    "on the second shift",
    "while the trucks waited",
    "just before closing",
];
const CITIES: [&str; 20] = [
    "Lisbon",
    "Oslo",
    "Nairobi",
    "Quito",
    "Hanoi",
    "Tbilisi",
    "Perth",
    "Montreal",
    "Cusco",
    "Tunis",
    "Riga",
    "Busan",
    "Accra",
    "Lyon",
    "Bergen",
    "Kraków",
    "Valparaíso",
    "Osaka",
    "Ghent",
    "Porto",
];

/// One filler sentence of a warehouse log.
fn filler(rng: &mut ChaCha8Rng) -> String {
    use rand::Rng;
    let pick = |rng: &mut ChaCha8Rng, xs: &[&'static str]| xs[rng.random_range(0..xs.len())];
    format!(
        "{} {} {} at {} {}.",
        pick(rng, &NAMES),
        pick(rng, &VERBS),
        pick(rng, &THINGS),
        pick(rng, &PLACES),
        pick(rng, &TIMES)
    )
}

/// Questions about a fact (a vault code) hidden at five depths in logs of
/// two lengths. The hard set uses longer logs and hides codes for other
/// vaults too, so the model has to find the right one.
fn context_set(difficulty: Difficulty) -> String {
    use rand::Rng;
    let mut rng = ChaCha8Rng::seed_from_u64(SEED ^ eval_hash("context"));
    let (lengths, decoys) = match difficulty {
        Difficulty::Standard => ([4096usize, 8192], 0),
        Difficulty::Hard => ([16384, 32768], 4),
    };
    let mut lines = Vec::new();
    for tokens in lengths {
        for depth in [0.1, 0.3, 0.5, 0.7, 0.9] {
            for rep in 0..5 {
                // About four characters per token of English.
                let mut sentences = Vec::new();
                let mut chars = 0;
                while chars < tokens * 4 {
                    let s = filler(&mut rng);
                    chars += s.len() + 1;
                    sentences.push(s);
                }
                let mut cities = CITIES.to_vec();
                cities.shuffle(&mut rng);
                let code = rng.random_range(100_000..10_000_000u32);
                let at = ((sentences.len() as f64) * depth) as usize;
                for decoy in &cities[1..=decoys] {
                    let other = rng.random_range(100_000..10_000_000u32);
                    let pos = rng.random_range(0..sentences.len());
                    sentences.insert(pos, format!("The vault code for {decoy} is {other}."));
                }
                sentences.insert(
                    at.min(sentences.len()),
                    format!("The vault code for {} is {code}.", cities[0]),
                );
                let log: String = sentences
                    .chunks(6)
                    .map(|p| p.join(" "))
                    .collect::<Vec<_>>()
                    .join("\n\n");
                lines.push(
                    json!({
                        "id": format!("context-{}k-{}-{rep}", tokens / 1024, (depth * 100.0) as usize),
                        "cat": "context", "check": "number", "answer": code.to_string(),
                        "prompt": format!(
                            "Here is a warehouse log.\n\n<log>\n{log}\n</log>\n\nWhat is the vault code for {}?",
                            cities[0]
                        ),
                        "how": "Reply with the code on its own line as \"Answer: <number>\".",
                        "max_tokens": 64,
                    })
                    .to_string(),
                );
            }
        }
    }
    lines.shuffle(&mut rng);
    lines.join("\n") + "\n"
}

fn gsm8k(r: &Json) -> Option<Json> {
    let row = &r["row"];
    let answer = row["answer"]
        .as_str()?
        .rsplit("####")
        .next()?
        .trim()
        .replace(',', "");
    Some(json!({
        "id": format!("gsm8k-{}", r["row_idx"]), "cat": "math", "check": "number",
        "answer": answer, "prompt": row["question"].as_str()?.trim(),
    }))
}

fn mmlu(r: &Json) -> Option<Json> {
    let row = &r["row"];
    let letters = ['A', 'B', 'C', 'D'];
    let options: Vec<String> = row["choices"]
        .as_array()?
        .iter()
        .zip(letters)
        .map(|(c, l)| format!("{l}. {}", c.as_str().unwrap_or("")))
        .collect();
    let answer = letters.get(row["answer"].as_u64()? as usize)?;
    Some(json!({
        "id": format!("mmlu-{}", r["row_idx"]), "cat": "knowledge", "check": "choice",
        "answer": answer.to_string(),
        "prompt": format!(
            "A question about {}.\n\n{}\n\n{}",
            row["subject"].as_str()?.replace('_', " "),
            row["question"].as_str()?.trim(),
            options.join("\n")
        ),
    }))
}

/// Only prompts whose every instruction [`eval`] can verify.
fn ifeval(r: &Json) -> Option<Json> {
    let row = &r["row"];
    let ids = row["instruction_id_list"].as_array()?;
    let kwargs = row["kwargs"].as_array()?;
    let rules: Vec<Json> = ids
        .iter()
        .zip(kwargs)
        .map(|(id, kw)| {
            let mut rule: Map<String, Json> = kw
                .as_object()
                .map(|o| {
                    o.iter()
                        .filter(|(_, v)| !v.is_null())
                        .map(|(k, v)| (k.clone(), v.clone()))
                        .collect()
                })
                .unwrap_or_default();
            rule.insert("id".into(), id.clone());
            Json::Object(rule)
        })
        .collect();
    (rules.len() == ids.len()).then(|| {
        json!({
            "id": format!("ifeval-{}", row["key"]), "cat": "instructions", "check": "rules",
            "rules": rules, "prompt": row["prompt"],
        })
    })
}

fn cruxeval(r: &Json) -> Option<Json> {
    let row = &r["row"];
    Some(json!({
        "id": format!("crux-{}", row["id"].as_str()?), "cat": "code", "check": "literal",
        "answer": row["output"].as_str()?,
        "prompt": format!(
            "Here is a Python function:\n\n```python\n{}\n```\n\nWhat does `f({})` return?",
            row["code"].as_str()?,
            row["input"].as_str()?
        ),
    }))
}

/// `item` as one JSON line, if `eval` can read and grade it.
fn checked_line(item: Json) -> Option<String> {
    let line = item.to_string();
    eval::parse_items(&line).ok().map(|_| line)
}

/// Sleeps `secs`, a second at a time so a stop is noticed.
fn wait(secs: u64, stop: &AtomicBool) -> Result<(), String> {
    for _ in 0..secs {
        if stop.load(Ordering::SeqCst) {
            return Err("stopped".into());
        }
        std::thread::sleep(Duration::from_secs(1));
    }
    Ok(())
}

/// GET `url` with the system curl, paced, retrying (longer after HTTP 429).
/// `label` names the download in progress lines and errors.
fn curl(
    url: &str,
    label: &str,
    progress: &impl Fn(String),
    stop: &AtomicBool,
) -> Result<Vec<u8>, String> {
    // A Hugging Face token raises the rate limit; it goes through stdin so it
    // never shows in the process list.
    let token = std::env::var("HF_TOKEN").ok().filter(|t| !t.is_empty());
    let mut last = String::new();
    for attempt in 0..ATTEMPTS {
        if stop.load(Ordering::SeqCst) {
            return Err("stopped".into());
        }
        std::thread::sleep(PACE);
        let mut cmd = Command::new("/usr/bin/curl");
        cmd.args(["-sSfL", "--max-time", "120"]);
        if token.is_some() {
            cmd.args(["-H", "@-"]);
        }
        let mut child = cmd
            .arg(url)
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .map_err(|e| format!("could not run curl: {e}"))?;
        if let (Some(t), Some(mut stdin)) = (&token, child.stdin.take()) {
            use std::io::Write;
            let _ = writeln!(stdin, "Authorization: Bearer {t}");
        }
        let out = child.wait_with_output().map_err(|e| format!("curl: {e}"))?;
        if out.status.success() {
            return Ok(out.stdout);
        }
        last = String::from_utf8_lossy(&out.stderr).trim().to_string();
        let limited = last.contains("429");
        let secs = if limited {
            15 * (attempt + 1)
        } else {
            2 + 3 * attempt
        };
        if attempt + 1 < ATTEMPTS {
            progress(format!(
                "{label} · {}, retrying in {secs}s",
                if limited {
                    "Hugging Face is rate-limiting"
                } else {
                    "request failed"
                }
            ));
            wait(secs, stop)?;
        }
    }
    Err(if last.contains("429") {
        format!(
            "{label}: Hugging Face kept rate-limiting the download (HTTP 429). Try again in a few minutes (finished sets are kept), or set HF_TOKEN."
        )
    } else {
        format!("{label}: {last}")
    })
}

/// One page of rows from the datasets-server API.
fn fetch(
    d: &Dataset,
    offset: usize,
    length: usize,
    progress: &impl Fn(String),
    stop: &AtomicBool,
) -> Result<Json, String> {
    let url = format!(
        "{API}?dataset={}&config={}&split={}&offset={offset}&length={length}",
        d.name.replace('/', "%2F"),
        d.config,
        d.split
    );
    let body = curl(&url, d.cat, progress, stop)?;
    serde_json::from_slice(&body).map_err(|e| format!("{}: {e}", d.name))
}

/// The seeded sample of one dataset as JSON lines.
fn sample(d: &Dataset, progress: &impl Fn(String), stop: &AtomicBool) -> Result<String, String> {
    let mut rng = ChaCha8Rng::seed_from_u64(SEED ^ eval_hash(d.cat));
    let first = fetch(d, 0, 100, progress, stop)?;
    let total = first["num_rows_total"].as_u64().unwrap_or(0) as usize;
    if total == 0 {
        return Err(format!("{}: no rows", d.name));
    }
    let mut lines = Vec::new();
    if total <= FETCH_ALL {
        let mut rows: Vec<Json> = first["rows"].as_array().cloned().unwrap_or_default();
        while rows.len() < total {
            progress(format!("{} · {}/{total} rows", d.cat, rows.len()));
            let page = fetch(d, rows.len(), 100, progress, stop)?;
            let batch = page["rows"].as_array().cloned().unwrap_or_default();
            if batch.is_empty() {
                break;
            }
            rows.extend(batch);
        }
        rows.shuffle(&mut rng);
        if let Some(rank) = d.rank {
            // The hardest usable rows (stable, so ties keep the shuffle)...
            let mut usable: Vec<(usize, String)> = rows
                .iter()
                .filter_map(|r| Some((rank(r), (d.convert)(r).and_then(checked_line)?)))
                .collect();
            usable.sort_by_key(|(k, _)| std::cmp::Reverse(*k));
            usable.truncate(PER_SET);
            // ...shuffled again, so a quick run's first questions are a fair
            // sample of them rather than the very hardest.
            usable.shuffle(&mut rng);
            lines.extend(usable.into_iter().map(|(_, l)| l));
        } else {
            lines.extend(
                rows.iter()
                    .filter_map(|r| (d.convert)(r).and_then(checked_line))
                    .take(PER_SET),
            );
        }
    } else {
        // Too big to fetch whole, and sorted by subject: a few rows from each
        // of many random places.
        let mut windows: Vec<usize> = (0..total.div_ceil(WINDOW)).collect();
        windows.shuffle(&mut rng);
        for w in windows {
            if lines.len() >= PER_SET {
                break;
            }
            progress(format!("{} · {}/{PER_SET} questions", d.cat, lines.len()));
            let page = fetch(d, w * WINDOW, WINDOW, progress, stop)?;
            lines.extend(
                page["rows"]
                    .as_array()
                    .into_iter()
                    .flatten()
                    .filter_map(|r| (d.convert)(r).and_then(checked_line)),
            );
        }
        lines.truncate(PER_SET);
    }
    if lines.is_empty() {
        return Err(format!("{}: no usable questions", d.name));
    }
    Ok(lines.join("\n") + "\n")
}

pub fn eval_hash(s: &str) -> u64 {
    s.bytes().fold(0xcbf2_9ce4_8422_2325, |h, b| {
        (h ^ b as u64).wrapping_mul(0x0100_0000_01b3)
    })
}

/// A finished set waiting for the others, so a failed install resumes.
fn part_path(dir: &Path, cat: &str) -> PathBuf {
    dir.join(format!("{cat}.jsonl.part"))
}

/// Downloads every set into `dir`. Each finished set is kept as a `.part`
/// file, and the question files are only replaced once all of them are in.
pub fn install(
    dir: &Path,
    difficulty: Difficulty,
    progress: impl Fn(String),
    stop: &AtomicBool,
) -> Result<usize, String> {
    fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    for d in datasets(difficulty) {
        let part = part_path(dir, d.cat);
        if part.exists() {
            continue;
        }
        progress(format!("{} · downloading {}", d.cat, d.name));
        let text = sample(d, &progress, stop)?;
        fs::write(&part, text).map_err(|e| format!("{}: {e}", part.display()))?;
    }
    let tools = part_path(dir, "tools");
    if !tools.exists() {
        let text = sample_bfcl(difficulty, &progress, stop)?;
        fs::write(&tools, text).map_err(|e| format!("{}: {e}", tools.display()))?;
    }
    let context = part_path(dir, "context");
    if !context.exists() {
        progress("context · writing the long-context questions".into());
        fs::write(&context, context_set(difficulty))
            .map_err(|e| format!("{}: {e}", context.display()))?;
    }
    let mut count = 0;
    for (cat, _) in CATEGORIES {
        let part = part_path(dir, cat);
        let path = dir.join(format!("{cat}.jsonl"));
        count += fs::read_to_string(&part).map_or(0, |t| t.lines().count());
        fs::rename(&part, &path).map_err(|e| format!("{}: {e}", path.display()))?;
    }
    Ok(count)
}

/// Deletes the question files and unfinished downloads (and the folder, if
/// nothing else is in it).
pub fn remove(dir: &Path) -> Result<(), String> {
    for (cat, _) in CATEGORIES {
        for path in [dir.join(format!("{cat}.jsonl")), part_path(dir, cat)] {
            if path.exists() {
                fs::remove_file(&path).map_err(|e| format!("{}: {e}", path.display()))?;
            }
        }
    }
    fs::remove_dir(dir).ok();
    Ok(())
}

pub enum InstallEvent {
    Progress(String),
    Done(Result<usize, String>),
}

pub struct Installing {
    pub rx: Receiver<InstallEvent>,
    pub stop: Arc<AtomicBool>,
    pub handle: Option<JoinHandle<()>>,
}

pub fn spawn_install(dir: PathBuf, difficulty: Difficulty) -> Result<Installing, String> {
    let (tx, rx) = channel();
    let stop = Arc::new(AtomicBool::new(false));
    let flag = stop.clone();
    let handle = std::thread::Builder::new()
        .name("ferrum-eval-install".into())
        .spawn(move || {
            let progress = |s| {
                let _ = tx.send(InstallEvent::Progress(s));
            };
            let result = install(&dir, difficulty, progress, &flag);
            let _ = tx.send(InstallEvent::Done(result));
        })
        .map_err(|e| e.to_string())?;
    Ok(Installing {
        rx,
        stop,
        handle: Some(handle),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(idx: usize, row: Json) -> Json {
        json!({"row_idx": idx, "row": row})
    }

    #[test]
    fn converts_rows_into_gradable_items() {
        let g = gsm8k(&row(
            4,
            json!({"question": " How many? ", "answer": "work...\n#### 1,416"}),
        ))
        .and_then(checked_line)
        .unwrap();
        let item = &eval::parse_items(&g).unwrap()[0];
        assert_eq!(item.id, "gsm8k-4");
        assert_eq!(item.check, eval::Check::Number(1416.0));
        assert_eq!(item.prompt, "How many?");

        let m = mmlu(&row(
            9,
            json!({"question": "Q?", "subject": "high_school_physics", "choices": ["a", "b", "c", "d"], "answer": 2}),
        ))
        .and_then(checked_line)
        .unwrap();
        let item = &eval::parse_items(&m).unwrap()[0];
        assert_eq!(item.check, eval::Check::Choice('C'));
        assert!(
            item.prompt
                .starts_with("A question about high school physics.")
        );
        assert!(item.prompt.ends_with("C. c\nD. d"));

        let c = cruxeval(&row(
            0,
            json!({"id": "sample_3", "code": "def f(x):\n    return x * 2", "input": "4", "output": "8"}),
        ))
        .and_then(checked_line)
        .unwrap();
        assert!(c.contains("crux-sample_3") && c.contains("f(4)"));
    }

    #[test]
    fn converts_the_hard_sets() {
        let m = |level: u64, answer: &str| {
            math500(&row(
                3,
                json!({"problem": "Find x.", "answer": answer, "level": level.to_string()}),
            ))
            .and_then(checked_line)
        };
        let line = m(5, "1,024").unwrap();
        let item = &eval::parse_items(&line).unwrap()[0];
        assert_eq!(item.check, eval::Check::Number(1024.0));
        assert_eq!(item.max_tokens, 1024);
        assert_eq!(m(5, "\\frac{1}{2}"), None, "LaTeX answers are skipped");
        assert_eq!(m(2, "7"), None, "easy levels are skipped");
        assert_eq!(math_level(&row(0, json!({"level": 4}))), 4);
        assert_eq!(math_level(&row(0, json!({"level": "5"}))), 5);

        let p = mmlu_pro(&row(
            0,
            json!({"question_id": 70, "question": "Q?", "category": "law",
                   "options": ["a","b","c","d","e","f","g","h","i","j"], "answer": "J"}),
        ))
        .and_then(checked_line)
        .unwrap();
        let item = &eval::parse_items(&p).unwrap()[0];
        assert_eq!(item.check, eval::Check::Choice('J'));
        assert!(item.prompt.ends_with("J. j"));
        assert_eq!(code_length(&row(0, json!({"code": "abcd"}))), 4);
        assert_eq!(
            rule_count(&row(0, json!({"instruction_id_list": ["a", "b", "c"]}))),
            3
        );
    }

    #[test]
    fn converts_coding_and_tool_sets() {
        let m = mbpp(&row(
            0,
            json!({"task_id": 2, "prompt": "Write add.", "test_imports": ["import math"],
                   "test_list": ["assert add(1, 2) == 3", "assert add(0, 0) == 0"]}),
        ))
        .and_then(checked_line)
        .unwrap();
        let item = &eval::parse_items(&m).unwrap()[0];
        assert!(item.prompt.contains("assert add(1, 2) == 3"));
        assert!(!item.prompt.contains("add(0, 0)"), "only one test is shown");
        assert_eq!(
            item.check,
            eval::Check::Program(
                "import math\nassert add(1, 2) == 3\nassert add(0, 0) == 0".into()
            )
        );
        let h = humaneval(&row(
            7,
            json!({"prompt": "def f(x):\n", "test": "def check(c):\n    assert c(1)", "entry_point": "f"}),
        ))
        .and_then(checked_line)
        .unwrap();
        assert!(h.contains("check(f)") && h.contains("humaneval-7"));

        let q = json!({"id": "parallel_0", "question": [[{"role": "user", "content": "Play two songs."}]],
            "function": [{"name": "spotify.play", "description": "d",
                "parameters": {"type": "dict", "properties": {"t": {"type": "float"}, "a": {"type": "any"}}}}]});
        let truth = json!([{"spotify.play": {"t": [1.5]}}, {"spotify.play": {"t": [2.0]}}]);
        let line = bfcl_item(&q, &truth).and_then(checked_line).unwrap();
        let item = &eval::parse_items(&line).unwrap()[0];
        assert_eq!(item.prompt, "Play two songs.");
        let f = &item.tools[0]["function"];
        assert_eq!(f["name"], "spotify_play");
        assert_eq!(f["parameters"]["type"], "object");
        assert_eq!(f["parameters"]["properties"]["t"]["type"], "number");
        assert!(f["parameters"]["properties"]["a"].get("type").is_none());
        match &item.check {
            eval::Check::Calls(calls) => {
                assert_eq!(calls.len(), 2);
                assert_eq!(calls[0].name, "spotify_play");
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn generates_long_context_questions() {
        let standard = context_set(Difficulty::Standard);
        assert_eq!(
            standard,
            context_set(Difficulty::Standard),
            "the same every time"
        );
        let items = eval::parse_items(&standard).unwrap();
        assert_eq!(items.len(), 50);
        for item in &items {
            let eval::Check::Number(code) = item.check else {
                panic!("{:?}", item.check)
            };
            let fact = format!(" is {code}.");
            assert_eq!(item.prompt.matches(&fact).count(), 1, "{}", item.id);
            assert!(item.message().ends_with("\"Answer: <number>\"."));
        }
        let hard = eval::parse_items(&context_set(Difficulty::Hard)).unwrap();
        let longest = hard.iter().map(|i| i.prompt.len()).max().unwrap();
        assert!(longest > 32768 * 4 && longest < 32768 * 5, "{longest}");
        // Hard logs hide codes for other vaults too.
        assert!(hard[0].prompt.matches("The vault code for").count() == 5);
    }

    #[test]
    fn keeps_only_verifiable_ifeval_prompts() {
        let ok = row(
            0,
            json!({
                "key": 1001, "prompt": "Write a poem.",
                "instruction_id_list": ["punctuation:no_comma", "length_constraints:number_words"],
                "kwargs": [{"num_words": null}, {"relation": "less than", "num_words": 50, "letter": null}],
            }),
        );
        let line = ifeval(&ok).and_then(checked_line).unwrap();
        assert_eq!(
            eval::parse_items(&line).unwrap()[0].check,
            eval::Check::Rules(vec![
                eval::Rule::NoComma,
                eval::Rule::Words { less: true, n: 50 }
            ])
        );
        let unsupported = row(
            1,
            json!({
                "key": 1002, "prompt": "p",
                "instruction_id_list": ["language:response_language"],
                "kwargs": [{"language": "fr"}],
            }),
        );
        assert_eq!(ifeval(&unsupported).and_then(checked_line), None);
    }

    #[test]
    fn removes_only_the_question_files() {
        let dir = std::env::temp_dir().join(format!("ferrum-sets-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        for (cat, _) in CATEGORIES {
            fs::write(dir.join(format!("{cat}.jsonl")), "x").unwrap();
        }
        fs::write(part_path(&dir, "math"), "x").unwrap();
        fs::write(dir.join("notes.txt"), "keep").unwrap();
        remove(&dir).unwrap();
        assert!(!dir.join("math.jsonl").exists());
        assert!(!part_path(&dir, "math").exists());
        assert!(dir.join("notes.txt").exists(), "other files stay");
        fs::remove_file(dir.join("notes.txt")).unwrap();
        remove(&dir).unwrap();
        assert!(!dir.exists(), "an empty folder goes too");
    }
}
