//! Installing and removing the intelligence question sets: a fixed, seeded
//! sample of GSM8K (MIT), MMLU (MIT), IFEval (Apache-2.0) and CRUXEval (MIT),
//! fetched with the system `curl` from the Hugging Face datasets-server API
//! and written as the JSON lines [`eval`] reads.
use crate::eval::{self, CATEGORIES};
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
}

const DATASETS: [Dataset; 4] = [
    Dataset {
        cat: "math",
        name: "openai/gsm8k",
        config: "main",
        split: "test",
        convert: gsm8k,
    },
    Dataset {
        cat: "knowledge",
        name: "cais/mmlu",
        config: "all",
        split: "test",
        convert: mmlu,
    },
    Dataset {
        cat: "instructions",
        name: "google/IFEval",
        config: "default",
        split: "train",
        convert: ifeval,
    },
    Dataset {
        cat: "code",
        name: "cruxeval-org/cruxeval",
        config: "default",
        split: "test",
        convert: cruxeval,
    },
];

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
        cmd.args(["-sSfL", "--max-time", "60"]);
        if token.is_some() {
            cmd.args(["-H", "@-"]);
        }
        let mut child = cmd
            .arg(&url)
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
            return serde_json::from_slice(&out.stdout).map_err(|e| format!("{}: {e}", d.name));
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
                "{} · {}, retrying in {secs}s",
                d.cat,
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
            "{}: Hugging Face kept rate-limiting the download (HTTP 429). Try again in a few minutes (finished sets are kept), or set HF_TOKEN.",
            d.name
        )
    } else {
        format!("{}: {last}", d.name)
    })
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
        lines.extend(
            rows.iter()
                .filter_map(|r| (d.convert)(r).and_then(checked_line))
                .take(PER_SET),
        );
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

fn eval_hash(s: &str) -> u64 {
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
pub fn install(dir: &Path, progress: impl Fn(String), stop: &AtomicBool) -> Result<usize, String> {
    fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    for d in &DATASETS {
        let part = part_path(dir, d.cat);
        if part.exists() {
            continue;
        }
        progress(format!("{} · downloading {}", d.cat, d.name));
        let text = sample(d, &progress, stop)?;
        fs::write(&part, text).map_err(|e| format!("{}: {e}", part.display()))?;
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

pub fn spawn_install(dir: PathBuf) -> Result<Installing, String> {
    let (tx, rx) = channel();
    let stop = Arc::new(AtomicBool::new(false));
    let flag = stop.clone();
    let handle = std::thread::Builder::new()
        .name("ferrum-eval-install".into())
        .spawn(move || {
            let progress = |s| {
                let _ = tx.send(InstallEvent::Progress(s));
            };
            let result = install(&dir, progress, &flag);
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
