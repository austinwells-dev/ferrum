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
/// Splits up to this size are fetched whole; larger ones row by row.
const FETCH_ALL: usize = 2000;

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

fn fetch(d: &Dataset, offset: usize, length: usize, stop: &AtomicBool) -> Result<Json, String> {
    let url = format!(
        "{API}?dataset={}&config={}&split={}&offset={offset}&length={length}",
        d.name.replace('/', "%2F"),
        d.config,
        d.split
    );
    let mut last = String::new();
    for attempt in 0..5u64 {
        if stop.load(Ordering::SeqCst) {
            return Err("stopped".into());
        }
        let out = Command::new("/usr/bin/curl")
            .args(["-sSfL", "--max-time", "60", &url])
            .output()
            .map_err(|e| format!("could not run curl: {e}"))?;
        if out.status.success() {
            return serde_json::from_slice(&out.stdout).map_err(|e| format!("{}: {e}", d.name));
        }
        // Rate limits and hiccups: back off and retry.
        last = String::from_utf8_lossy(&out.stderr).trim().to_string();
        std::thread::sleep(Duration::from_secs(2 + 3 * attempt));
    }
    Err(format!("{}: {last}", d.name))
}

/// The seeded sample of one dataset as JSON lines.
fn sample(d: &Dataset, progress: &impl Fn(String), stop: &AtomicBool) -> Result<String, String> {
    let mut rng = ChaCha8Rng::seed_from_u64(SEED ^ eval_hash(d.cat));
    let first = fetch(d, 0, 100, stop)?;
    let total = first["num_rows_total"].as_u64().unwrap_or(0) as usize;
    if total == 0 {
        return Err(format!("{}: no rows", d.name));
    }
    let mut lines = Vec::new();
    if total <= FETCH_ALL {
        let mut rows: Vec<Json> = first["rows"].as_array().cloned().unwrap_or_default();
        while rows.len() < total {
            progress(format!("{} · {}/{total} rows", d.cat, rows.len()));
            let page = fetch(d, rows.len(), 100, stop)?;
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
        // Too big to fetch whole (and sorted by subject): pick rows at random.
        let mut order: Vec<usize> = (0..total).collect();
        order.shuffle(&mut rng);
        for idx in order {
            if lines.len() == PER_SET {
                break;
            }
            progress(format!("{} · {}/{PER_SET} questions", d.cat, lines.len()));
            let page = fetch(d, idx, 1, stop)?;
            if let Some(line) = page["rows"]
                .get(0)
                .and_then(d.convert)
                .and_then(checked_line)
            {
                lines.push(line);
            }
        }
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

/// Downloads every set into `dir`. Nothing is replaced unless all succeed.
pub fn install(dir: &Path, progress: impl Fn(String), stop: &AtomicBool) -> Result<usize, String> {
    let mut texts = Vec::new();
    for d in &DATASETS {
        progress(format!("{} · downloading {}", d.cat, d.name));
        texts.push((d.cat, sample(d, &progress, stop)?));
    }
    fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    let mut count = 0;
    for (cat, text) in &texts {
        let path = dir.join(format!("{cat}.jsonl"));
        let tmp = path.with_extension("jsonl.tmp");
        fs::write(&tmp, text)
            .and_then(|_| fs::rename(&tmp, &path))
            .map_err(|e| format!("{}: {e}", path.display()))?;
        count += text.lines().count();
    }
    Ok(count)
}

/// Deletes the question files (and the folder, if nothing else is in it).
pub fn remove(dir: &Path) -> Result<(), String> {
    for (cat, _) in CATEGORIES {
        let path = dir.join(format!("{cat}.jsonl"));
        if path.exists() {
            fs::remove_file(&path).map_err(|e| format!("{}: {e}", path.display()))?;
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
        fs::write(dir.join("notes.txt"), "keep").unwrap();
        remove(&dir).unwrap();
        assert!(!dir.join("math.jsonl").exists());
        assert!(dir.join("notes.txt").exists(), "other files stay");
        fs::remove_file(dir.join("notes.txt")).unwrap();
        remove(&dir).unwrap();
        assert!(!dir.exists(), "an empty folder goes too");
    }
}
