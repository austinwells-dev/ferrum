//! Intelligence evaluation: question sets whose answers a program can check,
//! scored from greedy, no-thinking generations. The runtime exposes no
//! log-probabilities, so every item is graded from the generated text.
//!
//! Items are JSON lines:
//! `{"id", "cat", "prompt", "check": "number"|"choice"|"literal"|"rules",
//!   "answer"?, "rules"?, "max_tokens"?}`
//! `number` compares the final number (GSM8K), `choice` an option letter
//! (MMLU), `literal` a Python value (CRUXEval output prediction) and `rules`
//! a list of IFEval instructions that must all hold.
//!
//! Two uses:
//! - The answer check guards `≈` rows of the benchmark: a drafter whose output
//!   left plain decoding late (a near-tie flip) must still get the same
//!   questions right. Its questions are bundled.
//! - Intelligence runs score a model on the suite installed from the TUI
//!   ([`crate::evalsets`]: GSM8K, MMLU, IFEval, CRUXEval samples): quick runs
//!   ask a fixed subset and predict the full score, full runs measure it.
use crate::bench::{self, Candidate, Source};
use crate::*;
use ferrum::hybrid::runtime::Runtime;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
    mpsc::{Receiver, Sender, channel},
};
use std::thread::JoinHandle;

/// Math word problems for the answer check. Their worked answers run a few
/// hundred tokens, long enough for late near-tie flips to show up.
pub const CHECK_ITEMS: &str = include_str!("eval/check.jsonl");

/// Questions a drafter may get wrong beyond those it gets right that plain
/// decoding missed.
pub const CHECK_SLACK: usize = 1;

#[derive(Clone, Debug, PartialEq)]
pub enum Check {
    Number(f64),
    Choice(char),
    Literal(String),
    Rules(Vec<Rule>),
}

/// The IFEval instructions we can verify (`id` in brackets).
#[derive(Clone, Debug, PartialEq)]
pub enum Rule {
    /// `punctuation:no_comma`
    NoComma,
    /// `change_case:english_lowercase`
    Lowercase,
    /// `change_case:english_capital`
    Uppercase,
    /// `length_constraints:number_words`
    Words { less: bool, n: usize },
    /// `length_constraints:number_paragraphs` (separated by `***`)
    Paragraphs(usize),
    /// `detectable_format:number_bullet_lists`
    Bullets(usize),
    /// `detectable_format:number_highlighted_sections`
    Highlights(usize),
    /// `detectable_format:title` (`<<title>>`)
    Title,
    /// `detectable_format:json_format`
    Json,
    /// `keywords:existence`
    Keywords(Vec<String>),
    /// `keywords:forbidden_words`
    Forbidden(Vec<String>),
    /// `keywords:letter_frequency`
    Letter { letter: char, less: bool, n: usize },
    /// `startend:end_checker`
    EndsWith(String),
    /// `startend:quotation`
    Quoted,
    /// `detectable_content:postscript`
    Postscript(String),
}

#[derive(Clone, Debug)]
pub struct Item {
    pub id: String,
    pub prompt: String,
    pub check: Check,
    pub max_tokens: usize,
}

impl Item {
    /// The user message: the prompt plus how to state the answer.
    pub fn message(&self) -> String {
        let how = match self.check {
            Check::Number(_) => {
                "Work it out step by step, briefly, then give the final number on its own line as \"Answer: <number>\"."
            }
            Check::Choice(_) => {
                "Reply with the letter of the correct option on its own line as \"Answer: <letter>\"."
            }
            Check::Literal(_) => {
                "Give the exact value as a Python literal on the last line as \"Answer: <value>\"."
            }
            Check::Rules(_) => return self.prompt.clone(),
        };
        format!("{}\n\n{how}", self.prompt)
    }
}

pub fn parse_items(text: &str) -> Result<Vec<Item>, String> {
    text.lines()
        .enumerate()
        .filter(|(_, l)| !l.trim().is_empty())
        .map(|(n, l)| parse_item(l).map_err(|e| format!("line {}: {e}", n + 1)))
        .collect()
}

/// The bundled answer-check set (validated by the tests).
pub fn check_items() -> Vec<Item> {
    parse_items(CHECK_ITEMS).unwrap_or_default()
}

fn parse_item(line: &str) -> Result<Item, String> {
    let v: Json = serde_json::from_str(line).map_err(|e| e.to_string())?;
    let text = |k: &str| {
        v[k].as_str()
            .map(str::to_string)
            .ok_or_else(|| format!("missing {k}"))
    };
    let check = match v["check"].as_str() {
        Some("number") => {
            let a = text("answer")?;
            Check::Number(
                a.replace(',', "")
                    .parse()
                    .map_err(|_| format!("answer {a:?} is not a number"))?,
            )
        }
        Some("choice") => {
            let a = text("answer")?;
            match a.chars().collect::<Vec<_>>()[..] {
                [c @ 'A'..='J'] => Check::Choice(c),
                _ => return Err(format!("answer {a:?} is not an option letter")),
            }
        }
        Some("literal") => Check::Literal(text("answer")?),
        Some("rules") => Check::Rules(
            v["rules"]
                .as_array()
                .ok_or("missing rules")?
                .iter()
                .map(parse_rule)
                .collect::<Result<_, _>>()?,
        ),
        other => return Err(format!("unknown check {other:?}")),
    };
    let max_tokens = v["max_tokens"].as_u64().map_or(
        match check {
            Check::Number(_) => 512,
            Check::Choice(_) => 128,
            Check::Literal(_) => 192,
            Check::Rules(_) => 512,
        },
        |n| n as usize,
    );
    Ok(Item {
        id: text("id")?,
        prompt: text("prompt")?,
        check,
        max_tokens,
    })
}

/// One rule from IFEval's `instruction_id` plus its kwargs, flattened.
fn parse_rule(v: &Json) -> Result<Rule, String> {
    let id = v["id"].as_str().ok_or("rule without id")?;
    let n = |k: &str| {
        v[k].as_u64()
            .map(|n| n as usize)
            .ok_or_else(|| format!("{id}: missing {k}"))
    };
    let s = |k: &str| {
        v[k].as_str()
            .map(str::to_string)
            .ok_or_else(|| format!("{id}: missing {k}"))
    };
    let less = |k: &str| match v[k].as_str() {
        Some("less than") => Ok(true),
        Some("at least") => Ok(false),
        other => Err(format!("{id}: relation {other:?}")),
    };
    let list = |k: &str| {
        v[k].as_array()
            .map(|a| {
                a.iter()
                    .filter_map(|w| w.as_str().map(str::to_string))
                    .collect()
            })
            .ok_or_else(|| format!("{id}: missing {k}"))
    };
    Ok(match id {
        "punctuation:no_comma" => Rule::NoComma,
        "change_case:english_lowercase" => Rule::Lowercase,
        "change_case:english_capital" => Rule::Uppercase,
        "length_constraints:number_words" => Rule::Words {
            less: less("relation")?,
            n: n("num_words")?,
        },
        "length_constraints:number_paragraphs" => Rule::Paragraphs(n("num_paragraphs")?),
        "detectable_format:number_bullet_lists" => Rule::Bullets(n("num_bullets")?),
        "detectable_format:number_highlighted_sections" => Rule::Highlights(n("num_highlights")?),
        "detectable_format:title" => Rule::Title,
        "detectable_format:json_format" => Rule::Json,
        "keywords:existence" => Rule::Keywords(list("keywords")?),
        "keywords:forbidden_words" => Rule::Forbidden(list("forbidden_words")?),
        "keywords:letter_frequency" => Rule::Letter {
            letter: s("letter")?
                .chars()
                .next()
                .ok_or_else(|| format!("{id}: empty letter"))?
                .to_ascii_lowercase(),
            less: less("let_relation")?,
            n: n("let_frequency")?,
        },
        "startend:end_checker" => Rule::EndsWith(s("end_phrase")?),
        "startend:quotation" => Rule::Quoted,
        "detectable_content:postscript" => Rule::Postscript(s("postscript_marker")?),
        other => return Err(format!("unsupported rule {other}")),
    })
}

// ---- grading ----

#[derive(Clone, Debug, PartialEq)]
pub struct Graded {
    pub correct: bool,
    /// The answer as extracted (None when none could be found), used to tell
    /// whether two runs answered alike.
    pub answer: Option<String>,
}

pub fn grade(check: &Check, text: &str) -> Graded {
    match check {
        Check::Number(gold) => {
            let got = final_number(text);
            Graded {
                correct: got.is_some_and(|g| (g - gold).abs() <= 1e-6 * gold.abs().max(1.0)),
                answer: got.map(|g| g.to_string()),
            }
        }
        Check::Choice(gold) => {
            let got = final_choice(text);
            Graded {
                correct: got == Some(*gold),
                answer: got.map(String::from),
            }
        }
        Check::Literal(gold) => {
            let got = final_literal(text);
            Graded {
                correct: got
                    .as_deref()
                    .is_some_and(|g| normalize_literal(g) == normalize_literal(gold)),
                answer: got,
            }
        }
        Check::Rules(rules) => {
            let held: String = rules
                .iter()
                .map(|r| if r.holds(text) { '1' } else { '0' })
                .collect();
            Graded {
                correct: !held.contains('0'),
                answer: Some(held),
            }
        }
    }
}

/// Text after the last answer marker ("Answer:", "answer is", `\boxed{`).
fn after_marker(text: &str) -> Option<&str> {
    let lower = text.to_ascii_lowercase();
    ["answer:", "answer is", "\\boxed{"]
        .iter()
        .filter_map(|m| lower.rfind(m).map(|i| i + m.len()))
        .max()
        .map(|i| &text[i..])
}

/// Numbers in reading order; thousands commas are dropped.
fn numbers(text: &str) -> Vec<f64> {
    let b = text.as_bytes();
    let mut out = Vec::new();
    let mut i = 0;
    while i < b.len() {
        if !b[i].is_ascii_digit() {
            i += 1;
            continue;
        }
        let mut s = String::new();
        if i > 0 && b[i - 1] == b'-' && (i < 2 || !b[i - 2].is_ascii_alphanumeric()) {
            s.push('-');
        }
        while i < b.len() {
            let c = b[i];
            let digit_next = b.get(i + 1).is_some_and(u8::is_ascii_digit);
            if c.is_ascii_digit() {
                s.push(c as char);
            } else if c == b',' && digit_next {
            } else if c == b'.' && digit_next && !s.contains('.') {
                s.push('.');
            } else {
                break;
            }
            i += 1;
        }
        if let Ok(v) = s.parse() {
            out.push(v);
        }
    }
    out
}

/// The first number after the answer marker, else the last number.
fn final_number(text: &str) -> Option<f64> {
    after_marker(text)
        .and_then(|rest| numbers(rest).first().copied())
        .or_else(|| numbers(text).last().copied())
}

/// An option letter standing alone at the start of `s` (after `*`, `(` ...).
fn leading_letter(s: &str) -> Option<char> {
    let mut chars = s
        .trim_start_matches([' ', '\t', '\n', '*', '(', '[', ':'])
        .chars();
    let c = chars.next()?;
    let alone = chars.next().is_none_or(|n| !n.is_alphanumeric());
    (('A'..='J').contains(&c) && alone).then_some(c)
}

fn final_choice(text: &str) -> Option<char> {
    match after_marker(text) {
        Some(rest) => leading_letter(rest),
        None => leading_letter(text.trim()),
    }
}

fn final_literal(text: &str) -> Option<String> {
    let rest = after_marker(text)?;
    let mut line = rest.lines().next().unwrap_or("").trim();
    // Peel markdown and a closing full stop: "`'abc'`." -> "'abc'".
    loop {
        let next = line.trim_matches(['*', '`', ' ']);
        let next = match next.strip_suffix('.') {
            Some(s) if s.ends_with(['\'', '"', ']', ')', '}', '`', '*']) => s,
            _ => next,
        };
        if next == line {
            break;
        }
        line = next;
    }
    (!line.is_empty()).then(|| line.to_string())
}

/// One quote style and no spaces after separators, so `["a", 1]` matches `['a',1]`.
fn normalize_literal(s: &str) -> String {
    let mut out = String::new();
    let mut prev = ' ';
    for c in s.trim().chars() {
        if c == ' ' && matches!(prev, ',' | ':') {
            continue;
        }
        out.push(if c == '"' { '\'' } else { c });
        prev = c;
    }
    out
}

/// Non-overlapping `delim…delim` spans on one line with non-blank content
/// (IFEval's highlight regexes).
fn wrapped(text: &str, delim: &str) -> usize {
    let mut count = 0;
    let mut i = 0;
    while let Some(start) = text[i..].find(delim).map(|p| i + p) {
        let from = start + delim.len();
        let close = text[from..].find(delim).map(|p| from + p);
        match close {
            Some(end) if !text[from..end].contains(['*', '\n']) => {
                count += usize::from(!text[from..end].trim().is_empty());
                i = end + delim.len();
            }
            _ => i = start + 1,
        }
    }
    count
}

fn has_word(lower: &str, word: &str) -> bool {
    let word = word.to_lowercase();
    lower.match_indices(&word).any(|(i, w)| {
        let before = lower[..i].chars().next_back();
        let after = lower[i + w.len()..].chars().next();
        !before.is_some_and(char::is_alphanumeric) && !after.is_some_and(char::is_alphanumeric)
    })
}

impl Rule {
    pub fn holds(&self, text: &str) -> bool {
        let lower = text.to_lowercase();
        let cmp = |less: bool, count: usize, n: usize| if less { count < n } else { count >= n };
        match self {
            Rule::NoComma => !text.contains(','),
            Rule::Lowercase => text == lower,
            Rule::Uppercase => text == text.to_uppercase(),
            Rule::Words { less, n } => cmp(
                *less,
                text.split_whitespace()
                    .filter(|w| w.chars().any(char::is_alphanumeric))
                    .count(),
                *n,
            ),
            Rule::Paragraphs(n) => text.split("***").filter(|p| !p.trim().is_empty()).count() == *n,
            Rule::Bullets(n) => {
                text.lines()
                    .map(str::trim_start)
                    .filter(|l| (l.starts_with('*') && !l.starts_with("**")) || l.starts_with('-'))
                    .count()
                    == *n
            }
            Rule::Highlights(n) => wrapped(text, "*") + wrapped(text, "**") >= *n,
            Rule::Title => text.find("<<").is_some_and(|i| {
                text[i + 2..]
                    .find(">>")
                    .is_some_and(|j| j > 0 && !text[i + 2..i + 2 + j].contains('\n'))
            }),
            Rule::Json => {
                let t = text.trim();
                let t = t
                    .strip_prefix("```json")
                    .or_else(|| t.strip_prefix("```"))
                    .unwrap_or(t);
                let t = t.strip_suffix("```").unwrap_or(t);
                serde_json::from_str::<Json>(t.trim()).is_ok()
            }
            Rule::Keywords(words) => words.iter().all(|w| lower.contains(&w.to_lowercase())),
            Rule::Forbidden(words) => !words.iter().any(|w| has_word(&lower, w)),
            Rule::Letter { letter, less, n } => {
                cmp(*less, lower.chars().filter(|c| c == letter).count(), *n)
            }
            Rule::EndsWith(phrase) => lower.trim_end().ends_with(&phrase.to_lowercase()),
            Rule::Quoted => {
                let t = text.trim();
                t.len() > 1 && t.starts_with('"') && t.ends_with('"')
            }
            Rule::Postscript(marker) => lower.contains(&marker.to_lowercase()),
        }
    }
}

// ---- running ----

/// Asks every item in a fresh session and grades the answers. `progress` gets
/// the number answered so far.
pub fn answer_items(
    rt: &mut Runtime,
    items: &[Item],
    stop: &AtomicBool,
    mut progress: impl FnMut(usize),
) -> Result<Vec<Graded>, String> {
    let mut out = Vec::with_capacity(items.len());
    for item in items {
        progress(out.len());
        let s = crate::bench::generate(rt, &item.message(), item.max_tokens, None, stop)
            .map_err(|e| format!("{}: {e}", item.id))?;
        if stop.load(Ordering::SeqCst) {
            return Err("stopped".into());
        }
        out.push(grade(&item.check, &s.text));
    }
    Ok(out)
}

/// How a drafter's answers compare with plain decoding's on the same items.
#[derive(Clone, Debug, PartialEq)]
pub struct Verdict {
    pub n: usize,
    /// Right with plain decoding, wrong with the drafter.
    pub lost: usize,
    /// Wrong with plain decoding, right with the drafter.
    pub gained: usize,
    /// Same extracted answer as plain decoding.
    pub same: usize,
}

impl Verdict {
    pub fn passed(&self) -> bool {
        self.n > 0 && self.lost <= self.gained + CHECK_SLACK
    }
}

pub fn compare_answers(base: &[Graded], run: &[Graded]) -> Verdict {
    let mut v = Verdict {
        n: base.len().min(run.len()),
        lost: 0,
        gained: 0,
        same: 0,
    };
    for (b, r) in base.iter().zip(run) {
        v.lost += usize::from(b.correct && !r.correct);
        v.gained += usize::from(!b.correct && r.correct);
        v.same += usize::from(b.answer == r.answer);
    }
    v
}

// ---- plain-decoding answers, cached per model ----

/// What plain decoding's answers depend on: the model file, the K/V cache
/// format (q8 rounds differently) and the item set.
#[derive(Clone, Debug, PartialEq)]
pub struct CheckKey {
    pub model: String,
    pub size: u64,
    pub modified: u64,
    pub kv_cache: String,
    pub items: u64,
}

/// FNV-1a, enough to notice the bundled items changed.
fn fingerprint(text: &str) -> u64 {
    text.bytes().fold(0xcbf2_9ce4_8422_2325, |h, b| {
        (h ^ b as u64).wrapping_mul(0x0100_0000_01b3)
    })
}

impl CheckKey {
    pub fn new(model: &Path, kv_cache: KvCache) -> Self {
        let meta = fs::metadata(model).ok();
        Self {
            model: model.display().to_string(),
            size: meta.as_ref().map_or(0, |m| m.len()),
            modified: meta
                .and_then(|m| m.modified().ok())
                .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                .map_or(0, |d| d.as_secs()),
            kv_cache: format!("{kv_cache:?}"),
            items: fingerprint(CHECK_ITEMS),
        }
    }

    fn to_json(&self) -> Json {
        json!({
            "model": self.model, "size": self.size, "modified": self.modified,
            "kv_cache": self.kv_cache, "items": self.items.to_string(),
        })
    }
}

pub fn cache_path() -> PathBuf {
    home().join(".config/ferrum/answer-check.json")
}

fn read_cache(path: &Path) -> Vec<Json> {
    fs::read_to_string(path)
        .ok()
        .and_then(|t| serde_json::from_str::<Json>(&t).ok())
        .and_then(|j| j["baselines"].as_array().cloned())
        .unwrap_or_default()
}

pub fn cached_baseline(path: &Path, key: &CheckKey) -> Option<Vec<Graded>> {
    let key = key.to_json();
    let entry = read_cache(path).into_iter().find(|e| e["key"] == key)?;
    entry["answers"].as_array().map(|a| {
        a.iter()
            .map(|g| Graded {
                correct: g["correct"].as_bool().unwrap_or(false),
                answer: g["answer"].as_str().map(str::to_string),
            })
            .collect()
    })
}

/// Remember plain decoding's answers, newest first, at most 20 models.
pub fn store_baseline(path: &Path, key: &CheckKey, answers: &[Graded]) {
    let key = key.to_json();
    let mut entries = read_cache(path);
    entries.retain(|e| e["key"] != key);
    let answers: Vec<Json> = answers
        .iter()
        .map(|g| json!({"correct": g.correct, "answer": g.answer}))
        .collect();
    entries.insert(0, json!({"key": key, "answers": answers}));
    entries.truncate(20);
    if let Some(dir) = path.parent() {
        fs::create_dir_all(dir).ok();
    }
    fs::write(
        path,
        serde_json::to_string_pretty(&json!({"baselines": entries})).unwrap_or_default(),
    )
    .ok();
}

// ---- the intelligence suite ----

/// Question sets, one `<name>.jsonl` each in the suite folder.
pub const CATEGORIES: [(&str, &str); 4] = [
    ("math", "Math · GSM8K"),
    ("knowledge", "Knowledge · MMLU"),
    ("instructions", "Instructions · IFEval"),
    ("code", "Code reading · CRUXEval"),
];

/// Questions per category in a quick run: the first ones of each file, which
/// the installer shuffled with a fixed seed.
pub const QUICK_PER_CAT: usize = 15;

/// `$FERRUM_EVAL_DIR`, else `~/.config/ferrum/eval`.
pub fn suite_dir() -> PathBuf {
    std::env::var_os("FERRUM_EVAL_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| home().join(".config/ferrum/eval"))
}

#[derive(Clone, Debug)]
pub struct Suite {
    /// Items per category, in `CATEGORIES` order.
    pub sets: Vec<Vec<Item>>,
    /// Changes whenever a question file does.
    pub fingerprint: String,
}

impl Suite {
    pub fn load(dir: &Path) -> Result<Suite, String> {
        let (mut sets, mut missing, mut all) = (Vec::new(), Vec::new(), String::new());
        for (cat, _) in CATEGORIES {
            let path = dir.join(format!("{cat}.jsonl"));
            let text = fs::read_to_string(&path).unwrap_or_default();
            let items = parse_items(&text).map_err(|e| format!("{}: {e}", path.display()))?;
            if items.is_empty() {
                missing.push(cat);
            }
            all.push_str(&text);
            sets.push(items);
        }
        if !missing.is_empty() {
            return Err(format!(
                "no {} questions in {}",
                missing.join(", "),
                dir.display()
            ));
        }
        Ok(Suite {
            sets,
            fingerprint: format!("{:016x}", fingerprint(&all)),
        })
    }

    fn asked(&self, cat: usize, quick: bool) -> usize {
        let n = self.sets[cat].len();
        if quick { n.min(QUICK_PER_CAT) } else { n }
    }

    /// A run's questions as (category, item), taking the categories in turn
    /// so a run stopped part-way still covers each.
    pub fn questions(&self, quick: bool) -> Vec<(usize, &Item)> {
        let longest = (0..self.sets.len())
            .map(|c| self.asked(c, quick))
            .max()
            .unwrap_or(0);
        let mut out = Vec::new();
        for i in 0..longest {
            for c in 0..self.sets.len() {
                if i < self.asked(c, quick) {
                    out.push((c, &self.sets[c][i]));
                }
            }
        }
        out
    }

    /// Empty tallies, one per category.
    pub fn tallies(&self) -> Vec<Tally> {
        self.sets
            .iter()
            .map(|s| Tally {
                right: 0,
                n: 0,
                total: s.len(),
            })
            .collect()
    }
}

/// `right` of `n` questions asked, from a category of `total`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Tally {
    pub right: usize,
    pub n: usize,
    pub total: usize,
}

impl Tally {
    pub fn share(&self) -> f64 {
        self.right as f64 / self.n.max(1) as f64
    }
}

/// The mean of the category shares, with the 95% margin of predicting the
/// whole suite from the questions asked (0 when all of them were).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Score {
    pub share: f64,
    pub margin: f64,
    pub predicted: bool,
}

impl Score {
    pub fn label(&self) -> String {
        if self.predicted {
            format!("~{:.0}% ±{:.0}", self.share * 100.0, self.margin * 100.0)
        } else {
            format!("{:.0}%", self.share * 100.0)
        }
    }
}

pub fn score(tallies: &[Tally]) -> Option<Score> {
    let asked: Vec<&Tally> = tallies.iter().filter(|t| t.n > 0).collect();
    if asked.is_empty() {
        return None;
    }
    let k = asked.len() as f64;
    let share = asked.iter().map(|t| t.share()).sum::<f64>() / k;
    let var = asked
        .iter()
        .map(|t| {
            // Smoothed, so 15 of 15 still leaves some doubt, and corrected
            // for sampling without replacement from `total`.
            let p = (t.right as f64 + 1.0) / (t.n as f64 + 2.0);
            let fpc = if t.total > 1 {
                t.total.saturating_sub(t.n) as f64 / (t.total - 1) as f64
            } else {
                0.0
            };
            p * (1.0 - p) / t.n as f64 * fpc
        })
        .sum::<f64>()
        / (k * k);
    Some(Score {
        share,
        margin: 1.96 * var.sqrt(),
        predicted: tallies.iter().any(|t| t.n < t.total),
    })
}

// ---- intelligence runs ----

/// What an intelligence run measures.
#[derive(Clone)]
pub struct EvalPlan {
    pub model: PathBuf,
    pub quick: bool,
    /// Reasoning budget per question (None: thinking off).
    pub thinking: Option<usize>,
    /// Drafter and depth (None: plain decoding). Greedy speculation gives
    /// plain decoding's answers up to near-tie flips, only faster.
    pub draft: Option<(Candidate, usize)>,
    pub context: usize,
    pub kv_cache: KvCache,
}

impl EvalPlan {
    pub fn draft_label(&self) -> String {
        match &self.draft {
            Some((c, d)) => format!("{} · depth {d} · {}", c.label(), c.quant),
            None => "plain decoding".into(),
        }
    }

    /// Rough run time: typical answers at a typical 30 tok/s.
    pub fn estimate_secs(&self, suite: &Suite) -> usize {
        let tokens: usize = suite
            .questions(self.quick)
            .iter()
            .map(|(_, i)| i.max_tokens / 2 + self.thinking.map_or(0, |b| b * 3 / 4))
            .sum();
        30 + tokens / 30
    }
}

pub enum EvalEvent {
    Stage(String),
    Loaded {
        context: usize,
        weights_gib: f64,
    },
    Answered {
        cat: usize,
        id: String,
        correct: bool,
        answer: Option<String>,
    },
    Failed(String),
    Done,
}

pub struct EvalRunning {
    pub rx: Receiver<EvalEvent>,
    pub stop: Arc<AtomicBool>,
    pub handle: Option<JoinHandle<()>>,
}

pub fn spawn(plan: EvalPlan, suite: Suite) -> Result<EvalRunning, String> {
    let (tx, rx) = channel();
    let stop = Arc::new(AtomicBool::new(false));
    let flag = stop.clone();
    let handle = std::thread::Builder::new()
        .name("ferrum-eval".into())
        .spawn(move || run(plan, suite, tx, flag))
        .map_err(|e| e.to_string())?;
    Ok(EvalRunning {
        rx,
        stop,
        handle: Some(handle),
    })
}

fn run(plan: EvalPlan, suite: Suite, tx: Sender<EvalEvent>, stop: Arc<AtomicBool>) {
    let send = |e: EvalEvent| {
        let _ = tx.send(e);
    };
    let bench_plan = bench::Plan {
        model: plan.model.clone(),
        candidates: Vec::new(),
        quick: plan.quick,
        tokens: 0,
        chunks: false,
        context: plan.context,
        kv_cache: plan.kv_cache,
    };
    let plain = Candidate {
        source: Source::Baseline,
        quant: "q4_0",
        on: true,
    };
    let (cand, depth) = match &plan.draft {
        Some((c, d)) => (c, Some(*d)),
        None => (&plain, None),
    };
    send(EvalEvent::Stage(format!("loading {}", plan.draft_label())));
    let loaded = bench::load(&bench_plan, cand, depth, None)
        .and_then(|mut rt| rt.warmup().map(|_| rt).map_err(|e| e.to_string()));
    let mut rt = match loaded {
        Ok(rt) => rt,
        Err(e) => {
            send(EvalEvent::Failed(e));
            send(EvalEvent::Done);
            return;
        }
    };
    send(EvalEvent::Loaded {
        context: rt.loaded.plan.context,
        weights_gib: rt.loaded.plan.weights as f64 / (1u64 << 30) as f64,
    });
    rt.session.depth_cap = depth;
    rt.session.speculate = depth.is_some();
    let questions = suite.questions(plan.quick);
    for (n, (cat, item)) in questions.iter().enumerate() {
        if stop.load(Ordering::SeqCst) {
            break;
        }
        send(EvalEvent::Stage(format!(
            "{} · question {}/{}",
            CATEGORIES[*cat].1,
            n + 1,
            questions.len()
        )));
        match bench::generate(
            &mut rt,
            &item.message(),
            item.max_tokens,
            plan.thinking,
            &stop,
        ) {
            // A stopped generation is cut short: do not grade it.
            Ok(_) if stop.load(Ordering::SeqCst) => break,
            Ok(s) => {
                let g = grade(&item.check, &s.text);
                send(EvalEvent::Answered {
                    cat: *cat,
                    id: item.id.clone(),
                    correct: g.correct,
                    answer: g.answer,
                });
            }
            Err(e) => {
                send(EvalEvent::Failed(format!("{}: {e}", item.id)));
                break;
            }
        }
    }
    send(EvalEvent::Done);
}

// ---- saved intelligence results ----

#[derive(Clone, Debug, PartialEq)]
pub struct EvalResult {
    pub model: String,
    pub when: u64,
    pub quick: bool,
    pub thinking: Option<usize>,
    /// `EvalPlan::draft_label`.
    pub draft: String,
    pub kv_cache: String,
    /// `Suite::fingerprint`: results of different question files never mix.
    pub suite: String,
    /// One per category, in `CATEGORIES` order.
    pub tallies: Vec<Tally>,
    /// Questions answered wrong.
    pub wrong: Vec<String>,
}

impl EvalResult {
    pub fn score(&self) -> Option<Score> {
        score(&self.tallies)
    }

    fn same_setup(&self, o: &Self) -> bool {
        (
            &self.model,
            self.quick,
            self.thinking,
            &self.draft,
            &self.kv_cache,
            &self.suite,
        ) == (
            &o.model,
            o.quick,
            o.thinking,
            &o.draft,
            &o.kv_cache,
            &o.suite,
        )
    }

    /// "quick · thinking off · plain decoding"
    pub fn setup_label(&self) -> String {
        format!(
            "{} · thinking {} · {}",
            if self.quick { "quick" } else { "full" },
            self.thinking.map_or("off".to_string(), |b| format!("≤{b}")),
            self.draft
        )
    }

    fn to_json(&self) -> Json {
        let tallies: Map<String, Json> = CATEGORIES
            .iter()
            .zip(&self.tallies)
            .map(|((cat, _), t)| {
                (
                    cat.to_string(),
                    json!({"right": t.right, "n": t.n, "total": t.total}),
                )
            })
            .collect();
        json!({
            "model": self.model, "when": self.when, "quick": self.quick,
            "thinking": self.thinking, "draft": self.draft, "kv_cache": self.kv_cache,
            "suite": self.suite, "tallies": tallies, "wrong": self.wrong,
        })
    }

    fn from_json(r: &Json) -> Self {
        let n = |v: &Json| v.as_u64().unwrap_or(0) as usize;
        Self {
            model: r["model"].as_str().unwrap_or("").into(),
            when: r["when"].as_u64().unwrap_or(0),
            quick: r["quick"].as_bool().unwrap_or(true),
            thinking: r["thinking"].as_u64().map(|b| b as usize),
            draft: r["draft"].as_str().unwrap_or("").into(),
            kv_cache: r["kv_cache"].as_str().unwrap_or("").into(),
            suite: r["suite"].as_str().unwrap_or("").into(),
            tallies: CATEGORIES
                .iter()
                .map(|(cat, _)| {
                    let t = &r["tallies"][cat];
                    Tally {
                        right: n(&t["right"]),
                        n: n(&t["n"]),
                        total: n(&t["total"]),
                    }
                })
                .collect(),
            wrong: r["wrong"]
                .as_array()
                .map(|a| {
                    a.iter()
                        .filter_map(|w| w.as_str().map(str::to_string))
                        .collect()
                })
                .unwrap_or_default(),
        }
    }
}

pub fn results_path() -> PathBuf {
    home().join(".config/ferrum/eval.json")
}

pub fn load_results(path: &Path) -> Vec<EvalResult> {
    fs::read_to_string(path)
        .ok()
        .and_then(|t| serde_json::from_str::<Json>(&t).ok())
        .and_then(|j| {
            j["runs"]
                .as_array()
                .map(|a| a.iter().map(EvalResult::from_json).collect())
        })
        .unwrap_or_default()
}

/// Keep the newest result per setup, and at most 60.
pub fn record(results: &mut Vec<EvalResult>, run: EvalResult) {
    results.retain(|r| !r.same_setup(&run));
    results.insert(0, run);
    results.truncate(60);
}

pub fn write_results(path: &Path, results: &[EvalResult]) {
    let runs: Vec<Json> = results.iter().map(EvalResult::to_json).collect();
    if let Some(dir) = path.parent() {
        fs::create_dir_all(dir).ok();
    }
    fs::write(
        path,
        serde_json::to_string_pretty(&json!({"runs": runs})).unwrap_or_default(),
    )
    .ok();
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rule(json: Json) -> Rule {
        parse_rule(&json).unwrap()
    }

    #[test]
    fn bundled_check_items_are_valid() {
        let items = parse_items(CHECK_ITEMS).unwrap();
        assert_eq!(items.len(), 30);
        let ids: HashSet<&str> = items.iter().map(|i| i.id.as_str()).collect();
        assert_eq!(ids.len(), items.len(), "ids are unique");
        assert!(items.iter().all(|i| matches!(i.check, Check::Number(_))));
        assert!(items[0].message().ends_with("\"Answer: <number>\"."));
    }

    #[test]
    fn parses_every_check_kind() {
        let text = r#"
{"id":"a","cat":"math","prompt":"p","check":"number","answer":"1,416"}
{"id":"b","cat":"knowledge","prompt":"p","check":"choice","answer":"C"}
{"id":"c","cat":"code","prompt":"p","check":"literal","answer":"'ab'","max_tokens":64}
{"id":"d","cat":"instructions","prompt":"p","check":"rules","rules":[{"id":"punctuation:no_comma"},{"id":"length_constraints:number_words","relation":"at least","num_words":3}]}
"#;
        let items = parse_items(text).unwrap();
        assert_eq!(items[0].check, Check::Number(1416.0));
        assert_eq!(items[1].check, Check::Choice('C'));
        assert_eq!(items[2].max_tokens, 64);
        assert_eq!(
            items[3].check,
            Check::Rules(vec![Rule::NoComma, Rule::Words { less: false, n: 3 }])
        );
        assert_eq!(items[3].message(), "p");
        assert!(
            parse_items(r#"{"id":"x","cat":"c","prompt":"p","check":"choice","answer":"Z"}"#)
                .is_err()
        );
        assert!(
            parse_items(r#"{"id":"x","cat":"c","prompt":"p","check":"rules","rules":[{"id":"language:response_language"}]}"#)
                .unwrap_err()
                .contains("unsupported")
        );
    }

    #[test]
    fn grades_numbers() {
        let g = |t: &str| grade(&Check::Number(1416.0), t);
        assert!(g("so 1,326 + 90 = 1,416 - wait.\nAnswer: 1416").correct);
        assert!(g("**Answer:** $1,416.00").correct);
        assert!(g("The total is \\boxed{1416}.").correct);
        assert!(g("Without a marker the last number counts: 1416").correct);
        // The marker wins over a later stray number.
        assert!(!g("Answer: 1400 (rounded from 1416)").correct);
        assert_eq!(g("nothing here").answer, None);
        assert!(grade(&Check::Number(-3.0), "Answer: -3").correct);
        assert!(grade(&Check::Number(32.4), "Answer: $32.40").correct);
        assert_eq!(numbers("2.5 h, 1.2.3, x-4"), vec![2.5, 1.2, 3.0, 4.0]);
    }

    #[test]
    fn grades_choices_and_literals() {
        let c = |t: &str| grade(&Check::Choice('B'), t).correct;
        assert!(c("Answer: B"));
        assert!(c("The answer is (B) because ..."));
        assert!(c("**Answer: B**"));
        assert!(c("B. Paris"));
        assert!(!c("Answer: Because of A"));
        assert!(!c("Answer: A"));
        let l = |gold: &str, t: &str| grade(&Check::Literal(gold.into()), t).correct;
        assert!(l("['a', 1]", "It returns a list.\nAnswer: [\"a\",1]"));
        assert!(l("'abc'", "Answer: `'abc'`."));
        assert!(l("{'x': 2}", "Answer: {'x':2}"));
        assert!(!l("'abc'", "Answer: 'abd'"));
        assert!(!l("3", "it is 3"), "a literal needs the marker");
    }

    #[test]
    fn checks_ifeval_rules() {
        let words = rule(
            json!({"id":"length_constraints:number_words","relation":"less than","num_words":4}),
        );
        assert!(words.holds("one two three"));
        assert!(!words.holds("one two three four"));
        assert!(Rule::NoComma.holds("no commas here"));
        assert!(!Rule::NoComma.holds("a, b"));
        assert!(Rule::Lowercase.holds("all lower 123"));
        assert!(Rule::Uppercase.holds("ALL UPPER!"));
        assert!(!Rule::Uppercase.holds("Not"));
        assert!(Rule::Paragraphs(2).holds("first\n***\nsecond"));
        assert!(!Rule::Paragraphs(3).holds("first\n***\nsecond"));
        assert!(Rule::Bullets(2).holds("intro\n* one\n- two\n**bold** line"));
        assert!(Rule::Highlights(2).holds("an *important* and **bold** part"));
        assert!(!Rule::Highlights(1).holds("a ** b"));
        assert!(Rule::Title.holds("<<My Title>>\nbody"));
        assert!(!Rule::Title.holds("<<>> body"));
        assert!(Rule::Json.holds("```json\n{\"a\": 1}\n```"));
        assert!(!Rule::Json.holds("{a: 1}"));
        assert!(Rule::Keywords(vec!["Rust".into(), "metal".into()]).holds("rust on Metal"));
        let forbidden = Rule::Forbidden(vec!["cat".into()]);
        assert!(forbidden.holds("concatenate"));
        assert!(!forbidden.holds("the Cat sat"));
        let letter = rule(
            json!({"id":"keywords:letter_frequency","letter":"Z","let_relation":"at least","let_frequency":2}),
        );
        assert!(letter.holds("zigzag"));
        assert!(!letter.holds("zag"));
        assert!(
            Rule::EndsWith("Any other questions?".into()).holds("text. any other questions?\n")
        );
        assert!(Rule::Quoted.holds("  \"quoted\" "));
        assert!(!Rule::Quoted.holds("\""));
        assert!(Rule::Postscript("P.S.".into()).holds("bye\n\np.s. see you"));
        let g = grade(&Check::Rules(vec![Rule::NoComma, Rule::Quoted]), "\"a, b\"");
        assert!(!g.correct);
        assert_eq!(g.answer.as_deref(), Some("01"));
    }

    #[test]
    fn verdicts_allow_one_net_loss() {
        let g = |correct: bool, a: &str| Graded {
            correct,
            answer: Some(a.into()),
        };
        let base = vec![g(true, "1"), g(true, "2"), g(false, "9"), g(true, "4")];
        let same = compare_answers(&base, &base);
        assert_eq!((same.lost, same.gained, same.same), (0, 0, 4));
        assert!(same.passed());
        let one = compare_answers(
            &base,
            &[g(false, "0"), g(true, "2"), g(false, "9"), g(true, "4")],
        );
        assert_eq!((one.lost, one.same), (1, 3));
        assert!(one.passed());
        let two = compare_answers(
            &base,
            &[g(false, "0"), g(false, "0"), g(false, "9"), g(true, "4")],
        );
        assert!(!two.passed());
        // A loss offset by a gain is noise, not a regression.
        let swap = compare_answers(
            &base,
            &[g(false, "0"), g(false, "0"), g(true, "3"), g(true, "4")],
        );
        assert_eq!((swap.lost, swap.gained), (2, 1));
        assert!(swap.passed());
        assert!(!compare_answers(&[], &[]).passed());
    }

    /// Headless answer check of one drafter against plain decoding, for
    /// drafter and kernel changes. Baseline answers are recomputed, not cached.
    ///
    /// FERRUM_CHECK_MODEL=m.gguf FERRUM_CHECK_DRAFTER=dir|mtp [FERRUM_CHECK_DEPTH=4]
    /// [FERRUM_CHECK_QUANT=q8_0] cargo test --release --bin ferrum-tui -- --ignored --nocapture keeps_its_answers
    #[test]
    #[ignore = "needs a model and a drafter on disk"]
    fn drafter_keeps_its_answers() {
        use crate::bench::{Candidate, Plan, Source, answer_check};
        let var = |k: &str| std::env::var(k).ok();
        let model = PathBuf::from(var("FERRUM_CHECK_MODEL").expect("FERRUM_CHECK_MODEL"));
        let drafter = var("FERRUM_CHECK_DRAFTER").expect("FERRUM_CHECK_DRAFTER");
        let depth: usize = var("FERRUM_CHECK_DEPTH").map_or(4, |d| d.parse().unwrap());
        let cand = |source| Candidate {
            source,
            quant: if var("FERRUM_CHECK_QUANT").as_deref() == Some("q8_0") {
                "q8_0"
            } else {
                "q4_0"
            },
            on: true,
        };
        let draft = cand(if drafter == "mtp" {
            Source::Mtp
        } else {
            Source::Drafter {
                path: drafter.into(),
                label: "drafter".into(),
                kind: "drafter",
                max_depth: depth,
            }
        });
        let plan = Plan {
            model,
            candidates: Vec::new(),
            quick: true,
            tokens: 0,
            chunks: false,
            context: 8192,
            kv_cache: KvCache::default(),
        };
        let items = check_items();
        let stop = AtomicBool::new(false);
        let quiet = |_| {};
        let base =
            answer_check(&plan, &cand(Source::Baseline), None, &items, &quiet, &stop).unwrap();
        for (item, g) in items.iter().zip(&base).filter(|(_, g)| !g.correct) {
            eprintln!("plain decoding missed {}: {:?}", item.id, g.answer);
        }
        let run = answer_check(&plan, &draft, Some(depth), &items, &quiet, &stop).unwrap();
        let v = compare_answers(&base, &run);
        let right = base.iter().filter(|g| g.correct).count();
        eprintln!("plain decoding {right}/{}; drafter {v:?}", items.len());
        assert!(v.passed(), "the drafter lost answers: {v:?}");
    }

    fn suite(sizes: [usize; 4]) -> Suite {
        let item = |c: usize, i: usize| Item {
            id: format!("{c}-{i}"),
            prompt: "p".into(),
            check: Check::Number(1.0),
            max_tokens: 100,
        };
        Suite {
            sets: sizes
                .iter()
                .enumerate()
                .map(|(c, &n)| (0..n).map(|i| item(c, i)).collect())
                .collect(),
            fingerprint: "f".into(),
        }
    }

    #[test]
    fn loads_the_suite_from_a_folder() {
        let dir = std::env::temp_dir().join(format!("ferrum-suite-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let line = |id: &str| {
            format!(r#"{{"id":"{id}","cat":"c","prompt":"p","check":"number","answer":"1"}}"#)
        };
        for (cat, _) in &CATEGORIES[..3] {
            fs::write(dir.join(format!("{cat}.jsonl")), line(cat)).unwrap();
        }
        let err = Suite::load(&dir).unwrap_err();
        assert!(err.contains("code"), "{err}");
        fs::write(dir.join("code.jsonl"), line("code")).unwrap();
        let s = Suite::load(&dir).unwrap();
        assert_eq!(s.sets.iter().map(Vec::len).sum::<usize>(), 4);
        let before = s.fingerprint.clone();
        fs::write(dir.join("code.jsonl"), line("code2")).unwrap();
        assert_ne!(Suite::load(&dir).unwrap().fingerprint, before);
        fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn questions_take_categories_in_turn() {
        let s = suite([20, 3, 20, 20]);
        let quick = s.questions(true);
        assert_eq!(quick.len(), 15 + 3 + 15 + 15);
        let first: Vec<usize> = quick.iter().take(5).map(|(c, _)| *c).collect();
        assert_eq!(first, vec![0, 1, 2, 3, 0]);
        assert_eq!(s.questions(false).len(), 63);
        assert_eq!(s.tallies()[1].total, 3);
    }

    #[test]
    fn scores_predict_from_a_sample() {
        let t = |right, n, total| Tally { right, n, total };
        let full = score(&[t(80, 100, 100), t(60, 100, 100)]).unwrap();
        assert!((full.share - 0.7).abs() < 1e-9);
        assert_eq!(full.margin, 0.0);
        assert!(!full.predicted);
        assert_eq!(full.label(), "70%");
        let quick = score(&[t(12, 15, 100), t(9, 15, 100)]).unwrap();
        assert!(quick.predicted);
        assert!((quick.share - 0.7).abs() < 1e-9);
        assert!(
            quick.margin > 0.05 && quick.margin < 0.25,
            "{}",
            quick.margin
        );
        assert!(quick.label().starts_with("~70% ±"));
        // A perfect sample still leaves some doubt.
        assert!(score(&[t(15, 15, 100)]).unwrap().margin > 0.0);
        // Unasked categories do not count.
        assert_eq!(score(&[t(5, 10, 10), t(0, 0, 10)]).unwrap().share, 0.5);
        assert_eq!(score(&[t(0, 0, 10)]), None);
    }

    #[test]
    fn saves_results_per_setup() {
        let dir = std::env::temp_dir().join(format!("ferrum-evalres-{}", std::process::id()));
        let path = dir.join("eval.json");
        let run = |quick, right| EvalResult {
            model: "/m.gguf".into(),
            when: 5,
            quick,
            thinking: Some(1024),
            draft: "plain decoding".into(),
            kv_cache: "F16".into(),
            suite: "abc".into(),
            tallies: vec![
                Tally {
                    right,
                    n: 15,
                    total: 100
                };
                4
            ],
            wrong: vec!["gsm8k-1".into()],
        };
        let mut results = Vec::new();
        record(&mut results, run(true, 10));
        record(&mut results, run(false, 70));
        record(&mut results, run(true, 12));
        assert_eq!(results.len(), 2, "same setup replaced");
        assert_eq!(results[0].tallies[0].right, 12);
        write_results(&path, &results);
        assert_eq!(load_results(&path), results);
        assert_eq!(
            results[1].setup_label(),
            "full · thinking ≤1024 · plain decoding"
        );
        fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn caches_baselines_per_key() {
        let dir = std::env::temp_dir().join(format!("ferrum-eval-{}", std::process::id()));
        let path = dir.join("answer-check.json");
        let key = CheckKey {
            model: "/m.gguf".into(),
            size: 10,
            modified: 1,
            kv_cache: "F16".into(),
            items: 7,
        };
        let answers = vec![
            Graded {
                correct: true,
                answer: Some("4".into()),
            },
            Graded {
                correct: false,
                answer: None,
            },
        ];
        assert_eq!(cached_baseline(&path, &key), None);
        store_baseline(&path, &key, &answers);
        assert_eq!(cached_baseline(&path, &key), Some(answers.clone()));
        let other = CheckKey {
            kv_cache: "Q8".into(),
            ..key.clone()
        };
        assert_eq!(cached_baseline(&path, &other), None);
        store_baseline(&path, &key, &answers[..1]);
        assert_eq!(read_cache(&path).len(), 1, "replaced, not duplicated");
        fs::remove_dir_all(dir).ok();
    }
}
