//! What the coding agent is told, and how its context is kept small: the
//! system prompt, ponytail (minimal-code) modes, a repo map, and compaction of
//! the history when the context fills.
use crate::coding::{IGNORED, Shared, TodoState, Touch};
use crate::store::Store;
use crate::*;
use std::collections::BTreeMap;

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Ponytail {
    Off,
    Lite,
    Full,
    Ultra,
}

impl Ponytail {
    pub fn parse(s: &str) -> Self {
        match s {
            "off" => Self::Off,
            "lite" => Self::Lite,
            "ultra" => Self::Ultra,
            _ => Self::Full,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Lite => "lite",
            Self::Full => "full",
            Self::Ultra => "ultra",
        }
    }

    /// The guidance added to the system prompt.
    pub fn text(self) -> &'static str {
        match self {
            Self::Off => "",
            Self::Lite => {
                "PONYTAIL (lite): be lazy about the solution, never about reading. Read the code \
before you write any. Make the smallest change that works; do not add abstractions, options or \
files nobody asked for. Never skip validation, error handling, security or accessibility."
            }
            Self::Full => {
                "PONYTAIL (full): be lazy about the solution, never about reading. Read the code \
first, then climb this ladder and stop at the first rung that solves the problem:\n\
1. Does this need to exist? If not, skip it.\n\
2. Is it already in this codebase? Reuse it.\n\
3. Does the standard library do it? Use that.\n\
4. Does the platform or language have it built in? Use that.\n\
5. Does an already-installed dependency do it? Use that.\n\
6. Is it one line? Write one line.\n\
7. Only then write the minimum that works.\n\
Never skip validation, error handling, security or accessibility. When you knowingly take a \
shortcut or defer something, leave a comment starting `ponytail:` that says what was deferred."
            }
            Self::Ultra => {
                "PONYTAIL (ultra): be lazy about the solution, never about reading. Read the code \
first, then climb this ladder and stop at the first rung that solves the problem: (1) does it need \
to exist? (2) already in this codebase? (3) standard library? (4) built into the platform? \
(5) an installed dependency? (6) one line? (7) the minimum that works.\n\
Be aggressive: delete before you add. No new file, dependency, config option, class or abstraction \
unless the task cannot be done without it. Three similar lines beat a helper. If the code is \
over-engineered, say so and remove what is not needed instead of extending it. Never skip \
validation, error handling, security or accessibility. Mark every deferred shortcut with a \
comment starting `ponytail:`."
            }
        }
    }
}

/// The text of the project's own instructions, if it has any.
pub fn instructions(root: &Path) -> Option<String> {
    for name in ["AGENTS.md", "CLAUDE.md", ".ferrum/AGENTS.md"] {
        if let Ok(text) = fs::read_to_string(root.join(name)) {
            let text = text.trim();
            if !text.is_empty() {
                return Some(format!(
                    "{name}:\n{}",
                    crate::agent::clip_output(text, 4_000)
                ));
            }
        }
    }
    None
}

fn git(root: &Path, args: &[&str]) -> Option<String> {
    let out = Command::new("git")
        .arg("-C")
        .arg(root)
        .args(args)
        .output()
        .ok()?;
    out.status
        .success()
        .then(|| String::from_utf8_lossy(&out.stdout).trim().to_string())
}

/// Uncommitted changes (staged and not).
pub fn git_diff(root: &Path) -> Option<String> {
    git(root, &["diff", "HEAD", "--no-color"])
}

pub fn git_branch(root: &Path) -> Option<String> {
    git(root, &["rev-parse", "--abbrev-ref", "HEAD"]).filter(|b| !b.is_empty())
}

pub fn git_dirty(root: &Path) -> usize {
    git(root, &["status", "--porcelain"]).map_or(0, |s| s.lines().count())
}

/// A compact picture of the project: top-level layout, languages, git state.
pub fn repo_map(root: &Path) -> String {
    let mut counts: BTreeMap<String, usize> = BTreeMap::new();
    let mut top: Vec<(String, usize, bool)> = Vec::new();
    let Ok(entries) = fs::read_dir(root) else {
        return String::new();
    };
    let mut entries: Vec<_> = entries.flatten().collect();
    entries.sort_by_key(|e| (!e.path().is_dir(), e.file_name()));
    for e in entries {
        let name = e.file_name().to_string_lossy().to_string();
        if IGNORED.contains(&name.as_str()) || name.starts_with('.') {
            continue;
        }
        if e.path().is_dir() {
            let mut files = Vec::new();
            collect(&e.path(), &mut files, 4_000);
            for f in &files {
                if let Some(ext) = f.extension() {
                    *counts
                        .entry(ext.to_string_lossy().to_lowercase())
                        .or_default() += 1;
                }
            }
            top.push((name, files.len(), true));
        } else {
            if let Some(ext) = e.path().extension() {
                *counts
                    .entry(ext.to_string_lossy().to_lowercase())
                    .or_default() += 1;
            }
            top.push((name, 1, false));
        }
    }
    let mut out = String::new();
    if let Some(branch) = git_branch(root) {
        out += &format!(
            "git: branch {branch}, {} changed file(s)\n",
            git_dirty(root)
        );
    }
    let mut langs: Vec<_> = counts.into_iter().collect();
    langs.sort_by_key(|l| std::cmp::Reverse(l.1));
    let shown: Vec<String> = langs
        .iter()
        .take(6)
        .map(|(e, n)| format!(".{e} ×{n}"))
        .collect();
    if !shown.is_empty() {
        out += &format!("files: {}\n", shown.join(", "));
    }
    out += "layout:\n";
    for (name, n, dir) in top.iter().take(40) {
        if *dir {
            out += &format!("  {name}/ ({n} files)\n");
        } else {
            out += &format!("  {name}\n");
        }
    }
    crate::agent::clip_output(&out, 2_500)
}

fn collect(dir: &Path, out: &mut Vec<PathBuf>, cap: usize) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for e in entries.flatten() {
        if out.len() >= cap {
            return;
        }
        let name = e.file_name().to_string_lossy().to_string();
        let Ok(kind) = e.file_type() else { continue };
        if kind.is_dir() {
            if !IGNORED.contains(&name.as_str()) && !name.starts_with('.') {
                collect(&e.path(), out, cap);
            }
        } else if kind.is_file() {
            out.push(e.path());
        }
    }
}

/// Which common commands exist, so the model does not guess (`python` vs `python3`).
pub fn env_notes() -> String {
    let h = home();
    let dirs = [
        h.join(".cargo/bin"),
        h.join(".local/bin"),
        PathBuf::from("/opt/homebrew/bin"),
        PathBuf::from("/usr/local/bin"),
        PathBuf::from("/usr/bin"),
        PathBuf::from("/bin"),
    ];
    let has = |c: &str| dirs.iter().any(|d| d.join(c).exists());
    let found: Vec<&str> = [
        "python3", "python", "node", "npm", "cargo", "go", "git", "rg", "make",
    ]
    .into_iter()
    .filter(|c| has(c))
    .collect();
    let mut s = format!("commands available: {}\n", found.join(", "));
    if !has("python") && has("python3") {
        s += "there is no `python`; use `python3`\n";
    }
    s
}

/// The agent's standing instructions.
pub fn system_prompt(
    project: &Path,
    ponytail: Ponytail,
    plan: bool,
    network: bool,
    map: &str,
    project_notes: Option<&str>,
) -> String {
    let mut s = format!(
        "You are Ferrum Agent, a coding agent running on a local model in the user's terminal. \
You work in the project at {}. That folder is the only place you can change; commands run in a sandbox{}.\n\n\
HOW TO WORK\n\
- Act rather than narrate. Do not restate the task or announce what you are about to do.\n\
- Observe, find the likely cause, make the smallest fix, then test it. Prefer surgical edits to rewriting files.\n\
- Use targeted searches (grep, glob) and known paths first; read only the part of a file you need.\n\
- Once direct evidence shows the cause, stop exploring and fix it. Once a change works and the relevant check passes, stop.\n\
- After editing, run the smallest relevant check or test and read the result. Use `check` for a quick error scan.\n\
- Never change unrelated code or \"improve\" things you were not asked to.\n\
- For any task with several steps keep a plan with the `todo` tool: one item active at a time, mark items done as you go.\n\
- If you are blocked or the choice is the user's, use `ask_user` with one short question.\n\
- When finished, answer in a few lines: what you changed and how you checked it.\n\n\
CONTEXT\n\
Your context is small and precious. Long tool output is stored instead of shown: you get a short excerpt and a handle like #7. \
Use `ctx_search` or `ctx_read` to get more. Give `bash` an `intent` (what you are looking for) so only the relevant lines come back. \
Re-reading an unchanged file returns a notice; use the earlier output. Old tool output may be elided later; the handle still works.",
        project.display(),
        if network {
            ", with network access"
        } else {
            ", without network access"
        }
    );
    if plan {
        s += "\n\nPLAN MODE: you can only read, search and plan. Do not change anything. Explore, then write a clear, \
step-by-step plan (files to touch, what to change, how to verify) and stop.";
    }
    let p = ponytail.text();
    if !p.is_empty() {
        s += &format!("\n\n{p}");
    }
    if !map.is_empty() {
        s += &format!("\n\nPROJECT\n{map}");
    }
    if let Some(notes) = project_notes {
        s += &format!("\n\nPROJECT INSTRUCTIONS\n{notes}");
    }
    s
}

pub fn review_prompt(diff: &str) -> String {
    format!(
        "Ponytail review. Look at this diff for over-engineering: code that does not need to exist, \
duplicates something already in the codebase, reimplements the standard library or a platform feature, \
or adds an abstraction, option, file or dependency nobody needed. For each finding give file and line, \
what to delete or replace, and what to use instead. End with a short deletion list. Do not edit anything.\n\n```diff\n{}\n```",
        crate::agent::clip_output(diff, 24_000)
    )
}

pub const AUDIT_PROMPT: &str = "Ponytail audit. Audit the whole project, not just recent changes, for code that does not need to exist: \
dead code, duplicated logic, reimplemented library features, speculative abstractions, unused options and dependencies. \
Explore with list_dir, glob and grep, and read only what you must. Report a ranked list: file, what to remove or simplify, and roughly how many lines it saves. \
Do not edit anything.";

pub const DEBT_PROMPT: &str = "Ponytail debt. Use grep to find every comment containing `ponytail:` in the project. \
Turn them into a ledger: file:line, what was deferred, the risk of leaving it, and the smallest fix. Order by risk. \
If there are none, say so. Do not edit anything.";

// ---- compaction ----

/// Text of a message, whether its content is a string or a list of parts.
fn text_of(message: &Json) -> String {
    match &message["content"] {
        Json::String(s) => s.clone(),
        Json::Array(parts) => parts
            .iter()
            .filter_map(|p| p["text"].as_str())
            .collect::<Vec<_>>()
            .join("\n"),
        _ => String::new(),
    }
}

/// Stub out the content of older tool results (all but the last `keep`),
/// storing the full text first so it can be fetched back. Returns
/// (messages stubbed, characters saved).
pub fn evict_old(history: &mut [Json], store: &mut Store, keep: usize) -> (usize, usize) {
    let tool_idx: Vec<usize> = history
        .iter()
        .enumerate()
        .filter(|(_, m)| m["role"] == "tool")
        .map(|(i, _)| i)
        .collect();
    let cut = tool_idx.len().saturating_sub(keep);
    let (mut count, mut saved) = (0, 0);
    for &i in &tool_idx[..cut] {
        let text = text_of(&history[i]);
        if text.len() <= 500 || text.starts_with("[elided") {
            continue;
        }
        let name = history[i]["name"].as_str().unwrap_or("tool").to_string();
        let id = text
            .strip_prefix("[output #")
            .and_then(|r| r.split(':').next())
            .and_then(|n| n.parse::<usize>().ok())
            .unwrap_or_else(|| store.add(&format!("earlier {name} output"), &text));
        let stub = format!(
            "[elided to save context: {} characters from {name}; stored as output #{id}. ctx_read(id={id}) or ctx_search(query) brings it back]",
            text.len()
        );
        saved += text.len().saturating_sub(stub.len());
        history[i]["content"] = Json::String(stub);
        count += 1;
    }
    store.saved += saved;
    (count, saved)
}

/// A structured summary of the session so far, with no model call.
pub fn snapshot(history: &[Json], shared: &Shared, project: &Path) -> String {
    let users: Vec<String> = history
        .iter()
        .filter(|m| m["role"] == "user")
        .map(text_of)
        .filter(|t| !t.starts_with("<session-snapshot>"))
        .collect();
    let mut s = String::from(
        "<session-snapshot>\nThe conversation was compacted to save context. What matters:\n",
    );
    s += &format!("project: {}\n", project.display());
    if let Some(first) = users.first() {
        s += &format!(
            "original request: {}\n",
            crate::agent::clip_output(first.trim(), 600)
        );
    }
    if users.len() > 1
        && let Some(last) = users.last()
    {
        s += &format!(
            "latest request: {}\n",
            crate::agent::clip_output(last.trim(), 600)
        );
    }
    if !shared.todos.is_empty() {
        s += "plan:\n";
        for t in &shared.todos {
            let mark = match t.state {
                TodoState::Done => "[x]",
                TodoState::Active => "[>]",
                TodoState::Pending => "[ ]",
            };
            s += &format!("  {mark} {}\n", t.text);
        }
    }
    let edited: Vec<&String> = shared
        .touched
        .iter()
        .filter(|(_, t)| **t == Touch::Edited)
        .map(|(p, _)| p)
        .collect();
    if !edited.is_empty() {
        s += &format!(
            "files you changed: {}\n",
            edited
                .iter()
                .map(|p| p.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        );
    }
    let read: Vec<&String> = shared
        .touched
        .iter()
        .filter(|(_, t)| **t == Touch::Read)
        .map(|(p, _)| p)
        .take(12)
        .collect();
    if !read.is_empty() {
        s += &format!(
            "files you read: {}\n",
            read.iter()
                .map(|p| p.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        );
    }
    // The last few commands and how they ended.
    let mut results: BTreeMap<String, String> = BTreeMap::new();
    for m in history.iter().filter(|m| m["role"] == "tool") {
        if let Some(id) = m["tool_call_id"].as_str() {
            results.insert(id.to_string(), text_of(m));
        }
    }
    let mut commands = Vec::new();
    for m in history.iter() {
        for call in m["tool_calls"].as_array().into_iter().flatten() {
            if call["function"]["name"] == "bash" {
                let args: Json = call["function"]["arguments"]
                    .as_str()
                    .and_then(|a| serde_json::from_str(a).ok())
                    .unwrap_or_else(|| call["function"]["arguments"].clone());
                let cmd = args["command"]
                    .as_str()
                    .unwrap_or("")
                    .lines()
                    .next()
                    .unwrap_or("")
                    .to_string();
                let outcome = call["id"]
                    .as_str()
                    .and_then(|id| results.get(id))
                    .map(|r| {
                        if r.starts_with("error:")
                            || r.contains("[exit code") && !r.contains("[exit code 0]")
                        {
                            "failed"
                        } else {
                            "ok"
                        }
                    })
                    .unwrap_or("?");
                commands.push(format!(
                    "  {outcome}: {}",
                    crate::agent::clip_output(&cmd, 100)
                ));
            }
        }
    }
    if !commands.is_empty() {
        s += "recent commands:\n";
        for c in commands.iter().rev().take(6).rev() {
            s += &format!("{c}\n");
        }
    }
    if let Some(err) = results.values().rev().find(|r| r.starts_with("error:")) {
        s += &format!(
            "last error: {}\n",
            crate::agent::clip_output(err.lines().next().unwrap_or(""), 240)
        );
    }
    if !shared.store.is_empty() {
        s += &format!(
            "{} long outputs are stored; ctx_search(query) and ctx_read(id) retrieve them.\n",
            shared.store.len()
        );
    }
    s += "</session-snapshot>";
    crate::agent::clip_output(&s, 4_500)
}

/// Replace everything before the latest user message with a snapshot.
/// Returns the characters removed from the history.
pub fn compact(history: &mut Vec<Json>, shared: &Shared, project: &Path) -> usize {
    let Some(last_user) = history
        .iter()
        .rposition(|m| m["role"] == "user" && !text_of(m).starts_with("<session-snapshot>"))
    else {
        return 0;
    };
    let before: usize = history.iter().map(|m| text_of(m).len()).sum();
    let snap = snapshot(history, shared, project);
    let mut next = vec![
        json!({"role": "user", "content": snap}),
        json!({"role": "assistant", "content": "Understood. I'll continue from the snapshot."}),
    ];
    next.extend(history.drain(last_user..));
    *history = next;
    let after: usize = history.iter().map(|m| text_of(m).len()).sum();
    before.saturating_sub(after)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn session() -> Vec<Json> {
        let mut h = vec![json!({"role": "user", "content": "fix the add function in main.py"})];
        for i in 1..=8 {
            h.push(json!({"role": "assistant", "content": "", "tool_calls": [{
                "id": format!("call_{i}"), "type": "function",
                "function": {"name": "bash", "arguments": json!({"command": format!("cat file{i}")}).to_string()}}]}));
            h.push(json!({"role": "tool", "tool_call_id": format!("call_{i}"), "name": "bash",
                "content": format!("{}\n[exit code {}]", "x".repeat(900), if i == 5 { 2 } else { 0 })}));
        }
        h.push(json!({"role": "user", "content": "now run the tests"}));
        h
    }

    #[test]
    fn evicts_old_outputs_but_keeps_recent_ones() {
        let mut h = session();
        let mut store = Store::default();
        let (n, saved) = evict_old(&mut h, &mut store, 3);
        assert_eq!(n, 5);
        assert!(saved > 3_500, "{saved}");
        let tools: Vec<String> = h
            .iter()
            .filter(|m| m["role"] == "tool")
            .map(text_of)
            .collect();
        assert!(
            tools[0].starts_with("[elided") && tools[0].contains("#1"),
            "{}",
            tools[0]
        );
        assert!(tools[7].starts_with("xxx"), "recent outputs stay");
        assert_eq!(store.len(), 5);
        assert!(store.read(1, 1, 2).unwrap().contains("xxx"));
        // Evicting again changes nothing.
        assert_eq!(evict_old(&mut h, &mut store, 3).0, 0);
    }

    #[test]
    fn snapshot_keeps_the_essentials() {
        let h = session();
        let mut shared = Shared::default();
        shared.touch("main.py", Touch::Edited);
        shared.touch("util.py", Touch::Read);
        shared.todos.push(crate::coding::Todo {
            text: "run tests".into(),
            state: TodoState::Active,
        });
        let s = snapshot(&h, &shared, Path::new("/proj"));
        for want in [
            "fix the add function",
            "main.py",
            "util.py",
            "[>] run tests",
            "failed: cat file5",
        ] {
            assert!(s.contains(want), "missing {want:?} in\n{s}");
        }
    }

    #[test]
    fn compaction_keeps_the_latest_request() {
        let mut h = session();
        let shared = Shared::default();
        let removed = compact(&mut h, &shared, Path::new("/proj"));
        assert!(removed > 3_500, "{removed}");
        assert_eq!(h.len(), 3);
        assert!(text_of(&h[0]).starts_with("<session-snapshot>"));
        assert_eq!(text_of(&h[2]), "now run the tests");
        // A second compaction does not nest snapshots.
        compact(&mut h, &shared, Path::new("/proj"));
        assert_eq!(
            h.iter()
                .filter(|m| text_of(m).starts_with("<session-snapshot>"))
                .count(),
            1
        );
    }

    #[test]
    fn prompts_follow_the_mode() {
        let p = |m| {
            system_prompt(
                Path::new("/p"),
                m,
                false,
                true,
                "layout:\n  src/\n",
                Some("be nice"),
            )
        };
        assert!(!p(Ponytail::Off).contains("PONYTAIL"));
        assert!(p(Ponytail::Lite).contains("lite") && !p(Ponytail::Lite).contains("ladder"));
        assert!(p(Ponytail::Full).contains("Does the standard library"));
        assert!(p(Ponytail::Ultra).contains("delete before you add"));
        assert!(p(Ponytail::Full).contains("be nice") && p(Ponytail::Full).contains("src/"));
        assert!(
            system_prompt(Path::new("/p"), Ponytail::Full, true, false, "", None)
                .contains("PLAN MODE")
        );
    }

    #[test]
    fn maps_a_project() {
        let dir = std::env::temp_dir().join(format!("ferrum-map-test-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(dir.join("src")).unwrap();
        fs::create_dir_all(dir.join("node_modules/x")).unwrap();
        fs::write(dir.join("src/a.rs"), "").unwrap();
        fs::write(dir.join("src/b.rs"), "").unwrap();
        fs::write(dir.join("Cargo.toml"), "").unwrap();
        fs::write(dir.join("AGENTS.md"), "use tabs\n").unwrap();
        let map = repo_map(&dir);
        assert!(
            map.contains("src/ (2 files)") && map.contains("Cargo.toml") && map.contains(".rs ×2"),
            "{map}"
        );
        assert!(!map.contains("node_modules"), "{map}");
        assert!(instructions(&dir).unwrap().contains("use tabs"));
        let _ = fs::remove_dir_all(&dir);
    }
}
