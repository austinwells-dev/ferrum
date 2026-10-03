//! The coding agent's tools. They build on the sandboxed file and shell
//! tools in `agent.rs` and add search, a todo list, background processes,
//! project checks and web search, and they route long output through the
//! context store so it does not flood the model.
use crate::agent::{Sandbox, clip_output};
use crate::store::Store;
use crate::*;
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::io::{BufRead, BufReader};
use std::process::Stdio;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

/// The agent's sandbox keeps command output whole; the store decides what the model sees.
pub const AGENT_OUTPUT_LIMIT: usize = 400_000;

/// Folders no search or listing should wander into.
pub const IGNORED: [&str; 9] = [
    ".git",
    "node_modules",
    "target",
    ".venv",
    "venv",
    "__pycache__",
    ".build",
    "dist",
    ".next",
];

#[derive(Clone, Copy, PartialEq)]
pub enum TodoState {
    Pending,
    Active,
    Done,
}

#[derive(Clone)]
pub struct Todo {
    pub text: String,
    pub state: TodoState,
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Touch {
    Read,
    Edited,
}

pub struct Proc {
    pub name: String,
    pub command: String,
    child: std::process::Child,
    lines: Arc<Mutex<VecDeque<String>>>,
    pub started: Instant,
}

impl Proc {
    pub fn running(&mut self) -> bool {
        matches!(self.child.try_wait(), Ok(None))
    }

    fn kill(&mut self) {
        let _ = Command::new("/bin/kill")
            .args(["-KILL", &format!("-{}", self.child.id())])
            .status();
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// Everything the tools share with each other and with the screen.
#[derive(Default)]
pub struct Shared {
    pub store: Store,
    pub todos: Vec<Todo>,
    pub touched: BTreeMap<String, Touch>,
    pub procs: Vec<Proc>,
    /// (path, offset, limit) -> hash of what the model was shown.
    reads: HashMap<(String, usize, usize), u64>,
    recent: VecDeque<String>,
    /// Old tool outputs removed from the history, and bytes saved doing so.
    pub evicted: usize,
    pub evicted_bytes: usize,
    pub compactions: usize,
}

impl Drop for Shared {
    fn drop(&mut self) {
        for p in &mut self.procs {
            p.kill();
        }
    }
}

pub type SharedRef = Arc<Mutex<Shared>>;

pub fn new_shared() -> SharedRef {
    Arc::new(Mutex::new(Shared::default()))
}

impl Shared {
    /// True when this exact call was already made twice in a row of recent calls.
    pub fn is_loop(&mut self, name: &str, args: &Map<String, Json>) -> bool {
        let sig = format!("{name}:{}", Json::Object(args.clone()));
        self.recent.push_back(sig.clone());
        while self.recent.len() > 8 {
            self.recent.pop_front();
        }
        self.recent.iter().filter(|s| **s == sig).count() >= 3
    }

    pub fn new_message(&mut self) {
        self.recent.clear();
    }

    /// The model no longer has earlier file contents in view.
    pub fn forget_reads(&mut self) {
        self.reads.clear();
    }

    /// Forget the conversation's state; background processes keep running.
    pub fn reset_conversation(&mut self) {
        self.store = Store::default();
        self.todos.clear();
        self.touched.clear();
        self.reads.clear();
        self.recent.clear();
        self.evicted = 0;
        self.evicted_bytes = 0;
        self.compactions = 0;
    }

    pub fn touch(&mut self, path: &str, how: Touch) {
        let entry = self.touched.entry(path.to_string()).or_insert(how);
        if how == Touch::Edited {
            *entry = Touch::Edited;
        }
    }

    /// Everything a resumed session needs besides the messages.
    pub fn snapshot(&self) -> Json {
        json!({
            "todos": self.todos.iter().map(|t| json!({
                "text": t.text,
                "state": match t.state { TodoState::Done => "done", TodoState::Active => "active", TodoState::Pending => "pending" },
            })).collect::<Vec<_>>(),
            "touched": self.touched.iter().map(|(p, t)| (p.clone(), json!(if *t == Touch::Edited { "edited" } else { "read" }))).collect::<Map<String, Json>>(),
            "store": self.store.dump().into_iter().map(|(l, t)| json!({"label": l, "text": t})).collect::<Vec<_>>(),
            "saved": self.store.saved,
            "evicted": self.evicted,
            "evicted_bytes": self.evicted_bytes,
            "compactions": self.compactions,
        })
    }

    pub fn restore(&mut self, j: &Json) {
        self.reset_conversation();
        for t in j["todos"].as_array().into_iter().flatten() {
            self.todos.push(Todo {
                text: t["text"].as_str().unwrap_or("").into(),
                state: match t["state"].as_str() {
                    Some("done") => TodoState::Done,
                    Some("active") => TodoState::Active,
                    _ => TodoState::Pending,
                },
            });
        }
        for (p, how) in j["touched"].as_object().into_iter().flatten() {
            self.touched.insert(
                p.clone(),
                if how == "edited" {
                    Touch::Edited
                } else {
                    Touch::Read
                },
            );
        }
        for item in j["store"].as_array().into_iter().flatten() {
            self.store.add(
                item["label"].as_str().unwrap_or(""),
                item["text"].as_str().unwrap_or(""),
            );
        }
        self.store.saved = j["saved"].as_u64().unwrap_or(0) as usize;
        self.evicted = j["evicted"].as_u64().unwrap_or(0) as usize;
        self.evicted_bytes = j["evicted_bytes"].as_u64().unwrap_or(0) as usize;
        self.compactions = j["compactions"].as_u64().unwrap_or(0) as usize;
    }

    pub fn todo_counts(&self) -> (usize, usize) {
        (
            self.todos
                .iter()
                .filter(|t| t.state == TodoState::Done)
                .count(),
            self.todos.len(),
        )
    }
}

fn def(name: &str, description: &str, properties: Json, required: &[&str]) -> Json {
    json!({
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required}
        }
    })
}

/// Tools that never change anything: what plan mode allows.
pub fn read_only(name: &str) -> bool {
    matches!(
        name,
        "read_file"
            | "list_dir"
            | "glob"
            | "grep"
            | "ctx_search"
            | "ctx_read"
            | "todo"
            | "ask_user"
            | "web_search"
            | "fetch_url"
            | "process_output"
            | "process_list"
            | "check"
    )
}

pub fn needs_approval(name: &str, mode: ToolMode) -> bool {
    match mode {
        ToolMode::Off | ToolMode::Auto => false,
        ToolMode::Ask => matches!(
            name,
            "bash"
                | "write_file"
                | "edit_file"
                | "fetch_url"
                | "web_search"
                | "process_start"
                | "check"
        ),
        ToolMode::Edits => matches!(name, "bash" | "fetch_url" | "web_search" | "process_start"),
    }
}

pub fn tool_defs(plan: bool, network: bool) -> Vec<Json> {
    let mut tools = vec![
        def(
            "read_file",
            "Read a text file with line numbers. Use offset/limit for big files. Re-reading an unchanged range returns a short notice.",
            json!({
                "path": {"type": "string", "description": "Path relative to the project."},
                "offset": {"type": "integer", "description": "First line (1-based, default 1)."},
                "limit": {"type": "integer", "description": "Number of lines (default 400)."}
            }),
            &["path"],
        ),
        def(
            "grep",
            "Search file contents with a regular expression (ripgrep). Returns file:line:text. Prefer this to reading whole files.",
            json!({
                "pattern": {"type": "string", "description": "Regular expression."},
                "path": {"type": "string", "description": "File or folder to search (default: the project)."},
                "glob": {"type": "string", "description": "Only files matching this glob, e.g. *.rs."},
                "ignore_case": {"type": "boolean", "description": "Case-insensitive search."},
                "context": {"type": "integer", "description": "Lines of context around each match."}
            }),
            &["pattern"],
        ),
        def(
            "glob",
            "Find files by name pattern, e.g. **/*.py or src/**/test_*.rs.",
            json!({
                "pattern": {"type": "string", "description": "Glob pattern."},
                "path": {"type": "string", "description": "Folder to search (default: the project)."}
            }),
            &["pattern"],
        ),
        def(
            "list_dir",
            "Show a folder as a tree.",
            json!({
                "path": {"type": "string", "description": "Folder (default: the project)."},
                "depth": {"type": "integer", "description": "How many levels to show, 1-3 (default 2)."}
            }),
            &[],
        ),
        def(
            "todo",
            "Keep your plan: replace the whole task list. Use it for any task with several steps, keep exactly one item active, and mark items done as you finish them.",
            json!({"items": {"type": "array", "description": "Objects with content and status (pending, active or done).",
                "items": {"type": "object", "properties": {
                    "content": {"type": "string"},
                    "status": {"type": "string", "enum": ["pending", "active", "done"]}}}}}),
            &["items"],
        ),
        def(
            "ctx_search",
            "Search the full text of earlier long tool output that was stored instead of shown (ranked by relevance).",
            json!({
                "query": {"type": "string", "description": "Words to look for."},
                "id": {"type": "integer", "description": "Only search stored output #id."}
            }),
            &["query"],
        ),
        def(
            "ctx_read",
            "Read lines of stored output #id.",
            json!({
                "id": {"type": "integer", "description": "The #number from the stored-output notice."},
                "offset": {"type": "integer", "description": "First line (1-based)."},
                "limit": {"type": "integer", "description": "Number of lines (default 80)."}
            }),
            &["id"],
        ),
        def(
            "ask_user",
            "Ask the user one short question when you are blocked or a choice is theirs to make. Your turn ends and their reply continues the work.",
            json!({"question": {"type": "string", "description": "The question."}}),
            &["question"],
        ),
        def(
            "check",
            "Check the project for errors quickly (cargo check, python syntax, go vet or node --check, whichever applies).",
            json!({"path": {"type": "string", "description": "Subfolder to check (default: the project)."}}),
            &[],
        ),
        def(
            "process_list",
            "List background processes started with process_start.",
            json!({}),
            &[],
        ),
        def(
            "process_output",
            "Show the latest output of a background process.",
            json!({
                "name": {"type": "string", "description": "The process name."},
                "lines": {"type": "integer", "description": "How many recent lines (default 40)."},
                "intent": {"type": "string", "description": "Only the parts about this."}
            }),
            &["name"],
        ),
    ];
    if network {
        tools.push(def(
            "web_search",
            "Search the web; returns titles, links and snippets. Follow up with fetch_url.",
            json!({"query": {"type": "string", "description": "Search terms."}}),
            &["query"],
        ));
        tools.push(def(
            "fetch_url",
            "Download a web page or file and return it as text (HTML reduced to readable text). Long pages are stored; use intent to get only what you need.",
            json!({
                "url": {"type": "string", "description": "http(s) address."},
                "intent": {"type": "string", "description": "What you are looking for in the page."},
                "max_chars": {"type": "integer", "description": "Cap on characters (default 12000)."}
            }),
            &["url"],
        ));
    }
    if !plan {
        tools.insert(
            0,
            def(
                "bash",
                "Run a shell command in the project (sandboxed: only the project can be written). Long output is stored and summarised; pass intent to get just the relevant parts.",
                json!({
                    "command": {"type": "string", "description": "The command."},
                    "timeout_secs": {"type": "integer", "description": "Give up after this long (default 60, max 300)."},
                    "intent": {"type": "string", "description": "What you are looking for in the output."}
                }),
                &["command"],
            ),
        );
        tools.insert(
            2,
            def(
                "edit_file",
                "Replace exact text in a file. old_text must match once, or set replace_all. Prefer this to rewriting the file.",
                json!({
                    "path": {"type": "string", "description": "Path relative to the project."},
                    "old_text": {"type": "string", "description": "Exact text to find."},
                    "new_text": {"type": "string", "description": "Replacement."},
                    "replace_all": {"type": "boolean", "description": "Replace every occurrence."}
                }),
                &["path", "old_text", "new_text"],
            ),
        );
        tools.insert(
            3,
            def(
                "write_file",
                "Create a file or replace it completely (parent folders are created).",
                json!({
                    "path": {"type": "string", "description": "Path relative to the project."},
                    "content": {"type": "string", "description": "Full contents."}
                }),
                &["path", "content"],
            ),
        );
        tools.push(def(
            "process_start",
            "Start a long-running command (dev server, watcher, test loop) in the background.",
            json!({
                "name": {"type": "string", "description": "A short name to refer to it."},
                "command": {"type": "string", "description": "The command."}
            }),
            &["name", "command"],
        ));
        tools.push(def(
            "process_stop",
            "Stop a background process.",
            json!({"name": {"type": "string", "description": "The process name."}}),
            &["name"],
        ));
    }
    tools
}

fn arg_str<'a>(args: &'a Map<String, Json>, key: &str) -> Result<&'a str, String> {
    args.get(key)
        .and_then(Json::as_str)
        .ok_or_else(|| format!("missing required argument {key}"))
}

fn arg_int(args: &Map<String, Json>, key: &str) -> Option<i64> {
    args.get(key).and_then(|v| {
        v.as_i64()
            .or_else(|| v.as_str().and_then(|s| s.trim().parse().ok()))
    })
}

fn arg_bool(args: &Map<String, Json>, key: &str) -> bool {
    args.get(key)
        .map(|v| v.as_bool().unwrap_or(v.as_str() == Some("true")))
        .unwrap_or(false)
}

fn intent(args: &Map<String, Json>) -> Option<String> {
    args.get("intent")
        .and_then(Json::as_str)
        .map(str::to_string)
        .filter(|s| !s.trim().is_empty())
}

fn hash(text: &str) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in text.bytes() {
        h = (h ^ b as u64).wrapping_mul(0x100000001b3);
    }
    h
}

/// `*` matches within a name, `**` across folders, `?` one character.
pub fn glob_match(pattern: &str, text: &str) -> bool {
    fn go(p: &[char], t: &[char]) -> bool {
        match p.first() {
            None => t.is_empty(),
            Some('*') if p.get(1) == Some(&'*') => {
                let rest = if p.get(2) == Some(&'/') {
                    &p[3..]
                } else {
                    &p[2..]
                };
                (0..=t.len()).any(|i| go(rest, &t[i..]))
            }
            Some('*') => {
                let mut i = 0;
                loop {
                    if go(&p[1..], &t[i..]) {
                        return true;
                    }
                    if i >= t.len() || t[i] == '/' {
                        return false;
                    }
                    i += 1;
                }
            }
            Some('?') => !t.is_empty() && t[0] != '/' && go(&p[1..], &t[1..]),
            Some(c) => t.first() == Some(c) && go(&p[1..], &t[1..]),
        }
    }
    let (p, t): (Vec<char>, Vec<char>) = (pattern.chars().collect(), text.chars().collect());
    go(&p, &t)
}

fn walk(dir: &Path, out: &mut Vec<PathBuf>, cap: usize) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    let mut entries: Vec<_> = entries.flatten().collect();
    entries.sort_by_key(|e| e.file_name());
    for e in entries {
        if out.len() >= cap {
            return;
        }
        let name = e.file_name().to_string_lossy().to_string();
        let path = e.path();
        let Ok(kind) = e.file_type() else { continue };
        if kind.is_dir() {
            if !IGNORED.contains(&name.as_str()) && !name.starts_with(".git") {
                walk(&path, out, cap);
            }
        } else if kind.is_file() {
            out.push(path);
        }
    }
}

fn tree(dir: &Path, depth: usize, level: usize, out: &mut String, count: &mut usize) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    let mut entries: Vec<_> = entries
        .flatten()
        .filter(|e| {
            let n = e.file_name().to_string_lossy().to_string();
            !IGNORED.contains(&n.as_str()) && n != ".DS_Store"
        })
        .collect();
    entries.sort_by_key(|e| (!e.path().is_dir(), e.file_name()));
    for e in entries {
        if *count >= 250 {
            if *count == 250 {
                *out += &format!("{}…\n", "  ".repeat(level));
                *count += 1;
            }
            return;
        }
        *count += 1;
        let name = e.file_name().to_string_lossy().to_string();
        if e.path().is_dir() {
            *out += &format!("{}{name}/\n", "  ".repeat(level));
            if level + 1 < depth {
                tree(&e.path(), depth, level + 1, out, count);
            }
        } else {
            *out += &format!("{}{name}\n", "  ".repeat(level));
        }
    }
}

pub fn url_encode(s: &str) -> String {
    s.bytes()
        .map(|b| match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                (b as char).to_string()
            }
            b' ' => "+".to_string(),
            _ => format!("%{b:02X}"),
        })
        .collect()
}

fn url_decode(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut out = Vec::new();
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'%' if i + 2 < bytes.len() => {
                let hex = std::str::from_utf8(&bytes[i + 1..i + 3]).unwrap_or("");
                if let Ok(v) = u8::from_str_radix(hex, 16) {
                    out.push(v);
                    i += 3;
                } else {
                    out.push(b'%');
                    i += 1;
                }
            }
            b'+' => {
                out.push(b' ');
                i += 1;
            }
            b => {
                out.push(b);
                i += 1;
            }
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// Text between the first `>` after `from` and the next `</a>`.
fn link_text(block: &str, from: usize) -> String {
    block[from..]
        .find('>')
        .and_then(|g| {
            let r = &block[from + g + 1..];
            r.find("</a>")
                .map(|e| crate::agent::html_to_text(&r[..e]).trim().to_string())
        })
        .unwrap_or_default()
}

/// Results from DuckDuckGo's HTML page: (title, url, snippet).
pub fn parse_search(html: &str) -> Vec<(String, String, String)> {
    const MARK: &str = "class=\"result__a\"";
    let mut out = Vec::new();
    let starts: Vec<usize> = html.match_indices(MARK).map(|(i, _)| i).collect();
    for (n, &start) in starts.iter().enumerate() {
        let end = starts.get(n + 1).copied().unwrap_or(html.len());
        let block = &html[start..end];
        let href = block
            .find("href=\"")
            .and_then(|h| {
                let r = &block[h + 6..];
                r.find('"').map(|e| r[..e].to_string())
            })
            .unwrap_or_default();
        let title = link_text(block, 0);
        let snippet = block
            .find("class=\"result__snippet\"")
            .map(|s| link_text(block, s))
            .unwrap_or_default();
        let url = match href.split("uddg=").nth(1) {
            Some(enc) => url_decode(enc.split('&').next().unwrap_or("")),
            None => href.replace("&amp;", "&"),
        };
        if !url.is_empty() && !title.is_empty() {
            out.push((title, url, snippet));
        }
    }
    out
}

impl Sandbox {
    fn project_script(
        &self,
        script: &str,
        secs: u64,
        cancel: &AtomicBool,
    ) -> Result<String, String> {
        let (code, out, killed) = self.run(
            &["/bin/bash", "-c", &format!("exec 2>&1\n{script}")],
            Duration::from_secs(secs),
            cancel,
        )?;
        if killed {
            return Err(format!("{}\n[timed out or cancelled]", out.trim_end()));
        }
        let text = out.trim_end().to_string();
        match code {
            Some(0) => Ok(if text.is_empty() {
                "(no output)".into()
            } else {
                text
            }),
            Some(1) if script.starts_with("rg ") => Ok("no matches".into()),
            c => Err(format!("{text}\n[exit code {}]", c.unwrap_or(-1))),
        }
    }
}

/// Run one agent tool call. Returns (succeeded, text for the model).
pub fn execute(
    shared: &SharedRef,
    sb: &Sandbox,
    name: &str,
    args: &Map<String, Json>,
    cancel: &AtomicBool,
) -> (bool, String) {
    let result = run(shared, sb, name, args, cancel);
    match result {
        Ok(text) => (true, text),
        Err(e) => (false, format!("error: {e}")),
    }
}

fn lock(shared: &SharedRef) -> std::sync::MutexGuard<'_, Shared> {
    shared.lock().unwrap_or_else(|e| e.into_inner())
}

fn run(
    shared: &SharedRef,
    sb: &Sandbox,
    name: &str,
    args: &Map<String, Json>,
    cancel: &AtomicBool,
) -> Result<String, String> {
    match name {
        "bash" => {
            let command = arg_str(args, "command")?.to_string();
            let (ok, text) = match sb.bash(args, cancel) {
                Ok(t) => (true, t),
                Err(t) => (false, t),
            };
            let shown = lock(shared).store.offload(
                &format!(
                    "bash: {}",
                    clip_output(command.lines().next().unwrap_or(""), 60)
                ),
                &text,
                intent(args).as_deref(),
            );
            if ok { Ok(shown) } else { Err(shown) }
        }
        "read_file" => {
            let rel = arg_str(args, "path")?.to_string();
            let path = sb.resolve(&rel)?;
            let offset = arg_int(args, "offset").unwrap_or(1).max(1) as usize;
            let limit = arg_int(args, "limit").unwrap_or(400).clamp(1, 2000) as usize;
            let content =
                fs::read_to_string(&path).map_err(|e| format!("{}: {e}", sb.rel(&path)))?;
            let key = (sb.rel(&path), offset, limit);
            let h = hash(&content);
            let mut g = lock(shared);
            if g.reads.get(&key) == Some(&h) {
                return Ok(format!(
                    "[{} (lines {}-{}) is unchanged since you read it earlier; use that output]",
                    key.0,
                    offset,
                    offset + limit - 1
                ));
            }
            let text = sb.read_file(args)?;
            g.reads.insert(key.clone(), h);
            g.touch(&key.0, Touch::Read);
            Ok(text)
        }
        "write_file" => {
            let text = sb.write_file(args)?;
            let path = sb.resolve(arg_str(args, "path")?)?;
            let mut g = lock(shared);
            g.touch(&sb.rel(&path), Touch::Edited);
            g.reads.clear();
            Ok(text)
        }
        "edit_file" => edit(shared, sb, args),
        "list_dir" => {
            let path = sb.resolve(args.get("path").and_then(Json::as_str).unwrap_or(""))?;
            let depth = arg_int(args, "depth").unwrap_or(2).clamp(1, 3) as usize;
            let mut out = String::new();
            let mut count = 0;
            tree(&path, depth, 0, &mut out, &mut count);
            Ok(if out.is_empty() {
                "(empty directory)".into()
            } else {
                out
            })
        }
        "glob" => {
            let pattern = arg_str(args, "pattern")?;
            let root = sb.resolve(args.get("path").and_then(Json::as_str).unwrap_or(""))?;
            let mut files = Vec::new();
            walk(&root, &mut files, 20_000);
            let by_name = !pattern.contains('/');
            let mut found: Vec<String> = files
                .iter()
                .filter_map(|f| {
                    let rel = f.strip_prefix(&root).ok()?.display().to_string();
                    let subject = if by_name {
                        f.file_name()?.to_string_lossy().to_string()
                    } else {
                        rel.clone()
                    };
                    glob_match(pattern, &subject).then_some(rel)
                })
                .collect();
            if found.is_empty() {
                return Ok("no files match".into());
            }
            let total = found.len();
            found.truncate(200);
            let mut out = found.join("\n");
            if total > 200 {
                out += &format!("\n… and {} more", total - 200);
            }
            Ok(out)
        }
        "grep" => {
            let pattern = arg_str(args, "pattern")?;
            let path = sb.resolve(args.get("path").and_then(Json::as_str).unwrap_or(""))?;
            let rel = {
                let r = sb.rel(&path);
                if r.is_empty() { ".".to_string() } else { r }
            };
            let mut cmd = String::from(
                "rg --line-number --no-heading --color never --max-columns 240 --max-count 60",
            );
            if arg_bool(args, "ignore_case") {
                cmd += " -i";
            }
            if let Some(n) = arg_int(args, "context") {
                cmd += &format!(" -C {}", n.clamp(0, 5));
            }
            if let Some(g) = args.get("glob").and_then(Json::as_str) {
                cmd += &format!(" -g {}", shell_quote(g));
            }
            cmd += &format!(" -- {} {}", shell_quote(pattern), shell_quote(&rel));
            let text = sb.project_script(&cmd, 30, cancel)?;
            Ok(lock(shared)
                .store
                .offload(&format!("grep {pattern}"), &text, None))
        }
        "todo" => {
            let items = match args.get("items") {
                Some(Json::Array(a)) => a.clone(),
                Some(Json::String(s)) => {
                    serde_json::from_str::<Vec<Json>>(s).map_err(|e| e.to_string())?
                }
                _ => return Err("items must be a list".into()),
            };
            let todos: Vec<Todo> = items
                .iter()
                .filter_map(|i| {
                    let text = i["content"].as_str().or_else(|| i.as_str())?.to_string();
                    let state = match i["status"].as_str().unwrap_or("pending") {
                        "done" | "completed" => TodoState::Done,
                        "active" | "in_progress" => TodoState::Active,
                        _ => TodoState::Pending,
                    };
                    Some(Todo { text, state })
                })
                .collect();
            let mut g = lock(shared);
            g.todos = todos;
            let (done, total) = g.todo_counts();
            Ok(format!("plan updated: {done} of {total} done"))
        }
        "ctx_search" => Ok(lock(shared).store.search_text(
            arg_str(args, "query")?,
            arg_int(args, "id").map(|i| i.max(0) as usize),
        )),
        "ctx_read" => {
            let id = arg_int(args, "id")
                .ok_or("missing required argument id")?
                .max(0) as usize;
            lock(shared).store.read(
                id,
                arg_int(args, "offset").unwrap_or(1).max(1) as usize,
                arg_int(args, "limit").unwrap_or(80).clamp(1, 400) as usize,
            )
        }
        "ask_user" => Ok(
            "the question was shown to the user; their reply comes in their next message".into(),
        ),
        "check" => check(shared, sb, args, cancel),
        "web_search" => {
            if !sb.network {
                return Err("the network is turned off for this session".into());
            }
            crate::search::search(&sb.search, arg_str(args, "query")?)
        }
        "fetch_url" => {
            let mut a = args.clone();
            // The page goes to the store; only the relevant parts come back.
            a.insert("max_chars".into(), json!(60_000));
            let text = sb.fetch_url(&a, cancel)?;
            Ok(lock(shared).store.offload(
                &format!(
                    "fetch {}",
                    clip_output(arg_str(args, "url").unwrap_or(""), 60)
                ),
                &text,
                intent(args).as_deref(),
            ))
        }
        "process_start" => {
            let pname = arg_str(args, "name")?.to_string();
            let command = arg_str(args, "command")?.to_string();
            let mut g = lock(shared);
            g.procs.retain_mut(|p| p.running() || p.name != pname);
            if g.procs.iter().any(|p| p.name == pname) {
                return Err(format!("{pname} is already running; stop it first"));
            }
            let mut c = sb.command(&["/bin/bash", "-c", &format!("exec 2>&1\n{command}")]);
            let mut child = c.spawn().map_err(|e| e.to_string())?;
            let out = child.stdout.take().ok_or("no output pipe")?;
            let lines = Arc::new(Mutex::new(VecDeque::new()));
            let sink = lines.clone();
            std::thread::spawn(move || {
                for line in BufReader::new(out).lines().map_while(Result::ok) {
                    let mut l = sink.lock().unwrap_or_else(|e| e.into_inner());
                    l.push_back(line);
                    while l.len() > 2000 {
                        l.pop_front();
                    }
                }
            });
            g.procs.push(Proc {
                name: pname.clone(),
                command,
                child,
                lines,
                started: Instant::now(),
            });
            Ok(format!("started {pname}; read it with process_output"))
        }
        "process_output" => {
            let pname = arg_str(args, "name")?;
            let n = arg_int(args, "lines").unwrap_or(40).clamp(1, 400) as usize;
            let mut g = lock(shared);
            let p = g
                .procs
                .iter_mut()
                .find(|p| p.name == pname)
                .ok_or_else(|| format!("no process named {pname}"))?;
            let state = if p.running() { "running" } else { "exited" };
            let lines: Vec<String> = {
                let l = p.lines.lock().unwrap_or_else(|e| e.into_inner());
                l.iter().skip(l.len().saturating_sub(n)).cloned().collect()
            };
            let text = format!("[{pname}: {state}]\n{}", lines.join("\n"));
            Ok(g.store
                .offload(&format!("process {pname}"), &text, intent(args).as_deref()))
        }
        "process_list" => {
            let mut g = lock(shared);
            if g.procs.is_empty() {
                return Ok("no background processes".into());
            }
            let mut out = String::new();
            for p in &mut g.procs {
                let state = if p.running() { "running" } else { "exited" };
                out += &format!("{}  {state}  {}\n", p.name, p.command);
            }
            Ok(out)
        }
        "process_stop" => {
            let pname = arg_str(args, "name")?;
            let mut g = lock(shared);
            let i = g
                .procs
                .iter()
                .position(|p| p.name == pname)
                .ok_or_else(|| format!("no process named {pname}"))?;
            let mut p = g.procs.remove(i);
            p.kill();
            Ok(format!("stopped {pname}"))
        }
        other => Err(format!("unknown tool {other}")),
    }
}

fn edit(shared: &SharedRef, sb: &Sandbox, args: &Map<String, Json>) -> Result<String, String> {
    let rel = arg_str(args, "path")?;
    let path = sb.resolve(rel)?;
    let (old, new) = (arg_str(args, "old_text")?, arg_str(args, "new_text")?);
    if old.is_empty() {
        return Err("old_text is empty".into());
    }
    let text = fs::read_to_string(&path).map_err(|e| format!("{}: {e}", sb.rel(&path)))?;
    let all = arg_bool(args, "replace_all");
    let mut n = text.matches(old).count();
    let mut base = text.clone();
    let mut old = old.to_string();
    if n == 0 {
        // Models often drift on trailing whitespace; match ignoring it.
        let strip = |s: &str| s.lines().map(str::trim_end).collect::<Vec<_>>().join("\n");
        let (b, o) = (strip(&text), strip(&old));
        if !o.is_empty() && b.matches(&o).count() == 1 {
            base = b;
            old = o;
            n = 1;
        }
    }
    let updated = match n {
        0 => {
            let first = old.lines().next().unwrap_or("").trim();
            let hint = text
                .lines()
                .enumerate()
                .find(|(_, l)| !first.is_empty() && l.contains(first))
                .map(|(i, _)| format!("; the first line of old_text appears near line {}", i + 1))
                .unwrap_or_default();
            return Err(format!(
                "old_text was not found{hint}. Read the file again and copy exactly"
            ));
        }
        1 => base.replacen(&old, new, 1),
        _ if all => base.replace(&old, new),
        n => {
            return Err(format!(
                "old_text matches {n} places; add surrounding lines or set replace_all"
            ));
        }
    };
    fs::write(&path, &updated).map_err(|e| format!("{}: {e}", sb.rel(&path)))?;
    let mut g = lock(shared);
    g.touch(&sb.rel(&path), Touch::Edited);
    g.reads.clear();
    Ok(format!(
        "edited {}: {} occurrence(s), {} lines replaced with {}",
        sb.rel(&path),
        if all { n } else { 1 },
        old.lines().count().max(1),
        new.lines().count().max(1)
    ))
}

fn check(
    shared: &SharedRef,
    sb: &Sandbox,
    args: &Map<String, Json>,
    cancel: &AtomicBool,
) -> Result<String, String> {
    let dir = sb.resolve(args.get("path").and_then(Json::as_str).unwrap_or(""))?;
    let has = |f: &str| dir.join(f).exists();
    let script = if has("Cargo.toml") {
        "cargo check --message-format short --color never 2>&1 | tail -80".to_string()
    } else if has("go.mod") {
        "go vet ./... 2>&1 | tail -80".to_string()
    } else if has("package.json") && has("tsconfig.json") {
        "npx --no-install tsc --noEmit 2>&1 | tail -80".to_string()
    } else {
        // Syntax-only checks that write nothing.
        r#"
failed=0
for f in $(rg --files -g '*.py' | head -400); do
  python3 - "$f" <<'PY' || failed=1
import ast, sys
try:
    ast.parse(open(sys.argv[1], encoding="utf-8").read(), sys.argv[1])
except SyntaxError as e:
    print(f"{sys.argv[1]}:{e.lineno}: SyntaxError: {e.msg}"); sys.exit(1)
PY
done
for f in $(rg --files -g '*.js' -g '*.mjs' | head -200); do node --check "$f" 2>&1 || failed=1; done
if [ -z "$(rg --files -g '*.py' -g '*.js' -g '*.mjs' | head -1)" ]; then echo "no project checks apply (no Cargo.toml, go.mod, tsconfig.json, .py or .js files)"; fi
[ $failed -eq 0 ] && echo "ok: no syntax errors found"
exit $failed
"#
        .to_string()
    };
    let wrapped = format!("cd {}\n{script}", shell_quote(&dir.display().to_string()));
    let text = sb.project_script(&wrapped, 240, cancel);
    let (ok, text) = match text {
        Ok(t) => (true, t),
        Err(t) => (false, t),
    };
    let shown = lock(shared).store.offload("check", &text, None);
    if ok { Ok(shown) } else { Err(shown) }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn project(tag: &str) -> (Sandbox, SharedRef, PathBuf) {
        let dir =
            std::env::temp_dir().join(format!("ferrum-coding-test-{}-{tag}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let proj = dir.join("proj");
        fs::create_dir_all(proj.join("src")).unwrap();
        fs::write(
            proj.join("src/main.py"),
            "def add(a, b):\n    return a - b\n\nprint(add(1, 2))\n",
        )
        .unwrap();
        fs::write(
            proj.join("src/util.py"),
            "import os\n\ndef helper():\n    return os.getcwd()\n",
        )
        .unwrap();
        fs::write(proj.join("README.md"), "# demo\n").unwrap();
        fs::create_dir_all(proj.join("node_modules/x")).unwrap();
        fs::write(proj.join("node_modules/x/ignored.py"), "x = 1\n").unwrap();
        let mut sb =
            Sandbox::with_state(fs::canonicalize(&proj).unwrap(), dir.join("state"), false)
                .unwrap();
        sb.output_limit = AGENT_OUTPUT_LIMIT;
        (sb, new_shared(), dir)
    }

    fn call(sb: &Sandbox, shared: &SharedRef, name: &str, args: Json) -> (bool, String) {
        execute(
            shared,
            sb,
            name,
            args.as_object().unwrap(),
            &AtomicBool::new(false),
        )
    }

    #[test]
    fn globs() {
        assert!(glob_match("*.py", "main.py"));
        assert!(!glob_match("*.py", "src/main.py"));
        assert!(glob_match("**/*.py", "src/deep/main.py"));
        assert!(glob_match("src/**/test_*.rs", "src/a/b/test_x.rs"));
        assert!(glob_match("a?c", "abc") && !glob_match("a?c", "a/c"));
    }

    #[test]
    fn finds_reads_and_edits() {
        let (sb, shared, dir) = project("edit");
        let (ok, out) = call(&sb, &shared, "glob", json!({"pattern": "*.py"}));
        assert!(
            ok && out.contains("src/main.py") && !out.contains("ignored"),
            "{out}"
        );
        let (ok, out) = call(
            &sb,
            &shared,
            "grep",
            json!({"pattern": "return", "glob": "*.py"}),
        );
        assert!(
            ok && out.contains("main.py") && out.contains("a - b"),
            "{out}"
        );
        let (ok, out) = call(&sb, &shared, "list_dir", json!({"depth": 2}));
        assert!(
            ok && out.contains("src/") && !out.contains("node_modules"),
            "{out}"
        );
        let (ok, first) = call(&sb, &shared, "read_file", json!({"path": "src/main.py"}));
        assert!(ok && first.contains("a - b"));
        let (_, again) = call(&sb, &shared, "read_file", json!({"path": "src/main.py"}));
        assert!(again.contains("unchanged"), "{again}");
        let (ok, out) = call(
            &sb,
            &shared,
            "edit_file",
            json!({"path": "src/main.py", "old_text": "a - b", "new_text": "a + b"}),
        );
        assert!(ok, "{out}");
        let (_, after) = call(&sb, &shared, "read_file", json!({"path": "src/main.py"}));
        assert!(
            after.contains("a + b"),
            "an edit must invalidate the read cache: {after}"
        );
        let g = lock(&shared);
        assert_eq!(g.touched.get("src/main.py"), Some(&Touch::Edited));
        drop(g);
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn a_session_snapshot_round_trips() {
        let mut g = Shared::default();
        g.todos.push(Todo {
            text: "fix it".into(),
            state: TodoState::Active,
        });
        g.todos.push(Todo {
            text: "test it".into(),
            state: TodoState::Pending,
        });
        g.touch("a.py", Touch::Edited);
        g.touch("b.py", Touch::Read);
        let id = g.store.add("bash: ls", "one\ntwo\nthree\n");
        g.store.saved = 1234;
        g.compactions = 2;
        let saved = g.snapshot();
        let mut back = Shared::default();
        back.restore(&saved);
        assert_eq!(back.todos.len(), 2);
        assert!(
            back.todos[0].state == TodoState::Active && back.todos[1].state == TodoState::Pending
        );
        assert_eq!(back.touched.get("a.py"), Some(&Touch::Edited));
        assert_eq!(back.touched.get("b.py"), Some(&Touch::Read));
        // Stored output keeps its id, so elided-output stubs still resolve.
        assert!(back.store.read(id, 2, 1).unwrap().contains("two"));
        assert_eq!((back.store.saved, back.compactions), (1234, 2));
    }

    #[test]
    fn keeps_a_todo_list_and_stops_loops() {
        let (sb, shared, dir) = project("todo");
        let (ok, out) = call(
            &sb,
            &shared,
            "todo",
            json!({"items": [{"content": "fix add", "status": "done"}, {"content": "run it", "status": "active"}]}),
        );
        assert!(ok && out.contains("1 of 2"), "{out}");
        let args = json!({"path": "x"}).as_object().unwrap().clone();
        let mut g = lock(&shared);
        assert!(!g.is_loop("read_file", &args));
        assert!(!g.is_loop("read_file", &args));
        assert!(
            g.is_loop("read_file", &args),
            "third identical call is a loop"
        );
        g.new_message();
        assert!(!g.is_loop("read_file", &args));
        drop(g);
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn offloads_long_bash_output_and_searches_it() {
        let (sb, shared, dir) = project("ctx");
        let (ok, out) = call(
            &sb,
            &shared,
            "bash",
            json!({"command": "for i in $(seq 1 600); do echo \"line $i padding padding padding\"; done; echo 'FATAL: disk quota exceeded'", "intent": "fatal quota"}),
        );
        assert!(ok, "{out}");
        assert!(
            out.contains("output #1") && out.contains("quota exceeded"),
            "{out}"
        );
        assert!(out.len() < 3_500, "{}", out.len());
        let (_, found) = call(&sb, &shared, "ctx_search", json!({"query": "disk quota"}));
        assert!(found.contains("quota exceeded"), "{found}");
        let (_, read) = call(
            &sb,
            &shared,
            "ctx_read",
            json!({"id": 1, "offset": 600, "limit": 3}),
        );
        assert!(read.contains("line 600"), "{read}");
        assert!(lock(&shared).store.saved > 10_000);
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn checks_python_syntax() {
        let (sb, shared, dir) = project("check");
        let (ok, out) = call(&sb, &shared, "check", json!({}));
        assert!(ok, "{out}");
        fs::write(sb.workspace.join("src/broken.py"), "def f(:\n  pass\n").unwrap();
        let (ok, out) = call(&sb, &shared, "check", json!({}));
        assert!(
            !ok && out.contains("broken.py") && out.contains("SyntaxError"),
            "{out}"
        );
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn background_processes() {
        let (sb, shared, dir) = project("proc");
        let (ok, out) = call(
            &sb,
            &shared,
            "process_start",
            json!({"name": "ticker", "command": "for i in 1 2 3; do echo tick $i; sleep 0.2; done; sleep 30"}),
        );
        assert!(ok, "{out}");
        std::thread::sleep(Duration::from_millis(900));
        let (ok, out) = call(&sb, &shared, "process_output", json!({"name": "ticker"}));
        assert!(
            ok && out.contains("running") && out.contains("tick 3"),
            "{out}"
        );
        let (_, list) = call(&sb, &shared, "process_list", json!({}));
        assert!(list.contains("ticker"), "{list}");
        let (ok, _) = call(&sb, &shared, "process_stop", json!({"name": "ticker"}));
        assert!(ok);
        assert!(!call(&sb, &shared, "process_output", json!({"name": "ticker"})).0);
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn parses_search_results() {
        let html = r#"<div class="result"><a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fa%3Fx%3D1&amp;rut=abc">Example <b>Title</b></a>
<a class="result__snippet" href="x">A short <b>snippet</b> here.</a></div>
<div class="result"><a class="result__a" href="https://rust-lang.org/">Rust</a><a class="result__snippet">Fast and safe.</a></div>"#;
        let r = parse_search(html);
        assert_eq!(r.len(), 2, "{r:?}");
        assert_eq!(r[0].1, "https://example.com/a?x=1");
        assert!(r[0].0.contains("Example") && r[0].2.contains("short"));
        assert_eq!(r[1].1, "https://rust-lang.org/");
    }

    #[test]
    fn plan_mode_has_no_mutating_tools() {
        let names = |plan| -> Vec<String> {
            tool_defs(plan, true)
                .iter()
                .map(|t| t["function"]["name"].as_str().unwrap().to_string())
                .collect()
        };
        let plan = names(true);
        assert!(plan.iter().all(|n| read_only(n)), "{plan:?}");
        assert!(!plan.contains(&"bash".to_string()));
        assert!(names(false).contains(&"edit_file".to_string()));
    }
}
