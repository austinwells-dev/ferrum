//! Tools the chat agent can call. Commands and downloads run inside a macOS
//! Seatbelt sandbox (`sandbox-exec` with `sandbox.sb`): the filesystem is
//! read-only except for the workspace folder, secrets are hidden, and the
//! network can be switched off. File tools never leave the workspace.
use crate::*;
use std::io::{Read, Write};
use std::os::unix::fs::OpenOptionsExt;
use std::os::unix::process::CommandExt;
use std::process::Stdio;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::channel;

const PROFILE: &str = include_str!("sandbox.sb");
const MAX_OUTPUT: usize = 12_000;
/// macOS `open(2)` flag: fail if any component of the path is a symlink.
const O_NOFOLLOW_ANY: i32 = 0x2000_0000;
/// `ELOOP`, what `O_NOFOLLOW_ANY` reports on meeting a symlink.
const ELOOP: i32 = 62;

/// Read a file the agent may have planted, refusing a symlink in its place
/// (the TUI reads outside the sandbox, so a link could reach a secret).
pub fn read_no_follow(path: &Path) -> std::io::Result<String> {
    const O_NOFOLLOW: i32 = 0x100;
    let mut text = String::new();
    fs::OpenOptions::new()
        .read(true)
        .custom_flags(O_NOFOLLOW)
        .open(path)?
        .read_to_string(&mut text)?;
    Ok(text)
}

/// `program args…` in a sandbox that can write nothing and reach no network.
/// For commands the TUI runs itself in the agent's project (git): the agent
/// can rewrite `.git/config`, and settings like `core.fsmonitor` run programs.
pub fn confined(argv: &[&str]) -> Command {
    let mut c = Command::new("/usr/bin/sandbox-exec");
    c.arg("-p")
        .arg(PROFILE)
        .args(["-D", "WORKSPACE=/private/var/empty"])
        .args(["-D", "STATE=/private/var/empty"])
        .arg("-D")
        .arg(format!("HOME={}", home().display()))
        .args(["-D", "NETWORK=off"])
        .args(argv)
        .stdin(Stdio::null());
    c
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum ToolMode {
    Off,
    Ask,
    /// Edits and reads run freely; commands and downloads ask.
    Edits,
    Auto,
}

impl ToolMode {
    pub fn parse(s: &str) -> Self {
        match s {
            "ask" => Self::Ask,
            "edits" => Self::Edits,
            "auto" => Self::Auto,
            _ => Self::Off,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Ask => "ask",
            Self::Edits => "edits",
            Self::Auto => "auto",
        }
    }
}

/// Tools that change things or reach the network ask first in `ask` mode.
pub fn needs_approval(name: &str) -> bool {
    matches!(
        name,
        "bash" | "write_file" | "edit_file" | "fetch_url" | "web_search" | imagetool::NAME
    )
}

pub fn def(name: &str, description: &str, properties: Json, required: &[&str]) -> Json {
    json!({
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required}
        }
    })
}

pub fn tool_defs(network: bool) -> Vec<Json> {
    let mut tools = vec![
        def(
            "bash",
            "Run a shell command (bash) in the sandboxed workspace and return its combined output and exit code. \
             Files can only be written inside the workspace.",
            json!({
                "command": {"type": "string", "description": "The command to run."},
                "timeout_secs": {"type": "integer", "description": "Give up after this many seconds (default 60, max 300)."}
            }),
            &["command"],
        ),
        def(
            "read_file",
            "Read a text file from the workspace with line numbers.",
            json!({
                "path": {"type": "string", "description": "Path relative to the workspace."},
                "offset": {"type": "integer", "description": "First line to read (1-based, default 1)."},
                "limit": {"type": "integer", "description": "Number of lines (default 400)."}
            }),
            &["path"],
        ),
        def(
            "write_file",
            "Create or overwrite a file in the workspace (parent folders are created).",
            json!({
                "path": {"type": "string", "description": "Path relative to the workspace."},
                "content": {"type": "string", "description": "Full contents of the file."}
            }),
            &["path", "content"],
        ),
        def(
            "edit_file",
            "Replace one exact piece of text in a workspace file. old_text must match exactly once.",
            json!({
                "path": {"type": "string", "description": "Path relative to the workspace."},
                "old_text": {"type": "string", "description": "Exact text to find."},
                "new_text": {"type": "string", "description": "Replacement text."}
            }),
            &["path", "old_text", "new_text"],
        ),
        def(
            "list_dir",
            "List the files and folders in a workspace directory.",
            json!({"path": {"type": "string", "description": "Directory relative to the workspace (default: the workspace itself)."}}),
            &[],
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
            "Download a web page or file over HTTP(S) and return it as text (HTML is reduced to readable text).",
            json!({
                "url": {"type": "string", "description": "The http:// or https:// address."},
                "max_chars": {"type": "integer", "description": "Truncate the result to this many characters (default 12000)."}
            }),
            &["url"],
        ));
    }
    tools
}

pub fn system_prompt(workspace: &Path, network: bool) -> String {
    format!(
        "You can use tools. They work inside a sandboxed workspace folder, {}. \
         Relative paths are relative to that folder, and nothing outside it can be changed. \
         Use tools to look at files, write and edit code, run commands{} and check your work by running it. \
         Take small steps, and when you are done give a short answer saying what you did.",
        workspace.display(),
        if network { ", fetch web pages" } else { "" }
    )
}

/// A one-line description of a call, for the transcript.
pub fn summary(name: &str, args: &Map<String, Json>) -> String {
    let s = |k: &str| args.get(k).and_then(Json::as_str).unwrap_or("");
    match name {
        "bash" => s("command").lines().next().unwrap_or("").to_string(),
        "fetch_url" => s("url").to_string(),
        "web_search" => s("query").to_string(),
        imagetool::NAME => s("prompt").to_string(),
        "list_dir" if s("path").is_empty() => ".".to_string(),
        _ => s("path").to_string(),
    }
}

pub fn arg_str<'a>(args: &'a Map<String, Json>, key: &str) -> Result<&'a str, String> {
    args.get(key)
        .and_then(Json::as_str)
        .ok_or_else(|| format!("missing required argument {key}"))
}

pub fn arg_int(args: &Map<String, Json>, key: &str) -> Option<i64> {
    args.get(key).and_then(|v| {
        v.as_i64()
            .or_else(|| v.as_str().and_then(|s| s.trim().parse().ok()))
    })
}

/// Keep the head and tail of long output.
pub fn clip_output(s: &str, max: usize) -> String {
    if s.len() <= max {
        return s.to_string();
    }
    let mut head = max / 2;
    while !s.is_char_boundary(head) {
        head -= 1;
    }
    let mut tail = s.len() - max / 2;
    while !s.is_char_boundary(tail) {
        tail += 1;
    }
    format!(
        "{}\n… [{} bytes omitted] …\n{}",
        &s[..head],
        tail - head,
        &s[tail..]
    )
}

#[derive(Clone)]
pub struct Sandbox {
    /// The only folder the agent can change (besides `state`).
    pub workspace: PathBuf,
    /// Scratch home, temp and attachment folders; outside the project for agents.
    pub state: PathBuf,
    /// Seatbelt confinement is applied (`sandbox-exec -p PROFILE`) when on. When
    /// off the sandbox runs commands with the same scrubbed environment and
    /// workspace cwd but no Seatbelt — an unrestricted shell (network only if
    /// `network` is on). Off is deliberate; the default is on.
    pub confined: bool,
    pub network: bool,
    pub search: crate::search::Config,
    /// Command output longer than this is clipped (agents offload it instead).
    pub output_limit: usize,
}

impl Sandbox {
    /// Create (if needed) and canonicalize the workspace; Seatbelt matches real paths.
    pub fn new(workspace: &str, network: bool) -> Result<Self, String> {
        let dir = expand(if workspace.trim().is_empty() {
            "~/ferrum-workspace"
        } else {
            workspace.trim()
        });
        fs::create_dir_all(&dir)
            .map_err(|e| format!("cannot create workspace {}: {e}", dir.display()))?;
        let workspace = fs::canonicalize(&dir).map_err(|e| e.to_string())?;
        Self::with_state(workspace.clone(), workspace, network)
    }

    /// A sandbox whose scratch folders live in `state` (created if needed).
    pub fn with_state(workspace: PathBuf, state: PathBuf, network: bool) -> Result<Self, String> {
        for sub in [".tmp", ".home", ".attachments"] {
            fs::create_dir_all(state.join(sub))
                .map_err(|e| format!("cannot create {}: {e}", state.display()))?;
        }
        let state = fs::canonicalize(&state).map_err(|e| e.to_string())?;
        Ok(Self {
            workspace,
            state,
            confined: true,
            network,
            search: Default::default(),
            output_limit: MAX_OUTPUT,
        })
    }

    pub fn attachments(&self) -> PathBuf {
        self.state.join(".attachments")
    }

    /// Resolve a path the model gave us, refusing anything outside the workspace.
    pub fn resolve(&self, p: &str) -> Result<PathBuf, String> {
        let p = p.trim();
        if p.is_empty() {
            return Ok(self.workspace.clone());
        }
        let raw = expand(p);
        let joined = if raw.is_absolute() {
            raw
        } else {
            self.workspace.join(raw)
        };
        let mut norm = PathBuf::new();
        for comp in joined.components() {
            match comp {
                std::path::Component::ParentDir => {
                    norm.pop();
                }
                std::path::Component::CurDir => {}
                c => norm.push(c),
            }
        }
        let mut probe = norm.clone();
        // A dangling symlink exists here, so it is canonicalized (and fails)
        // rather than skipped: writing through it would land outside.
        while fs::symlink_metadata(&probe).is_err() {
            if !probe.pop() {
                break;
            }
        }
        let real = fs::canonicalize(&probe).map_err(|e| format!("{p}: {e}"))?;
        if !real.starts_with(&self.workspace) && !real.starts_with(self.attachments()) {
            return Err(format!(
                "{p} is outside the workspace ({})",
                self.workspace.display()
            ));
        }
        Ok(norm)
    }

    /// Open a resolved path without following symlinks: the real location is
    /// checked again, then opened with `O_NOFOLLOW_ANY`, so a link swapped in
    /// after `resolve` (by a background process) is refused, not followed.
    fn open(&self, path: &Path, write: bool) -> Result<fs::File, String> {
        let rel = self.rel(path);
        let real = match fs::canonicalize(path) {
            Ok(real) => real,
            Err(_) => {
                let name = path
                    .file_name()
                    .ok_or_else(|| format!("{rel} is not a file"))?;
                let parent = path.parent().unwrap_or(&self.workspace);
                fs::canonicalize(parent)
                    .map_err(|e| format!("{rel}: {e}"))?
                    .join(name)
            }
        };
        if !real.starts_with(&self.workspace) && !real.starts_with(self.attachments()) {
            return Err(format!("{rel} is outside the workspace"));
        }
        let mut options = fs::OpenOptions::new();
        if write {
            options.write(true).create(true).truncate(true);
        } else {
            options.read(true);
        }
        options
            .custom_flags(O_NOFOLLOW_ANY)
            .open(&real)
            .map_err(|e| match e.raw_os_error() {
                Some(ELOOP) => format!("{rel}: refusing to follow a symbolic link"),
                _ => format!("{rel}: {e}"),
            })
    }

    pub fn read_bytes(&self, path: &Path) -> Result<Vec<u8>, String> {
        let mut bytes = Vec::new();
        self.open(path, false)?
            .read_to_end(&mut bytes)
            .map_err(|e| format!("{}: {e}", self.rel(path)))?;
        Ok(bytes)
    }

    pub fn read_text(&self, path: &Path) -> Result<String, String> {
        String::from_utf8(self.read_bytes(path)?)
            .map_err(|_| format!("{} is not UTF-8 text", self.rel(path)))
    }

    pub fn write_text(&self, path: &Path, text: &str) -> Result<(), String> {
        self.write_bytes(path, text.as_bytes())
    }

    pub fn write_bytes(&self, path: &Path, bytes: &[u8]) -> Result<(), String> {
        self.open(path, true)?
            .write_all(bytes)
            .map_err(|e| format!("{}: {e}", self.rel(path)))
    }

    pub fn rel(&self, path: &Path) -> String {
        path.strip_prefix(&self.workspace)
            .map(|p| p.display().to_string())
            .unwrap_or_else(|_| path.display().to_string())
    }

    /// `program args…` confined in `sandbox-exec` with a scrubbed environment and
    /// the workspace as cwd, or unconfined when `confined` is off: the same
    /// command, environment and cwd but no Seatbelt, so it is an unrestricted
    /// shell (network only when `network` is on). `argv` is never empty at a
    /// call site, matching the pre-existing assumption here.
    pub fn command(&self, argv: &[&str]) -> Command {
        let program = if self.confined {
            "/usr/bin/sandbox-exec"
        } else {
            argv[0]
        };
        // Confined, `argv` follows the `-D` directives; unconfined, it is the
        // whole command, so skip the leading program token to avoid doubling it.
        let tail = if self.confined { argv } else { &argv[1..] };
        let mut c = Command::new(program);
        if self.confined {
            c.arg("-p")
                .arg(PROFILE)
                .arg("-D")
                .arg(format!("WORKSPACE={}", self.workspace.display()))
                .arg("-D")
                .arg(format!("STATE={}", self.state.display()))
                .arg("-D")
                .arg(format!("HOME={}", home().display()))
                .arg("-D")
                .arg(format!(
                    "NETWORK={}",
                    if self.network { "on" } else { "off" }
                ));
        }
        c.args(tail)
            .current_dir(&self.workspace)
            .env_clear()
            .env(
                "PATH",
                format!(
                    "{h}/.cargo/bin:{h}/.local/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin",
                    h = home().display()
                ),
            )
            .env("HOME", self.state.join(".home"))
            .env("TMPDIR", self.state.join(".tmp"))
            .env("CARGO_HOME", home().join(".cargo"))
            .env("RUSTUP_HOME", home().join(".rustup"))
            .env("LANG", "en_US.UTF-8")
            .env("TERM", "dumb")
            .env("PAGER", "cat")
            .env("GIT_PAGER", "cat")
            .env("NO_COLOR", "1")
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .process_group(0);
        c
    }

    /// Run a command line; returns (exit code or None if killed, combined output).
    pub fn run(
        &self,
        argv: &[&str],
        timeout: Duration,
        cancel: &AtomicBool,
    ) -> Result<(Option<i32>, String, bool), String> {
        let mut child = self
            .command(argv)
            .spawn()
            .map_err(|e| format!("could not start sandbox-exec: {e}"))?;
        let mut stdout = child.stdout.take().ok_or("no stdout")?;
        let (tx, rx) = channel::<Vec<u8>>();
        let reader = std::thread::spawn(move || {
            let mut buf = [0u8; 8192];
            while let Ok(n) = stdout.read(&mut buf) {
                if n == 0 || tx.send(buf[..n].to_vec()).is_err() {
                    break;
                }
            }
        });
        let started = Instant::now();
        let mut out: Vec<u8> = Vec::new();
        let (mut code, mut killed) = (None, false);
        loop {
            while let Ok(chunk) = rx.try_recv() {
                if out.len() < 400_000 {
                    out.extend_from_slice(&chunk);
                }
            }
            match child.try_wait() {
                Ok(Some(status)) => {
                    code = status.code();
                    break;
                }
                Ok(None) => {}
                Err(_) => break,
            }
            if cancel.load(Ordering::SeqCst) || started.elapsed() > timeout {
                killed = true;
                let _ = Command::new("/bin/kill")
                    .args(["-KILL", &format!("-{}", child.id())])
                    // The group may be gone already: keep kill's complaint off the screen.
                    .stdout(Stdio::null())
                    .stderr(Stdio::null())
                    .status();
                let _ = child.kill();
                let _ = child.wait();
                break;
            }
            std::thread::sleep(Duration::from_millis(25));
        }
        // Jobs the command left in the background (`server &`) would hold the
        // output pipe open and hang this call: end the whole group.
        if !killed {
            let _ = Command::new("/bin/kill")
                .args(["-KILL", &format!("-{}", child.id())])
                // The group may be gone already: keep kill's complaint off the screen.
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status();
        }
        // Whatever escaped the group may still hold the pipe; stop waiting soon.
        let deadline = Instant::now() + Duration::from_secs(2);
        while let Ok(chunk) = rx.recv_timeout(deadline.saturating_duration_since(Instant::now())) {
            if out.len() < 400_000 {
                out.extend_from_slice(&chunk);
            }
        }
        drop(reader);
        Ok((code, String::from_utf8_lossy(&out).into_owned(), killed))
    }

    /// Run one tool call. Returns (succeeded, text for the model).
    pub fn execute(
        &self,
        name: &str,
        args: &Map<String, Json>,
        cancel: &AtomicBool,
    ) -> (bool, String) {
        let result = match name {
            "bash" => self.bash(args, cancel),
            "read_file" => self.read_file(args),
            "write_file" => self.write_file(args),
            "edit_file" => self.edit_file(args),
            "list_dir" => self.list_dir(args),
            "fetch_url" => self.fetch_url(args, cancel),
            "web_search" if !self.network => Err("the network is turned off for this chat".into()),
            "web_search" => {
                crate::search::search(&self.search, arg_str(args, "query").unwrap_or(""))
            }
            other => Err(format!("unknown tool {other}")),
        };
        match result {
            Ok(text) => (true, text),
            Err(e) => (false, format!("error: {e}")),
        }
    }

    pub fn bash(&self, args: &Map<String, Json>, cancel: &AtomicBool) -> Result<String, String> {
        let command = arg_str(args, "command")?;
        let secs = arg_int(args, "timeout_secs").unwrap_or(60).clamp(1, 300) as u64;
        let script = format!("exec 2>&1\n{command}");
        let (code, out, killed) = self.run(
            &["/bin/bash", "-c", &script],
            Duration::from_secs(secs),
            cancel,
        )?;
        let mut text = clip_output(out.trim_end(), self.output_limit);
        if !text.is_empty() {
            text.push('\n');
        }
        if killed {
            text += &if cancel.load(Ordering::SeqCst) {
                "[cancelled by the user]".to_string()
            } else {
                format!("[timed out after {secs}s and was killed]")
            };
            Err(text)
        } else {
            let code = code.unwrap_or(-1);
            text += &format!("[exit code {code}]");
            if code == 0 { Ok(text) } else { Err(text) }
        }
    }

    pub fn read_file(&self, args: &Map<String, Json>) -> Result<String, String> {
        let path = self.resolve(arg_str(args, "path")?)?;
        let bytes = self.read_bytes(&path)?;
        if bytes.contains(&0) {
            return Err(format!("{} is a binary file", self.rel(&path)));
        }
        let text = String::from_utf8_lossy(&bytes);
        let offset = arg_int(args, "offset").unwrap_or(1).max(1) as usize;
        let limit = arg_int(args, "limit").unwrap_or(400).clamp(1, 2000) as usize;
        let total = text.lines().count();
        let mut out = String::new();
        for (i, line) in text.lines().enumerate().skip(offset - 1).take(limit) {
            out += &format!("{:>5}\t{line}\n", i + 1);
        }
        if out.is_empty() {
            return Ok(format!("(empty: the file has {total} lines)"));
        }
        let shown_to = (offset - 1 + limit).min(total);
        if shown_to < total {
            out += &format!(
                "… {} more lines (use offset {})\n",
                total - shown_to,
                shown_to + 1
            );
        }
        Ok(clip_output(&out, 24_000))
    }

    pub fn write_file(&self, args: &Map<String, Json>) -> Result<String, String> {
        let path = self.resolve(arg_str(args, "path")?)?;
        let content = arg_str(args, "content")?;
        if let Some(dir) = path.parent() {
            fs::create_dir_all(dir).map_err(|e| e.to_string())?;
        }
        self.write_text(&path, content)?;
        Ok(format!(
            "wrote {} bytes ({} lines) to {}",
            content.len(),
            content.lines().count(),
            self.rel(&path)
        ))
    }

    pub fn edit_file(&self, args: &Map<String, Json>) -> Result<String, String> {
        let path = self.resolve(arg_str(args, "path")?)?;
        let (old, new) = (arg_str(args, "old_text")?, arg_str(args, "new_text")?);
        if old.is_empty() {
            return Err("old_text is empty".into());
        }
        let text = self.read_text(&path)?;
        match text.matches(old).count() {
            0 => Err("old_text was not found; read the file again and copy it exactly".into()),
            1 => {
                self.write_text(&path, &text.replacen(old, new, 1))?;
                Ok(format!(
                    "edited {}: replaced {} lines with {}",
                    self.rel(&path),
                    old.lines().count().max(1),
                    new.lines().count().max(1)
                ))
            }
            n => Err(format!(
                "old_text matches {n} places; include more surrounding text to make it unique"
            )),
        }
    }

    pub fn list_dir(&self, args: &Map<String, Json>) -> Result<String, String> {
        let path = self.resolve(args.get("path").and_then(Json::as_str).unwrap_or(""))?;
        let mut entries: Vec<(bool, String, u64)> = fs::read_dir(&path)
            .map_err(|e| format!("{}: {e}", self.rel(&path)))?
            .flatten()
            .filter_map(|e| {
                let name = e.file_name().to_string_lossy().to_string();
                if path == self.workspace
                    && self.state == self.workspace
                    && matches!(name.as_str(), ".home" | ".tmp")
                {
                    return None;
                }
                let meta = e.metadata().ok()?;
                Some((meta.is_dir(), name, meta.len()))
            })
            .collect();
        entries.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
        if entries.is_empty() {
            return Ok("(empty directory)".into());
        }
        let total = entries.len();
        let mut out = String::new();
        for (dir, name, size) in entries.into_iter().take(300) {
            if dir {
                out += &format!("{name}/\n");
            } else {
                out += &format!("{name}  ({size} bytes)\n");
            }
        }
        if total > 300 {
            out += &format!("… and {} more\n", total - 300);
        }
        Ok(out)
    }

    pub fn fetch_url(
        &self,
        args: &Map<String, Json>,
        cancel: &AtomicBool,
    ) -> Result<String, String> {
        if !self.network {
            return Err("the network is turned off for this chat".into());
        }
        let url = arg_str(args, "url")?.trim();
        if !(url.starts_with("http://") || url.starts_with("https://")) {
            return Err("only http:// and https:// URLs can be fetched".into());
        }
        let max = arg_int(args, "max_chars")
            .unwrap_or(12_000)
            .clamp(500, 60_000) as usize;
        let (code, out, killed) = self.run(
            &[
                "/usr/bin/curl",
                "-sSL",
                "--max-time",
                "30",
                "--max-filesize",
                "8000000",
                "-A",
                "Mozilla/5.0 (Macintosh) ferrum",
                "-H",
                "Accept: text/html,text/plain,application/json;q=0.9,*/*;q=0.5",
                "-w",
                "\n%{content_type}",
                "--",
                url,
            ],
            Duration::from_secs(35),
            cancel,
        )?;
        if killed {
            return Err("fetch cancelled or timed out".into());
        }
        if code != Some(0) {
            return Err(format!(
                "curl failed (exit {}): {}",
                code.unwrap_or(-1),
                out.trim()
            ));
        }
        let (body, content_type) = out.rsplit_once('\n').unwrap_or((&out, ""));
        let text = if content_type.contains("html") || body.trim_start().starts_with("<!") {
            html_to_text(body)
        } else {
            body.to_string()
        };
        let text = text.trim();
        let mut result = format!("{url} ({content_type})\n\n");
        if text.chars().count() > max {
            result += &text.chars().take(max).collect::<String>();
            result += "\n… [truncated]";
        } else {
            result += text;
        }
        Ok(result)
    }
}

/// `&#39;` and `&#x27;` style character references.
fn decode_numeric_entities(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(at) = rest.find("&#") {
        out += &rest[..at];
        let tail = &rest[at + 2..];
        let decoded = tail.find(';').filter(|&end| end <= 8).and_then(|end| {
            let digits = &tail[..end];
            let code = match digits.strip_prefix(['x', 'X']) {
                Some(hex) => u32::from_str_radix(hex, 16).ok(),
                None => digits.parse().ok(),
            };
            code.and_then(char::from_u32)
                .filter(|c| !c.is_control() || c.is_whitespace())
                .map(|c| (c, end))
        });
        match decoded {
            Some((c, end)) => {
                out.push(c);
                rest = &tail[end + 1..];
            }
            None => {
                out += "&#";
                rest = tail;
            }
        }
    }
    out + rest
}

/// Reduce HTML to readable text: drop scripts and styles, keep block breaks.
pub fn html_to_text(html: &str) -> String {
    let lower = html.to_ascii_lowercase();
    let mut keep = String::with_capacity(html.len());
    let mut i = 0;
    while i < html.len() {
        let rest = &lower[i..];
        let skip = ["<script", "<style", "<noscript", "<svg", "<head"]
            .iter()
            .find(|t| rest.starts_with(**t));
        if let Some(tag) = skip {
            let close = format!("</{}", &tag[1..]);
            match rest.find(&close) {
                Some(end) => {
                    let after = i + end;
                    i = html[after..]
                        .find('>')
                        .map_or(html.len(), |g| after + g + 1);
                }
                None => break,
            }
            continue;
        }
        let ch = html[i..].chars().next().unwrap_or(' ');
        keep.push(ch);
        i += ch.len_utf8();
    }
    let mut out = String::with_capacity(keep.len());
    let mut in_tag = false;
    let mut tag = String::new();
    for c in keep.chars() {
        match c {
            '<' => {
                in_tag = true;
                tag.clear();
            }
            '>' if in_tag => {
                in_tag = false;
                let t = tag.trim_start_matches('/').to_ascii_lowercase();
                let name = t
                    .split(|c: char| !c.is_ascii_alphanumeric())
                    .next()
                    .unwrap_or("");
                if matches!(
                    name,
                    "p" | "div"
                        | "br"
                        | "li"
                        | "tr"
                        | "h1"
                        | "h2"
                        | "h3"
                        | "h4"
                        | "h5"
                        | "h6"
                        | "section"
                        | "article"
                        | "ul"
                        | "ol"
                        | "table"
                        | "pre"
                        | "blockquote"
                ) {
                    out.push('\n');
                }
            }
            _ if in_tag => tag.push(c),
            _ => out.push(c),
        }
    }
    let decoded = out
        .replace("&nbsp;", " ")
        .replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&#39;", "'")
        .replace("&rsquo;", "'")
        .replace("&mdash;", "—");
    let decoded = decode_numeric_entities(&decoded);
    let mut result = String::new();
    let mut blank = 0;
    for line in decoded.lines() {
        let line = line.split_whitespace().collect::<Vec<_>>().join(" ");
        if line.is_empty() {
            blank += 1;
            if blank == 1 && !result.is_empty() {
                result.push('\n');
            }
        } else {
            blank = 0;
            result += &line;
            result.push('\n');
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sandbox(tag: &str, network: bool) -> (Sandbox, PathBuf) {
        let dir =
            std::env::temp_dir().join(format!("ferrum-agent-test-{}-{tag}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let sb = Sandbox::new(&dir.display().to_string(), network).expect("sandbox");
        (sb, dir)
    }

    fn call(sb: &Sandbox, name: &str, args: Json) -> (bool, String) {
        let map = args.as_object().cloned().unwrap_or_default();
        sb.execute(name, &map, &AtomicBool::new(false))
    }

    #[test]
    fn writes_only_inside_the_workspace() {
        let (sb, dir) = sandbox("write", false);
        let (ok, out) = call(
            &sb,
            "bash",
            json!({"command": "echo hi > inside.txt && cat inside.txt"}),
        );
        assert!(ok, "{out}");
        assert!(out.contains("hi"));
        let outside = home().join("ferrum-sandbox-escape-test.txt");
        let (ok, out) = call(
            &sb,
            "bash",
            json!({"command": format!("echo no > {}", outside.display())}),
        );
        assert!(!ok && !outside.exists(), "{out}");
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn sandbox_off_disables_the_confinement() {
        // On by default, then turned off flips every restriction back on.
        let (mut sb, dir) = sandbox("off", false);
        assert!(sb.confined, "the sandbox is confined by default");
        sb.confined = false;
        // With Seatbelt off, a command can write outside the workspace and read
        // a system file the confined sandbox would block.
        let outside = home().join("ferrum-sandbox-off-test.txt");
        let _ = fs::remove_file(&outside);
        let (ok, out) = call(
            &sb,
            "bash",
            json!({"command": format!(
                "echo yes > {} && cat {}",
                outside.display(),
                outside.display()
            )}),
        );
        assert!(
            ok && out.contains("yes"),
            "writing outside the workspace: {out}"
        );
        let (ok, out) = call(&sb, "bash", json!({"command": "cat /etc/hosts"}));
        assert!(ok, "/etc/hosts should be readable when unconfined: {out}");
        let _ = fs::remove_file(&outside);
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn hides_secrets_and_can_cut_the_network() {
        let (sb, dir) = sandbox("net", false);
        if home().join(".ssh").exists() {
            let (ok, _) = call(&sb, "bash", json!({"command": "ls ~/.ssh"}));
            let (ok2, out) = call(
                &sb,
                "bash",
                json!({"command": format!("ls {}", home().join(".ssh").display())}),
            );
            assert!(!ok2, "{out}");
            let _ = ok;
        }
        let (ok, out) = call(
            &sb,
            "bash",
            json!({"command": "curl -sS -m 5 -o /dev/null https://example.com", "timeout_secs": 10}),
        );
        assert!(!ok, "network should be off: {out}");
        // The clipboard and keychain are system services outside the sandbox.
        let (ok, out) = call(&sb, "bash", json!({"command": "pbpaste"}));
        assert!(!ok, "clipboard should be unreachable: {out}");
        let (_, out) = call(&sb, "bash", json!({"command": "security list-keychains"}));
        assert!(!out.contains("login.keychain"), "keychain: {out}");
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn file_tools_stay_in_the_workspace() {
        let (sb, dir) = sandbox("files", false);
        assert!(
            call(
                &sb,
                "write_file",
                json!({"path": "a/b.txt", "content": "one\ntwo\nthree\n"})
            )
            .0
        );
        let (ok, out) = call(&sb, "read_file", json!({"path": "a/b.txt", "offset": 2}));
        assert!(ok && out.contains("two") && !out.contains("one"), "{out}");
        assert!(!call(&sb, "read_file", json!({"path": "../../etc/hosts"})).0);
        assert!(
            !call(
                &sb,
                "write_file",
                json!({"path": "/etc/ferrum-nope", "content": "x"})
            )
            .0
        );
        // A symlink pointing out of the workspace is refused as well.
        let _ = std::os::unix::fs::symlink("/etc", sb.workspace.join("link"));
        assert!(!call(&sb, "read_file", json!({"path": "link/hosts"})).0);
        let (ok, out) = call(
            &sb,
            "edit_file",
            json!({"path": "a/b.txt", "old_text": "two", "new_text": "2"}),
        );
        assert!(ok, "{out}");
        assert!(
            fs::read_to_string(sb.workspace.join("a/b.txt"))
                .unwrap()
                .contains("\n2\n")
        );
        let (ok, out) = call(
            &sb,
            "edit_file",
            json!({"path": "a/b.txt", "old_text": "e", "new_text": "E"}),
        );
        assert!(!ok && out.contains("matches"), "{out}");
        let (ok, out) = call(&sb, "list_dir", json!({}));
        assert!(ok && out.contains("a/") && !out.contains(".home"), "{out}");
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn git_settings_and_hooks_stay_read_only() {
        let (sb, dir) = sandbox("git", false);
        let init = Command::new("git")
            .args(["init", "-q"])
            .arg(&sb.workspace)
            .status();
        if init.is_ok_and(|s| s.success()) {
            for command in [
                "git config core.fsmonitor 'touch pwned'",
                "echo x > .git/hooks/post-checkout",
                "mv .git .git-old",
            ] {
                let (ok, out) = call(&sb, "bash", json!({ "command": command }));
                assert!(!ok, "{command}: {out}");
            }
            let (ok, out) = call(&sb, "bash", json!({"command": "echo x > f && git add f"}));
            assert!(ok, "{out}");
        }
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn file_tools_refuse_dangling_symlinks() {
        let (sb, dir) = sandbox("dangling", false);
        let outside = dir.with_extension("outside");
        let _ = fs::remove_file(&outside);
        std::os::unix::fs::symlink(&outside, sb.workspace.join("out")).unwrap();
        let (ok, out) = call(&sb, "write_file", json!({"path": "out", "content": "x"}));
        assert!(!ok && !outside.exists(), "{out}");
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn background_jobs_do_not_hold_a_command_open() {
        let (sb, dir) = sandbox("background", false);
        let started = Instant::now();
        let (ok, out) = call(&sb, "bash", json!({"command": "sleep 30 & echo started"}));
        assert!(ok && out.contains("started"), "{out}");
        assert!(started.elapsed() < Duration::from_secs(10));
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn bash_reports_exit_codes_and_timeouts() {
        let (sb, dir) = sandbox("bash", false);
        let (ok, out) = call(
            &sb,
            "bash",
            json!({"command": "echo out; echo err >&2; exit 3"}),
        );
        assert!(
            !ok && out.contains("out") && out.contains("err") && out.contains("exit code 3"),
            "{out}"
        );
        let (ok, out) = call(
            &sb,
            "bash",
            json!({"command": "sleep 20", "timeout_secs": 1}),
        );
        assert!(!ok && out.contains("timed out"), "{out}");
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn html_becomes_text() {
        let text = html_to_text(
            "<html><head><title>x</title><style>p{}</style></head><body><h1>Hi &amp; bye</h1><script>var a=1;</script><p>one   two</p><ul><li>a</li><li>b</li></ul></body></html>",
        );
        assert!(
            text.contains("Hi & bye") && text.contains("one two"),
            "{text}"
        );
        assert_eq!(
            html_to_text("<p>it&#39;s &#x27;ok&#x27; &#xZZ; &#1;</p>"),
            "it's 'ok' &#xZZ; &#1;\n"
        );
        assert!(!text.contains("var a") && !text.contains("p{}"), "{text}");
    }

    #[test]
    fn clip_output_keeps_both_ends() {
        let s = "a".repeat(100) + &"z".repeat(100);
        let c = clip_output(&s, 40);
        assert!(c.starts_with("aaaa") && c.ends_with("zzzz") && c.contains("omitted"));
    }
}
