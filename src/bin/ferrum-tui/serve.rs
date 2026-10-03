//! Serve inside the TUI: runs `ferrum-server` as a child process, shows its
//! log live and whether it is accepting connections yet.
use crate::*;
use std::cell::Cell;
use std::io::{BufRead, BufReader};
use std::net::{SocketAddr, TcpStream};
use std::process::{Child, Stdio};
use std::sync::mpsc::{Receiver, channel};

const MAX_LINES: usize = 4000;

pub struct Serve {
    child: Child,
    rx: Receiver<String>,
    pub lines: Vec<String>,
    pub started: Instant,
    pub ready: bool,
    probed: Instant,
    pub exit: Option<String>,
    pub endpoint: String,
    pub local: String,
    pub name: String,
    pub detail: String,
    pub key_set: bool,
    addr: Option<SocketAddr>,
    follow: bool,
    top: Cell<usize>,
    max_top: Cell<usize>,
    page: Cell<usize>,
}

fn strip_ansi(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\x1b' {
            if chars.peek() == Some(&'[') {
                for n in chars.by_ref() {
                    if n.is_ascii_alphabetic() {
                        break;
                    }
                }
            }
        } else if c != '\r' {
            out.push(c);
        }
    }
    out
}

impl Serve {
    pub fn start(app: &App) -> Result<Serve, String> {
        let (bin, args) = app.command();
        let program = std::env::current_exe()
            .ok()
            .and_then(|p| p.parent().map(|d| d.join(bin)))
            .filter(|p| p.exists())
            .unwrap_or_else(|| PathBuf::from(bin));
        let mut child = Command::new(&program)
            .args(&args)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| format!("could not start {bin}: {e}"))?;
        let (tx, rx) = channel();
        if let Some(out) = child.stdout.take() {
            let tx = tx.clone();
            std::thread::spawn(move || {
                for line in BufReader::new(out).lines().map_while(Result::ok) {
                    if tx.send(line).is_err() {
                        break;
                    }
                }
            });
        }
        if let Some(err) = child.stderr.take() {
            std::thread::spawn(move || {
                for line in BufReader::new(err).lines().map_while(Result::ok) {
                    if tx.send(line).is_err() {
                        break;
                    }
                }
            });
        }
        let host = match app.value("host") {
            "" => "127.0.0.1",
            h => h,
        };
        let port = match app.value("port") {
            "" => "8080",
            p => p,
        };
        let local_host = if host == "0.0.0.0" { "127.0.0.1" } else { host };
        let model = app.selected().map(|m| m.name.clone()).unwrap_or_default();
        let name = match app.value("alias") {
            "" => model,
            a => a.to_string(),
        };
        let draft = app.value("draft");
        let ctx = match app.value("context") {
            "" => "auto",
            c => c,
        };
        let detail = format!(
            "context {ctx}{}",
            if draft.is_empty() || draft == "off" {
                String::new()
            } else {
                format!(" · ⚡ {}", app.draft_label(draft))
            }
        );
        Ok(Serve {
            child,
            rx,
            lines: Vec::new(),
            started: Instant::now(),
            ready: false,
            probed: Instant::now(),
            exit: None,
            endpoint: format!("http://{host}:{port}/v1"),
            local: format!("http://{local_host}:{port}"),
            name,
            detail,
            key_set: !app.value("api_key").is_empty(),
            addr: format!("{local_host}:{port}").parse().ok(),
            follow: true,
            top: Cell::new(0),
            max_top: Cell::new(0),
            page: Cell::new(10),
        })
    }

    pub fn pump(&mut self) {
        while let Ok(line) = self.rx.try_recv() {
            self.lines.push(strip_ansi(&line));
            if self.lines.len() > MAX_LINES {
                self.lines.drain(..MAX_LINES / 4);
            }
        }
        if self.exit.is_none()
            && let Ok(Some(status)) = self.child.try_wait()
        {
            self.exit = Some(status.to_string());
            self.ready = false;
        }
        if !self.ready && self.exit.is_none() && self.probed.elapsed() > Duration::from_millis(400)
        {
            self.probed = Instant::now();
            if let Some(addr) = self.addr {
                self.ready = healthy(addr);
            }
        }
    }

    pub fn stop(&mut self) {
        if self.exit.is_none() {
            let _ = self.child.kill();
            let _ = self.child.wait();
            self.exit = Some("stopped".into());
        }
    }

    fn scroll(&mut self, up: bool, amount: usize) {
        let top = if self.follow {
            self.max_top.get()
        } else {
            self.top.get()
        };
        let next = if up {
            top.saturating_sub(amount)
        } else {
            (top + amount).min(self.max_top.get())
        };
        self.follow = !up && next >= self.max_top.get();
        self.top.set(next);
    }
}

impl Drop for Serve {
    fn drop(&mut self) {
        self.stop();
    }
}

impl App {
    pub fn leave_serve(&mut self) {
        if let Some(mut s) = self.serve.take() {
            s.stop();
        }
        self.screen = Screen::Pick;
    }

    pub fn serve_key(&mut self, key: KeyEvent) {
        let Some(s) = self.serve.as_mut() else {
            self.screen = Screen::Home;
            return;
        };
        let page = s.page.get().max(2) - 1;
        match key.code {
            KeyCode::Char('q') | KeyCode::Esc => self.leave_serve(),
            KeyCode::PageUp => s.scroll(true, page),
            KeyCode::PageDown => s.scroll(false, page),
            KeyCode::Up | KeyCode::Char('k') => s.scroll(true, 1),
            KeyCode::Down | KeyCode::Char('j') => s.scroll(false, 1),
            KeyCode::End | KeyCode::Char('G') => s.follow = true,
            KeyCode::Char('c') => {
                s.lines.clear();
                s.follow = true;
            }
            _ => {}
        }
    }

    pub fn draw_serve(&self, f: &mut Frame) {
        let Some(s) = self.serve.as_ref() else {
            return;
        };
        let t = self.elapsed();
        let [head, info, logs, footer] = Layout::vertical([
            Constraint::Length(2),
            Constraint::Length(9),
            Constraint::Min(5),
            Constraint::Length(1),
        ])
        .areas(f.area());
        let (state, color) = match (&s.exit, s.ready) {
            (Some(_), _) => ("STOPPED", BAD),
            (None, true) => ("LISTENING", GOOD),
            _ => ("LOADING", GOLD),
        };
        header(f, head, &["Serve", &s.name], vec![pill(state, color)]);

        let block = panel("Endpoint", s.ready);
        let inner = block.inner(info);
        f.render_widget(block, info);
        let label = |k: &str| Span::styled(format!("  {k:<10}"), Style::new().fg(DIM));
        let val = |v: String| Span::styled(v, Style::new().fg(TEXT));
        let auth = if s.key_set {
            " -H 'Authorization: Bearer <your key>'"
        } else {
            ""
        };
        let curl = format!(
            "curl {}/v1/chat/completions -H 'Content-Type: application/json'{auth} -d '{{\"model\":\"{}\",\"messages\":[{{\"role\":\"user\",\"content\":\"Hello\"}}]}}'",
            s.local, s.name
        );
        let status_line = match (&s.exit, s.ready) {
            (Some(e), _) => Line::from(vec![
                label("status"),
                Span::styled(format!("server exited ({e})"), Style::new().fg(BAD)),
            ]),
            (None, true) => Line::from(vec![
                label("status"),
                Span::styled("● accepting connections", Style::new().fg(GOOD)),
                Span::styled(
                    format!("  up {}", fmt_dur(s.started.elapsed())),
                    Style::new().fg(DIM),
                ),
            ]),
            _ => Line::from(vec![
                label("status"),
                Span::styled(
                    format!("{} loading the model", spinner(t)),
                    Style::new().fg(GOLD),
                ),
                Span::styled(
                    format!("  {}", fmt_dur(s.started.elapsed())),
                    Style::new().fg(DIM),
                ),
            ]),
        };
        let lines = vec![
            Line::from(vec![
                label("endpoint"),
                Span::styled(
                    s.endpoint.clone(),
                    Style::new().fg(EMBER).add_modifier(Modifier::BOLD),
                ),
                Span::styled(
                    if s.key_set {
                        "   🔒 api key required"
                    } else {
                        ""
                    },
                    Style::new().fg(DIM),
                ),
            ]),
            Line::from(vec![
                label("model"),
                val(s.name.clone()),
                Span::styled(format!("  ·  {}", s.detail), Style::new().fg(DIM)),
            ]),
            status_line,
            Line::default(),
            Line::from(vec![
                label("try it"),
                Span::styled(curl, Style::new().fg(GOLD)),
            ]),
            Line::from(vec![
                label("also"),
                Span::styled(
                    "/v1/messages (Anthropic) · /v1/models · /health · /metrics",
                    Style::new().fg(DIM),
                ),
            ]),
        ];
        f.render_widget(Paragraph::new(lines).wrap(Wrap { trim: false }), inner);

        let block = panel("Log", false);
        let inner = block.inner(logs);
        f.render_widget(block, logs);
        let width = inner.width.saturating_sub(2) as usize;
        let mut rows: Vec<Line> = Vec::new();
        for line in &s.lines {
            let style = if line.contains(" ERR ") {
                Style::new().fg(BAD)
            } else if line.contains(" WRN ") {
                Style::new().fg(GOLD)
            } else if line.contains(" DBG ") {
                Style::new().fg(DIM)
            } else {
                Style::new().fg(TEXT)
            };
            let chars: Vec<Sc> = line.chars().map(|c| (c, style)).collect();
            for row in wrap_styled(&chars, width, 4) {
                rows.push(to_line(&row, vec![Span::raw(" ")]));
            }
        }
        if rows.is_empty() {
            rows.push(Line::styled(" waiting for output…", Style::new().fg(FAINT)));
        }
        let h = inner.height as usize;
        let max_top = rows.len().saturating_sub(h);
        s.max_top.set(max_top);
        s.page.set(h);
        let top = if s.follow {
            max_top
        } else {
            s.top.get().min(max_top)
        };
        s.top.set(top);
        f.render_widget(
            Paragraph::new(rows.into_iter().skip(top).take(h).collect::<Vec<_>>()),
            inner,
        );
        keys_footer(
            f,
            footer,
            &[
                ("pgup/pgdn", "scroll"),
                ("end", "follow"),
                ("c", "clear log"),
                ("q", "stop server"),
            ],
            &self.status,
        );
    }
}

/// `GET /health` is 200 only once the model is loaded and warm.
fn healthy(addr: SocketAddr) -> bool {
    use std::io::{Read, Write};
    let Ok(mut stream) = TcpStream::connect_timeout(&addr, Duration::from_millis(30)) else {
        return false;
    };
    let _ = stream.set_read_timeout(Some(Duration::from_millis(150)));
    if stream
        .write_all(b"GET /health HTTP/1.0\r\nHost: localhost\r\n\r\n")
        .is_err()
    {
        return false;
    }
    let mut head = [0u8; 16];
    let n = stream.read(&mut head).unwrap_or(0);
    String::from_utf8_lossy(&head[..n]).contains(" 200")
}

fn fmt_dur(d: Duration) -> String {
    let s = d.as_secs();
    if s >= 3600 {
        format!("{}h{:02}m", s / 3600, s % 3600 / 60)
    } else if s >= 60 {
        format!("{}m{:02}s", s / 60, s % 60)
    } else {
        format!("{s}s")
    }
}
