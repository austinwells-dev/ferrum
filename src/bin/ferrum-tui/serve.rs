//! Serve inside the TUI: runs `ferrum-server` as a child process, shows its
//! log live and whether it is accepting connections yet.
use crate::layout;
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
        let mut command = Command::new(&program);
        match app.value("api_key") {
            "" => command.env_remove("FERRUM_API_KEY"),
            key => command.env("FERRUM_API_KEY", key),
        };
        let mut child = command
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

#[cfg(test)]
impl Serve {
    /// A listening server for drawing tests, backed by a process that exits at once.
    pub fn fake(lines: Vec<String>) -> Self {
        Self {
            child: Command::new("/usr/bin/true").spawn().expect("spawn true"),
            rx: channel().1,
            lines,
            started: Instant::now(),
            ready: true,
            probed: Instant::now(),
            exit: None,
            endpoint: "http://127.0.0.1:8080/v1".into(),
            local: "http://127.0.0.1:8080".into(),
            name: "Qwen3.8-27B-UD-Q4_K_XL".into(),
            detail: "16.7 GiB · context 32768 · ⚡ z-lab/Qwen3.8-27B-DFlash2 (dflash) depth 7"
                .into(),
            key_set: true,
            addr: None,
            follow: true,
            top: Cell::new(0),
            max_top: Cell::new(0),
            page: Cell::new(0),
        }
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
        self.mode = Mode::Serve;
        self.screen = Screen::Pick;
    }

    pub fn serve_key(&mut self, key: KeyEvent) {
        let armed = self.armed.as_deref() == Some("stop server");
        let Some(s) = self.serve.as_mut() else {
            self.open_pick(Mode::Serve);
            return;
        };
        let page = s.page.get().max(2) - 1;
        match key.code {
            // The server keeps running on other tabs.
            KeyCode::Esc => self.go_tab(Tab::Home),
            KeyCode::Char('x') | KeyCode::Char('q') if s.exit.is_some() || armed => {
                self.leave_serve()
            }
            KeyCode::Char('x') | KeyCode::Char('q') => {
                self.status = Some(("press x again to stop the server".into(), false));
                self.confirm = Some("stop server".into());
            }
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

    pub fn draw_serve(&self, f: &mut Frame, body: Rect, footer: Rect) {
        let Some(s) = self.serve.as_ref() else {
            return;
        };
        let t = self.elapsed();
        let state = match (&s.exit, s.ready) {
            (Some(_), _) => "stopped",
            (None, true) => "listening",
            _ => "loading",
        };
        let stats = request_stats(&s.lines);
        let auth = if s.key_set {
            " -H 'Authorization: Bearer $KEY'"
        } else {
            ""
        };
        let curl = format!(
            "curl {}/v1/chat/completions -H 'Content-Type: application/json'{auth} -d '{{\"model\":\"{}\",\"messages\":[{{\"role\":\"user\",\"content\":\"Hello\"}}]}}'",
            s.local, s.name
        );
        let status = match (&s.exit, s.ready) {
            (Some(e), _) => (format!("server exited ({e})"), BAD),
            (None, true) => (
                format!("● accepting · up {}", fmt_dur(s.started.elapsed())),
                GOOD,
            ),
            _ => (
                format!(
                    "{} loading the model · {}",
                    spinner(t),
                    fmt_dur(s.started.elapsed())
                ),
                GOLD,
            ),
        };
        // Wide: endpoint, traffic and routes in a column, the log beside it.
        let (left, logs) = match layout::columns(body, self.prefs.serve, 44, 50, self.prefs.compact)
        {
            (l, Some(r)) => (Some(l), r),
            (all, None) => (None, all),
        };
        let logs = match left {
            Some(l) => {
                let [ep, traffic, routes] = Layout::vertical([
                    Constraint::Length(8),
                    Constraint::Length(10),
                    Constraint::Min(4),
                ])
                .areas(l);
                let w = inner_w(l);
                let inner = layout::pane(f, ep, "Endpoint", true, state);
                let lines = vec![
                    Line::styled(
                        format!(" {}", s.endpoint),
                        Style::new().fg(EMBER).add_modifier(Modifier::BOLD),
                    ),
                    Line::styled(
                        if s.key_set {
                            " api key required"
                        } else {
                            " open: no api key"
                        },
                        Style::new().fg(DIM),
                    ),
                    Line::default(),
                    layout::kv("status", status.0.clone(), Style::new().fg(status.1), 9),
                    layout::kv(
                        "model",
                        clip(&s.name, w.saturating_sub(11)),
                        Style::new().fg(TEXT),
                        9,
                    ),
                    layout::kv(
                        "setup",
                        clip(&s.detail, w.saturating_sub(11)),
                        Style::new().fg(DIM),
                        9,
                    ),
                ];
                f.render_widget(Paragraph::new(lines), inner);
                self.draw_traffic(f, traffic, &stats);
                let inner = layout::pane(f, routes, "Routes", false, "");
                let mut lines: Vec<Line> = [
                    ("/v1/chat/completions", "OpenAI chat"),
                    ("/v1/messages", "Anthropic messages"),
                    ("/v1/models", "the loaded model"),
                    ("/health", "200 once warm"),
                    ("/metrics", "Prometheus counters"),
                ]
                .iter()
                .map(|(r, what)| {
                    Line::from(vec![
                        Span::styled(format!(" {r:<22}"), Style::new().fg(TEXT)),
                        Span::styled(*what, Style::new().fg(FAINT)),
                    ])
                })
                .collect();
                lines.push(Line::default());
                lines.push(layout::heading("try it"));
                lines.extend(layout::wrapped(&curl, 1, w, Style::new().fg(GOLD)));
                f.render_widget(Paragraph::new(lines), inner);
                logs
            }
            None => {
                let [info, logs] =
                    Layout::vertical([Constraint::Length(7), Constraint::Min(5)]).areas(logs);
                let inner = layout::pane(f, info, "Endpoint", true, state);
                let lines = vec![
                    Line::styled(
                        format!(" {}", s.endpoint),
                        Style::new().fg(EMBER).add_modifier(Modifier::BOLD),
                    ),
                    layout::kv("status", status.0.clone(), Style::new().fg(status.1), 9),
                    layout::kv("model", s.name.clone(), Style::new().fg(TEXT), 9),
                    layout::kv(
                        "traffic",
                        match stats.last() {
                            Some(_) => format!(
                                "{} requests · {:.1} tok/s on average",
                                stats.len(),
                                stats.iter().map(|r| r.1).sum::<f64>() / stats.len() as f64
                            ),
                            None => "no requests yet".into(),
                        },
                        Style::new().fg(DIM),
                        9,
                    ),
                ];
                f.render_widget(Paragraph::new(lines), inner);
                logs
            }
        };

        let note = if s.follow {
            "following · end"
        } else {
            "paused · end follows"
        };
        let inner = layout::pane(f, logs, "Log", false, note);
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
        self.footer(
            f,
            footer,
            &[
                ("↑↓ pgup/pgdn", "scroll"),
                ("end", "follow"),
                ("c", "clear log"),
                ("x", "stop server"),
                ("esc", "home, keep it running"),
            ],
        );
    }

    /// Requests answered: count, tokens, speed, and a column per request.
    fn draw_traffic(&self, f: &mut Frame, area: Rect, stats: &[(usize, f64)]) {
        let inner = layout::pane(f, area, "Traffic", false, "decode tok/s per request");
        let n = stats.len();
        let tokens: usize = stats.iter().map(|r| r.0).sum();
        let avg = if n > 0 {
            stats.iter().map(|r| r.1).sum::<f64>() / n as f64
        } else {
            0.0
        };
        let mut lines = vec![
            layout::kv("requests", n.to_string(), Style::new().fg(Color::White), 10),
            layout::kv(
                "generated",
                format!("{} tokens", crate::agentui::kilo(tokens)),
                Style::new().fg(Color::White),
                10,
            ),
            layout::kv(
                "average",
                if n > 0 {
                    format!("{avg:.1} tok/s")
                } else {
                    "–".into()
                },
                Style::new().fg(GOOD),
                10,
            ),
        ];
        let h = inner.height.saturating_sub(3) as usize;
        let w = inner.width.saturating_sub(2) as usize;
        if n == 0 || h == 0 {
            lines.push(Line::styled(
                " requests appear here",
                Style::new().fg(FAINT),
            ));
            f.render_widget(Paragraph::new(lines), inner);
            return;
        }
        // One column per request, newest on the right, in eighths of a row.
        const BLOCKS: [&str; 9] = [" ", "▁", "▂", "▃", "▄", "▅", "▆", "▇", "█"];
        let shown = &stats[n.saturating_sub(w)..];
        let top = shown.iter().map(|r| r.1).fold(1.0, f64::max);
        for row in (0..h).rev() {
            let text: String = shown
                .iter()
                .map(|r| {
                    let units = (r.1 / top * (h * 8) as f64).round() as usize;
                    BLOCKS[units.saturating_sub(row * 8).min(8)]
                })
                .collect();
            lines.push(Line::styled(format!(" {text}"), Style::new().fg(GOOD)));
        }
        f.render_widget(Paragraph::new(lines), inner);
    }
}

/// (generated tokens, decode tok/s) of each finished request in the log,
/// from the server's "done … generated N tok in T s (R tok/s)" lines.
pub fn request_stats(lines: &[String]) -> Vec<(usize, f64)> {
    lines
        .iter()
        .filter(|l| l.contains("done (") && l.contains("generated "))
        .filter_map(|l| {
            let rest = &l[l.find("generated ")? + "generated ".len()..];
            let tokens: usize = rest.split_whitespace().next()?.parse().ok()?;
            let at = rest.find(" tok/s")?;
            let open = rest[..at].rfind('(')?;
            let tps: f64 = rest[open + 1..at].trim().parse().ok()?;
            Some((tokens, tps))
        })
        .collect()
}

impl Serve {
    /// ":8080" for the tab bar.
    pub fn port_label(&self) -> String {
        match self.local.rsplit_once(':') {
            Some((_, port)) => format!(":{port}"),
            None => "server".into(),
        }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_request_speed_from_the_log() {
        let lines = vec![
            "12:00:01 INF [7] done (stop) in 9.10 s: prompt 412 tok (380 cached), prefill 32 tok in 0.05 s (640 tok/s); generated 431 tok in 9.02 s (47.8 tok/s), accept 3.1".to_string(),
            "12:00:02 INF listening on 127.0.0.1:8080".to_string(),
        ];
        assert_eq!(request_stats(&lines), vec![(431, 47.8)]);
    }
}
