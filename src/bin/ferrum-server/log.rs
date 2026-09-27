//! Timestamped, optionally coloured log lines on stderr.
use std::{
    io::{IsTerminal, Write},
    sync::OnceLock,
    time::{SystemTime, UNIX_EPOCH},
};

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Level {
    Debug,
    Info,
    Warn,
    Error,
}

struct Config {
    offset_seconds: i64,
    colour: bool,
    verbose: bool,
}

static CONFIG: OnceLock<Config> = OnceLock::new();

pub fn init(verbose: bool) {
    // Local UTC offset from `date +%z` (e.g. "+1000"), read once at startup.
    let offset_seconds = std::process::Command::new("date")
        .arg("+%z")
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .and_then(|s| {
            let s = s.trim();
            let sign = if s.starts_with('-') { -1 } else { 1 };
            let digits = s.trim_start_matches(['+', '-']);
            let hours: i64 = digits.get(..2)?.parse().ok()?;
            let minutes: i64 = digits.get(2..4)?.parse().ok()?;
            Some(sign * (hours * 3600 + minutes * 60))
        })
        .unwrap_or(0);
    let _ = CONFIG.set(Config {
        offset_seconds,
        colour: std::io::stderr().is_terminal(),
        verbose,
    });
}

pub fn verbose() -> bool {
    CONFIG.get().is_some_and(|c| c.verbose)
}

fn stamp() -> String {
    let offset = CONFIG.get().map_or(0, |c| c.offset_seconds);
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    let secs = now.as_secs() as i64 + offset;
    let day = secs.rem_euclid(86_400);
    format!(
        "{:02}:{:02}:{:02}.{:03}",
        day / 3600,
        day / 60 % 60,
        day % 60,
        now.subsec_millis()
    )
}

pub fn write(level: Level, request: Option<u64>, message: &str) {
    if level == Level::Debug && !verbose() {
        return;
    }
    let colour = CONFIG.get().is_some_and(|c| c.colour);
    let (tag, code) = match level {
        Level::Debug => ("DBG", "2"),
        Level::Info => ("INF", "0"),
        Level::Warn => ("WRN", "33"),
        Level::Error => ("ERR", "31"),
    };
    let id = request.map_or_else(String::new, |id| format!("[req {id}] "));
    let line = if colour {
        format!(
            "\x1b[2m{}\x1b[0m \x1b[{code}m{tag}\x1b[0m {id}{message}\n",
            stamp()
        )
    } else {
        format!("{} {tag} {id}{message}\n", stamp())
    };
    let _ = std::io::stderr().lock().write_all(line.as_bytes());
}

#[macro_export]
macro_rules! info { ($id:expr, $($t:tt)*) => { $crate::log::write($crate::log::Level::Info, $id, &format!($($t)*)) } }
#[macro_export]
macro_rules! warn { ($id:expr, $($t:tt)*) => { $crate::log::write($crate::log::Level::Warn, $id, &format!($($t)*)) } }
#[macro_export]
macro_rules! error { ($id:expr, $($t:tt)*) => { $crate::log::write($crate::log::Level::Error, $id, &format!($($t)*)) } }
#[macro_export]
macro_rules! debug { ($id:expr, $($t:tt)*) => { $crate::log::write($crate::log::Level::Debug, $id, &format!($($t)*)) } }

/// Thousands separators for token counts in logs.
pub fn n(v: usize) -> String {
    let s = v.to_string();
    let mut out = String::new();
    for (i, c) in s.chars().enumerate() {
        if i > 0 && (s.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(c);
    }
    out
}
