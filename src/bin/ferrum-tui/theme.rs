use crate::*;

pub const EMBER: Color = Color::Rgb(255, 138, 61);
pub const GOLD: Color = Color::Rgb(255, 200, 120);
pub const TEXT: Color = Color::Rgb(226, 226, 236);
pub const DIM: Color = Color::Rgb(122, 122, 140);
pub const FAINT: Color = Color::Rgb(70, 70, 86);
pub const GOOD: Color = Color::Rgb(124, 220, 164);
pub const BAD: Color = Color::Rgb(255, 104, 104);
pub const SELECTED: Color = Color::Rgb(38, 33, 31);

pub fn home() -> PathBuf {
    std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_default()
}

pub fn tilde(path: &Path) -> String {
    let h = home();
    match path.strip_prefix(&h) {
        Ok(rest) if !h.as_os_str().is_empty() => format!("~/{}", rest.display()),
        _ => path.display().to_string(),
    }
}

pub fn expand(input: &str) -> PathBuf {
    match input.strip_prefix("~/") {
        Some(rest) => home().join(rest),
        None => PathBuf::from(input),
    }
}

pub fn gib(bytes: u64) -> String {
    format!("{:.1} GiB", bytes as f64 / (1u64 << 30) as f64)
}

pub fn clip(s: &str, width: usize) -> String {
    if s.chars().count() <= width {
        return s.to_string();
    }
    let keep = width.saturating_sub(1);
    format!("{}…", s.chars().take(keep).collect::<String>())
}

pub fn quant_of(name: &str) -> Option<String> {
    name.split(['-', '.', ' ']).find_map(|t| {
        let u = t.to_ascii_uppercase();
        let digit_after = |p: &str| {
            u.strip_prefix(p)
                .is_some_and(|r| r.starts_with(|c: char| c.is_ascii_digit()))
        };
        (digit_after("Q")
            || digit_after("IQ")
            || matches!(u.as_str(), "BF16" | "F16" | "F32")
            || u.starts_with("MXFP"))
        .then_some(u)
    })
}

pub fn fmt_num(n: f64, int: bool) -> String {
    if int {
        format!("{}", n.round() as i64)
    } else {
        let s = format!("{:.3}", (n * 1000.0).round() / 1000.0);
        s.trim_end_matches('0').trim_end_matches('.').to_string()
    }
}

pub fn shell_quote(s: &str) -> String {
    if !s.is_empty()
        && s.chars()
            .all(|c| c.is_ascii_alphanumeric() || "/._-:=,@~+".contains(c))
    {
        s.to_string()
    } else {
        format!("'{}'", s.replace('\'', "'\\''"))
    }
}

pub const CODE_BG: Color = Color::Rgb(27, 27, 36);
