//! A small syntax highlighter for fenced code in the chat. It colours
//! comments, strings, numbers, keywords, types and calls; it does not parse.
use crate::text::Sc;
use crate::*;

struct Spec {
    kw: &'static [&'static str],
    line: &'static [&'static str],
    block: bool,
    /// ' starts a string (not in Rust, where it is mostly a lifetime).
    single: bool,
    backtick: bool,
    /// key: value or key = value at the start of a line
    keys: bool,
}

const C_KW: &[&str] = &[
    "if",
    "else",
    "for",
    "while",
    "do",
    "switch",
    "case",
    "break",
    "continue",
    "return",
    "struct",
    "enum",
    "union",
    "typedef",
    "static",
    "const",
    "extern",
    "void",
    "int",
    "char",
    "float",
    "double",
    "long",
    "short",
    "unsigned",
    "signed",
    "sizeof",
    "class",
    "public",
    "private",
    "protected",
    "namespace",
    "using",
    "template",
    "typename",
    "new",
    "delete",
    "this",
    "true",
    "false",
    "nullptr",
    "NULL",
    "auto",
    "bool",
    "virtual",
    "override",
    "inline",
    "include",
    "define",
    "import",
    "package",
    "interface",
    "extends",
    "implements",
    "final",
    "throw",
    "throws",
    "try",
    "catch",
    "boolean",
    "byte",
    "abstract",
    "synchronized",
    "null",
];
const RUST_KW: &[&str] = &[
    "fn", "let", "mut", "pub", "use", "mod", "struct", "enum", "impl", "trait", "for", "while",
    "loop", "if", "else", "match", "return", "break", "continue", "const", "static", "ref", "move",
    "async", "await", "dyn", "where", "type", "unsafe", "extern", "crate", "self", "Self", "super",
    "as", "in", "true", "false", "Some", "None", "Ok", "Err",
];
const PY_KW: &[&str] = &[
    "def", "class", "if", "elif", "else", "for", "while", "return", "import", "from", "as", "with",
    "try", "except", "finally", "raise", "pass", "break", "continue", "lambda", "yield", "in",
    "is", "not", "and", "or", "None", "True", "False", "self", "async", "await", "global",
    "nonlocal", "del", "assert", "print",
];
const JS_KW: &[&str] = &[
    "function",
    "const",
    "let",
    "var",
    "if",
    "else",
    "for",
    "while",
    "do",
    "return",
    "class",
    "extends",
    "new",
    "this",
    "import",
    "export",
    "from",
    "default",
    "async",
    "await",
    "try",
    "catch",
    "finally",
    "throw",
    "switch",
    "case",
    "break",
    "continue",
    "typeof",
    "instanceof",
    "in",
    "of",
    "null",
    "undefined",
    "true",
    "false",
    "interface",
    "type",
    "enum",
    "implements",
    "public",
    "private",
    "readonly",
    "yield",
    "void",
];
const GO_KW: &[&str] = &[
    "func",
    "var",
    "const",
    "type",
    "struct",
    "interface",
    "map",
    "chan",
    "go",
    "defer",
    "if",
    "else",
    "for",
    "range",
    "switch",
    "case",
    "default",
    "return",
    "break",
    "continue",
    "package",
    "import",
    "select",
    "nil",
    "true",
    "false",
    "fallthrough",
    "goto",
];
const SH_KW: &[&str] = &[
    "if", "then", "else", "elif", "fi", "for", "while", "do", "done", "case", "esac", "function",
    "in", "return", "exit", "export", "local", "set", "unset", "source", "echo", "cd", "true",
    "false",
];
const SQL_KW: &[&str] = &[
    "select",
    "from",
    "where",
    "insert",
    "into",
    "values",
    "update",
    "set",
    "delete",
    "create",
    "table",
    "drop",
    "alter",
    "join",
    "left",
    "right",
    "inner",
    "outer",
    "on",
    "group",
    "by",
    "order",
    "having",
    "limit",
    "as",
    "and",
    "or",
    "not",
    "null",
    "in",
    "is",
    "like",
    "distinct",
    "union",
    "index",
    "primary",
    "key",
    "foreign",
    "references",
    "SELECT",
    "FROM",
    "WHERE",
    "INSERT",
    "INTO",
    "VALUES",
    "UPDATE",
    "SET",
    "DELETE",
    "CREATE",
    "TABLE",
    "DROP",
    "ALTER",
    "JOIN",
    "LEFT",
    "RIGHT",
    "INNER",
    "OUTER",
    "ON",
    "GROUP",
    "BY",
    "ORDER",
    "HAVING",
    "LIMIT",
    "AS",
    "AND",
    "OR",
    "NOT",
    "NULL",
    "IN",
    "IS",
    "LIKE",
    "DISTINCT",
    "UNION",
];
const DATA_KW: &[&str] = &[
    "true", "false", "null", "yes", "no", "True", "False", "None",
];

fn spec(lang: &str) -> Spec {
    let s = |kw, line, block, single, backtick, keys| Spec {
        kw,
        line,
        block,
        single,
        backtick,
        keys,
    };
    match lang.trim().to_lowercase().as_str() {
        "rust" | "rs" => s(RUST_KW, &["//"], true, false, false, false),
        "python" | "py" | "python3" => s(PY_KW, &["#"], false, true, false, false),
        "js" | "javascript" | "jsx" | "ts" | "typescript" | "tsx" | "mjs" => {
            s(JS_KW, &["//"], true, true, true, false)
        }
        "go" | "golang" => s(GO_KW, &["//"], true, true, true, false),
        "c" | "h" | "cpp" | "c++" | "cc" | "hpp" | "java" | "kotlin" | "swift" | "cs"
        | "csharp" | "php" | "scala" | "dart" | "zig" => s(C_KW, &["//"], true, true, false, false),
        "sh" | "bash" | "zsh" | "shell" | "console" | "fish" => {
            s(SH_KW, &["#"], false, true, true, false)
        }
        "sql" => s(SQL_KW, &["--"], true, true, false, false),
        "json" | "jsonc" | "json5" => s(DATA_KW, &["//"], false, false, false, true),
        "yaml" | "yml" => s(DATA_KW, &["#"], false, true, false, true),
        "toml" | "ini" | "conf" => s(DATA_KW, &["#", ";"], false, true, false, true),
        "ruby" | "rb" | "makefile" | "make" | "dockerfile" => {
            s(&[], &["#"], false, true, false, false)
        }
        "lua" => s(&[], &["--"], true, true, false, false),
        _ => s(&[], &[], false, true, false, false),
    }
}

/// Highlighting state carried from one line of a block to the next.
#[derive(Default)]
pub struct Hl {
    lang: String,
    in_block: bool,
}

impl Hl {
    pub fn new(lang: &str) -> Self {
        Self {
            lang: lang.to_string(),
            in_block: false,
        }
    }

    /// One line of code as styled characters on the code background.
    pub fn line(&mut self, text: &str) -> Vec<Sc> {
        let sp = spec(&self.lang);
        let base = Style::new().fg(TEXT).bg(CODE_BG);
        let st = |c: Color| Style::new().fg(c).bg(CODE_BG);
        let ch: Vec<char> = text.chars().collect();
        let mut out: Vec<Sc> = Vec::with_capacity(ch.len());
        let at = |i: usize, pat: &str| {
            pat.chars()
                .enumerate()
                .all(|(k, c)| ch.get(i + k) == Some(&c))
        };
        let is_c = matches!(
            self.lang.to_lowercase().as_str(),
            "c" | "h" | "cpp" | "c++" | "cc" | "hpp"
        );
        let mut i = 0;
        let mut first = true;
        while i < ch.len() {
            let c = ch[i];
            if self.in_block {
                let start = i;
                while i < ch.len() && !at(i, "*/") {
                    i += 1;
                }
                if i < ch.len() {
                    i += 2;
                    self.in_block = false;
                }
                out.extend(
                    ch[start..i]
                        .iter()
                        .map(|&c| (c, st(COMMENT).add_modifier(Modifier::ITALIC))),
                );
                continue;
            }
            if sp.line.iter().any(|m| at(i, m)) {
                out.extend(
                    ch[i..]
                        .iter()
                        .map(|&c| (c, st(COMMENT).add_modifier(Modifier::ITALIC))),
                );
                break;
            }
            if sp.block && at(i, "/*") {
                self.in_block = true;
                out.push(('/', st(COMMENT)));
                out.push(('*', st(COMMENT)));
                i += 2;
                continue;
            }
            if c.is_whitespace() {
                out.push((c, base));
                i += 1;
                continue;
            }
            let quote = c == '"' || (c == '\'' && sp.single) || (c == '`' && sp.backtick);
            if c == '\'' && !sp.single {
                // Rust: 'x' and '\n' are chars; 'a is a lifetime
                let close = if ch.get(i + 1) == Some(&'\\') {
                    i + 3
                } else {
                    i + 2
                };
                if ch.get(close) == Some(&'\'') {
                    out.extend(ch[i..=close].iter().map(|&c| (c, st(GOOD))));
                    i = close + 1;
                } else {
                    out.push((c, st(TEAL)));
                    i += 1;
                }
                continue;
            }
            if quote {
                let start = i;
                i += 1;
                while i < ch.len() && ch[i] != c {
                    if ch[i] == '\\' {
                        i += 1;
                    }
                    i += 1;
                }
                i = (i + 1).min(ch.len());
                let mut j = i;
                while ch.get(j).is_some_and(|c| *c == ' ') {
                    j += 1;
                }
                let key = sp.keys && ch.get(j) == Some(&':');
                let color = if key { BLUE } else { GOOD };
                out.extend(ch[start..i].iter().map(|&c| (c, st(color))));
                continue;
            }
            if c.is_ascii_digit() && (i == 0 || !(ch[i - 1].is_alphanumeric() || ch[i - 1] == '_'))
            {
                let start = i;
                while i < ch.len() && (ch[i].is_alphanumeric() || ch[i] == '.' || ch[i] == '_') {
                    i += 1;
                }
                out.extend(ch[start..i].iter().map(|&c| (c, st(GOLD))));
                continue;
            }
            if c == '$'
                && matches!(ch.get(i + 1), Some(n) if n.is_alphanumeric() || *n == '_' || *n == '{')
            {
                let start = i;
                i += 1;
                while i < ch.len()
                    && (ch[i].is_alphanumeric() || ch[i] == '_' || ch[i] == '{' || ch[i] == '}')
                {
                    i += 1;
                }
                out.extend(ch[start..i].iter().map(|&c| (c, st(TEAL))));
                continue;
            }
            if c.is_alphabetic() || c == '_' || (c == '#' && is_c && first) {
                let start = i;
                i += 1;
                while i < ch.len() && (ch[i].is_alphanumeric() || ch[i] == '_') {
                    i += 1;
                }
                let word: String = ch[start..i].iter().collect();
                let mut j = i;
                while ch.get(j) == Some(&' ') {
                    j += 1;
                }
                let color = if c == '#' || sp.kw.contains(&word.as_str()) {
                    if matches!(
                        word.as_str(),
                        "true"
                            | "false"
                            | "null"
                            | "None"
                            | "True"
                            | "False"
                            | "nil"
                            | "NULL"
                            | "nullptr"
                            | "undefined"
                            | "Some"
                            | "Ok"
                            | "Err"
                    ) {
                        GOLD
                    } else {
                        EMBER
                    }
                } else if (sp.keys
                    && matches!(ch.get(j), Some(':') | Some('='))
                    && first_token(&out))
                    || ch.get(i) == Some(&'(')
                    || (ch.get(i) == Some(&'!') && self.lang.starts_with('r'))
                {
                    BLUE
                } else if word.chars().next().is_some_and(char::is_uppercase)
                    && word.chars().any(char::is_lowercase)
                {
                    TEAL
                } else {
                    TEXT
                };
                out.extend(ch[start..i].iter().map(|&c| (c, st(color))));
                first = false;
                continue;
            }
            first = false;
            let color = if "{}[]()<>=+-*/%&|^!~?:;,.@".contains(c) {
                DIM
            } else {
                TEXT
            };
            out.push((c, st(color)));
            i += 1;
        }
        out
    }
}

/// True when nothing but whitespace or a dash (YAML list) came before.
fn first_token(out: &[Sc]) -> bool {
    out.iter().all(|(c, _)| c.is_whitespace() || *c == '-')
}

#[cfg(test)]
mod tests {
    use super::*;

    fn colors(hl: &mut Hl, line: &str) -> Vec<(String, Color)> {
        let mut runs: Vec<(String, Color)> = Vec::new();
        for (c, s) in hl.line(line) {
            let color = s.fg.unwrap_or(Color::Reset);
            match runs.last_mut() {
                Some((t, col)) if *col == color => t.push(c),
                _ => runs.push((c.to_string(), color)),
            }
        }
        runs
    }

    fn has(runs: &[(String, Color)], text: &str, color: Color) -> bool {
        runs.iter().any(|(t, c)| t.trim() == text && *c == color)
    }

    #[test]
    fn rust_line() {
        let r = colors(&mut Hl::new("rust"), "let x: Vec<u8> = foo(42); // note");
        assert!(has(&r, "let", EMBER), "{r:?}");
        assert!(has(&r, "Vec", TEAL));
        assert!(has(&r, "foo", BLUE));
        assert!(has(&r, "42", GOLD));
        assert!(has(&r, "// note", COMMENT));
    }

    #[test]
    fn strings_lifetimes_and_block_comments() {
        let mut hl = Hl::new("rust");
        let r = colors(&mut hl, "fn f<'a>(s: &'a str) -> char { 'x' } /* open");
        assert!(has(&r, "'x'", GOOD), "{r:?}");
        assert!(has(&r, "'", TEAL));
        let next = colors(&mut hl, "still comment */ let y = \"s\";");
        assert!(has(&next, "still comment */", COMMENT), "{next:?}");
        assert!(has(&next, "let", EMBER) && has(&next, "\"s\"", GOOD));
    }

    #[test]
    fn json_keys_and_shell_vars() {
        let r = colors(
            &mut Hl::new("json"),
            "  \"name\": \"ferrum\", \"n\": 3, \"ok\": true",
        );
        assert!(
            has(&r, "\"name\"", BLUE) && has(&r, "\"ferrum\"", GOOD),
            "{r:?}"
        );
        assert!(has(&r, "3", GOLD) && has(&r, "true", GOLD));
        let s = colors(&mut Hl::new("bash"), "echo $HOME # hi");
        assert!(has(&s, "$HOME", TEAL) && has(&s, "# hi", COMMENT), "{s:?}");
    }

    #[test]
    fn unknown_language_only_colours_strings_and_numbers() {
        let r = colors(&mut Hl::new(""), "if x == \"a\" { 5 }");
        assert!(has(&r, "\"a\"", GOOD) && has(&r, "5", GOLD));
        assert!(!r.iter().any(|(_, c)| *c == EMBER));
    }
}
