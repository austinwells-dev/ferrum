//! llama.cpp's GBNF grammar text compiled to a `Grammar`
//! (`Grammar::from_gbnf`): rules `name ::= ...` of quoted literals, character
//! classes, `.`, rule names, groups, alternatives and the repetitions `*`,
//! `+`, `?` and `{m,n}`. The rule `root` is the output.
//!
//! Not supported, and refused: token references (`<[id]>`, `<name>`), and
//! left recursion, including a repetition of something that can be empty
//! (the matcher expands rules eagerly and would not end).
use super::grammar::{Grammar, Seq, Sym};
use crate::{Error, Result};
use std::collections::BTreeMap;

/// Longest grammar text accepted.
const MAX_TEXT: usize = 64 << 10;
/// Largest count in `{m,n}`; each copy is expanded.
const MAX_REPEAT: usize = 256;
/// Deepest nesting of groups.
const MAX_DEPTH: usize = 64;

impl Grammar {
    /// Output that matches the GBNF grammar `text` from its `root` rule.
    pub fn from_gbnf(text: &str) -> Result<Self> {
        if text.len() > MAX_TEXT {
            return Err(fail("the grammar is longer than 64 KiB"));
        }
        let mut p = Parser {
            chars: text.chars().collect(),
            at: 0,
            rules: vec![None],
            names: BTreeMap::from([("root".to_owned(), 0)]),
        };
        p.space(true);
        while p.at < p.chars.len() {
            p.rule()?;
        }
        let mut rules = Vec::with_capacity(p.rules.len());
        for (id, rule) in p.rules.into_iter().enumerate() {
            let name = || {
                p.names
                    .iter()
                    .find(|(_, v)| **v == id)
                    .map(|(k, _)| k.as_str())
            };
            rules.push(rule.ok_or_else(|| {
                fail(&format!(
                    "rule {} is used but not defined",
                    name().unwrap_or("?")
                ))
            })?);
        }
        left_recursion(&rules)?;
        Ok(Grammar {
            rules,
            ignored: Vec::new(),
        })
    }
}

fn fail(message: &str) -> Error {
    Error::Parameter(format!("grammar: {message}"))
}

struct Parser {
    chars: Vec<char>,
    at: usize,
    /// Rules by id; None until defined (named ones may be used first).
    rules: Vec<Option<Vec<Seq>>>,
    names: BTreeMap<String, usize>,
}

impl Parser {
    fn peek(&self) -> Option<char> {
        self.chars.get(self.at).copied()
    }

    fn error(&self, message: &str) -> Error {
        let line = 1 + self.chars[..self.at.min(self.chars.len())]
            .iter()
            .filter(|&&c| c == '\n')
            .count();
        fail(&format!("line {line}: {message}"))
    }

    /// Skip spaces and `#` comments, and newlines too when `newlines`.
    fn space(&mut self, newlines: bool) {
        while let Some(c) = self.peek() {
            match c {
                '#' => {
                    while self.peek().is_some_and(|c| c != '\n') {
                        self.at += 1;
                    }
                }
                ' ' | '\t' => self.at += 1,
                '\r' | '\n' if newlines => self.at += 1,
                _ => break,
            }
        }
    }

    fn name(&mut self) -> Result<String> {
        let start = self.at;
        while self
            .peek()
            .is_some_and(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
        {
            self.at += 1;
        }
        if self.at == start {
            return Err(self.error("expected a rule name"));
        }
        Ok(self.chars[start..self.at].iter().collect())
    }

    fn id(&mut self, name: &str) -> usize {
        if let Some(&id) = self.names.get(name) {
            return id;
        }
        self.rules.push(None);
        self.names.insert(name.to_owned(), self.rules.len() - 1);
        self.rules.len() - 1
    }

    fn add(&mut self, alternatives: Vec<Seq>) -> usize {
        self.rules.push(Some(alternatives));
        self.rules.len() - 1
    }

    /// `name ::= alternatives`, up to the end of its line.
    fn rule(&mut self) -> Result<()> {
        let name = self.name()?;
        self.space(false);
        for expected in "::=".chars() {
            if self.peek() != Some(expected) {
                return Err(self.error(&format!("expected ::= after {name}")));
            }
            self.at += 1;
        }
        self.space(true);
        let alternatives = self.alternatives(0)?;
        match self.peek() {
            None | Some('\n' | '\r') => {}
            Some(c) => return Err(self.error(&format!("unexpected {c:?}"))),
        }
        self.space(true);
        let id = self.id(&name);
        if self.rules[id].is_some() {
            return Err(self.error(&format!("rule {name} is defined twice")));
        }
        self.rules[id] = Some(alternatives);
        Ok(())
    }

    /// Sequences separated by `|`. Inside a group (`depth` > 0) they may
    /// span lines; a rule's own end at its line's end, except after a `|`.
    fn alternatives(&mut self, depth: usize) -> Result<Vec<Seq>> {
        let mut out = vec![self.sequence(depth)?];
        while self.peek() == Some('|') {
            self.at += 1;
            self.space(true);
            out.push(self.sequence(depth)?);
        }
        Ok(out)
    }

    fn sequence(&mut self, depth: usize) -> Result<Seq> {
        let mut seq = Seq::new();
        // Where the element a repetition applies to starts.
        let mut last = 0;
        while let Some(c) = self.peek() {
            match c {
                '"' => {
                    self.at += 1;
                    last = seq.len();
                    loop {
                        match self.peek() {
                            None | Some('\n') => return Err(self.error("unterminated string")),
                            Some('"') => break,
                            Some(_) => {
                                let c = self.character()?;
                                seq.push(Sym::Chars(vec![(c, c)], false));
                            }
                        }
                    }
                    self.at += 1;
                }
                '[' => {
                    self.at += 1;
                    last = seq.len();
                    seq.push(self.class()?);
                }
                '.' => {
                    self.at += 1;
                    last = seq.len();
                    seq.push(Sym::Chars(Vec::new(), true));
                }
                '(' => {
                    if depth >= MAX_DEPTH {
                        return Err(self.error("groups are nested too deeply"));
                    }
                    self.at += 1;
                    self.space(true);
                    let group = self.alternatives(depth + 1)?;
                    if self.peek() != Some(')') {
                        return Err(self.error("expected )"));
                    }
                    self.at += 1;
                    last = seq.len();
                    let id = self.add(group);
                    seq.push(Sym::Rule(id));
                }
                '*' | '+' | '?' | '{' => {
                    if last == seq.len() {
                        return Err(self.error(&format!("{c:?} has nothing to repeat")));
                    }
                    self.at += 1;
                    let (min, max) = match c {
                        '*' => (0, None),
                        '+' => (1, None),
                        '?' => (0, Some(1)),
                        _ => self.bounds()?,
                    };
                    let item = seq.split_off(last);
                    self.repeat(&mut seq, item, min, max);
                    // A second operator would repeat nothing.
                    last = seq.len();
                }
                '<' => {
                    return Err(self.error("token references (<...>) are not supported"));
                }
                c if c.is_ascii_alphanumeric() || c == '-' || c == '_' => {
                    let name = self.name()?;
                    last = seq.len();
                    let id = self.id(&name);
                    seq.push(Sym::Rule(id));
                }
                _ => break,
            }
            self.space(depth > 0);
        }
        Ok(seq)
    }

    /// `min` copies of `item`, then up to `max - min` more (any number
    /// without a `max`), appended to `seq`.
    fn repeat(&mut self, seq: &mut Seq, item: Seq, min: usize, max: Option<usize>) {
        let item = match item.as_slice() {
            [single] => single.clone(),
            _ => Sym::Rule(self.add(vec![item])),
        };
        seq.extend(std::iter::repeat_n(item.clone(), min));
        match max {
            None => {
                let star = self.rules.len();
                self.add(vec![vec![], vec![item, Sym::Rule(star)]]);
                seq.push(Sym::Rule(star));
            }
            Some(max) => {
                // item? nested: (item (item (...)?)?)?
                let mut rest = None;
                for _ in min..max {
                    let mut more = vec![item.clone()];
                    more.extend(rest.map(Sym::Rule));
                    rest = Some(self.add(vec![vec![], more]));
                }
                seq.extend(rest.map(Sym::Rule));
            }
        }
    }

    /// After `{`: `m}`, `m,}` or `m,n}`.
    fn bounds(&mut self) -> Result<(usize, Option<usize>)> {
        let number = |p: &mut Self| -> Result<Option<usize>> {
            p.space(false);
            let start = p.at;
            while p.peek().is_some_and(|c| c.is_ascii_digit()) {
                p.at += 1;
            }
            if p.at == start {
                return Ok(None);
            }
            let text: String = p.chars[start..p.at].iter().collect();
            match text.parse::<usize>() {
                Ok(n) if n <= MAX_REPEAT => Ok(Some(n)),
                _ => Err(p.error("a repetition count is at most 256")),
            }
        };
        let min = number(self)?.ok_or_else(|| self.error("expected a count after {"))?;
        self.space(false);
        let max = match self.peek() {
            Some(',') => {
                self.at += 1;
                number(self)?
            }
            _ => Some(min),
        };
        self.space(false);
        if self.peek() != Some('}') {
            return Err(self.error("expected }"));
        }
        self.at += 1;
        if max.is_some_and(|max| max < min) {
            return Err(self.error("a repetition's maximum is below its minimum"));
        }
        Ok((min, max))
    }

    /// One character of a literal or class, with its escape resolved.
    fn character(&mut self) -> Result<char> {
        let c = self.peek().ok_or_else(|| self.error("unexpected end"))?;
        self.at += 1;
        if c != '\\' {
            return Ok(c);
        }
        let escape = self.peek().ok_or_else(|| self.error("unexpected end"))?;
        self.at += 1;
        let hex = |p: &mut Self, digits: usize| -> Result<char> {
            let text: String = p.chars.iter().skip(p.at).take(digits).collect();
            p.at += digits;
            (text.chars().count() == digits)
                .then(|| u32::from_str_radix(&text, 16).ok())
                .flatten()
                .and_then(char::from_u32)
                .ok_or_else(|| p.error("bad hexadecimal escape"))
        };
        match escape {
            'n' => Ok('\n'),
            'r' => Ok('\r'),
            't' => Ok('\t'),
            'x' => hex(self, 2),
            'u' => hex(self, 4),
            'U' => hex(self, 8),
            '\\' | '"' | '[' | ']' | '-' | '^' => Ok(escape),
            other => Err(self.error(&format!("unknown escape \\{other}"))),
        }
    }

    /// After `[`: the class up to `]`.
    fn class(&mut self) -> Result<Sym> {
        let negated = self.peek() == Some('^');
        if negated {
            self.at += 1;
        }
        let mut ranges = Vec::new();
        loop {
            match self.peek() {
                None | Some('\n') => return Err(self.error("unterminated character class")),
                Some(']') => break,
                Some(_) => {
                    let low = self.character()?;
                    let ranged = self.peek() == Some('-')
                        && self.chars.get(self.at + 1).is_some_and(|&c| c != ']');
                    let high = if ranged {
                        self.at += 1;
                        self.character()?
                    } else {
                        low
                    };
                    if high < low {
                        return Err(self.error("a character range is backwards"));
                    }
                    ranges.push((low, high));
                }
            }
        }
        self.at += 1;
        Ok(Sym::Chars(ranges, negated))
    }
}

/// Refuse a rule that can begin with itself, directly, through other rules,
/// or behind symbols that can match nothing: `Matcher` expands the first
/// symbol of every alternative before reading a character.
fn left_recursion(rules: &[Vec<Seq>]) -> Result<()> {
    let mut nullable = vec![false; rules.len()];
    loop {
        let mut changed = false;
        for (id, rule) in rules.iter().enumerate() {
            let empty = rule.iter().any(|seq| {
                seq.iter()
                    .all(|sym| matches!(sym, Sym::Rule(r) if nullable[*r]))
            });
            if empty && !nullable[id] {
                nullable[id] = true;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    // The rules each rule can start with.
    let first: Vec<Vec<usize>> = rules
        .iter()
        .map(|rule| {
            let mut out = Vec::new();
            for seq in rule {
                for sym in seq {
                    let Sym::Rule(r) = sym else { break };
                    out.push(*r);
                    if !nullable[*r] {
                        break;
                    }
                }
            }
            out
        })
        .collect();
    // 0 unvisited, 1 on the path, 2 done.
    let mut state = vec![0u8; rules.len()];
    for start in 0..rules.len() {
        if state[start] != 0 {
            continue;
        }
        let mut path = vec![(start, 0)];
        state[start] = 1;
        while let Some((rule, next)) = path.last_mut() {
            let Some(&target) = first[*rule].get(*next) else {
                state[*rule] = 2;
                path.pop();
                continue;
            };
            *next += 1;
            match state[target] {
                0 => {
                    state[target] = 1;
                    path.push((target, 0));
                }
                1 => {
                    return Err(fail(
                        "a rule can begin with itself (left recursion, or a repetition of \
                         something that can be empty), which is not supported",
                    ));
                }
                _ => {}
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::grammar::Matcher;
    use super::*;

    fn matches(grammar: &Grammar, text: &str) -> bool {
        let mut m = Matcher::new(grammar);
        m.accept_str(grammar, text) && m.complete()
    }

    #[test]
    fn literals_classes_groups_and_repetitions() {
        let g = Grammar::from_gbnf(
            r##"
# a list of moves
root   ::= move (", " move)* "\n"?
move   ::= piece? [a-h] [1-8] ("+" | "#")? | "O-O" ("-O"){0,1}
piece  ::= [KQRBN]
"##,
        )
        .unwrap();
        for ok in ["e4", "Nf3, e5\n", "O-O, O-O-O", "Qh7#, a1+"] {
            assert!(matches(&g, ok), "{ok}");
        }
        for bad in ["", "e9", "Nf3,e5", "O-O-O-O", "Xe4", "e4\n\n"] {
            assert!(!matches(&g, bad), "{bad}");
        }
    }

    #[test]
    fn escapes_negation_any_and_counts() {
        let g = Grammar::from_gbnf(
            "root ::= \"\\\"\" [^\"\\\\\\x00-\\x1f]{2,4} \"\\\"\" . [\\u00e9\\]]+ \"\\t\"{2}\n",
        )
        .unwrap();
        assert!(matches(&g, "\"ab\"!é\t\t"));
        assert!(matches(&g, "\"abcd\"\né]]\t\t"));
        for bad in [
            "\"a\"!é\t\t",
            "\"abcde\"!é\t\t",
            "\"ab\"é\t\t",
            "\"ab\"!é\t",
        ] {
            assert!(!matches(&g, bad), "{bad:?}");
        }
    }

    #[test]
    fn rules_may_be_used_before_they_are_defined_and_recurse() {
        let g = Grammar::from_gbnf(
            "root ::= value\nvalue ::= [0-9]+ | \"(\" value (\",\" value)* \")\"\n",
        )
        .unwrap();
        assert!(matches(&g, "((1,22),(3))"));
        assert!(!matches(&g, "((1,22),(3)"));
    }

    #[test]
    fn alternatives_continue_after_a_bar_and_inside_groups() {
        let g = Grammar::from_gbnf("root ::= \"a\" |\n  \"b\" | (\n \"c\"\n | \"d\" )\n").unwrap();
        for ok in ["a", "b", "c", "d"] {
            assert!(matches(&g, ok), "{ok}");
        }
        assert!(!matches(&g, "ab"));
    }

    #[test]
    fn refuses_what_it_cannot_run() {
        let refused = |text: &str| Grammar::from_gbnf(text).unwrap_err().to_string();
        assert!(refused("start ::= \"a\"\n").contains("root is used but not defined"));
        assert!(refused("root ::= other\n").contains("other is used but not defined"));
        assert!(refused("root ::= \"a\"\nroot ::= \"b\"\n").contains("defined twice"));
        assert!(refused("root ::= root \"a\" | \"a\"\n").contains("begin with itself"));
        assert!(
            refused("root ::= a\na ::= b \"x\"\nb ::= \"y\"? a\n").contains("begin with itself")
        );
        assert!(refused("root ::= (\"a\"?)*\n").contains("begin with itself"));
        assert!(refused("root ::= <[1000]>\n").contains("token references"));
        assert!(refused("root ::= \"a\"{2,1}\n").contains("below its minimum"));
        assert!(refused("root ::= \"a\"{999}\n").contains("at most 256"));
        assert!(refused("root ::= * \"a\"\n").contains("nothing to repeat"));
        assert!(refused("root ::= \"abc\n").contains("unterminated string"));
        assert!(refused("root ::= \"a\" )\n").contains("line 1: unexpected"));
    }
}
