//! Constrained output: a JSON Schema (or "any JSON object") compiled to a
//! character grammar, and a matcher that says which text may come next.
//!
//! The grammar is a set of rules, each a choice of sequences of character
//! classes and rule references (what llama.cpp's GBNF describes). `Matcher`
//! keeps every position the text so far could be at, as stacks of return
//! addresses, so alternatives that share a prefix (`anyOf`) need no
//! backtracking. `Session` asks it which tokens may follow
//! (`GenerationHooks::allows`).
//!
//! What a schema may use is listed in docs/server.md ("Structured output").
//! Objects write their properties in the schema's order, as llama.cpp's
//! converter does: the model does not choose an order.
use crate::{Error, Result};
use serde_json::Value as Json;
use std::collections::BTreeMap;

/// Most array bounds (`minItems`/`maxItems`) expanded into rules.
const MAX_ARRAY_BOUND: usize = 64;

#[derive(Debug, Clone, PartialEq)]
enum Sym {
    /// One character in (or, negated, outside) these inclusive ranges.
    Chars(Vec<(char, char)>, bool),
    Rule(usize),
}

impl Sym {
    fn matches(&self, c: char) -> bool {
        match self {
            Sym::Chars(ranges, negated) => {
                ranges.iter().any(|&(lo, hi)| lo <= c && c <= hi) != *negated
            }
            Sym::Rule(_) => false,
        }
    }
}

type Seq = Vec<Sym>;

/// A compiled grammar; rule 0 is the root.
#[derive(Debug)]
pub struct Grammar {
    rules: Vec<Vec<Seq>>,
    /// Schema keywords that were accepted but are not enforced.
    pub ignored: Vec<String>,
}

fn one(c: char) -> Sym {
    Sym::Chars(vec![(c, c)], false)
}

fn lit(text: &str) -> Seq {
    text.chars().map(one).collect()
}

struct Builder<'a> {
    rules: Vec<Vec<Seq>>,
    ignored: Vec<String>,
    root: &'a Json,
    /// `$ref` targets already given a rule (recursive schemas refer back).
    refs: BTreeMap<String, usize>,
    ws: usize,
    string: usize,
    number: usize,
    integer: usize,
    any: usize,
}

impl<'a> Builder<'a> {
    fn new(root: &'a Json) -> Self {
        let mut b = Self {
            rules: vec![Vec::new()],
            ignored: Vec::new(),
            root,
            refs: BTreeMap::new(),
            ws: 0,
            string: 0,
            number: 0,
            integer: 0,
            any: 0,
        };
        // Optional whitespace, bounded so a model cannot pad forever: one
        // space, or a newline and up to 20 spaces or tabs of indentation.
        let mut indent = b.add(vec![vec![]]);
        for _ in 0..20 {
            indent = b.add(vec![
                vec![],
                vec![
                    Sym::Chars(vec![(' ', ' '), ('\t', '\t')], false),
                    Sym::Rule(indent),
                ],
            ]);
        }
        b.ws = b.add(vec![
            vec![],
            vec![one(' ')],
            vec![one('\n'), Sym::Rule(indent)],
        ]);
        let hex = Sym::Chars(vec![('0', '9'), ('a', 'f'), ('A', 'F')], false);
        let escape = b.add(vec![
            vec![Sym::Chars(
                "\"\\/bfnrt".chars().map(|c| (c, c)).collect(),
                false,
            )],
            vec![one('u'), hex.clone(), hex.clone(), hex.clone(), hex],
        ]);
        let chars = b.rules.len();
        b.add(vec![
            vec![],
            vec![
                Sym::Chars(vec![('"', '"'), ('\\', '\\'), ('\0', '\x1f')], true),
                Sym::Rule(chars),
            ],
            vec![one('\\'), Sym::Rule(escape), Sym::Rule(chars)],
        ]);
        b.string = b.add(vec![vec![one('"'), Sym::Rule(chars), one('"')]]);
        let digit = Sym::Chars(vec![('0', '9')], false);
        let digits = b.rules.len();
        b.add(vec![vec![], vec![digit.clone(), Sym::Rule(digits)]]);
        let int = b.add(vec![
            vec![one('0')],
            vec![Sym::Chars(vec![('1', '9')], false), Sym::Rule(digits)],
        ]);
        let sign = b.add(vec![vec![], vec![one('-')]]);
        b.integer = b.add(vec![vec![Sym::Rule(sign), Sym::Rule(int)]]);
        let frac = b.add(vec![
            vec![],
            vec![one('.'), digit.clone(), Sym::Rule(digits)],
        ]);
        let exp_sign = b.add(vec![
            vec![],
            vec![Sym::Chars(vec![('+', '+'), ('-', '-')], false)],
        ]);
        let exp = b.add(vec![
            vec![],
            vec![
                Sym::Chars(vec![('e', 'e'), ('E', 'E')], false),
                Sym::Rule(exp_sign),
                digit,
                Sym::Rule(digits),
            ],
        ]);
        b.number = b.add(vec![vec![
            Sym::Rule(b.integer),
            Sym::Rule(frac),
            Sym::Rule(exp),
        ]]);
        // Any JSON value; its object and array refer back to it.
        b.any = b.rules.len();
        b.rules.push(Vec::new());
        let object = b.map_of(b.any);
        let array = b.array_of(b.any, 0, None);
        b.rules[b.any] = vec![
            vec![Sym::Rule(object)],
            vec![Sym::Rule(array)],
            vec![Sym::Rule(b.string)],
            vec![Sym::Rule(b.number)],
            lit("true"),
            lit("false"),
            lit("null"),
        ];
        b
    }

    fn add(&mut self, alternatives: Vec<Seq>) -> usize {
        self.rules.push(alternatives);
        self.rules.len() - 1
    }

    /// `{ "key": value, ... }` with any keys.
    fn map_of(&mut self, value: usize) -> usize {
        let ws = Sym::Rule(self.ws);
        let pair = self.add(vec![vec![
            Sym::Rule(self.string),
            ws.clone(),
            one(':'),
            ws.clone(),
            Sym::Rule(value),
            ws.clone(),
        ]]);
        let more = self.rules.len();
        self.add(vec![
            vec![],
            vec![one(','), ws.clone(), Sym::Rule(pair), Sym::Rule(more)],
        ]);
        let members = self.add(vec![vec![], vec![Sym::Rule(pair), Sym::Rule(more)]]);
        self.add(vec![vec![one('{'), ws, Sym::Rule(members), one('}')]])
    }

    /// `[ value, ... ]` with `min` to `max` items.
    fn array_of(&mut self, value: usize, min: usize, max: Option<usize>) -> usize {
        let ws = Sym::Rule(self.ws);
        let item = vec![Sym::Rule(value), ws.clone()];
        let next = |mut tail: Seq| -> Seq {
            let mut seq = vec![one(','), ws.clone()];
            seq.extend(item.clone());
            seq.append(&mut tail);
            seq
        };
        // `tail` matches what may follow the items so far.
        let mut tail = match max {
            None => {
                let more = self.rules.len();
                self.add(vec![vec![], next(vec![Sym::Rule(more)])]);
                more
            }
            Some(max) => {
                let mut tail = self.add(vec![vec![]]);
                for _ in min.max(1)..max {
                    tail = self.add(vec![vec![], next(vec![Sym::Rule(tail)])]);
                }
                tail
            }
        };
        for _ in 1..min {
            tail = self.add(vec![next(vec![Sym::Rule(tail)])]);
        }
        let mut first = item;
        first.push(Sym::Rule(tail));
        let items = if min == 0 {
            self.add(vec![vec![], first])
        } else {
            self.add(vec![first])
        };
        self.add(vec![vec![one('['), ws, Sym::Rule(items), one(']')]])
    }

    fn fail<T>(&self, message: String) -> Result<T> {
        Err(Error::Parameter(format!("response schema: {message}")))
    }

    fn resolve(&self, reference: &str) -> Result<&'a Json> {
        let Some(path) = reference.strip_prefix("#/") else {
            if reference == "#" {
                return Ok(self.root);
            }
            return self.fail(format!("$ref {reference:?} is not a local reference"));
        };
        let mut at = self.root;
        for part in path.split('/') {
            let key = part.replace("~1", "/").replace("~0", "~");
            at = match at.get(&key) {
                Some(next) => next,
                None => return self.fail(format!("$ref {reference:?} does not exist")),
            };
        }
        Ok(at)
    }

    /// The rule that matches a value of `schema`.
    fn value(&mut self, schema: &'a Json) -> Result<usize> {
        let schema = match schema {
            Json::Bool(true) => return Ok(self.any),
            Json::Object(o) => o,
            _ => return self.fail("a schema must be an object or true".into()),
        };
        for key in schema.keys() {
            match key.as_str() {
                "type"
                | "enum"
                | "const"
                | "properties"
                | "required"
                | "additionalProperties"
                | "items"
                | "minItems"
                | "maxItems"
                | "anyOf"
                | "oneOf"
                | "$ref"
                | "$defs"
                | "definitions"
                | "title"
                | "description"
                | "default"
                | "examples"
                | "$schema"
                | "$id"
                | "$comment"
                | "strict"
                | "nullable" => {}
                // Narrow what is valid, but output that ignores them is
                // still the right shape; the caller is told.
                "format" | "pattern" | "minLength" | "maxLength" | "minimum" | "maximum"
                | "exclusiveMinimum" | "exclusiveMaximum" | "multipleOf" | "uniqueItems"
                | "minProperties" | "maxProperties" => {
                    if !self.ignored.contains(key) {
                        self.ignored.push(key.clone());
                    }
                }
                other => return self.fail(format!("{other:?} is not supported")),
            }
        }
        if let Some(reference) = schema.get("$ref") {
            let Some(reference) = reference.as_str() else {
                return self.fail("$ref must be a string".into());
            };
            if let Some(&rule) = self.refs.get(reference) {
                return Ok(rule);
            }
            // Reserve the rule first: the target may refer to itself.
            let rule = self.add(Vec::new());
            self.refs.insert(reference.to_owned(), rule);
            let target = self.resolve(reference)?;
            let inner = self.value(target)?;
            self.rules[rule] = vec![vec![Sym::Rule(inner)]];
            return Ok(rule);
        }
        if let Some(value) = schema.get("const") {
            return Ok(self.add(vec![lit(&value.to_string())]));
        }
        if let Some(values) = schema.get("enum") {
            let Some(values) = values.as_array().filter(|v| !v.is_empty()) else {
                return self.fail("enum must be a non-empty array".into());
            };
            return Ok(self.add(values.iter().map(|v| lit(&v.to_string())).collect()));
        }
        for key in ["anyOf", "oneOf"] {
            if let Some(options) = schema.get(key) {
                let Some(options) = options.as_array().filter(|v| !v.is_empty()) else {
                    return self.fail(format!("{key} must be a non-empty array"));
                };
                let mut alternatives = Vec::new();
                for option in options {
                    alternatives.push(vec![Sym::Rule(self.value(option)?)]);
                }
                return Ok(self.add(alternatives));
            }
        }
        let mut types: Vec<&str> = match schema.get("type") {
            Some(Json::String(t)) => vec![t.as_str()],
            Some(Json::Array(list)) => list.iter().filter_map(Json::as_str).collect(),
            None if schema.contains_key("properties") => vec!["object"],
            None if schema.contains_key("items") => vec!["array"],
            None => return Ok(self.any),
            Some(_) => return self.fail("type must be a string or a list of strings".into()),
        };
        if schema.get("nullable") == Some(&Json::Bool(true)) && !types.contains(&"null") {
            types.push("null");
        }
        let mut alternatives = Vec::new();
        for ty in types {
            alternatives.push(match ty {
                "string" => vec![Sym::Rule(self.string)],
                "number" => vec![Sym::Rule(self.number)],
                "integer" => vec![Sym::Rule(self.integer)],
                "boolean" => {
                    let rule = self.add(vec![lit("true"), lit("false")]);
                    vec![Sym::Rule(rule)]
                }
                "null" => lit("null"),
                "object" => vec![Sym::Rule(self.object(schema)?)],
                "array" => vec![Sym::Rule(self.array(schema)?)],
                other => return self.fail(format!("unknown type {other:?}")),
            });
        }
        if alternatives.is_empty() {
            return self.fail("type lists no types".into());
        }
        Ok(self.add(alternatives))
    }

    fn array(&mut self, schema: &'a serde_json::Map<String, Json>) -> Result<usize> {
        let bound = |key: &str| -> Result<Option<usize>> {
            match schema.get(key) {
                None => Ok(None),
                Some(v) => match v.as_u64() {
                    Some(n) if n as usize <= MAX_ARRAY_BOUND => Ok(Some(n as usize)),
                    _ => self.fail(format!("{key} must be an integer up to {MAX_ARRAY_BOUND}")),
                },
            }
        };
        let (min, max) = (bound("minItems")?.unwrap_or(0), bound("maxItems")?);
        if max.is_some_and(|max| max < min) {
            return self.fail("maxItems is below minItems".into());
        }
        let items = match schema.get("items") {
            Some(items) => self.value(items)?,
            None => self.any,
        };
        if max == Some(0) {
            let ws = Sym::Rule(self.ws);
            return Ok(self.add(vec![vec![one('['), ws, one(']')]]));
        }
        Ok(self.array_of(items, min, max))
    }

    fn object(&mut self, schema: &'a serde_json::Map<String, Json>) -> Result<usize> {
        let properties: Vec<(&'a String, &'a Json)> = match schema.get("properties") {
            Some(Json::Object(p)) => p.iter().collect(),
            None => Vec::new(),
            Some(_) => return self.fail("properties must be an object".into()),
        };
        let required: Vec<&str> = match schema.get("required") {
            Some(Json::Array(list)) => list.iter().filter_map(Json::as_str).collect(),
            None => Vec::new(),
            Some(_) => return self.fail("required must be an array".into()),
        };
        if let Some(missing) = required
            .iter()
            .find(|r| !properties.iter().any(|(name, _)| name == *r))
        {
            return self.fail(format!(
                "required property {missing:?} is not in properties"
            ));
        }
        if properties.is_empty() {
            // A free-form object: any keys, values of additionalProperties.
            let value = match schema.get("additionalProperties") {
                Some(Json::Bool(false)) => {
                    let ws = Sym::Rule(self.ws);
                    return Ok(self.add(vec![vec![one('{'), ws, one('}')]]));
                }
                Some(extra @ Json::Object(_)) => self.value(extra)?,
                _ => self.any,
            };
            return Ok(self.map_of(value));
        }
        let ws = Sym::Rule(self.ws);
        let mut pairs = Vec::new();
        for (name, property) in properties {
            let value = self.value(property)?;
            let mut pair = lit(&Json::String((*name).clone()).to_string());
            pair.extend([
                ws.clone(),
                one(':'),
                ws.clone(),
                Sym::Rule(value),
                ws.clone(),
            ]);
            pairs.push((pair, required.contains(&name.as_str())));
        }
        // `first` is what may follow with nothing written yet, `rest` after
        // at least one property; an optional property may be skipped.
        let mut first = self.add(vec![vec![]]);
        let mut rest = first;
        for (pair, required) in pairs.into_iter().rev() {
            let mut with = pair.clone();
            with.push(Sym::Rule(rest));
            let mut first_alternatives = vec![with.clone()];
            let mut after = vec![one(','), ws.clone()];
            after.extend(with);
            let mut rest_alternatives = vec![after];
            if !required {
                first_alternatives.push(vec![Sym::Rule(first)]);
                rest_alternatives.push(vec![Sym::Rule(rest)]);
            }
            first = self.add(first_alternatives);
            rest = self.add(rest_alternatives);
        }
        Ok(self.add(vec![vec![one('{'), ws, Sym::Rule(first), one('}')]]))
    }

    fn finish(mut self, value: usize) -> Grammar {
        self.rules[0] = vec![vec![Sym::Rule(self.ws), Sym::Rule(value)]];
        Grammar {
            rules: self.rules,
            ignored: self.ignored,
        }
    }
}

impl Grammar {
    /// Output that is one JSON object (OpenAI's `{"type": "json_object"}`).
    pub fn json_object() -> Self {
        let root = Json::Null;
        let mut b = Builder::new(&root);
        let object = b.map_of(b.any);
        b.finish(object)
    }

    /// Output that is one JSON value matching `schema`.
    pub fn from_schema(schema: &Json) -> Result<Self> {
        let mut b = Builder::new(schema);
        let value = b.value(schema)?;
        Ok(b.finish(value))
    }

    /// From an OpenAI `response_format` value. `Ok(None)`: plain text.
    pub fn from_response_format(format: &Json) -> Result<Option<Self>> {
        let fail = |m: &str| Err(Error::Parameter(format!("response_format: {m}")));
        match format {
            Json::Null => Ok(None),
            Json::Object(o) => match o.get("type").and_then(Json::as_str) {
                Some("text") => Ok(None),
                Some("json_object") => match o.get("schema") {
                    // llama-server's form: a schema beside the type.
                    Some(schema) => Self::from_schema(schema).map(Some),
                    None => Ok(Some(Self::json_object())),
                },
                Some("json_schema") => {
                    match o
                        .get("json_schema")
                        .and_then(|j| j.get("schema"))
                        .or_else(|| o.get("schema"))
                    {
                        Some(schema) => Self::from_schema(schema).map(Some),
                        None => fail("json_schema needs json_schema.schema"),
                    }
                }
                _ => fail("type must be text, json_object or json_schema"),
            },
            _ => fail("must be an object"),
        }
    }
}

/// A position in a sequence: (rule, alternative, index of the next symbol).
type Pos = (u32, u32, u32);

/// Where text matched so far can be in a `Grammar`: every stack ends at a
/// character class the next character may match, or is empty (the text is a
/// complete match).
#[derive(Clone, Debug)]
pub struct Matcher {
    stacks: Vec<Vec<Pos>>,
}

impl Matcher {
    pub fn new(grammar: &Grammar) -> Self {
        let mut stacks = Vec::new();
        for alternative in 0..grammar.rules[0].len() {
            expand(grammar, vec![(0, alternative as u32, 0)], &mut stacks);
        }
        Self { stacks }
    }

    /// The text so far is a complete match (more may still be allowed, as
    /// after the digits of a bare number).
    pub fn complete(&self) -> bool {
        self.stacks.iter().any(Vec::is_empty)
    }

    /// Whether `c` may come next; advances past it when it may.
    pub fn accept(&mut self, grammar: &Grammar, c: char) -> bool {
        let mut next = Vec::new();
        for stack in &self.stacks {
            let Some(&(rule, alternative, index)) = stack.last() else {
                continue;
            };
            let sym = &grammar.rules[rule as usize][alternative as usize][index as usize];
            if sym.matches(c) {
                let mut stack = stack.clone();
                stack.last_mut().expect("non-empty").2 += 1;
                expand(grammar, stack, &mut next);
            }
        }
        if next.is_empty() {
            return false;
        }
        self.stacks = next;
        true
    }

    /// Whether all of `text` may come next; advances past it when it may,
    /// and is unchanged when not.
    pub fn accept_str(&mut self, grammar: &Grammar, text: &str) -> bool {
        let mut trial = self.clone();
        if text.chars().all(|c| trial.accept(grammar, c)) {
            *self = trial;
            return true;
        }
        false
    }

    /// Whether all of `text` may come next.
    pub fn allows(&self, grammar: &Grammar, text: &str) -> bool {
        let mut chars = text.chars();
        let Some(first) = chars.next() else {
            return false;
        };
        // Most tokens fail on their first character: check it before
        // copying the stacks.
        let fits = self.stacks.iter().any(|stack| {
            stack.last().is_some_and(|&(rule, alternative, index)| {
                grammar.rules[rule as usize][alternative as usize][index as usize].matches(first)
            })
        });
        fits && self.clone().accept_str(grammar, text)
    }
}

/// Add to `out` every stack `stack` stands for once its top is a character
/// class (or it is empty): finished sequences return to their caller, and a
/// rule reference becomes one stack per alternative.
fn expand(grammar: &Grammar, mut stack: Vec<Pos>, out: &mut Vec<Vec<Pos>>) {
    while let Some(&(rule, alternative, index)) = stack.last() {
        let seq = &grammar.rules[rule as usize][alternative as usize];
        match seq.get(index as usize) {
            None => {
                stack.pop();
            }
            Some(Sym::Chars(..)) => break,
            Some(&Sym::Rule(target)) => {
                // Return to the symbol after the reference.
                stack.last_mut().expect("non-empty").2 += 1;
                for alternative in 0..grammar.rules[target].len() {
                    let mut branch = stack.clone();
                    branch.push((target as u32, alternative as u32, 0));
                    expand(grammar, branch, out);
                }
                return;
            }
        }
    }
    if !out.contains(&stack) {
        out.push(stack);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// Whether `text` is a complete match of `grammar`.
    fn matches(grammar: &Grammar, text: &str) -> bool {
        let mut m = Matcher::new(grammar);
        m.accept_str(grammar, text) && m.complete()
    }

    #[test]
    fn json_object_accepts_objects_only() {
        let g = Grammar::json_object();
        for ok in [
            "{}",
            "{\"a\": 1}",
            " {\"a\":[1, 2.5e-3, true, null, \"x\\n\\u00e9\"], \"b\": {\"c\": {}}}",
            "{\n  \"k\": \"v\",\n  \"n\": -0.5\n}",
        ] {
            assert!(matches(&g, ok), "{ok}");
        }
        for bad in [
            "[]",
            "1",
            "{\"a\": 01}",
            "{\"a\": }",
            "{a: 1}",
            "{\"a\": 1,}",
            "{\"a\": \"x\ny\"}",
            "{} x",
            "{\"a\": 1}}",
        ] {
            assert!(!matches(&g, bad), "{bad}");
        }
        // An unfinished object is a prefix, not a match.
        let mut m = Matcher::new(&g);
        assert!(m.accept_str(&g, "{\"a\": [1, "));
        assert!(!m.complete());
        assert!(m.allows(&g, "2]}"));
        assert!(!m.allows(&g, "]"));
        assert!(!m.allows(&g, ""));
    }

    #[test]
    fn schema_objects_keep_property_order_and_required() {
        let g = Grammar::from_schema(&json!({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
                "tags": {"type": "array", "items": {"type": "string"}, "maxItems": 2},
                "kind": {"enum": ["cat", "dog", 3]}
            },
            "required": ["name", "kind"]
        }))
        .unwrap();
        for ok in [
            "{\"name\": \"Rex\", \"kind\": \"dog\"}",
            "{\"name\":\"Rex\",\"age\":4,\"kind\":3}",
            "{\"name\": \"\", \"tags\": [], \"kind\": \"cat\"}",
            "{\"name\": \"a\", \"age\": -1, \"tags\": [\"x\", \"y\"], \"kind\": \"cat\"}",
        ] {
            assert!(matches(&g, ok), "{ok}");
        }
        for bad in [
            "{\"kind\": \"dog\"}",
            "{\"name\": \"Rex\"}",
            "{\"kind\": \"dog\", \"name\": \"Rex\"}",
            "{\"name\": \"Rex\", \"age\": 4.5, \"kind\": \"dog\"}",
            "{\"name\": \"Rex\", \"kind\": \"bird\"}",
            "{\"name\": \"Rex\", \"kind\": \"dog\", \"extra\": 1}",
            "{\"name\": \"a\", \"tags\": [\"x\", \"y\", \"z\"], \"kind\": \"cat\"}",
        ] {
            assert!(!matches(&g, bad), "{bad}");
        }
    }

    #[test]
    fn alternatives_arrays_and_references() {
        let g = Grammar::from_schema(&json!({
            "$defs": {
                "node": {
                    "type": "object",
                    "properties": {
                        "value": {"anyOf": [{"type": "integer"}, {"type": "number"}, {"type": "null"}]},
                        "next": {"$ref": "#/$defs/node"}
                    },
                    "required": ["value"]
                }
            },
            "type": "array",
            "items": {"$ref": "#/$defs/node"},
            "minItems": 1,
            "maxItems": 3
        }))
        .unwrap();
        assert!(matches(&g, "[{\"value\": 1}]"));
        assert!(matches(
            &g,
            "[{\"value\": 1.5, \"next\": {\"value\": null, \"next\": {\"value\": 2}}}, {\"value\": 0}]"
        ));
        assert!(!matches(&g, "[]"));
        assert!(!matches(&g, "[{\"value\": \"x\"}]"));
        assert!(!matches(
            &g,
            "[{\"value\": 1}, {\"value\": 1}, {\"value\": 1}, {\"value\": 1}]"
        ));
        let exact = Grammar::from_schema(
            &json!({"type": "array", "items": {"type": "boolean"}, "minItems": 2, "maxItems": 2}),
        )
        .unwrap();
        assert!(matches(&exact, "[true, false]"));
        assert!(!matches(&exact, "[true]"));
        assert!(!matches(&exact, "[true, false, true]"));
        let nullable = Grammar::from_schema(&json!({"type": ["string", "null"]})).unwrap();
        assert!(matches(&nullable, "null") && matches(&nullable, "\"x\""));
        assert!(!matches(&nullable, "1"));
        // A bare number is complete after any digit and may still grow.
        let number = Grammar::from_schema(&json!({"type": "number"})).unwrap();
        let mut m = Matcher::new(&number);
        assert!(m.accept_str(&number, "4"));
        assert!(m.complete() && m.allows(&number, "2") && !m.allows(&number, " "));
    }

    #[test]
    fn unsupported_and_ignored_keywords_are_reported() {
        let g = Grammar::from_schema(
            &json!({"type": "string", "pattern": "^a", "minLength": 2, "description": "x"}),
        )
        .unwrap();
        assert_eq!(g.ignored, ["pattern", "minLength"]);
        for bad in [
            json!({"allOf": [{"type": "string"}]}),
            json!({"type": "object", "patternProperties": {}}),
            json!({"type": "widget"}),
            json!({"$ref": "https://example.com/schema"}),
            json!({"$ref": "#/$defs/missing"}),
            json!({"type": "object", "properties": {"a": {}}, "required": ["b"]}),
            json!({"type": "array", "maxItems": 1000}),
        ] {
            assert!(Grammar::from_schema(&bad).is_err(), "{bad}");
        }
        assert!(
            Grammar::from_response_format(&json!({"type": "text"}))
                .unwrap()
                .is_none()
        );
        assert!(
            Grammar::from_response_format(&json!({"type": "json_object"}))
                .unwrap()
                .is_some()
        );
        assert!(Grammar::from_response_format(&json!({"type": "json_schema"})).is_err());
        let wrapped = Grammar::from_response_format(&json!({
            "type": "json_schema",
            "json_schema": {"name": "n", "strict": true, "schema": {"type": "integer"}}
        }))
        .unwrap()
        .unwrap();
        assert!(matches(&wrapped, "-12") && !matches(&wrapped, "1.5"));
    }
}
