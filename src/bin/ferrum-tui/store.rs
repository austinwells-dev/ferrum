//! Keeps long tool output out of the model's context. Output is stored in
//! full, split into chunks and ranked with BM25, so the model gets a short
//! excerpt plus a handle (`#7`) it can search or read from later.
use std::collections::HashMap;

/// Output up to this many characters goes to the model unchanged.
pub const INLINE_LIMIT: usize = 3_500;
const CHUNK_CHARS: usize = 1_000;
const HEAD_LINES: usize = 22;
const TAIL_LINES: usize = 12;

struct Item {
    label: String,
    lines: Vec<String>,
    bytes: usize,
}

struct Chunk {
    item: usize,
    first: usize,
    last: usize,
    terms: HashMap<String, u32>,
    len: usize,
}

#[derive(Default)]
pub struct Store {
    items: Vec<Item>,
    chunks: Vec<Chunk>,
    /// In how many chunks each term appears.
    df: HashMap<String, u32>,
    total_len: usize,
    /// Bytes kept out of the context so far.
    pub saved: usize,
}

pub struct Hit {
    pub item: usize,
    pub first: usize,
    pub last: usize,
    pub score: f32,
}

/// Lowercased words, with a light stem so "caching" finds "cache".
pub fn tokens(text: &str) -> Vec<String> {
    text.split(|c: char| !(c.is_alphanumeric() || c == '_'))
        .filter(|w| w.len() >= 2)
        .map(|w| {
            let w = w.to_lowercase();
            for suffix in ["ing", "ed", "es", "s"] {
                if w.len() > suffix.len() + 3 && w.ends_with(suffix) {
                    return w[..w.len() - suffix.len()].to_string();
                }
            }
            w
        })
        .collect()
}

impl Store {
    pub fn len(&self) -> usize {
        self.items.len()
    }

    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// Store `text` under a new id (1-based).
    pub fn add(&mut self, label: &str, text: &str) -> usize {
        let lines: Vec<String> = text.lines().map(str::to_string).collect();
        let id = self.items.len() + 1;
        let (mut first, mut size) = (0usize, 0usize);
        for (i, line) in lines.iter().enumerate() {
            size += line.len() + 1;
            if size >= CHUNK_CHARS || i + 1 == lines.len() {
                let mut terms: HashMap<String, u32> = HashMap::new();
                let mut len = 0;
                for l in &lines[first..=i] {
                    for t in tokens(l) {
                        *terms.entry(t).or_default() += 1;
                        len += 1;
                    }
                }
                for t in terms.keys() {
                    *self.df.entry(t.clone()).or_default() += 1;
                }
                self.total_len += len;
                self.chunks.push(Chunk {
                    item: id - 1,
                    first,
                    last: i,
                    terms,
                    len,
                });
                first = i + 1;
                size = 0;
            }
        }
        self.items.push(Item {
            label: label.to_string(),
            bytes: text.len(),
            lines,
        });
        id
    }

    /// BM25 over the chunks, optionally within one stored item.
    pub fn search(&self, query: &str, only: Option<usize>, limit: usize) -> Vec<Hit> {
        let query: Vec<String> = tokens(query);
        if query.is_empty() || self.chunks.is_empty() {
            return Vec::new();
        }
        let n = self.chunks.len() as f32;
        let avg = (self.total_len as f32 / n).max(1.0);
        let mut hits: Vec<Hit> = self
            .chunks
            .iter()
            .filter(|c| only.is_none_or(|id| c.item + 1 == id))
            .filter_map(|c| {
                let mut score = 0.0f32;
                for q in &query {
                    let tf = *c.terms.get(q).unwrap_or(&0) as f32;
                    if tf == 0.0 {
                        continue;
                    }
                    let df = *self.df.get(q).unwrap_or(&1) as f32;
                    let idf = ((n - df + 0.5) / (df + 0.5) + 1.0).ln();
                    score += idf * tf * 2.2 / (tf + 1.2 * (0.25 + 0.75 * c.len as f32 / avg));
                }
                (score > 0.0).then_some(Hit {
                    item: c.item + 1,
                    first: c.first,
                    last: c.last,
                    score,
                })
            })
            .collect();
        hits.sort_by(|a, b| b.score.total_cmp(&a.score));
        hits.truncate(limit);
        hits
    }

    /// Lines `offset..offset+limit` (1-based) of a stored item, numbered.
    pub fn read(&self, id: usize, offset: usize, limit: usize) -> Result<String, String> {
        let item = id
            .checked_sub(1)
            .and_then(|i| self.items.get(i))
            .ok_or_else(|| format!("no stored output #{id}"))?;
        let start = offset.max(1) - 1;
        let end = (start + limit.max(1)).min(item.lines.len());
        if start >= item.lines.len() {
            return Err(format!("output #{id} has only {} lines", item.lines.len()));
        }
        let mut out = String::new();
        for (i, line) in item.lines[start..end].iter().enumerate() {
            out += &format!("{:>5}\t{line}\n", start + i + 1);
        }
        if end < item.lines.len() {
            out += &format!(
                "… {} more lines (offset {})\n",
                item.lines.len() - end,
                end + 1
            );
        }
        Ok(out)
    }

    fn excerpt(&self, hit: &Hit) -> String {
        let item = &self.items[hit.item - 1];
        let mut out = format!("— #{} lines {}-{}\n", hit.item, hit.first + 1, hit.last + 1);
        for line in &item.lines[hit.first..=hit.last] {
            out += &crate::agent::clip_output(line, 300);
            out.push('\n');
        }
        out
    }

    /// What the model sees for a tool's output: all of it when short; otherwise
    /// the parts matching `intent`, or the head and tail, with a handle.
    pub fn offload(&mut self, label: &str, text: &str, intent: Option<&str>) -> String {
        if text.len() <= INLINE_LIMIT {
            return text.to_string();
        }
        let id = self.add(label, text);
        let item = &self.items[id - 1];
        let total = item.lines.len();
        let mut out = format!(
            "[output #{id}: {total} lines, {} bytes; the full text is stored, not shown]\n",
            item.bytes
        );
        let hits = intent
            .filter(|i| !i.trim().is_empty())
            .map(|i| self.search(i, Some(id), 4))
            .unwrap_or_default();
        if !hits.is_empty() {
            out += &format!("Parts matching “{}”:\n", intent.unwrap_or(""));
            let mut shown = hits;
            shown.sort_by_key(|h| h.first);
            for hit in &shown {
                out += &self.excerpt(hit);
            }
        } else {
            let item = &self.items[id - 1];
            for line in item.lines.iter().take(HEAD_LINES) {
                out += &crate::agent::clip_output(line, 300);
                out.push('\n');
            }
            if total > HEAD_LINES + TAIL_LINES {
                out += &format!("… {} lines omitted …\n", total - HEAD_LINES - TAIL_LINES);
            }
            for line in item
                .lines
                .iter()
                .skip(HEAD_LINES.max(total.saturating_sub(TAIL_LINES)))
            {
                out += &crate::agent::clip_output(line, 300);
                out.push('\n');
            }
        }
        out += &format!(
            "[ctx_search(query, id={id}) finds more; ctx_read(id={id}, offset, limit) reads lines]"
        );
        self.saved += text.len().saturating_sub(out.len());
        out
    }

    /// Search results as text for the model.
    pub fn search_text(&self, query: &str, only: Option<usize>) -> String {
        let hits = self.search(query, only, 5);
        if hits.is_empty() {
            return format!(
                "no stored output matches “{query}” ({} stored: {})",
                self.items.len(),
                self.items
                    .iter()
                    .enumerate()
                    .map(|(i, it)| format!("#{} {}", i + 1, it.label))
                    .collect::<Vec<_>>()
                    .join(", ")
            );
        }
        let mut out = String::new();
        for hit in &hits {
            out += &self.excerpt(hit);
        }
        out
    }

    pub fn label(&self, id: usize) -> Option<&str> {
        id.checked_sub(1)
            .and_then(|i| self.items.get(i))
            .map(|i| i.label.as_str())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn log() -> String {
        let mut text = String::new();
        for i in 0..400 {
            text += &format!("INFO request {i} handled in {} ms\n", 10 + i % 7);
        }
        text += "ERROR database connection refused while caching user sessions\n";
        for i in 0..200 {
            text += &format!("INFO worker {i} idle\n");
        }
        text
    }

    #[test]
    fn short_output_passes_through() {
        let mut s = Store::default();
        assert_eq!(s.offload("bash", "hello\n", None), "hello\n");
        assert!(s.is_empty());
    }

    #[test]
    fn long_output_is_stored_and_summarised() {
        let mut s = Store::default();
        let text = log();
        let shown = s.offload("bash: tail log", &text, None);
        assert!(shown.len() < text.len() / 5, "{}", shown.len());
        assert!(shown.contains("output #1") && shown.contains("omitted"));
        assert!(s.saved > text.len() / 2);
    }

    #[test]
    fn intent_finds_the_needle() {
        let mut s = Store::default();
        let shown = s.offload("bash", &log(), Some("database connection error"));
        assert!(shown.contains("connection refused"), "{shown}");
        assert!(shown.len() < 3_000, "{}", shown.len());
        let found = s.search_text("cache sessions", Some(1));
        assert!(found.contains("caching user sessions"), "{found}");
    }

    #[test]
    fn reads_ranges_and_reports_bad_ids() {
        let mut s = Store::default();
        s.add("x", "a\nb\nc\nd\n");
        let r = s.read(1, 2, 2).unwrap();
        assert!(
            r.contains("b") && r.contains("c") && !r.contains("\td"),
            "{r}"
        );
        assert!(s.read(9, 1, 1).is_err());
        assert!(s.read(1, 50, 1).is_err());
    }
}
