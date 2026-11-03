//! qwen4exp's n-gram hash embedding ("PLE"): every token gathers one row per
//! head of a 320M-row table that no Mac holds in memory, so the rows are read
//! from the file on the host as each token arrives.
//!
//! Head `h` of the n-gram of length `n` (2..=ngram) hashes the token and its
//! `n - 1` predecessors: `mixed = (t₀·m₀) ^ (t₁·m₁) ^ ...` in wrapping u64,
//! row `mixed % vocab[h] + offset[h]`. An EOS among the predecessors, or a
//! position before the start, cuts the context there: it and everything
//! before it read as EOS. The rows of a token's heads, side by side, are its
//! `heads × row_dim` embedding.
#![forbid(unsafe_code)]
use super::config::{HybridConfig, PleConfig};
use crate::{Error, Result, loader::gguf::GgufFile};
use std::{
    fs::File,
    os::unix::fs::FileExt,
    path::{Path, PathBuf},
};

const TABLE: &str = "per_layer_token_embd.weight";
const IQ4_NL: u32 = 20;
const IQ4_NL_BLOCK: usize = 18;
/// Threads that read a prompt's n-gram rows (SSDs want many reads in flight).
const GATHER_THREADS: usize = 16;
const KVALUES: [f32; 16] = [
    -127., -104., -83., -65., -49., -35., -22., -10., 1., 13., 25., 38., 53., 69., 89., 113.,
];

/// The table's rows for `tokens`, which follow `before` (oldest first; at
/// most `ngram - 1` are read): `heads` row ids per token, head-major.
pub fn rows(p: &PleConfig, before: &[u32], tokens: &[u32]) -> Vec<u64> {
    let context = p.ngram - 1;
    let mut seq: Vec<u32> = before[before.len().saturating_sub(context)..].to_vec();
    let lead = seq.len();
    seq.extend_from_slice(tokens);
    let eos = p.eos as u64;
    let mut out = Vec::with_capacity(tokens.len() * p.heads());
    for i in 0..tokens.len() {
        let at = lead + i;
        let mut ctx = vec![seq[at] as u64];
        let mut cut = false;
        for s in 1..p.ngram {
            let t = if cut || at < s {
                None
            } else {
                Some(seq[at - s])
            };
            cut = cut || t.is_none_or(|t| t == p.eos);
            ctx.push(if cut { eos } else { t.unwrap() as u64 });
        }
        for n in 2..=p.ngram {
            let mixed = ctx[..n]
                .iter()
                .zip(&p.multipliers)
                .fold(0u64, |acc, (&t, &m)| acc ^ t.wrapping_mul(m));
            for g in 0..p.heads_per_ngram {
                let h = (n - 2) * p.heads_per_ngram + g;
                out.push(mixed % p.vocab_sizes[h] + p.offsets[h]);
            }
        }
    }
    out
}

/// The IQ4_NL table, read a row at a time.
pub struct PleTable {
    file: File,
    base: u64,
    row_bytes: usize,
    row_dim: usize,
    rows: u64,
}

impl PleTable {
    /// Find the table in `path` or the other shards of its split.
    pub fn open(path: &Path, c: &HybridConfig) -> Result<Self> {
        let p = &c
            .qwen4exp
            .as_ref()
            .ok_or_else(|| Error::Config("no n-gram embedding".into()))?
            .ple;
        for candidate in shards(path) {
            let file = GgufFile::open_untagged(&candidate)?;
            let Ok(info) = file.tensor(TABLE) else {
                continue;
            };
            let bad = |message: String| Error::Weight {
                name: TABLE.into(),
                message,
            };
            if info.type_id != IQ4_NL
                || info.dimensions.len() != 2
                || info.dimensions[0] != p.row_dim
            {
                return Err(bad(format!(
                    "expected IQ4_NL rows of {}, found type {} dims {:?}",
                    p.row_dim, info.type_id, info.dimensions
                )));
            }
            let rows = info.dimensions[1] as u64;
            if rows < p.table_rows() || !p.row_dim.is_multiple_of(32) {
                return Err(bad(format!(
                    "{rows} rows, the heads need {}",
                    p.table_rows()
                )));
            }
            let row_bytes = p.row_dim / 32 * IQ4_NL_BLOCK;
            let base = file.tensor_data_offset() + info.offset;
            let file = File::open(&candidate)?;
            return Ok(Self {
                file,
                base,
                row_bytes,
                row_dim: p.row_dim,
                rows,
            });
        }
        Err(Error::Weight {
            name: TABLE.into(),
            message: format!("not in {} or its other shards", path.display()),
        })
    }

    /// Decode `rows` into `out` (`rows.len() × row_dim` values). A prompt's
    /// rows are thousands of small random reads: they run on several threads.
    pub fn gather(&self, rows: &[u64], out: &mut [f32]) -> Result<()> {
        assert_eq!(out.len(), rows.len() * self.row_dim);
        const PER_THREAD: usize = 256;
        if rows.len() <= PER_THREAD {
            return self.gather_serial(rows, out);
        }
        let threads = rows.len().div_ceil(PER_THREAD).min(GATHER_THREADS);
        let per = rows.len().div_ceil(threads);
        std::thread::scope(|scope| {
            let jobs: Vec<_> = rows
                .chunks(per)
                .zip(out.chunks_mut(per * self.row_dim))
                .map(|(r, o)| scope.spawn(move || self.gather_serial(r, o)))
                .collect();
            jobs.into_iter()
                .try_for_each(|j| j.join().expect("n-gram gather thread"))
        })
    }

    fn gather_serial(&self, rows: &[u64], out: &mut [f32]) -> Result<()> {
        let mut raw = vec![0u8; self.row_bytes];
        for (&r, dst) in rows.iter().zip(out.chunks_exact_mut(self.row_dim)) {
            if r >= self.rows {
                return Err(Error::Range(format!("n-gram row {r} past {}", self.rows)));
            }
            self.file
                .read_exact_at(&mut raw, self.base + r * self.row_bytes as u64)?;
            for (block, values) in raw.chunks_exact(IQ4_NL_BLOCK).zip(dst.chunks_exact_mut(32)) {
                let d = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
                for (j, &q) in block[2..].iter().enumerate() {
                    values[j] = d * KVALUES[(q & 15) as usize];
                    values[j + 16] = d * KVALUES[(q >> 4) as usize];
                }
            }
        }
        Ok(())
    }
}

/// `path` first, then the other shards of a `-NNNNN-of-MMMMM.gguf` split.
fn shards(path: &Path) -> Vec<PathBuf> {
    let mut out = vec![path.to_path_buf()];
    let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
    if let Some(at) = name.rfind("-of-")
        && at >= 6
        && let Ok(count) = name[at + 4..].trim_end_matches(".gguf").parse::<usize>()
    {
        let (stem, total) = (&name[..at - 6], &name[at + 4..]);
        for i in 1..=count {
            let sibling = path.with_file_name(format!("{stem}-{i:05}-of-{total}"));
            if sibling != path {
                out.push(sibling);
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> PleConfig {
        PleConfig {
            layers: vec![1],
            ngram: 3,
            heads_per_ngram: 2,
            row_dim: 32,
            conv_kernel: 4,
            eos: 9,
            multipliers: vec![23703573157769, 20109073645365, 8052911324071],
            vocab_sizes: vec![101, 103, 107, 109],
            offsets: vec![0, 101, 204, 311],
        }
    }

    #[test]
    fn hashes_follow_the_reference_and_eos_cuts() {
        let p = config();
        let r = rows(&p, &[5, 6], &[7]);
        let m = &p.multipliers;
        let bi = 7u64.wrapping_mul(m[0]) ^ 6u64.wrapping_mul(m[1]);
        let tri = bi ^ 5u64.wrapping_mul(m[2]);
        assert_eq!(
            r,
            vec![bi % 101, bi % 103 + 101, tri % 107 + 204, tri % 109 + 311]
        );
        // An EOS predecessor reads as EOS, and so does everything before it.
        let cut = rows(&p, &[5, 9], &[7]);
        let bi = 7u64.wrapping_mul(m[0]) ^ 9u64.wrapping_mul(m[1]);
        let tri = bi ^ 9u64.wrapping_mul(m[2]);
        assert_eq!(cut[2], tri % 107 + 204);
        // The first token of a sequence has only EOS before it.
        assert_eq!(rows(&p, &[], &[7]), cut);
        // One call over a sequence equals token-by-token calls.
        let whole = rows(&p, &[], &[3, 4, 5, 6]);
        let mut pieces = rows(&p, &[], &[3, 4]);
        pieces.extend(rows(&p, &[3, 4], &[5, 6]));
        assert_eq!(whole, pieces);
    }

    #[test]
    fn shard_names_are_found() {
        let s = shards(Path::new("/m/Q-00001-of-00002.gguf"));
        assert_eq!(
            s,
            vec![
                PathBuf::from("/m/Q-00001-of-00002.gguf"),
                PathBuf::from("/m/Q-00002-of-00002.gguf")
            ]
        );
    }
}
