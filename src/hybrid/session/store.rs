//! Sessions on disk. A save writes what a later load needs to resume the
//! conversation without prefilling it again: the tokens, the K/V rows of
//! full-attention layers (only the rows added since the previous save when
//! the cached sequence just grew), and the snapshots (recurrent state and
//! sliding-window rings), plus the live state at the end of the conversation,
//! so a load resumes exactly where it stopped. A cache saved for another
//! model file or session geometry (context, K/V format, rings) is ignored.
use super::Session;
use crate::{
    DType, Error, MetalDevice, Result, Tensor,
    hybrid::{
        HybridModel,
        engine::{ImageSpan, upload},
    },
};
use serde_json::{Value as Json, json};
use std::{
    fs,
    io::{Read, Seek, SeekFrom, Write},
    path::Path,
    rc::Rc,
};

/// 2: Q8 K/V rows are stored plane by plane (`HybridState::kv_planes`).
const VERSION: u64 = 2;
/// K/V rows uploaded per GPU copy on load.
const ROWS_PER_UPLOAD: usize = 4096;

/// What one save wrote.
#[derive(Debug, Clone, Copy, Default)]
pub struct SaveStats {
    pub bytes: u64,
    /// K/V rows appended (all of them when the file was rewritten).
    pub kv_rows: usize,
    /// Snapshot files written (ones already on disk are kept as they are).
    pub snapshots: usize,
}

fn io(e: std::io::Error) -> Error {
    Error::Cache(format!("session cache: {e}"))
}

fn read_tokens(path: &Path) -> Option<Vec<u32>> {
    let bytes = fs::read(path).ok()?;
    Some(
        bytes
            .chunks_exact(4)
            .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect(),
    )
}

/// Bytes per row of a one-row-per-position K/V cache.
fn row_bytes(t: &Tensor) -> usize {
    t.byte_size() / t.shape().dimensions()[0]
}

/// Checks the order of the K/V records in `kv.bin` as a load replays them:
/// each starts within what earlier records reached, and together they cover
/// the `rows` being loaded. A record may reach past `rows`, or start past
/// it, when the conversation got shorter after it was written.
struct Replay {
    rows: usize,
    capacity: usize,
    /// Furthest position any record reached.
    reach: usize,
    /// Rows below `rows` loaded so far (all of `0..covered`).
    covered: usize,
}

impl Replay {
    fn new(rows: usize, capacity: usize) -> Self {
        Self {
            rows,
            capacity,
            reach: 0,
            covered: 0,
        }
    }

    /// The record `from..to`: its rows `from..keep` are loaded, the rest
    /// skipped. Returns `keep`.
    fn take(&mut self, from: usize, to: usize) -> Result<usize> {
        if from > self.reach || from >= to || to > self.capacity {
            return Err(Error::Cache(
                "session cache: K/V records out of order".into(),
            ));
        }
        self.reach = self.reach.max(to);
        let keep = to.min(self.rows).max(from);
        self.covered = self.covered.max(keep);
        Ok(keep)
    }

    fn finish(&self) -> Result<()> {
        if self.covered < self.rows {
            return Err(Error::Cache("session cache: K/V rows missing".into()));
        }
        Ok(())
    }
}

impl Session {
    /// Everything a saved state depends on besides the model file: context
    /// capacity and the byte sizes of the K/V caches and of a snapshot.
    fn geometry(&self) -> String {
        let kv: Vec<usize> = self.state.kv_linear().map(Tensor::byte_size).collect();
        let snap: Vec<usize> = self
            .state
            .snapshot_tensors()
            .map(Tensor::byte_size)
            .collect();
        format!(
            "v{VERSION} capacity {} kv {kv:?} snapshot {snap:?}",
            self.capacity()
        )
    }

    /// Save to `dir` for a model identified by `identity` (its file, as
    /// `Runtime::cache_identity` describes it). Appends to an earlier save
    /// of the same conversation when it only grew since.
    pub fn save_to(&self, dir: &Path, identity: &str) -> Result<SaveStats> {
        fs::create_dir_all(dir).map_err(io)?;
        let key = format!("{identity} | {}", self.geometry());
        let meta_path = dir.join("meta.json");
        let old = fs::read(&meta_path)
            .ok()
            .and_then(|b| serde_json::from_slice::<Json>(&b).ok())
            .filter(|m| m["key"] == key.as_str());
        // Until the new meta.json is in place the folder holds a mix of two
        // saves; without a meta.json a load ignores it.
        let _ = fs::remove_file(&meta_path);
        let mut stats = SaveStats::default();
        let len = self.tokens.len();

        // K/V rows: append from where the rows on disk stop matching ours (a
        // later record overwrites the rows past that point), unless the file
        // would then hold more than twice the live rows.
        let saved_rows = old
            .as_ref()
            .and_then(|m| m["kv_rows"].as_u64())
            .map_or(0, |n| n as usize);
        let saved_tokens = read_tokens(&dir.join("tokens.bin")).unwrap_or_default();
        let common = saved_tokens
            .iter()
            .take(saved_rows)
            .zip(&self.tokens)
            .take_while(|(a, b)| a == b)
            .count();
        let row_total: usize = self.state.kv_linear().map(row_bytes).sum();
        let on_disk = fs::metadata(dir.join("kv.bin")).map_or(0, |m| m.len() as usize);
        let append = old.is_some()
            && common > 0
            && on_disk + (len - common) * row_total <= 2 * len * row_total + 4096;
        let from = if append { common } else { 0 };
        let mut kv = fs::OpenOptions::new()
            .create(true)
            .append(append)
            .write(true)
            .truncate(!append)
            .open(dir.join("kv.bin"))
            .map_err(io)?;
        if from < len {
            let mut record = Vec::with_capacity(16);
            record.extend((from as u64).to_le_bytes());
            record.extend((len as u64).to_le_bytes());
            kv.write_all(&record).map_err(io)?;
            stats.bytes += 16;
            for t in self.state.kv_linear() {
                for (at, row) in self.state.kv_planes(t) {
                    t.read_bytes(|b| kv.write_all(&b[at + from * row..at + len * row]))
                        .map_err(io)?;
                    stats.bytes += ((len - from) * row) as u64;
                }
            }
            stats.kv_rows = len - from;
        }
        kv.sync_data().map_err(io)?;

        // Snapshots: one file per (position, written). A file already on disk
        // is kept only within the prefix that still matches: past it, the
        // same name may hold another conversation's state.
        let mut snapshots = Vec::new();
        let mut keep = Vec::new();
        for s in self
            .snapshots
            .iter()
            .filter(|s| s.position > 0 && s.position <= len)
        {
            let name = format!("snap-{}-{}.bin", s.position, s.written);
            let path = dir.join(&name);
            if !(append && s.position <= common && path.exists()) {
                let mut file = fs::File::create(&path).map_err(io)?;
                for t in &s.tensors {
                    t.read_bytes(|b| file.write_all(b)).map_err(io)?;
                    stats.bytes += t.byte_size() as u64;
                }
                file.sync_data().map_err(io)?;
                stats.snapshots += 1;
            }
            snapshots.push(json!({
                "file": name,
                "position": s.position,
                "written": s.written,
                "pinned": s.pinned,
            }));
            keep.push(name);
        }
        for entry in fs::read_dir(dir).map_err(io)?.flatten() {
            let name = entry.file_name().to_string_lossy().into_owned();
            if name.starts_with("snap-") && !keep.contains(&name) {
                let _ = fs::remove_file(entry.path());
            }
        }

        // The live recurrent state and rings, rewritten every save: without
        // them a load resumes at the deepest snapshot and the next request
        // recomputes the reply after it.
        let live = self.state.len() == len && self.state.snapshot_tensors().next().is_some();
        if live {
            let mut file = fs::File::create(dir.join("live.bin")).map_err(io)?;
            for t in self.state.snapshot_tensors() {
                t.read_bytes(|b| file.write_all(b)).map_err(io)?;
                stats.bytes += t.byte_size() as u64;
            }
            file.sync_data().map_err(io)?;
        } else {
            let _ = fs::remove_file(dir.join("live.bin"));
        }

        let tokens: Vec<u8> = self.tokens.iter().flat_map(|t| t.to_le_bytes()).collect();
        fs::write(dir.join("tokens.bin"), &tokens).map_err(io)?;
        stats.bytes += tokens.len() as u64;
        let media: Vec<Json> = self
            .state
            .media()
            .iter()
            .filter(|m| m.end() <= len)
            .map(|m| {
                json!({"start": m.start, "rows": m.grid.0, "cols": m.grid.1,
                       "hash": m.hash, "linear": m.linear})
            })
            .collect();
        let meta = json!({
            "key": key,
            "kv_rows": len,
            "written": self.state.written(),
            "live": live,
            "media": media,
            "snapshots": snapshots,
        });
        let tmp = dir.join("meta.json.tmp");
        fs::write(&tmp, meta.to_string()).map_err(io)?;
        fs::rename(&tmp, &meta_path).map_err(io)?;
        Ok(stats)
    }

    /// Load what `save_to` wrote in `dir` for the same `identity`, replacing
    /// this session's state. Returns the tokens now cached (all of them, or
    /// the deepest snapshot's position when the live state is missing), or
    /// None when there is no usable save.
    pub fn load_from(
        &mut self,
        d: &MetalDevice,
        model: &HybridModel,
        dir: &Path,
        identity: &str,
    ) -> Result<Option<usize>> {
        let key = format!("{identity} | {}", self.geometry());
        let Some(meta) = fs::read(dir.join("meta.json"))
            .ok()
            .and_then(|b| serde_json::from_slice::<Json>(&b).ok())
            .filter(|m| m["key"] == key.as_str())
        else {
            return Ok(None);
        };
        let Some(tokens) = read_tokens(&dir.join("tokens.bin")) else {
            return Ok(None);
        };
        let rows = meta["kv_rows"].as_u64().unwrap_or(0) as usize;
        if rows == 0 || rows > tokens.len() || rows >= self.capacity() {
            return Ok(None);
        }
        let mut saved: Vec<(String, usize, usize, bool)> = meta["snapshots"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(|s| {
                Some((
                    s["file"].as_str()?.to_string(),
                    s["position"].as_u64()? as usize,
                    s["written"].as_u64()? as usize,
                    s["pinned"].as_bool().unwrap_or(false),
                ))
            })
            .filter(|s| s.1 > 0 && s.1 <= rows && dir.join(&s.0).exists())
            .collect();
        // The pinned snapshot, then the deepest ones, as many as fit the slots.
        saved.sort_by_key(|s| (!s.3, std::cmp::Reverse(s.1)));
        saved.truncate(self.snapshots.len());
        let needs_snapshot = self.state.snapshot_tensors().next().is_some();
        let state_bytes: usize = self.state.snapshot_tensors().map(Tensor::byte_size).sum();
        let live = meta["live"].as_bool() == Some(true)
            && fs::metadata(dir.join("live.bin")).is_ok_and(|m| m.len() as usize == state_bytes);
        let resume = if needs_snapshot && !live {
            match saved.iter().map(|s| s.1).max() {
                Some(p) => p,
                None => return Ok(None),
            }
        } else {
            rows
        };
        self.reset();
        let loaded = self.load_state(d, dir, &meta, &saved, rows, resume, live);
        if let Err(e) = loaded {
            self.reset();
            return Err(e);
        }
        self.tokens = tokens[..resume].to_vec();
        if let Some(drafter) = self.drafter.as_mut() {
            drafter.adopt(d, model, &self.tokens)?;
            self.drafter_rewind = None;
        }
        Ok(Some(resume))
    }

    #[allow(clippy::too_many_arguments)]
    fn load_state(
        &mut self,
        d: &MetalDevice,
        dir: &Path,
        meta: &Json,
        saved: &[(String, usize, usize, bool)],
        rows: usize,
        resume: usize,
        live: bool,
    ) -> Result<()> {
        // K/V rows, record by record (a later record may rewrite rows). A
        // record can reach past `rows` when the conversation got shorter
        // since it was written; those rows are skipped.
        let mut kv = fs::File::open(dir.join("kv.bin")).map_err(io)?;
        let mut replay = Replay::new(rows, self.capacity());
        let kv_tensors: Vec<Tensor> = self.state.kv_linear().cloned().collect();
        let mut header = [0u8; 16];
        while kv.read_exact(&mut header).is_ok() {
            let from = u64::from_le_bytes(header[..8].try_into().unwrap()) as usize;
            let to = u64::from_le_bytes(header[8..].try_into().unwrap()) as usize;
            let keep = replay.take(from, to)?;
            for t in &kv_tensors {
                for (at, row) in self.state.kv_planes(t) {
                    let mut start = from;
                    while start < keep {
                        let end = (start + ROWS_PER_UPLOAD).min(keep);
                        let mut buf = vec![0; (end - start) * row];
                        kv.read_exact(&mut buf).map_err(io)?;
                        // Each upload waits for its copy, so the staging
                        // buffer outlives it.
                        upload(d, t, at + start * row, &buf)?;
                        start = end;
                    }
                    kv.seek(SeekFrom::Current(((to - keep) * row) as i64))
                        .map_err(io)?;
                }
            }
        }
        replay.finish()?;
        let media = meta["media"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(|m| {
                // Cached images are only compared by hash and placed for
                // rotary positions; their embeddings are never read again.
                Some(ImageSpan {
                    start: m["start"].as_u64()? as usize,
                    grid: (m["rows"].as_u64()? as usize, m["cols"].as_u64()? as usize),
                    embedding: Rc::new(Tensor::zeros(d, [1], DType::F32).ok()?),
                    deepstack: Vec::new(),
                    hash: m["hash"].as_u64()?,
                    linear: m["linear"].as_bool()?,
                })
            })
            .filter(|m| m.end() <= resume)
            .collect();
        let written = meta["written"].as_u64().unwrap_or(rows as u64) as usize;
        self.state.adopt(rows, written.max(rows), media);
        for (slot, (file, position, written, pinned)) in self.snapshots.iter_mut().zip(saved) {
            let bytes = fs::read(dir.join(file)).map_err(io)?;
            let sizes: usize = slot.tensors.iter().map(Tensor::byte_size).sum();
            if bytes.len() != sizes {
                return Err(Error::Cache("session cache: snapshot size differs".into()));
            }
            let mut at = 0;
            for t in &slot.tensors {
                upload(d, t, 0, &bytes[at..at + t.byte_size()])?;
                at += t.byte_size();
            }
            self.clock += 1;
            slot.position = *position;
            slot.written = *written;
            slot.pinned = *pinned;
            slot.stamp = self.clock;
        }
        if live {
            let bytes = fs::read(dir.join("live.bin")).map_err(io)?;
            let mut at = 0;
            for t in self.state.snapshot_tensors() {
                upload(d, t, 0, &bytes[at..at + t.byte_size()])?;
                at += t.byte_size();
            }
        } else if let Some(s) = self.snapshots.iter().find(|s| s.position == resume) {
            self.state
                .restore_snapshot(d, &s.tensors, resume, s.written)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::Replay;

    fn replay(records: &[(usize, usize)], rows: usize) -> Option<Vec<usize>> {
        let mut r = Replay::new(rows, 1000);
        let keeps = records
            .iter()
            .map(|&(from, to)| r.take(from, to).ok())
            .collect::<Option<Vec<_>>>()?;
        r.finish().ok().map(|_| keeps)
    }

    #[test]
    fn records_replay_in_order() {
        // Grown twice: everything loads.
        assert_eq!(replay(&[(0, 100), (100, 120)], 120), Some(vec![100, 120]));
        // Rewound to 80 and grown to 90 after one save.
        assert_eq!(replay(&[(0, 100), (80, 90)], 90), Some(vec![90, 90]));
        // The same after two saves: the second record starts past the new
        // length and is skipped.
        assert_eq!(
            replay(&[(0, 100), (100, 120), (80, 90)], 90),
            Some(vec![90, 100, 90])
        );
        // A gap, an empty record, or rows past the capacity are refused.
        assert_eq!(replay(&[(0, 50), (60, 90)], 90), None);
        assert_eq!(replay(&[(0, 50), (50, 50)], 50), None);
        assert_eq!(replay(&[(0, 1001)], 90), None);
        // Records that stop short of the saved rows.
        assert_eq!(replay(&[(0, 50)], 90), None);
    }
}
