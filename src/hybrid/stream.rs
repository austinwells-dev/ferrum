//! Routed experts streamed from the GGUF into a resident arena of slots
//! (`metal::stream::ExpertArena`), for MoE models larger than memory.
//!
//! The forward pass stops after each layer's router, reads the chosen experts
//! back and calls `place`: experts already in a slot are hits, the rest are
//! read into the least recently used slots that no work in flight reads. The
//! layer's expert kernels then run on the arena with slot ids in place of
//! expert ids, so the numbers are those of a resident model.
#![forbid(unsafe_code)]
use super::config::HybridConfig;
use crate::{
    DType, Error, MetalDevice, Result, Tensor,
    loader::gguf::GgufFile,
    metal::stream::{ExpertArena, SlotRead, Ticket},
};
use std::{
    cell::{Cell, RefCell},
    collections::BTreeSet,
    fs::File,
    path::Path,
};

/// The three matrices of one expert, in arena part order.
pub const PARTS: [&str; 3] = ["ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"];
const EMPTY: u32 = u32::MAX;

#[derive(Clone, Copy, Debug, Default)]
pub struct StreamStats {
    /// Expert lookups (unique experts per layer per forward).
    pub lookups: u64,
    pub hits: u64,
    /// Hits on experts a read-ahead brought in.
    pub ahead_hits: u64,
    /// Experts read ahead of their layer, and the time spent waiting for them.
    pub ahead_reads: u64,
    pub ahead_wait: std::time::Duration,
    pub reads: u64,
    pub read_bytes: u64,
    pub read_time: std::time::Duration,
}

/// Rows of a forward that read ahead `k` experts each; a larger forward (a
/// prompt chunk, whose union of routes is large already) reads ahead only
/// the experts each row will use.
pub const AHEAD_ROWS: usize = 4;
/// Experts read ahead per row by default.
const AHEAD_K: usize = 16;

/// Device bytes of the read-ahead router scratch for forwards of `chunk` rows.
pub fn ahead_bytes(c: &HybridConfig, chunk: usize) -> usize {
    let Some(moe) = c.moe else { return 0 };
    let rows = chunk.max(AHEAD_ROWS);
    let k = AHEAD_K.max(moe.experts_used).min(moe.experts);
    let page = |n: usize| super::engine::page(n * 4);
    page(rows * moe.experts) + 2 * page(rows * k) + page(rows)
}

/// The next layer's router output on this layer's FFN input: its top `k`
/// experts per row, more than the layer will use, since the guess is a guess.
pub struct AheadScratch {
    pub k: usize,
    /// Rows the scratch holds (a prompt chunk).
    pub rows: usize,
    pub logits: Tensor,
    pub ids: Tensor,
    pub weights: Tensor,
    pub shared_gate: Tensor,
}

pub struct StreamedExperts {
    pub arena: ExpertArena,
    /// None when read-ahead is off.
    pub ahead_scratch: Option<AheadScratch>,
    experts: usize,
    /// Absolute file offset of expert 0 of part `p` of layer `l`.
    offsets: Vec<[u64; 3]>,
    /// `layer * experts + expert` to its slot, or `EMPTY`.
    slot_of: RefCell<Vec<u32>>,
    /// Slot to the `layer * experts + expert` it holds, or `EMPTY`.
    owner: RefCell<Vec<u32>>,
    last_use: RefCell<Vec<u64>>,
    /// Every slot as `(last_use, slot)`, oldest first.
    lru: RefCell<BTreeSet<(u64, usize)>>,
    /// Slots filled by a read-ahead and not yet used by their layer.
    ahead: RefCell<Vec<bool>>,
    /// The read-ahead in flight, if any.
    pending: RefCell<Option<(Ticket, Vec<usize>)>>,
    tick: Cell<u64>,
    stats: Cell<StreamStats>,
}

impl StreamedExperts {
    /// An arena of `slots` experts for every MoE layer of `file`, read from
    /// `path` with `threads` I/O threads, for forwards of up to `chunk` rows.
    pub fn open(
        device: &MetalDevice,
        path: &Path,
        file: &GgufFile,
        c: &HybridConfig,
        slots: usize,
        threads: usize,
        chunk: usize,
    ) -> Result<Self> {
        let moe = c
            .moe
            .ok_or_else(|| Error::Config("expert streaming needs an MoE model".into()))?;
        let mut part_bytes = None;
        let mut offsets = Vec::with_capacity(c.layers);
        for l in 0..c.layers {
            let mut layer = [0u64; 3];
            for (p, name) in PARTS.iter().enumerate() {
                let name = format!("blk.{l}.{name}.weight");
                let info = file.tensor(&name)?;
                let bytes = info.byte_len / moe.experts;
                if info.byte_len % moe.experts != 0 || *part_bytes.get_or_insert(bytes) != bytes {
                    return Err(Error::Weight {
                        name,
                        message: "streamed experts must all have one size".into(),
                    });
                }
                layer[p] = file.tensor_data_offset() + info.offset;
            }
            offsets.push(layer);
        }
        let part_bytes = part_bytes.ok_or_else(|| Error::Config("no layers".into()))?;
        let slots = slots.clamp(moe.experts_used, c.layers * moe.experts);
        let handle =
            File::open(path).map_err(|e| Error::Gguf(format!("{}: {e}", path.display())))?;
        let arena = ExpertArena::new(
            device,
            part_bytes,
            PARTS.len(),
            slots,
            vec![handle],
            threads,
        )?;
        let f = |n: usize| Tensor::zeros_resident(device, [n], DType::F32);
        // `FERRUM_STREAM_AHEAD=0` turns read-ahead off.
        let off = std::env::var("FERRUM_STREAM_AHEAD").is_ok_and(|v| v == "0");
        let k = AHEAD_K.max(moe.experts_used).min(moe.experts);
        let rows = chunk.max(AHEAD_ROWS);
        let ahead_scratch = match off {
            true => None,
            false => Some(AheadScratch {
                k,
                rows,
                logits: f(rows * moe.experts)?,
                ids: f(rows * k)?,
                weights: f(rows * k)?,
                shared_gate: f(rows)?,
            }),
        };
        Ok(Self {
            arena,
            ahead_scratch,
            experts: moe.experts,
            offsets,
            slot_of: RefCell::new(vec![EMPTY; c.layers * moe.experts]),
            owner: RefCell::new(vec![EMPTY; slots]),
            last_use: RefCell::new(vec![0; slots]),
            lru: RefCell::new((0..slots).map(|s| (0, s)).collect()),
            ahead: RefCell::new(vec![false; slots]),
            pending: RefCell::new(None),
            tick: Cell::new(0),
            stats: Cell::new(StreamStats::default()),
        })
    }

    pub fn stats(&self) -> StreamStats {
        self.stats.get()
    }
    pub fn reset_stats(&self) {
        self.stats.set(StreamStats::default());
    }

    /// Make the experts `ids` of `layer` resident and pin their slots for the
    /// work about to be encoded. Returns the slot of each id, in order.
    /// `prompt`: a prompt chunk, whose wrong read-ahead guesses go first.
    pub fn place(
        &self,
        device: &MetalDevice,
        layer: usize,
        ids: &[u32],
        prompt: bool,
    ) -> Result<Vec<u32>> {
        self.finish_ahead()?;
        let tick = self.tick.get() + 1;
        self.tick.set(tick);
        let mut stats = self.stats.get();
        let mut wanted: Vec<u32> = ids.to_vec();
        wanted.sort_unstable();
        wanted.dedup();
        if wanted.last().is_some_and(|&e| e as usize >= self.experts) {
            return Err(Error::Range(format!("expert id past {}", self.experts)));
        }
        let mut reads = Vec::new();
        let mut used = Vec::with_capacity(wanted.len());
        {
            let mut slot_of = self.slot_of.borrow_mut();
            let mut owner = self.owner.borrow_mut();
            let mut last_use = self.last_use.borrow_mut();
            let mut lru = self.lru.borrow_mut();
            // Hits first, so a miss never evicts an expert this layer needs.
            let mut ahead = self.ahead.borrow_mut();
            for &e in &wanted {
                let key = layer * self.experts + e as usize;
                if slot_of[key] != EMPTY {
                    let slot = slot_of[key] as usize;
                    touch(&mut lru, &mut last_use, slot, tick);
                    stats.hits += 1;
                    if std::mem::take(&mut ahead[slot]) {
                        stats.ahead_hits += 1;
                    }
                }
            }
            // Experts read ahead for this layer that a prompt chunk did not
            // route to were wrong guesses: make them the first victims, so the
            // chunk's misses do not push out experts the next chunk will use.
            // Decode keeps them: the next tokens often route to them.
            for &slot in &slot_of[layer * self.experts..(layer + 1) * self.experts] {
                if prompt && slot != EMPTY && std::mem::take(&mut ahead[slot as usize]) {
                    touch(&mut lru, &mut last_use, slot as usize, 0);
                }
            }
            for &e in &wanted {
                let key = layer * self.experts + e as usize;
                stats.lookups += 1;
                if slot_of[key] != EMPTY {
                    used.push(slot_of[key] as usize);
                    continue;
                }
                let victim = self.oldest(device, &lru, tick).ok_or_else(|| {
                    Error::Allocation(self.arena.part_bytes() * PARTS.len() * wanted.len())
                })?;
                if owner[victim] != EMPTY {
                    slot_of[owner[victim] as usize] = EMPTY;
                }
                owner[victim] = key as u32;
                slot_of[key] = victim as u32;
                touch(&mut lru, &mut last_use, victim, tick);
                used.push(victim);
                for (p, base) in self.offsets[layer].iter().enumerate() {
                    reads.push(SlotRead {
                        slot: victim,
                        part: p,
                        file: 0,
                        offset: base + (e as u64) * self.arena.part_bytes() as u64,
                    });
                }
            }
        }
        if !reads.is_empty() {
            let start = std::time::Instant::now();
            let ticket = self.arena.submit(device, &reads)?;
            if let Err(e) = self.arena.wait(ticket) {
                self.forget(reads.iter().map(|r| r.slot));
                return Err(e);
            }
            stats.reads += (reads.len() / PARTS.len()) as u64;
            stats.read_bytes += (reads.len() * self.arena.part_bytes()) as u64;
            stats.read_time += start.elapsed();
        }
        self.stats.set(stats);
        self.arena.pin(device, &used)?;
        let slot_of = self.slot_of.borrow();
        Ok(ids
            .iter()
            .map(|&e| slot_of[layer * self.experts + e as usize])
            .collect())
    }

    /// Start reading the experts `ids` of `layer` that are not resident,
    /// without waiting: a guess at the next layer's routes. Victims are slots
    /// neither pinned for work in flight nor used since the last `place`.
    pub fn read_ahead(&self, device: &MetalDevice, layer: usize, ids: &[u32]) -> Result<()> {
        self.finish_ahead()?;
        let tick = self.tick.get();
        let mut wanted: Vec<u32> = ids.to_vec();
        wanted.sort_unstable();
        wanted.dedup();
        let mut reads = Vec::new();
        let mut slots = Vec::new();
        {
            let mut slot_of = self.slot_of.borrow_mut();
            let mut owner = self.owner.borrow_mut();
            let mut last_use = self.last_use.borrow_mut();
            let mut lru = self.lru.borrow_mut();
            let mut ahead = self.ahead.borrow_mut();
            for &e in wanted.iter().filter(|&&e| (e as usize) < self.experts) {
                let key = layer * self.experts + e as usize;
                if slot_of[key] != EMPTY {
                    continue;
                }
                let Some(victim) = self.oldest(device, &lru, tick) else {
                    break;
                };
                if owner[victim] != EMPTY {
                    slot_of[owner[victim] as usize] = EMPTY;
                }
                owner[victim] = key as u32;
                slot_of[key] = victim as u32;
                // Newer than any slot `place` has not touched this tick, so the
                // next read-ahead does not take it back.
                touch(&mut lru, &mut last_use, victim, tick);
                ahead[victim] = true;
                slots.push(victim);
                for (p, base) in self.offsets[layer].iter().enumerate() {
                    reads.push(SlotRead {
                        slot: victim,
                        part: p,
                        file: 0,
                        offset: base + (e as u64) * self.arena.part_bytes() as u64,
                    });
                }
            }
        }
        if reads.is_empty() {
            return Ok(());
        }
        let ticket = match self.arena.submit(device, &reads) {
            Ok(t) => t,
            Err(e) => {
                self.forget(slots.into_iter());
                return Err(e);
            }
        };
        let mut stats = self.stats.get();
        stats.ahead_reads += slots.len() as u64;
        stats.reads += slots.len() as u64;
        stats.read_bytes += (reads.len() * self.arena.part_bytes()) as u64;
        self.stats.set(stats);
        *self.pending.borrow_mut() = Some((ticket, slots));
        Ok(())
    }

    /// The least recently used slot that was not used at `tick` and that no
    /// work in flight reads. Busy slots were used recently, so few are skipped.
    fn oldest(
        &self,
        device: &MetalDevice,
        lru: &BTreeSet<(u64, usize)>,
        tick: u64,
    ) -> Option<usize> {
        lru.iter()
            .take_while(|&&(used, _)| used < tick)
            .find(|&&(_, s)| !self.arena.busy(device, s))
            .map(|&(_, s)| s)
    }

    /// Complete the read-ahead in flight; on failure its slots are dropped.
    fn finish_ahead(&self) -> Result<()> {
        let Some((ticket, slots)) = self.pending.borrow_mut().take() else {
            return Ok(());
        };
        let start = std::time::Instant::now();
        let result = self.arena.wait(ticket);
        let mut stats = self.stats.get();
        stats.ahead_wait += start.elapsed();
        self.stats.set(stats);
        if result.is_err() {
            self.forget(slots.into_iter());
        }
        result
    }

    /// Drop the mapping of slots whose reads failed: their bytes are unknown.
    fn forget(&self, slots: impl Iterator<Item = usize>) {
        let mut slot_of = self.slot_of.borrow_mut();
        let mut owner = self.owner.borrow_mut();
        for s in slots {
            if owner[s] != EMPTY {
                slot_of[owner[s] as usize] = EMPTY;
                owner[s] = EMPTY;
            }
        }
    }
}

fn touch(lru: &mut BTreeSet<(u64, usize)>, last_use: &mut [u64], slot: usize, tick: u64) {
    lru.remove(&(last_use[slot], slot));
    last_use[slot] = tick;
    lru.insert((tick, slot));
}
