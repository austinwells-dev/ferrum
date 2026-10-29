//! Forward pass for Qwen3.5-family hybrid models on fixed, preallocated storage.
#![forbid(unsafe_code)]
use super::{
    config::{AttnSpec, HybridConfig, KvCache, Variant},
    weights::{
        AttentionWeights, DeltaWeights, DenseFfn, Ffn, Format, Matrix, Mixer, MoeFfn,
        ShortConvWeights, Weights,
    },
};
use crate::{DType, Error, MetalDevice, Result, Tensor, metal::MetalBuffer};
use std::cell::RefCell;

type Binding<'a> = (&'a MetalBuffer, usize, usize);

/// Little-endian kernel parameter block.
#[derive(Default)]
pub(crate) struct Params(pub(crate) Vec<u8>);
impl Params {
    pub(crate) fn u(mut self, v: usize) -> Result<Self> {
        let v =
            u32::try_from(v).map_err(|_| Error::Shape("kernel parameter exceeds u32".into()))?;
        self.0.extend_from_slice(&v.to_le_bytes());
        Ok(self)
    }
    pub(crate) fn f(mut self, v: f32) -> Self {
        self.0.extend_from_slice(&v.to_le_bytes());
        self
    }
    pub(crate) fn i(mut self, v: i32) -> Self {
        self.0.extend_from_slice(&v.to_le_bytes());
        self
    }
}

fn rows(t: &Tensor, start_row: usize, rows: usize, width: usize) -> Result<Tensor> {
    t.view(start_row * width, [rows, width])
}

/// An image occupying `grid.0 * grid.1` consecutive prompt positions whose
/// input embeddings come from the vision encoder instead of the token table.
#[derive(Clone)]
pub struct ImageSpan {
    /// Cache position of the first image token.
    pub start: usize,
    /// Token grid (rows, columns).
    pub grid: (usize, usize),
    /// `[grid.0 * grid.1, hidden]` F32 embeddings in raster order.
    pub embedding: std::rc::Rc<Tensor>,
    /// DeepStack features (Qwen3-VL): entry `i`, shaped like `embedding`, is
    /// added to the image's rows after decoder layer `i`. Usually empty.
    pub deepstack: Vec<std::rc::Rc<Tensor>>,
    /// Identity of the image content, for prefix reuse.
    pub hash: u64,
    /// One language-model position per token. Qwen's grid is false: those
    /// tokens share `max(rows, cols)` rotary positions.
    pub linear: bool,
}

impl ImageSpan {
    pub fn len(&self) -> usize {
        self.grid.0 * self.grid.1
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    pub fn end(&self) -> usize {
        self.start + self.len()
    }
    /// How many positions the image advances the rotary position less than it
    /// advances the cache. A 2-D grid shares `max(rows, cols)` positions.
    /// A linear span (LFM3) takes one position per token.
    fn rope_saving(&self) -> usize {
        if self.linear {
            0
        } else {
            self.len() - self.grid.0.max(self.grid.1)
        }
    }
}

/// Rotary position of the token at cache position `index` past images that
/// end at or before it (text keeps counting from where the last image's
/// position grid ended).
pub fn rope_position(media: &[ImageSpan], index: usize) -> usize {
    index
        - media
            .iter()
            .filter(|s| s.end() <= index)
            .map(ImageSpan::rope_saving)
            .sum::<usize>()
}

/// How rotary positions differ from cache positions for one forward chunk,
/// and, in a draft-tree verify, which rows each row sees.
#[derive(Default)]
pub(crate) struct RopePlan {
    /// Added to the cache position (non-positive: images compress positions).
    pub delta: i32,
    /// Per-row (time, height, width) positions when the chunk holds image
    /// tokens or draft-tree nodes.
    pub table: Option<Tensor>,
    /// Draft tree: the nodes each row sees and their depths (see
    /// `tree_plan_rows`).
    pub tree: Option<Tensor>,
    /// Cache position of the tree's root, the lowest position any row sits at.
    pub tree_root: usize,
}

/// Most rows a draft-tree verify takes: a row's ancestors are one `u32` mask.
pub(crate) const TREE_MAX_ROWS: usize = 32;

/// Plan of a draft-tree verify whose root sits at rotary position `base`:
/// node i is stored at the i-th new cache position but sits at `base +
/// depth(i)`, through the position table with all three axes equal.
pub(crate) fn tree_plan(
    d: &MetalDevice,
    pos: usize,
    base: i32,
    parents: &[Option<usize>],
) -> Result<RopePlan> {
    tree_plan_rows(d, pos, base, parents, 0..parents.len())
}

/// Plan of a forward over nodes `rows` of the draft tree with `parents`,
/// whose root sits at cache position `root` and rotary position `base`
/// (the forward starts at `root + rows.start`): earlier nodes were stored by
/// earlier forwards. The tree buffer
/// holds `rows.start`, then every node's mask and depth (`KeyMask` in
/// `hybrid.metal`). A forward of the root alone needs no tree.
pub(crate) fn tree_plan_rows(
    d: &MetalDevice,
    root: usize,
    base: i32,
    parents: &[Option<usize>],
    rows: std::ops::Range<usize>,
) -> Result<RopePlan> {
    let nodes = tree_rows(parents)?;
    if rows.is_empty() || rows.end > nodes.len() {
        return Err(Error::Parameter(format!(
            "tree rows {rows:?} of a {}-node tree",
            nodes.len()
        )));
    }
    let mut table = Vec::with_capacity(rows.len() * 3 * 4);
    for &(_, depth) in &nodes[rows.clone()] {
        let p = base + depth as i32;
        for v in [p, p, p] {
            table.extend_from_slice(&v.to_le_bytes());
        }
    }
    let table = Tensor::from_le_bytes(d, [rows.len() * 3], DType::F32, &table)?;
    if rows == (0..1) {
        return Ok(RopePlan {
            delta: 0,
            table: Some(table),
            tree: None,
            tree_root: root,
        });
    }
    let mut tree = Vec::with_capacity((rows.end + 1) * 2 * 4);
    for v in [rows.start as u32, 0] {
        tree.extend_from_slice(&v.to_le_bytes());
    }
    for &(mask, depth) in &nodes[..rows.end] {
        tree.extend_from_slice(&mask.to_le_bytes());
        tree.extend_from_slice(&depth.to_le_bytes());
    }
    Ok(RopePlan {
        delta: 0,
        table: Some(table),
        tree: Some(Tensor::from_le_bytes(
            d,
            [(rows.end + 1) * 2],
            DType::F32,
            &tree,
        )?),
        tree_root: root,
    })
}

/// Rows of a draft-tree verify, from each node's parent (`None` for the
/// root, row 0; parents come before their children). Returns per row the
/// mask of rows it sees (itself and its ancestors) and its depth, the layout
/// of `KeyMask` in `hybrid.metal`.
pub(crate) fn tree_rows(parents: &[Option<usize>]) -> Result<Vec<(u32, u32)>> {
    if parents.is_empty() || parents.len() > TREE_MAX_ROWS {
        return Err(Error::Parameter(format!(
            "a draft tree has 1..={TREE_MAX_ROWS} rows, got {}",
            parents.len()
        )));
    }
    let mut rows: Vec<(u32, u32)> = Vec::with_capacity(parents.len());
    for (i, parent) in parents.iter().enumerate() {
        let row = match (i, *parent) {
            (0, None) => (1, 0),
            (_, Some(p)) if i > 0 && p < i => (rows[p].0 | 1 << i, rows[p].1 + 1),
            _ => {
                return Err(Error::Parameter(format!(
                    "draft tree row {i}: only row 0 is the root and parents precede children"
                )));
            }
        };
        rows.push(row);
    }
    Ok(rows)
}

/// Conversation state: K/V (F16 or Q8) for attention layers and F32 recurrent state
/// for delta-rule layers, all allocated once for `capacity` positions.
pub struct HybridState {
    capacity: usize,
    len: usize,
    valid: bool,
    /// Rows written by a speculative verify and not yet committed.
    verified: usize,
    /// Parents of those rows when the verify was a draft tree (else empty).
    verified_tree: Vec<Option<usize>>,
    /// Images in the cached or incoming prompt, ordered by position.
    media: Vec<ImageSpan>,
    k: Vec<Option<Tensor>>,
    v: Vec<Option<Tensor>>,
    conv: Vec<Option<Tensor>>,
    ssm: Vec<Option<Tensor>>,
    /// Per layer: rows of its K/V ring (`ring_rows`), or 0 for one row per position.
    ring: Vec<usize>,
    /// Per layer: its sliding window (0 = full attention).
    window: Vec<usize>,
    /// Most positions one forward may add (`HybridModel::max_forward_rows`).
    max_rows: usize,
    /// One past the highest position whose K/V was ever written: ring rows of
    /// positions below `written - ring` have been overwritten.
    written: usize,
}

/// Rows of the K/V ring of sliding-window `layer` when a forward adds at most
/// `max_rows` positions, or 0 when a linear cache of `capacity` positions is
/// no larger (or the layer attends to everything). A forward at `pos` reads
/// keys from its first query's window start aligned down to 64 (at most
/// `window + 62` back) through its own rows, so that span must fit without
/// wrapping onto itself. Multiples of 64 keep aligned 64-key tiles contiguous.
pub fn ring_rows(c: &HybridConfig, layer: usize, capacity: usize, max_rows: usize) -> usize {
    let Some(window) = attn_spec(c, layer).window else {
        return 0;
    };
    let ring = (window + 62 + max_rows).next_multiple_of(64);
    if ring >= capacity.next_multiple_of(32) {
        0
    } else {
        ring
    }
}

/// Most positions one forward adds for prompt chunks of `chunk` rows: the
/// chunk itself, or the widest speculative verify.
pub fn max_forward_rows(chunk: usize) -> usize {
    chunk.max(MMS_MAX_ROWS)
}

impl HybridState {
    /// A state of `capacity` positions whose forwards add at most `max_rows`
    /// positions each (sliding-window layers size their K/V rings from it).
    pub fn new(
        d: &MetalDevice,
        c: &HybridConfig,
        capacity: usize,
        max_rows: usize,
    ) -> Result<Self> {
        if capacity == 0 || max_rows == 0 {
            return Err(Error::Cache("context capacity must be positive".into()));
        }
        let kv_row = c.kv_heads * c.head_dim;
        // Attention reads keys in blocks of 32; pad so a block never leaves storage.
        let kv_rows = capacity.next_multiple_of(32);
        let mut state = Self {
            capacity,
            len: 0,
            valid: true,
            verified: 0,
            verified_tree: Vec::new(),
            media: Vec::new(),
            k: Vec::new(),
            v: Vec::new(),
            conv: Vec::new(),
            ssm: Vec::new(),
            ring: Vec::new(),
            window: Vec::new(),
            max_rows,
            written: 0,
        };
        for layer in 0..c.layers {
            let ring = if c.is_attention(layer) {
                ring_rows(c, layer, capacity, max_rows)
            } else {
                0
            };
            state.ring.push(ring);
            state.window.push(attn_spec(c, layer).window.unwrap_or(0));
            if c.is_attention(layer) {
                let rows = if ring > 0 { ring } else { kv_rows };
                state.k.push(Some(c.kv_cache.alloc(d, rows, kv_row)?));
                state.v.push(Some(c.kv_cache.alloc(d, rows, kv_row)?));
                state.conv.push(None);
                state.ssm.push(None);
            } else if c.variant == Variant::Lfm3 {
                // Previous B*x rows only. No delta-rule matrix: `page(0)` would
                // still reserve 16 KiB, and the snapshot skips a missing SSM.
                state.k.push(None);
                state.v.push(None);
                state.conv.push(Some(Tensor::zeros_resident(
                    d,
                    [(c.conv_kernel - 1) * c.hidden],
                    DType::F32,
                )?));
                state.ssm.push(None);
            } else {
                state.k.push(None);
                state.v.push(None);
                state.conv.push(Some(Tensor::zeros_resident(
                    d,
                    [(c.conv_kernel - 1) * c.conv_channels()],
                    DType::F32,
                )?));
                state.ssm.push(Some(Tensor::zeros_resident(
                    d,
                    [c.ssm_v_heads * c.ssm_head_dim * c.ssm_head_dim],
                    DType::F32,
                )?));
            }
        }
        Ok(state)
    }
    pub fn len(&self) -> usize {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    pub fn capacity(&self) -> usize {
        self.capacity
    }
    pub fn byte_size(&self) -> usize {
        [&self.k, &self.v, &self.conv, &self.ssm]
            .iter()
            .flat_map(|v| v.iter().flatten())
            .map(|t| page(t.byte_size()))
            .sum()
    }
    /// Forget the conversation. Recurrent state is zeroed on the next forward.
    pub fn reset(&mut self) {
        self.len = 0;
        self.valid = true;
        self.verified = 0;
        self.verified_tree.clear();
        self.written = 0;
        self.media.clear();
    }
    /// Whether every K/V row a forward at `position` reads is still intact:
    /// a ring layer must not have overwritten the start of that window.
    pub(crate) fn can_resume(&self, position: usize) -> bool {
        self.can_resume_from(position, self.written)
    }
    /// `can_resume` for rings as they were when `written` positions had been
    /// written (a snapshot's own copy of them).
    pub(crate) fn can_resume_from(&self, position: usize, written: usize) -> bool {
        self.ring.iter().zip(&self.window).all(|(&ring, &window)| {
            ring == 0
                || (position + 1).saturating_sub(window) / 64 * 64 >= written.saturating_sub(ring)
        })
    }
    /// The images of the sequence being processed. Spans past `len` are the
    /// ones about to be prefilled; their embeddings replace the token rows.
    pub fn set_media(&mut self, media: Vec<ImageSpan>) {
        self.media = media;
    }
    pub fn media(&self) -> &[ImageSpan] {
        &self.media
    }
    /// Rotary positions for `m` rows starting at cache position `pos`.
    fn rope_plan(&self, d: &MetalDevice, pos: usize, m: usize) -> Result<RopePlan> {
        let delta = rope_position(&self.media, pos) as i64 - pos as i64;
        let delta = i32::try_from(delta)
            .map_err(|_| Error::Cache("image rotary offset exceeds i32".into()))?;
        if !self
            .media
            .iter()
            .any(|s| s.start < pos + m && s.end() > pos)
        {
            return Ok(RopePlan {
                delta,
                ..Default::default()
            });
        }
        let mut table = Vec::with_capacity(m * 3 * 4);
        for index in pos..pos + m {
            let (t, h, w) = match self
                .media
                .iter()
                .find(|s| s.start <= index && index < s.end())
            {
                Some(span) => {
                    let base = rope_position(&self.media, span.start) as i32;
                    let offset = index - span.start;
                    if span.linear {
                        let p = base + offset as i32;
                        (p, p, p)
                    } else {
                        (
                            base,
                            base + (offset / span.grid.1) as i32,
                            base + (offset % span.grid.1) as i32,
                        )
                    }
                }
                None => {
                    let p = rope_position(&self.media, index) as i32;
                    (p, p, p)
                }
            };
            for v in [t, h, w] {
                table.extend_from_slice(&v.to_le_bytes());
            }
        }
        Ok(RopePlan {
            delta: 0,
            table: Some(Tensor::from_le_bytes(d, [m * 3], DType::F32, &table)?),
            tree: None,
            tree_root: 0,
        })
    }
    /// Rotary positions and visibility for a draft-tree verify of the nodes
    /// with `parents` (see `tree_rows`) at cache positions `pos..`.
    fn tree_plan(
        &self,
        d: &MetalDevice,
        pos: usize,
        parents: &[Option<usize>],
    ) -> Result<RopePlan> {
        if self
            .media
            .iter()
            .any(|s| s.start < pos + parents.len() && s.end() > pos)
        {
            return Err(Error::Parameter(
                "a draft tree cannot hold image tokens".into(),
            ));
        }
        let base = i32::try_from(rope_position(&self.media, pos))
            .map_err(|_| Error::Cache("rotary position exceeds i32".into()))?;
        tree_plan(d, pos, base, parents)
    }
    /// Recurrent tensors (conv windows then delta-rule matrices) in a fixed order.
    pub(crate) fn recurrent(&self) -> impl Iterator<Item = &Tensor> {
        self.conv.iter().flatten().chain(self.ssm.iter().flatten())
    }
    /// Whether the model has recurrent state (only attention-only models, such
    /// as Flint, have none: their append-only K/V cache can resume anywhere).
    pub(crate) fn has_recurrent(&self) -> bool {
        self.recurrent().next().is_some()
    }
    /// What a snapshot copies: the recurrent tensors, then the K/V rings of
    /// sliding-window layers. A ring overwrites its oldest rows, so without
    /// its own copy a snapshot stops being resumable once the conversation
    /// has moved a window and a chunk past it.
    pub(crate) fn snapshot_tensors(&self) -> impl Iterator<Item = &Tensor> {
        let rings = self
            .ring
            .iter()
            .zip(self.k.iter().zip(&self.v))
            .filter(|(ring, _)| **ring > 0)
            .flat_map(|(_, (k, v))| k.iter().chain(v));
        self.recurrent().chain(rings)
    }
    /// K/V caches of full-attention layers (one row per position), K then V
    /// per layer: what a session saved to disk stores row by row.
    pub(crate) fn kv_linear(&self) -> impl Iterator<Item = &Tensor> {
        self.ring
            .iter()
            .zip(self.k.iter().zip(&self.v))
            .filter(|(ring, _)| **ring == 0)
            .flat_map(|(_, (k, v))| k.iter().chain(v))
    }
    pub(crate) fn written(&self) -> usize {
        self.written
    }
    /// Take `len` positions whose K/V rows were written from disk, with
    /// `written` for the rings a snapshot restore then copies in.
    pub(crate) fn adopt(&mut self, len: usize, written: usize, media: Vec<ImageSpan>) {
        self.len = len;
        self.written = written;
        self.media = media;
        self.valid = true;
        self.verified = 0;
        self.verified_tree.clear();
    }
    /// Buffers for one `save_snapshot`.
    pub(crate) fn alloc_snapshot(&self, d: &MetalDevice) -> Result<Vec<Tensor>> {
        self.snapshot_tensors()
            .map(|t| Tensor::zeros_resident(d, [t.byte_size() / 4], DType::F32))
            .collect()
    }
    /// Copy `snapshot_tensors` into `dst` (same order, same byte sizes).
    /// Returns the `written` mark the copied rings go with.
    pub(crate) fn save_snapshot(&self, d: &MetalDevice, dst: &[Tensor]) -> Result<usize> {
        let e = d.execution_with_shared_encoder(true)?;
        for (src, dst) in self.snapshot_tensors().zip(dst) {
            copy_bytes(d, src, dst)?;
        }
        e.finish()?;
        Ok(self.written)
    }
    /// Restore a snapshot saved at `position` with `written` (`save_snapshot`),
    /// or with `src` empty, only rewind to `position`: K/V rows of full
    /// attention up to `position` are unchanged because that cache is
    /// append-only.
    pub(crate) fn restore_snapshot(
        &mut self,
        d: &MetalDevice,
        src: &[Tensor],
        position: usize,
        written: usize,
    ) -> Result<()> {
        if position > self.len && self.valid {
            return Err(Error::Cache(
                "snapshot lies beyond the cached sequence".into(),
            ));
        }
        let rings = !src.is_empty() && src.len() > self.recurrent().count();
        let resumable = if rings {
            self.can_resume_from(position, written)
        } else {
            self.can_resume(position)
        };
        if !resumable {
            return Err(Error::Cache(
                "the sliding-window K/V ring no longer holds that position's window".into(),
            ));
        }
        self.valid = false;
        let e = d.execution_with_shared_encoder(true)?;
        let dst: Vec<&Tensor> = self.snapshot_tensors().collect();
        for (dst, src) in dst.into_iter().zip(src) {
            copy_bytes(d, src, dst)?;
        }
        e.finish()?;
        if rings {
            self.written = written;
        }
        self.len = position;
        self.valid = true;
        self.verified = 0;
        self.verified_tree.clear();
        Ok(())
    }
    pub(crate) fn is_valid(&self) -> bool {
        self.valid
    }
}

/// Per-chunk activation buffers, allocated once for the largest chunk.
pub(crate) struct Scratch {
    pub(crate) chunk: usize,
    pub(crate) x: [Tensor; 2],
    pub(crate) xn: Tensor,
    pub(crate) mix: Tensor,
    pub(crate) qg: Tensor,
    pub(crate) k: Tensor,
    pub(crate) v: Tensor,
    pub(crate) q: Tensor,
    /// Verify-width queries regrouped by KV head (`h_q_by_kv_head`).
    pub(crate) q_grouped: Tensor,
    pub(crate) att: Tensor,
    /// `[chunk, heads]` per-head output gate logits (Spark).
    pub(crate) head_gate: Tensor,
    pub(crate) attn_partial: Tensor,
    pub(crate) attn_ml: Tensor,
    pub(crate) qkv: Tensor,
    pub(crate) z: Tensor,
    pub(crate) alpha: Tensor,
    pub(crate) beta: Tensor,
    pub(crate) g: Tensor,
    pub(crate) b: Tensor,
    pub(crate) conv: Tensor,
    pub(crate) delta: Tensor,
    pub(crate) gated: Tensor,
    pub(crate) ffn_gate: Tensor,
    pub(crate) ffn_up: Tensor,
    pub(crate) ffn_act: Tensor,
    pub(crate) logits: Tensor,
    pub(crate) token: Tensor,
    pub(crate) candidates: Tensor,
    moe: Option<MoeScratch>,
}

/// Routing and expert buffers for `chunk * experts_used` routes.
struct MoeScratch {
    logits: Tensor,
    ids: Tensor,
    weights: Tensor,
    shared_gate: Tensor,
    offsets: Tensor,
    sorted: Tensor,
    position: Tensor,
    xg: Tensor,
    gate: Tensor,
    up: Tensor,
    act: Tensor,
    down: Tensor,
    shared: Tensor,
}

impl MoeScratch {
    fn tensors(&self) -> [&Tensor; 13] {
        [
            &self.logits,
            &self.ids,
            &self.weights,
            &self.shared_gate,
            &self.offsets,
            &self.sorted,
            &self.position,
            &self.xg,
            &self.gate,
            &self.up,
            &self.act,
            &self.down,
            &self.shared,
        ]
    }
}

/// Speculative-decoding buffers (`HybridModel::enable_speculation`).
pub(crate) struct Spec {
    /// Widest verify (anchor plus drafts).
    pub(crate) rows: usize,
    /// Layers whose outputs are captured, in feature-column order.
    pub(crate) aux: Vec<usize>,
    /// `[max(chunk, rows), aux.len() * hidden]`: aux-layer outputs of the
    /// rows of the latest forward.
    pub(crate) features: Option<Tensor>,
    /// `[max(chunk, rows), hidden]`: output-normed final hidden state of the
    /// rows of the latest forward (MTP input).
    pub(crate) hidden: Option<Tensor>,
    /// Per recurrent layer: pre-conv qkv, g and beta of the verify rows.
    tape_qkv: Vec<Tensor>,
    tape_g: Vec<Tensor>,
    tape_b: Vec<Tensor>,
    /// Recurrent-layer index of each layer (None for attention layers).
    recurrent_index: Vec<Option<usize>>,
    logits: Tensor,
    candidates: Tensor,
    tokens: Tensor,
    /// `[rows, gather_width]`: staging for `commit_path`.
    gather: Tensor,
}

impl Spec {
    fn byte_size(&self) -> usize {
        self.features
            .iter()
            .chain(&self.hidden)
            .chain(&self.tape_qkv)
            .chain(&self.tape_g)
            .chain(&self.tape_b)
            .chain([&self.logits, &self.candidates, &self.tokens, &self.gather])
            .map(|t| page(t.byte_size()))
            .sum()
    }
}

/// What speculation needs from the target (see `spec_bytes`).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SpecGeometry {
    /// Widest verify: anchor plus the most drafts.
    pub rows: usize,
    /// Target layers whose outputs feed the drafter.
    pub aux_layers: Vec<usize>,
    /// The drafter reads the output-normed final hidden state (MTP).
    pub final_hidden: bool,
}

/// Bytes `HybridModel::enable_speculation` allocates, page-rounded.
pub fn spec_bytes(c: &HybridConfig, chunk: usize, g: &SpecGeometry) -> usize {
    let f32s = |n: usize| page(n * 4);
    let capture = chunk.max(g.rows);
    let mut total = 0;
    if !g.aux_layers.is_empty() {
        total += f32s(capture * g.aux_layers.len() * c.hidden);
    }
    if g.final_hidden {
        total += f32s(capture * c.hidden);
    }
    total += if c.variant == Variant::Lfm3 {
        // One B*x tape per short-conv layer. No g/beta tapes.
        c.recurrent_layer_count() * f32s(g.rows * c.hidden)
    } else {
        c.recurrent_layer_count()
            * (f32s(g.rows * c.conv_channels()) + 2 * f32s(g.rows * c.ssm_v_heads))
    };
    total
        + f32s(g.rows * c.vocab)
        + f32s(g.rows * c.vocab.div_ceil(1024) * 64)
        + f32s(g.rows)
        + f32s(g.rows * gather_width(c, g))
}

/// Widest row `commit_path` stages: captured features, the hidden state, or
/// a K/V row (at most two bytes per element, so this overcounts).
fn gather_width(c: &HybridConfig, g: &SpecGeometry) -> usize {
    (g.aux_layers.len() * c.hidden)
        .max(c.hidden)
        .max(c.kv_heads * c.head_dim)
}

/// Gated depthwise conv of one LFM3 short-conv layer. `tape`, when set, receives
/// the B*x rows and the live state is left unchanged (speculative verify).
#[allow(clippy::too_many_arguments)]
fn run_short_conv(
    d: &MetalDevice,
    c: &HybridConfig,
    s: &Scratch,
    w: &ShortConvWeights,
    state: &HybridState,
    layer: usize,
    xn: &Tensor,
    out: &Tensor,
    m: usize,
    tape: Option<&Tensor>,
) -> Result<()> {
    let h = c.hidden;
    let packed = rows(&s.qkv, 0, m, 3 * h)?;
    HybridModel::project(d, &w.in_proj, xn, m, &packed)?;
    let y = rows(&s.conv, 0, m, h)?;
    let conv_state = state.conv[layer]
        .as_ref()
        .expect("short-conv layer has a conv window");
    // Unread unless the verify tapes B*x. A zero-width view is enough.
    let bx = match tape {
        Some(t) => rows(t, 0, m, h)?,
        None => s.z.view(0, [0])?,
    };
    d.dispatch_hybrid(
        "h_short_conv",
        &[packed.binding(), w.conv.binding()],
        &[y.binding(), conv_state.binding(), bx.binding()],
        &Params::default()
            .u(m)?
            .u(h)?
            .u(c.conv_kernel - 1)?
            .u(usize::from(tape.is_none()))?
            .u(usize::from(tape.is_some()))?
            .0,
        [h.div_ceil(256), 1, 1],
        [256, 1, 1],
        0,
    )?;
    HybridModel::project(d, &w.out_proj, &y, m, out)
}

/// Shift `n` recorded B*x rows into a short-conv window.
fn bx_state(
    d: &MetalDevice,
    c: &HybridConfig,
    bx: &Tensor,
    state: &HybridState,
    layer: usize,
    n: usize,
) -> Result<()> {
    let conv_state = state.conv[layer]
        .as_ref()
        .expect("short-conv layer has a conv window");
    d.dispatch_hybrid(
        "h_bx_state",
        &[bx.binding()],
        &[conv_state.binding()],
        &Params::default().u(n)?.u(c.hidden)?.u(c.conv_kernel - 1)?.0,
        [c.hidden.div_ceil(256), 1, 1],
        [256, 1, 1],
        0,
    )
}

/// Write `data` into `dst` at byte `offset` through a staging buffer.
pub(crate) fn upload(d: &MetalDevice, dst: &Tensor, offset: usize, data: &[u8]) -> Result<()> {
    let size = dst.dtype().size_bytes();
    if !offset.is_multiple_of(size) || !data.len().is_multiple_of(4) || data.is_empty() {
        return Err(Error::Shape("unaligned upload".into()));
    }
    let staged = Tensor::staged(d, [data.len() / 4], DType::F32, data)?;
    let view = dst.view(offset / size, [data.len() / size])?;
    copy_bytes(d, &staged, &view)
}

/// `copy` of the raw bytes of two same-sized tensors of any type (a
/// multiple of 4 bytes, as K/V rows and F32 state are).
fn copy_bytes(d: &MetalDevice, src: &Tensor, dst: &Tensor) -> Result<()> {
    let bytes = src.byte_size();
    if dst.byte_size() != bytes || !bytes.is_multiple_of(4) {
        return Err(Error::Shape("copy between different sizes".into()));
    }
    let n = bytes / 4;
    d.dispatch_hybrid(
        "h_copy",
        &[src.binding()],
        &[dst.binding()],
        &Params::default().u(n)?.0,
        [n.div_ceil(256), 1, 1],
        [256, 1, 1],
        0,
    )
}

fn copy(d: &MetalDevice, src: &Tensor, dst: &Tensor) -> Result<()> {
    let n = src.numel();
    if dst.numel() != n {
        return Err(Error::Shape("copy between different sizes".into()));
    }
    d.dispatch_hybrid(
        "h_copy",
        &[src.binding()],
        &[dst.binding()],
        &Params::default().u(n)?.0,
        [n.div_ceil(256), 1, 1],
        [256, 1, 1],
        0,
    )
}

/// Attention options of `layer`. Layers past the trunk (the MTP block) use the
/// Qwen3.5 layout: full window, RoPE, gated, no sinks.
/// `dst[r] = src[idx[r]]` for `n` rows of `width` (`src` rows `src_stride` apart).
pub(crate) fn gather_rows(
    d: &MetalDevice,
    src: &Tensor,
    idx: &Tensor,
    dst: &Tensor,
    n: usize,
    width: usize,
    src_stride: usize,
) -> Result<()> {
    d.dispatch_hybrid(
        "h_gather_rows",
        &[src.binding(), idx.binding()],
        &[dst.binding()],
        &Params::default().u(n)?.u(width)?.u(src_stride)?.0,
        [(n * width).div_ceil(256), 1, 1],
        [256, 1, 1],
        0,
    )
}

pub(crate) fn attn_spec(c: &HybridConfig, layer: usize) -> AttnSpec {
    c.attn.get(layer).copied().unwrap_or(AttnSpec {
        window: None,
        rope: true,
        sinks: false,
        gated: true,
    })
}

/// RoPE width (0 = none) and base of a layer with options `spec`:
/// sliding-window layers take `swa_rope` when the model has one.
pub(crate) fn rope_of(c: &HybridConfig, spec: AttnSpec) -> (usize, f32) {
    match (spec.rope, spec.window.and(c.swa_rope)) {
        (false, _) => (0, c.rope_theta),
        (true, Some(swa)) => swa,
        (true, None) => (c.rope_dims, c.rope_theta),
    }
}

/// Widest feed-forward activation of any layer: dense, expert or shared.
fn scratch_ffn(c: &HybridConfig) -> usize {
    match c.moe {
        Some(moe) => c.ffn.max(moe.shared_ffn).max(moe.expert_ffn),
        None => c.ffn,
    }
}

/// Bytes of every scratch allocation `Scratch::new` makes, page-rounded.
pub fn scratch_bytes(c: &HybridConfig, chunk: usize, logit_rows: usize) -> usize {
    let chunk = chunk.max(1);
    let logit_rows = logit_rows.clamp(1, chunk);
    let ffn = scratch_ffn(c);
    let f32s = |n: usize| page(n * 4);
    let mut total = 0;
    for n in [
        chunk * c.hidden,
        chunk * c.hidden,
        chunk * c.hidden,
        chunk * c.hidden,
        chunk * c.query_width(),
        chunk * c.kv_heads * c.head_dim,
        chunk * c.kv_heads * c.head_dim,
        chunk * c.heads * c.head_dim,
        if matches!(c.variant, Variant::Spark | Variant::Lfm3) {
            chunk * c.heads
        } else {
            1
        },
        MAX_SPLITS * MAX_SPLIT_ROWS * c.head_dim,
        MAX_SPLITS * MAX_SPLIT_ROWS * 2,
        qkv_width(c) * chunk,
        chunk * c.ssm_value_dim(),
        chunk * c.ssm_v_heads,
        chunk * c.ssm_v_heads,
        chunk * c.ssm_v_heads,
        chunk * c.ssm_v_heads,
        conv_out_width(c) * chunk,
        chunk * c.ssm_value_dim(),
        chunk * c.ssm_value_dim(),
        chunk * ffn,
        chunk * ffn,
        chunk * ffn,
        logit_rows * c.vocab,
        1,
        c.vocab.div_ceil(1024) * 64,
    ] {
        total += f32s(n);
    }
    total += page(chunk * c.heads * c.head_dim * 2); // F16 queries
    total += page(MPP_ATTENTION_MIN_TOKENS * c.heads * c.head_dim * 2); // regrouped verify queries
    if let Some(moe) = c.moe {
        let routes = chunk * moe.experts_used;
        for n in [
            chunk * moe.experts,
            routes,
            routes,
            chunk,
            moe.experts + 1,
            routes,
            routes,
            routes * c.hidden,
            routes * moe.expert_ffn,
            routes * moe.expert_ffn,
            routes * moe.expert_ffn,
            routes * c.hidden,
            chunk * c.hidden,
        ] {
            total += f32s(n);
        }
    }
    total
}

/// Bytes of a `HybridState` for `capacity` positions, page-rounded.
pub fn state_bytes(c: &HybridConfig, capacity: usize, max_rows: usize) -> usize {
    kv_bytes(c, capacity, max_rows) + recurrent_bytes(c)
}

/// K and V caches of every attention layer for `capacity` positions (ring
/// layers hold `ring_rows` rows whatever the capacity).
pub fn kv_bytes(c: &HybridConfig, capacity: usize, max_rows: usize) -> usize {
    let row = c.kv_cache.row_bytes(c.kv_heads * c.head_dim);
    (0..c.layers)
        .filter(|&l| c.is_attention(l))
        .map(|l| match ring_rows(c, l, capacity, max_rows) {
            0 => 2 * page(capacity.next_multiple_of(32) * row),
            ring => 2 * page(ring * row),
        })
        .sum()
}

/// The part of `kv_bytes` held by sliding-window rings, which does not grow
/// with the context once it exceeds the ring.
pub fn kv_ring_bytes(c: &HybridConfig, max_rows: usize) -> usize {
    let row = c.kv_cache.row_bytes(c.kv_heads * c.head_dim);
    (0..c.layers)
        .filter(|&l| c.is_attention(l))
        .map(|l| ring_rows(c, l, usize::MAX / 2, max_rows))
        .filter(|&ring| ring > 0)
        .map(|ring| 2 * page(ring * row))
        .sum()
}

/// Convolution windows and delta-rule matrices of every recurrent layer.
/// LFM3 counts only the short-conv window (`(l_cache - 1) * hidden`).
pub fn recurrent_bytes(c: &HybridConfig) -> usize {
    if c.variant == Variant::Lfm3 {
        let conv = page((c.conv_kernel - 1) * c.hidden * 4);
        return c.recurrent_layer_count() * conv;
    }
    let conv = page((c.conv_kernel - 1) * c.conv_channels() * 4);
    let ssm = page(c.ssm_v_heads * c.ssm_head_dim * c.ssm_head_dim * 4);
    c.recurrent_layer_count() * (conv + ssm)
}

/// Width of the packed QKV / short-conv in-projection scratch.
fn qkv_width(c: &HybridConfig) -> usize {
    if c.variant == Variant::Lfm3 {
        3 * c.hidden
    } else {
        c.conv_channels()
    }
}

/// Width of the conv-output scratch (delta channels, or the LFM3 gated conv).
fn conv_out_width(c: &HybridConfig) -> usize {
    if c.variant == Variant::Lfm3 {
        c.hidden
    } else {
        c.conv_channels()
    }
}

/// Metal rounds each allocation up to whole 16 KiB pages.
pub(crate) fn page(bytes: usize) -> usize {
    bytes.max(4).next_multiple_of(16384)
}

impl Scratch {
    fn new(d: &MetalDevice, c: &HybridConfig, chunk: usize, logit_rows: usize) -> Result<Self> {
        let f = |dims: &[usize]| Tensor::zeros_resident(d, dims, DType::F32);
        let ffn = scratch_ffn(c);
        Ok(Self {
            chunk,
            x: [f(&[chunk, c.hidden])?, f(&[chunk, c.hidden])?],
            xn: f(&[chunk, c.hidden])?,
            mix: f(&[chunk, c.hidden])?,
            qg: f(&[chunk, c.query_width()])?,
            k: f(&[chunk, c.kv_heads * c.head_dim])?,
            v: f(&[chunk, c.kv_heads * c.head_dim])?,
            q: Tensor::zeros_resident(d, [chunk, c.heads * c.head_dim], DType::F16)?,
            q_grouped: Tensor::zeros_resident(
                d,
                [MPP_ATTENTION_MIN_TOKENS, c.heads * c.head_dim],
                DType::F16,
            )?,
            att: f(&[chunk, c.heads * c.head_dim])?,
            // Spark and LFM3 multiply a per-head sigmoid gate onto attention.
            head_gate: f(&if matches!(c.variant, Variant::Spark | Variant::Lfm3) {
                [chunk, c.heads]
            } else {
                [1, 1]
            })?,
            attn_partial: f(&[MAX_SPLITS * MAX_SPLIT_ROWS * c.head_dim])?,
            attn_ml: f(&[MAX_SPLITS * MAX_SPLIT_ROWS * 2])?,
            qkv: f(&[chunk, qkv_width(c)])?,
            z: f(&[chunk, c.ssm_value_dim()])?,
            alpha: f(&[chunk, c.ssm_v_heads])?,
            beta: f(&[chunk, c.ssm_v_heads])?,
            g: f(&[chunk, c.ssm_v_heads])?,
            b: f(&[chunk, c.ssm_v_heads])?,
            conv: f(&[chunk, conv_out_width(c)])?,
            delta: f(&[chunk, c.ssm_value_dim()])?,
            gated: f(&[chunk, c.ssm_value_dim()])?,
            ffn_gate: f(&[chunk, ffn])?,
            ffn_up: f(&[chunk, ffn])?,
            ffn_act: f(&[chunk, ffn])?,
            logits: f(&[logit_rows, c.vocab])?,
            token: f(&[1])?,
            candidates: f(&[c.vocab.div_ceil(1024) * 64])?,
            moe: match c.moe {
                None => None,
                Some(moe) => {
                    let routes = chunk * moe.experts_used;
                    Some(MoeScratch {
                        logits: f(&[chunk, moe.experts])?,
                        ids: f(&[routes])?,
                        weights: f(&[routes])?,
                        shared_gate: f(&[chunk])?,
                        offsets: f(&[moe.experts + 1])?,
                        sorted: f(&[routes])?,
                        position: f(&[routes])?,
                        xg: f(&[routes, c.hidden])?,
                        gate: f(&[routes, moe.expert_ffn])?,
                        up: f(&[routes, moe.expert_ffn])?,
                        act: f(&[routes, moe.expert_ffn])?,
                        down: f(&[routes, c.hidden])?,
                        shared: f(&[chunk, c.hidden])?,
                    })
                }
            },
        })
    }
    fn byte_size(&self) -> usize {
        [
            &self.x[0],
            &self.x[1],
            &self.xn,
            &self.mix,
            &self.qg,
            &self.k,
            &self.v,
            &self.q,
            &self.q_grouped,
            &self.att,
            &self.head_gate,
            &self.attn_partial,
            &self.attn_ml,
            &self.qkv,
            &self.z,
            &self.alpha,
            &self.beta,
            &self.g,
            &self.b,
            &self.conv,
            &self.delta,
            &self.gated,
            &self.ffn_gate,
            &self.ffn_up,
            &self.ffn_act,
            &self.logits,
            &self.token,
            &self.candidates,
        ]
        .iter()
        .map(|t| page(t.byte_size()))
        .sum::<usize>()
            + self
                .moe
                .as_ref()
                .map_or(0, |m| m.tensors().iter().map(|t| page(t.byte_size())).sum())
    }
}

/// Logit adjustments for previously generated tokens (OpenAI / llama.cpp
/// semantics): repetition divides positive and multiplies negative logits;
/// presence and frequency subtract per occurrence.
#[derive(Debug, Clone, Default)]
pub struct Penalties {
    /// (token, occurrences) within the penalty window.
    pub tokens: Vec<(u32, u32)>,
    pub presence: f32,
    pub frequency: f32,
    pub repetition: f32,
}

impl Penalties {
    pub(crate) fn active(&self) -> bool {
        !self.tokens.is_empty()
            && (self.presence != 0. || self.frequency != 0. || self.repetition != 1.)
    }
}

/// What a forward call returns for its final token(s).
pub enum Output {
    /// Nothing (prefill of a prefix that will be continued).
    None,
    /// Greedy next token, selected on the GPU.
    Argmax,
    /// Final-position logits copied to the host.
    Logits,
    /// Sampling candidates: after penalties, the 32 largest logits of every
    /// 1024-token block (so the global top 32 is exact).
    Candidates(Penalties),
}

pub enum Produced {
    None,
    Token(u32),
    Logits(Vec<f32>),
    /// (token, logit) pairs, unordered across blocks.
    Candidates(Vec<(u32, f32)>),
}

pub struct HybridModel {
    pub config: HybridConfig,
    pub(crate) weights: Weights,
    pub(crate) scratch: RefCell<Scratch>,
    /// Rows of logits the scratch holds (1 unless evaluation asked for more).
    logit_rows: usize,
    pub(crate) spec: RefCell<Option<Spec>>,
}

/// Per-row outputs of a speculative verify.
pub enum RowOutput {
    /// Greedy token of every row, selected on the GPU.
    Argmax,
    /// Sampling candidates of every row (see `Output::Candidates`), each row
    /// with its own penalty window.
    Candidates(Vec<Penalties>),
    /// Full logits of every row (evaluation).
    Logits,
}

pub enum RowsProduced {
    Tokens(Vec<u32>),
    Candidates(Vec<Vec<(u32, f32)>>),
    Logits(Vec<Vec<f32>>),
}

impl HybridModel {
    pub fn new(
        d: &MetalDevice,
        config: HybridConfig,
        weights: Weights,
        chunk: usize,
        logit_rows: usize,
    ) -> Result<Self> {
        let chunk = chunk.max(1);
        let logit_rows = logit_rows.clamp(1, chunk);
        let scratch = Scratch::new(d, &config, chunk, logit_rows)?;
        Ok(Self {
            config,
            weights,
            scratch: RefCell::new(scratch),
            logit_rows,
            spec: RefCell::new(None),
        })
    }

    /// Allocate verify and feature-capture buffers for `g` (replacing any
    /// previous speculation setup).
    pub fn enable_speculation(&self, d: &MetalDevice, g: &SpecGeometry) -> Result<()> {
        let c = &self.config;
        if g.rows < 2 || g.rows > MMS_MAX_ROWS {
            return Err(Error::Parameter(format!(
                "verify width {} must be within 2..={MMS_MAX_ROWS}",
                g.rows
            )));
        }
        if let Some(&bad) = g.aux_layers.iter().find(|&&l| l >= c.layers) {
            return Err(Error::Parameter(format!(
                "aux layer {bad} beyond the {} trunk layers",
                c.layers
            )));
        }
        *self.spec.borrow_mut() = None;
        let f = |dims: &[usize]| Tensor::zeros_resident(d, dims, DType::F32);
        let capture = self.chunk().max(g.rows);
        let mut recurrent_index = Vec::with_capacity(c.layers);
        let (mut tape_qkv, mut tape_g, mut tape_b) = (Vec::new(), Vec::new(), Vec::new());
        for layer in 0..c.layers {
            if c.is_attention(layer) {
                recurrent_index.push(None);
            } else {
                recurrent_index.push(Some(tape_qkv.len()));
                if c.variant == Variant::Lfm3 {
                    tape_qkv.push(f(&[g.rows, c.hidden])?);
                } else {
                    tape_qkv.push(f(&[g.rows, c.conv_channels()])?);
                    tape_g.push(f(&[g.rows, c.ssm_v_heads])?);
                    tape_b.push(f(&[g.rows, c.ssm_v_heads])?);
                }
            }
        }
        *self.spec.borrow_mut() = Some(Spec {
            rows: g.rows,
            aux: g.aux_layers.clone(),
            features: if g.aux_layers.is_empty() {
                None
            } else {
                Some(f(&[capture, g.aux_layers.len() * c.hidden])?)
            },
            hidden: if g.final_hidden {
                Some(f(&[capture, c.hidden])?)
            } else {
                None
            },
            tape_qkv,
            tape_g,
            tape_b,
            recurrent_index,
            logits: f(&[g.rows, c.vocab])?,
            candidates: f(&[g.rows, c.vocab.div_ceil(1024) * 64])?,
            tokens: f(&[g.rows])?,
            gather: f(&[g.rows, gather_width(c, g)])?,
        });
        Ok(())
    }

    pub fn disable_speculation(&self) {
        *self.spec.borrow_mut() = None;
    }

    pub fn spec_bytes(&self) -> usize {
        self.spec.borrow().as_ref().map_or(0, Spec::byte_size)
    }
    pub fn weight_bytes(&self) -> usize {
        self.weights.byte_size()
    }
    pub fn scratch_bytes(&self) -> usize {
        self.scratch.borrow().byte_size()
    }
    pub fn chunk(&self) -> usize {
        self.scratch.borrow().chunk
    }
    /// Most positions one forward adds (sizes `HybridState` K/V rings).
    pub fn max_forward_rows(&self) -> usize {
        max_forward_rows(self.chunk())
    }

    /// Append `tokens` to `state` and produce `output` for the last token.
    pub fn forward(
        &self,
        d: &MetalDevice,
        state: &mut HybridState,
        tokens: &[u32],
        output: Output,
    ) -> Result<Produced> {
        if tokens.is_empty() {
            return Err(Error::Parameter(
                "forward requires at least one token".into(),
            ));
        }
        if !state.valid {
            return Err(Error::Cache(
                "state was invalidated by a failed forward; reset it".into(),
            ));
        }
        if state.len + tokens.len() > state.capacity {
            return Err(Error::Cache(format!(
                "context overflow: {} cached + {} new exceeds capacity {}",
                state.len,
                tokens.len(),
                state.capacity
            )));
        }
        if let Some(&id) = tokens.iter().find(|&&id| id as usize >= self.config.vocab) {
            return Err(Error::Token {
                id,
                vocab: self.config.vocab,
            });
        }
        let chunk = self.chunk();
        let mut produced = Produced::None;
        let pieces = tokens.chunks(chunk).count();
        // An uncommitted verify only wrote K/V rows past `len`; drop it.
        state.verified = 0;
        state.verified_tree.clear();
        for (index, piece) in tokens.chunks(chunk).enumerate() {
            let last = index + 1 == pieces;
            let out = if last { &output } else { &Output::None };
            state.valid = false;
            produced = self.forward_chunk(d, state, piece, out, None, false, None)?;
            state.len += piece.len();
            state.valid = true;
        }
        Ok(produced)
    }

    /// Evaluation: append `tokens` (at most `logit_rows` per call) and return
    /// the logits of every position as rows of `vocab` values.
    pub fn forward_all_logits(
        &self,
        d: &MetalDevice,
        state: &mut HybridState,
        tokens: &[u32],
    ) -> Result<Vec<f32>> {
        if tokens.len() > self.logit_rows || tokens.is_empty() {
            return Err(Error::Parameter(format!(
                "evaluation forward takes 1..={} tokens",
                self.logit_rows
            )));
        }
        if !state.valid || state.len + tokens.len() > state.capacity {
            return Err(Error::Cache("state invalid or full".into()));
        }
        state.valid = false;
        state.verified = 0;
        state.verified_tree.clear();
        let mut all = Vec::new();
        self.forward_chunk(d, state, tokens, &Output::None, Some(&mut all), false, None)?;
        state.len += tokens.len();
        state.valid = true;
        Ok(all)
    }

    /// Speculative verify: run `tokens` (anchor then drafts) after the cached
    /// sequence and return every row's output, without advancing the state.
    /// Attention K/V rows are written past `len`; recurrent layers only read
    /// their state and record their inputs so that `commit` can replay the
    /// accepted prefix. Captured features cover all verify rows.
    pub fn verify(
        &self,
        d: &MetalDevice,
        state: &mut HybridState,
        tokens: &[u32],
        output: RowOutput,
    ) -> Result<RowsProduced> {
        self.verify_rows(d, state, tokens, None, output)
    }

    /// Speculative verify of a draft tree: row i holds node `tokens[i]`, a
    /// child of row `parents[i]` (`None` for the anchor, row 0; parents come
    /// before their children). Each row sees the cached sequence and its own
    /// ancestors, at position `len + depth`, and recurrent layers start each
    /// node from its parent's state. `commit_path` keeps one root-to-node path.
    pub fn verify_tree(
        &self,
        d: &MetalDevice,
        state: &mut HybridState,
        tokens: &[u32],
        parents: &[Option<usize>],
        output: RowOutput,
    ) -> Result<RowsProduced> {
        if parents.len() != tokens.len() {
            return Err(Error::Parameter(format!(
                "{} parents for {} tree rows",
                parents.len(),
                tokens.len()
            )));
        }
        self.verify_rows(d, state, tokens, Some(parents), output)
    }

    fn verify_rows(
        &self,
        d: &MetalDevice,
        state: &mut HybridState,
        tokens: &[u32],
        parents: Option<&[Option<usize>]>,
        output: RowOutput,
    ) -> Result<RowsProduced> {
        let rows = self
            .spec
            .borrow()
            .as_ref()
            .map(|s| s.rows)
            .ok_or_else(|| Error::Parameter("speculation is not enabled".into()))?;
        if tokens.is_empty() || tokens.len() > rows {
            return Err(Error::Parameter(format!(
                "verify takes 1..={rows} tokens, got {}",
                tokens.len()
            )));
        }
        if !state.valid {
            return Err(Error::Cache(
                "state was invalidated by a failed forward; reset it".into(),
            ));
        }
        if state.len + tokens.len() > state.capacity {
            return Err(Error::Cache("verify rows exceed the context".into()));
        }
        if let Some(&id) = tokens.iter().find(|&&id| id as usize >= self.config.vocab) {
            return Err(Error::Token {
                id,
                vocab: self.config.vocab,
            });
        }
        if let RowOutput::Candidates(p) = &output
            && p.len() != tokens.len()
        {
            return Err(Error::Parameter("one penalty set per verify row".into()));
        }
        if parents.is_some() && self.config.variant == Variant::Lfm3 {
            return Err(Error::Parameter(
                "draft trees are not supported for this model".into(),
            ));
        }
        state.verified = 0;
        state.verified_tree.clear();
        // The recurrent state is never written, so a failure leaves it valid.
        let produced = self.forward_chunk(d, state, tokens, &Output::None, None, true, parents);
        let produced = match produced {
            Ok(_) => self.verify_outputs(d, tokens.len(), &output)?,
            Err(e) => return Err(e),
        };
        state.verified = tokens.len();
        if let Some(p) = parents {
            state.verified_tree = p.to_vec();
        }
        Ok(produced)
    }

    /// Keep the first `n` rows of the last verify: replay them through the
    /// recurrent layers and advance the sequence by `n`.
    pub fn commit(&self, d: &MetalDevice, state: &mut HybridState, n: usize) -> Result<()> {
        if n == 0 || n > state.verified {
            return Err(Error::Cache(format!(
                "commit of {n} rows after a verify of {}",
                state.verified
            )));
        }
        if !state.verified_tree.is_empty()
            && !(0..n).all(|i| i == 0 || state.verified_tree[i] == Some(i - 1))
        {
            return Err(Error::Cache(
                "the first rows of the last tree verify are not a path; use commit_path".into(),
            ));
        }
        let spec = self.spec.borrow();
        let spec = spec.as_ref().expect("verify implies speculation");
        let s = self.scratch.borrow();
        state.valid = false;
        let e = d.execution_with_shared_encoder(true)?;
        self.replay(d, &s, spec, state, n, None)?;
        e.finish()?;
        state.len += n;
        state.verified = 0;
        state.verified_tree.clear();
        state.valid = true;
        Ok(())
    }

    /// Keep the root-to-node `path` (verify rows, starting at the anchor's 0)
    /// of the last tree verify: its K/V rows, recurrent replay and captured
    /// features move to the next `path.len()` positions, and the sequence
    /// advances by that much.
    pub fn commit_path(
        &self,
        d: &MetalDevice,
        state: &mut HybridState,
        path: &[usize],
    ) -> Result<()> {
        let n = path.len();
        if path.first() != Some(&0) || n > state.verified {
            return Err(Error::Cache(format!(
                "commit of path {path:?} after a verify of {} rows",
                state.verified
            )));
        }
        if path.iter().enumerate().all(|(i, &row)| i == row) {
            return self.commit(d, state, n);
        }
        let tree = &state.verified_tree;
        if tree.is_empty()
            || path
                .windows(2)
                .any(|w| tree.get(w[1]).copied().flatten() != Some(w[0]))
        {
            return Err(Error::Cache(format!(
                "path {path:?} does not follow the last verify's tree"
            )));
        }
        let c = &self.config;
        let spec = self.spec.borrow();
        let spec = spec.as_ref().expect("verify implies speculation");
        let s = self.scratch.borrow();
        let index = |v: &[usize]| {
            Tensor::from_le_bytes(
                d,
                [v.len()],
                DType::F32,
                &v.iter()
                    .flat_map(|&i| (i as u32).to_le_bytes())
                    .collect::<Vec<_>>(),
            )
        };
        // Rows already in place (the path's chain prefix) stay where they are.
        let first = path
            .iter()
            .enumerate()
            .take_while(|&(i, &row)| i == row)
            .count();
        let from = index(&path[first..])?;
        let to = index(&(first..n).collect::<Vec<_>>())?;
        let path_t = index(path)?;
        state.valid = false;
        let e = d.execution_with_shared_encoder(true)?;
        self.replay(d, &s, spec, state, n, Some(&path_t))?;
        let moved = n - first;
        for layer in 0..c.layers {
            let (Some(kc), Some(vc)) = (&state.k[layer], &state.v[layer]) else {
                continue;
            };
            for cache in [kc, vc] {
                let args = Params::default()
                    .u(moved)?
                    .u(c.kv_heads * c.head_dim)?
                    .u(cache.shape().dimensions()[0])?
                    .u(usize::from(c.kv_cache == KvCache::Q8))?
                    .u(state.ring[layer])?
                    .u(state.len)?;
                let units = moved * c.kv_heads * c.head_dim;
                d.dispatch_hybrid(
                    "h_kv_gather",
                    &[cache.binding(), from.binding()],
                    &[spec.gather.binding()],
                    &args.0,
                    [units.div_ceil(256), 1, 1],
                    [256, 1, 1],
                    0,
                )?;
                d.dispatch_hybrid(
                    "h_kv_scatter",
                    &[spec.gather.binding(), to.binding()],
                    &[cache.binding()],
                    &args.0,
                    [units.div_ceil(256), 1, 1],
                    [256, 1, 1],
                    0,
                )?;
            }
        }
        // Captured rows the drafter ingests next: rows 0..n become the path's.
        for captured in spec.features.iter().chain(&spec.hidden) {
            let width = captured.numel() / captured.shape().dimensions()[0];
            gather_rows(d, captured, &path_t, &spec.gather, n, width, width)?;
            d.dispatch_hybrid(
                "h_copy_rows",
                &[spec.gather.binding()],
                &[captured.binding()],
                &Params::default()
                    .u(n)?
                    .u(width)?
                    .u(width)?
                    .u(width)?
                    .u(0)?
                    .0,
                [(n * width).div_ceil(256), 1, 1],
                [256, 1, 1],
                0,
            )?;
        }
        e.finish()?;
        state.len += n;
        state.verified = 0;
        state.verified_tree.clear();
        state.valid = true;
        Ok(())
    }

    /// Replay `n` verify rows into the recurrent state: tape rows `0..n`, or
    /// the tape rows `path` lists, gathered into the chunk scratch.
    fn replay(
        &self,
        d: &MetalDevice,
        s: &Scratch,
        spec: &Spec,
        state: &HybridState,
        n: usize,
        path: Option<&Tensor>,
    ) -> Result<()> {
        let c = &self.config;
        let channels = c.conv_channels();
        let heads = c.ssm_v_heads;
        for (layer, index) in spec.recurrent_index.iter().enumerate() {
            let Some(r) = *index else { continue };
            if matches!(&self.weights.layers[layer].mixer, Mixer::ShortConv(_)) {
                let hidden = c.hidden;
                let bx = match path {
                    None => rows(&spec.tape_qkv[r], 0, n, hidden)?,
                    Some(path) => {
                        let bx = rows(&s.conv, 0, n, hidden)?;
                        gather_rows(d, &spec.tape_qkv[r], path, &bx, n, hidden, hidden)?;
                        bx
                    }
                };
                bx_state(d, c, &bx, state, layer, n)?;
                continue;
            }
            let Mixer::Delta(w) = &self.weights.layers[layer].mixer else {
                unreachable!("recurrent index on an attention layer")
            };
            let (qkv, g, b) = match path {
                None => (
                    rows(&spec.tape_qkv[r], 0, n, channels)?,
                    rows(&spec.tape_g[r], 0, n, heads)?,
                    rows(&spec.tape_b[r], 0, n, heads)?,
                ),
                Some(path) => {
                    let (qkv, g, b) = (
                        rows(&s.qkv, 0, n, channels)?,
                        rows(&s.g, 0, n, heads)?,
                        rows(&s.b, 0, n, heads)?,
                    );
                    gather_rows(d, &spec.tape_qkv[r], path, &qkv, n, channels, channels)?;
                    gather_rows(d, &spec.tape_g[r], path, &g, n, heads, heads)?;
                    gather_rows(d, &spec.tape_b[r], path, &b, n, heads, heads)?;
                    (qkv, g, b)
                }
            };
            let conv = rows(&s.conv, 0, n, channels)?;
            let delta = rows(&s.delta, 0, n, c.ssm_value_dim())?;
            self.gdn_conv(d, &qkv, w, &conv, state, layer, n, true, None)?;
            self.gdn_recurrent(d, &conv, &g, &b, &delta, state, layer, n, true, None)?;
        }
        Ok(())
    }

    fn verify_outputs(
        &self,
        d: &MetalDevice,
        m: usize,
        output: &RowOutput,
    ) -> Result<RowsProduced> {
        let c = &self.config;
        let spec = self.spec.borrow();
        let spec = spec.as_ref().expect("verify implies speculation");
        let s = self.scratch.borrow();
        let xn = rows(&s.xn, 0, m, c.hidden)?;
        let logits = rows(&spec.logits, 0, m, c.vocab)?;
        let e = d.execution_with_shared_encoder(true)?;
        Self::project(d, &self.weights.output, &xn, m, &logits)?;
        match output {
            RowOutput::Logits => {
                e.finish()?;
                Ok(RowsProduced::Logits(
                    logits
                        .to_f32()
                        .chunks(c.vocab)
                        .map(<[f32]>::to_vec)
                        .collect(),
                ))
            }
            RowOutput::Argmax => {
                d.dispatch_hybrid(
                    "h_argmax",
                    &[logits.binding()],
                    &[spec.tokens.binding()],
                    &Params::default().u(c.vocab)?.0,
                    [m, 1, 1],
                    [1024, 1, 1],
                    0,
                )?;
                e.finish()?;
                Ok(RowsProduced::Tokens(
                    spec.tokens.to_f32()[..m]
                        .iter()
                        .map(|v| v.to_bits())
                        .collect(),
                ))
            }
            RowOutput::Candidates(penalties) => {
                let mut ids = Vec::new();
                let mut row_ids = Vec::new();
                let mut counts = Vec::new();
                let mut settings = None;
                for (row, p) in penalties.iter().enumerate() {
                    if !p.active() {
                        continue;
                    }
                    settings = Some((p.presence, p.frequency, p.repetition));
                    for &(t, n) in &p.tokens {
                        ids.push(t);
                        row_ids.push(row as u32);
                        counts.push(n as f32);
                    }
                }
                if let Some((presence, frequency, repetition)) = settings {
                    let bytes =
                        |v: &[u32]| v.iter().flat_map(|t| t.to_le_bytes()).collect::<Vec<_>>();
                    let ids_t = Tensor::from_le_bytes(d, [ids.len()], DType::F32, &bytes(&ids))?;
                    let rows_t =
                        Tensor::from_le_bytes(d, [ids.len()], DType::F32, &bytes(&row_ids))?;
                    let counts_t = Tensor::from_f32(d, [ids.len()], DType::F32, &counts)?;
                    d.dispatch_hybrid(
                        "h_penalize_rows",
                        &[ids_t.binding(), rows_t.binding(), counts_t.binding()],
                        &[logits.binding()],
                        &Params::default()
                            .u(ids.len())?
                            .u(c.vocab)?
                            .f(presence)
                            .f(frequency)
                            .f(repetition)
                            .0,
                        [ids.len().div_ceil(64), 1, 1],
                        [64, 1, 1],
                        0,
                    )?;
                }
                let blocks = c.vocab.div_ceil(1024);
                d.dispatch_hybrid(
                    "h_topk_blocks",
                    &[logits.binding()],
                    &[spec.candidates.binding()],
                    &Params::default().u(c.vocab)?.0,
                    [blocks, m, 1],
                    [256, 1, 1],
                    0,
                )?;
                e.finish()?;
                let raw = spec.candidates.to_f32();
                Ok(RowsProduced::Candidates(
                    raw.chunks(blocks * 64)
                        .take(m)
                        .map(|row| {
                            row.chunks_exact(2)
                                .filter(|pair| pair[0].is_finite())
                                .map(|pair| (pair[1].to_bits(), pair[0]))
                                .collect()
                        })
                        .collect(),
                ))
            }
        }
    }

    /// Copy `m` rows of a layer output into its column block of the features.
    pub(crate) fn capture(
        &self,
        d: &MetalDevice,
        src: &Tensor,
        dst: &Tensor,
        column: usize,
        m: usize,
    ) -> Result<()> {
        let h = self.config.hidden;
        let width = dst.numel() / dst.shape().dimensions()[0];
        d.dispatch_hybrid(
            "h_copy_rows",
            &[src.binding()],
            &[dst.binding()],
            &Params::default()
                .u(m)?
                .u(h)?
                .u(h)?
                .u(width)?
                .u(column * h)?
                .0,
            [(m * h).div_ceil(256), 1, 1],
            [256, 1, 1],
            0,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_chunk(
        &self,
        d: &MetalDevice,
        state: &mut HybridState,
        tokens: &[u32],
        output: &Output,
        all_logits: Option<&mut Vec<f32>>,
        verify: bool,
        tree: Option<&[Option<usize>]>,
    ) -> Result<Produced> {
        let c = &self.config;
        let s = self.scratch.borrow();
        let m = tokens.len();
        let pos = state.len;
        let h = c.hidden;
        if m > state.max_rows && state.ring.iter().any(|&r| r > 0) {
            return Err(Error::Cache(format!(
                "a forward of {m} rows exceeds the {} the K/V rings were sized for",
                state.max_rows
            )));
        }
        state.written = state.written.max(pos + m);
        let execution = d.execution_with_shared_encoder(true)?;
        if pos == 0 {
            for t in state.conv.iter().chain(&state.ssm).flatten() {
                self.fill_zero(d, t)?;
            }
        }
        let ids = Tensor::from_le_bytes(
            d,
            [m],
            DType::F32,
            &tokens
                .iter()
                .flat_map(|t| t.to_le_bytes())
                .collect::<Vec<_>>(),
        )?;
        let x0 = rows(&s.x[0], 0, m, h)?;
        let x1 = rows(&s.x[1], 0, m, h)?;
        let xn = rows(&s.xn, 0, m, h)?;
        let mix = rows(&s.mix, 0, m, h)?;
        Self::get_rows(d, &self.weights.embedding, &ids, &x0)?;
        // Image rows take the vision encoder's embeddings in place of the pad token's.
        for span in state
            .media
            .iter()
            .filter(|sp| sp.start < pos + m && sp.end() > pos)
        {
            if span.embedding.numel() != span.len() * h {
                return Err(Error::Shape(format!(
                    "image embedding has {} values, expected {} x {h}",
                    span.embedding.numel(),
                    span.len()
                )));
            }
            let (from, to) = (span.start.max(pos), span.end().min(pos + m));
            let src = rows(&span.embedding, from - span.start, to - from, h)?;
            let dst = rows(&x0, from - pos, to - from, h)?;
            d.dispatch_hybrid(
                "h_copy_rows",
                &[src.binding()],
                &[dst.binding()],
                &Params::default().u(to - from)?.u(h)?.u(h)?.u(h)?.u(0)?.0,
                [((to - from) * h).div_ceil(256), 1, 1],
                [256, 1, 1],
                0,
            )?;
        }
        let rope = match tree {
            Some(parents) => state.tree_plan(d, pos, parents)?,
            None => state.rope_plan(d, pos, m)?,
        };
        self.rmsnorm(d, &x0, &self.weights.layers[0].attn_norm, &xn, m)?;
        let spec = self.spec.borrow();
        let spec = spec.as_ref();
        let (mut cur, mut next) = (&x0, &x1);
        for (i, layer) in self.weights.layers.iter().enumerate() {
            match &layer.mixer {
                Mixer::Attention(a) => {
                    self.attention(d, &s, a, state, i, &xn, &mix, m, pos, &rope)?
                }
                Mixer::ShortConv(w) => {
                    let tape = match (verify, spec) {
                        (true, Some(sp)) => {
                            let r = sp.recurrent_index[i].expect("short conv is recurrent");
                            Some(&sp.tape_qkv[r])
                        }
                        _ => None,
                    };
                    run_short_conv(d, c, &s, w, state, i, &xn, &mix, m, tape)?;
                }
                Mixer::Delta(w) => {
                    let tape = match (verify, spec) {
                        (true, Some(sp)) => {
                            let r = sp.recurrent_index[i].expect("delta layer is recurrent");
                            Some((&sp.tape_qkv[r], &sp.tape_g[r], &sp.tape_b[r]))
                        }
                        _ => None,
                    };
                    self.delta(d, &s, w, state, i, &xn, &mix, m, tape, rope.tree.as_ref())?
                }
            }
            // LFM3 sandwich: norm the branch, then the fused residual + next
            // pre-norm. The dispatcher rejects an input that aliases an
            // output, so the fused norm lands in `mix` and is copied back to
            // `xn`, which is where the rest of the loop reads it.
            if let Some(post) = &layer.post_attn {
                self.rmsnorm(d, &mix, post, &xn, m)?;
                self.add_rmsnorm(d, cur, &xn, &layer.post_norm, next, &mix, m)?;
                copy(d, &mix, &xn)?;
            } else {
                self.add_rmsnorm(d, cur, &mix, &layer.post_norm, next, &xn, m)?;
            }
            std::mem::swap(&mut cur, &mut next);
            match &layer.ffn {
                Ffn::Dense(f) => self.dense_ffn(d, &s, f, &xn, &mix, m)?,
                Ffn::Moe(f) => {
                    self.dump(d, "attn_post_norm", i, &xn)?;
                    self.moe_ffn(d, &s, f, &xn, &mix, m)?;
                    self.dump(d, "ffn_out", i, &mix)?;
                }
            }
            let norm = self
                .weights
                .layers
                .get(i + 1)
                .map_or(&self.weights.output_norm, |l| &l.attn_norm);
            if let Some(post) = &layer.post_ffw {
                self.rmsnorm(d, &mix, post, &xn, m)?;
                self.add_rmsnorm(d, cur, &xn, norm, next, &mix, m)?;
                copy(d, &mix, &xn)?;
            } else {
                self.add_rmsnorm(d, cur, &mix, norm, next, &xn, m)?;
            }
            std::mem::swap(&mut cur, &mut next);
            if self.add_deepstack(d, state, i, cur, pos, m)? {
                self.rmsnorm(d, cur, norm, &xn, m)?;
            }
            self.dump(d, "l_out", i, cur)?;
            if let Some(sp) = spec
                && let Some(features) = &sp.features
                && let Some(column) = sp.aux.iter().position(|&l| l == i)
            {
                self.capture(d, cur, features, column, m)?;
            }
        }
        if let Some(hidden) = spec.and_then(|sp| sp.hidden.as_ref()) {
            self.capture(d, &xn, hidden, 0, m)?;
        }
        // xn now holds output_norm(final hidden) for every position.
        let produced = if let Some(all) = all_logits {
            let logits = rows(&s.logits, 0, m, c.vocab)?;
            Self::project(d, &self.weights.output, &xn, m, &logits)?;
            execution.finish()?;
            *all = logits.to_f32();
            Produced::None
        } else {
            match output {
                Output::None => {
                    execution.finish()?;
                    Produced::None
                }
                Output::Candidates(penalties) => {
                    let last = rows(&xn, m - 1, 1, h)?;
                    let logits = rows(&s.logits, 0, 1, c.vocab)?;
                    Self::project(d, &self.weights.output, &last, 1, &logits)?;
                    if penalties.active() {
                        let ids = Tensor::from_le_bytes(
                            d,
                            [penalties.tokens.len()],
                            DType::F32,
                            &penalties
                                .tokens
                                .iter()
                                .flat_map(|(t, _)| t.to_le_bytes())
                                .collect::<Vec<_>>(),
                        )?;
                        let counts = Tensor::from_f32(
                            d,
                            [penalties.tokens.len()],
                            DType::F32,
                            &penalties
                                .tokens
                                .iter()
                                .map(|&(_, n)| n as f32)
                                .collect::<Vec<_>>(),
                        )?;
                        d.dispatch_hybrid(
                            "h_penalize",
                            &[ids.binding(), counts.binding()],
                            &[logits.binding()],
                            &Params::default()
                                .u(penalties.tokens.len())?
                                .f(penalties.presence)
                                .f(penalties.frequency)
                                .f(penalties.repetition)
                                .0,
                            [penalties.tokens.len().div_ceil(64), 1, 1],
                            [64, 1, 1],
                            0,
                        )?;
                    }
                    let blocks = c.vocab.div_ceil(1024);
                    d.dispatch_hybrid(
                        "h_topk_blocks",
                        &[logits.binding()],
                        &[s.candidates.binding()],
                        &Params::default().u(c.vocab)?.0,
                        [blocks, 1, 1],
                        [256, 1, 1],
                        0,
                    )?;
                    execution.finish()?;
                    let raw = s.candidates.to_f32();
                    Produced::Candidates(
                        raw[..blocks * 64]
                            .chunks_exact(2)
                            .filter(|pair| pair[0].is_finite())
                            .map(|pair| (pair[1].to_bits(), pair[0]))
                            .collect(),
                    )
                }
                Output::Argmax | Output::Logits => {
                    let last = rows(&xn, m - 1, 1, h)?;
                    let logits = rows(&s.logits, 0, 1, c.vocab)?;
                    Self::project(d, &self.weights.output, &last, 1, &logits)?;
                    if matches!(output, Output::Argmax) {
                        d.dispatch_hybrid(
                            "h_argmax",
                            &[logits.binding()],
                            &[s.token.binding()],
                            &Params::default().u(c.vocab)?.0,
                            [1, 1, 1],
                            [1024, 1, 1],
                            0,
                        )?;
                        execution.finish()?;
                        Produced::Token(s.token.to_f32()[0].to_bits())
                    } else {
                        execution.finish()?;
                        Produced::Logits(logits.to_f32())
                    }
                }
            }
        };
        Ok(produced)
    }

    /// Add layer `layer`'s DeepStack features of the images in this chunk to
    /// their rows of `x`. Whether any were added.
    fn add_deepstack(
        &self,
        d: &MetalDevice,
        state: &HybridState,
        layer: usize,
        x: &Tensor,
        pos: usize,
        m: usize,
    ) -> Result<bool> {
        let h = self.config.hidden;
        let mut added = false;
        for span in state
            .media
            .iter()
            .filter(|sp| sp.start < pos + m && sp.end() > pos)
        {
            let Some(features) = span.deepstack.get(layer) else {
                continue;
            };
            if features.numel() != span.len() * h || features.dtype() != DType::F32 {
                return Err(Error::Shape(format!(
                    "DeepStack features of {} values for an image of {} x {h}",
                    features.numel(),
                    span.len()
                )));
            }
            let (from, to) = (span.start.max(pos), span.end().min(pos + m));
            let n = (to - from) * h;
            d.dispatch_hybrid(
                "h_accumulate",
                &[rows(features, from - span.start, to - from, h)?.binding()],
                &[rows(x, from - pos, to - from, h)?.binding()],
                &Params::default().u(n)?.0,
                [n.div_ceil(256), 1, 1],
                [256, 1, 1],
                0,
            )?;
            added = true;
        }
        Ok(added)
    }

    #[allow(clippy::too_many_arguments)]
    fn attention(
        &self,
        d: &MetalDevice,
        s: &Scratch,
        a: &AttentionWeights,
        state: &HybridState,
        layer: usize,
        xn: &Tensor,
        out: &Tensor,
        m: usize,
        pos: usize,
        rope: &RopePlan,
    ) -> Result<()> {
        let kc = state.k[layer]
            .as_ref()
            .expect("attention layer has K cache");
        let vc = state.v[layer]
            .as_ref()
            .expect("attention layer has V cache");
        self.attention_with(
            d,
            s,
            a,
            kc,
            vc,
            state.ring[layer],
            layer,
            xn,
            out,
            m,
            pos,
            rope,
        )
    }

    /// Attention of `m` rows at `pos..` against explicit K/V caches, which
    /// are rings of `ring` rows (see `ring_rows`) or linear (`ring` = 0).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn attention_with(
        &self,
        d: &MetalDevice,
        s: &Scratch,
        a: &AttentionWeights,
        kc: &Tensor,
        vc: &Tensor,
        ring: usize,
        layer: usize,
        xn: &Tensor,
        out: &Tensor,
        m: usize,
        pos: usize,
        rope: &RopePlan,
    ) -> Result<()> {
        let c = &self.config;
        let spec = attn_spec(c, layer);
        let q_width = c.heads * c.head_dim;
        let kv_width = c.kv_heads * c.head_dim;
        let qg = rows(&s.qg, 0, m, if spec.gated { 2 * q_width } else { q_width })?;
        let k = rows(&s.k, 0, m, kv_width)?;
        let v = rows(&s.v, 0, m, kv_width)?;
        let q = s.q.view(0, [m, q_width])?;
        let att = rows(&s.att, 0, m, q_width)?;
        Self::project(d, &a.q, xn, m, &qg)?;
        Self::project(d, &a.k, xn, m, &k)?;
        Self::project(d, &a.v, xn, m, &v)?;
        let (rope_dims, rope_theta) = rope_of(c, spec);
        let prep = Params::default()
            .u(m)?
            .u(c.heads)?
            .u(c.kv_heads)?
            .u(c.head_dim)?
            .u(rope_dims)?
            .u(pos)?
            .f(rope_theta)
            .f(c.eps)
            .u(kv_width)?
            .f(1. / (c.head_dim as f32).sqrt())
            .i(rope.delta)
            .u(usize::from(rope.table.is_some()))?
            .u(c.rope_sections[1])?
            .u(c.rope_sections[2])?
            .u(kc.shape().dimensions()[0])?
            .u(usize::from(spec.gated))?
            .u(ring)?
            .u(usize::from(a.q_norm.is_some()))?
            // Full-attention LFM3 layers divide the angle by rope_freqs.
            // Sliding layers, and every other variant, leave the formula as it is.
            .u(usize::from(
                c.variant == Variant::Lfm3
                    && spec.window.is_none()
                    && self.weights.rope_freqs.is_some(),
            ))?;
        let q8 = c.kv_cache == KvCache::Q8;
        d.dispatch_hybrid(
            if q8 { "h_attn_prep_q8" } else { "h_attn_prep" },
            &[
                qg.binding(),
                k.binding(),
                v.binding(),
                // Unread without QK norms.
                a.q_norm.as_ref().unwrap_or(&qg).binding(),
                a.k_norm.as_ref().unwrap_or(&qg).binding(),
                // Unread unless the plan carries a position table.
                rope.table.as_ref().unwrap_or(&qg).binding(),
                // Unread unless use_factors is set.
                self.weights.rope_freqs.as_ref().unwrap_or(&qg).binding(),
            ],
            &[q.binding(), kc.binding()],
            &prep.0,
            [c.heads + c.kv_heads, m, 1],
            [c.head_dim, 1, 1],
            0,
        )?;
        d.dispatch_hybrid(
            if q8 { "h_store_v_q8" } else { "h_store_v" },
            &[v.binding()],
            &[vc.binding()],
            &prep.0,
            [(m * kv_width).div_ceil(256), 1, 1],
            [256, 1, 1],
            0,
        )?;
        self.dump(d, "Qcur_full", layer, &qg)?;
        self.dump(d, "Kcur", layer, &k)?;
        self.dump(d, "Vcur", layer, &v)?;
        self.flash_attention(
            d,
            s,
            c.kv_cache,
            &q,
            kc,
            vc,
            &qg,
            &att,
            m,
            pos,
            spec,
            a.sinks.as_ref(),
            ring,
            rope.tree.as_ref().map(|t| (t, rope.tree_root)),
        )?;
        if let Some(gate) = &a.head_gate {
            let logits = rows(&s.head_gate, 0, m, c.heads)?;
            Self::project(d, gate, xn, m, &logits)?;
            d.dispatch_hybrid(
                "h_head_gate",
                &[logits.binding()],
                &[att.binding()],
                &Params::default().u(m * q_width)?.u(c.head_dim)?.0,
                [(m * q_width).div_ceil(256), 1, 1],
                [256, 1, 1],
                0,
            )?;
        }
        self.dump(d, "attn_gated", layer, &att)?;
        Self::project(d, &a.o, &att, m, out)?;
        self.dump(d, "attn_output", layer, out)
    }

    /// Causal attention of `m` new queries against the cache (format `kv`),
    /// written to `att`. With `spec.gated` the output is multiplied by
    /// sigmoid(gate) from `qg`; `spec.window` masks keys out of range, and
    /// `sinks` (per-head logits) join the softmax denominator. With `tree`
    /// (`RopePlan::tree`) the rows are draft-tree nodes that see only their
    /// ancestors among the tree's rows, stored from cache position `root`
    /// (the second member) on.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn flash_attention(
        &self,
        d: &MetalDevice,
        s: &Scratch,
        kv: KvCache,
        q: &Tensor,
        kc: &Tensor,
        vc: &Tensor,
        qg: &Tensor,
        att: &Tensor,
        m: usize,
        pos: usize,
        spec: AttnSpec,
        sinks: Option<&Tensor>,
        ring: usize,
        tree: Option<(&Tensor, usize)>,
    ) -> Result<()> {
        let c = &self.config;
        let group = c.heads / c.kv_heads;
        let (tree, tree_root) = (tree.map(|t| t.0), tree.map_or(pos, |t| t.1));
        let rows = m * group;
        let window = spec.window.unwrap_or(0);
        // No query reads keys before the window of the first query (of the
        // tree's root, which sits lowest). A ring starts at a multiple of 64
        // so tensor-op tiles never wrap.
        let first_key = spec
            .window
            .map_or(0, |w| (pos.min(tree_root) + 1).saturating_sub(w));
        let first_key = if ring > 0 {
            first_key / 64 * 64
        } else {
            first_key
        };
        let keys = pos + m - first_key;
        let kv_rows = kc.shape().dimensions()[0];
        let q8 = kv == KvCache::Q8;
        let wide = c.head_dim == 256;
        let legacy = flags().legacy_attention || flags().simd_attention;
        if q8 && (flags().reference_attention || legacy) {
            return Err(Error::Config(
                "the reference and legacy attention kernels read only an F16 K/V cache".into(),
            ));
        }
        if legacy && (!wide || !spec.gated || window != 0 || sinks.is_some() || ring != 0) {
            return Err(Error::Config(
                "the legacy attention kernels cover gated, full-window, sink-free 256-wide heads"
                    .into(),
            ));
        }
        // A one-row forward with a tree is a node whose ancestors earlier
        // forwards stored; it runs on the verify kernels, not decode.
        if tree.is_some() && (legacy || m > TREE_MAX_ROWS) {
            return Err(Error::Config(format!(
                "a draft-tree verify takes at most {TREE_MAX_ROWS} rows on the TensorOps \
                 or reference attention kernels"
            )));
        }
        let sinks_on = usize::from(sinks.is_some());
        let tree_on = usize::from(tree.is_some());
        // Bound (and ignored) when the layer has no sinks or the rows no tree.
        let sinks_t = sinks.unwrap_or(qg);
        let tree_t = tree.unwrap_or(qg);
        if flags().reference_attention {
            let args = Params::default()
                .u(m)?
                .u(c.heads)?
                .u(c.kv_heads)?
                .u(c.head_dim)?
                .u(pos)?
                .u(c.kv_heads * c.head_dim)?
                .f(1.)
                .u(window)?
                .u(usize::from(spec.gated))?
                .u(sinks_on)?
                .u(ring)?
                .u(tree_on)?;
            return d.dispatch_hybrid(
                "h_attention",
                &[
                    q.binding(),
                    kc.binding(),
                    vc.binding(),
                    qg.binding(),
                    sinks_t.binding(),
                    tree_t.binding(),
                ],
                &[att.binding()],
                &args.0,
                [c.heads, m, 1],
                [c.head_dim, 1, 1],
                0,
            );
        }
        let fa = |keys_per_split: usize, splits: usize| -> Result<Vec<u8>> {
            Ok(Params::default()
                .u(m)?
                .u(c.heads)?
                .u(c.kv_heads)?
                .u(group)?
                .u(pos)?
                .u(c.kv_heads * c.head_dim)?
                .u(keys_per_split)?
                .u(splits)?
                .u(kv_rows)?
                .u(c.head_dim)?
                .u(first_key)?
                .u(window)?
                .u(usize::from(spec.gated))?
                .u(sinks_on)?
                .u(ring)?
                .0)
        };
        // Sinks ride as the fifth input; the legacy kernels predate them. The
        // `_tree` variants take the tree as the sixth.
        let base = [q.binding(), kc.binding(), vc.binding(), qg.binding()];
        let sink_inputs = [base[0], base[1], base[2], base[3], sinks_t.binding()];
        let tree_inputs = [
            base[0],
            base[1],
            base[2],
            base[3],
            sinks_t.binding(),
            tree_t.binding(),
        ];
        if m >= MPP_ATTENTION_MIN_TOKENS && !flags().simd_attention {
            let args = fa(0, 1)?;
            let name = match (q8, legacy, wide, tree.is_some()) {
                (true, _, true, false) => "h_flash_attn_mpp_q8",
                (true, _, false, false) => "h_flash_attn_mpp_q8_d128",
                (false, false, true, false) => "h_flash_attn_mpp_staged",
                (false, false, false, false) => "h_flash_attn_mpp_staged_d128",
                (false, true, _, _) => "h_flash_attn_mpp",
                (true, _, true, true) => "h_flash_attn_mpp_q8_tree",
                (true, _, false, true) => "h_flash_attn_mpp_q8_d128_tree",
                (false, false, true, true) => "h_flash_attn_mpp_staged_tree",
                (false, false, false, true) => "h_flash_attn_mpp_staged_d128_tree",
            };
            // K/V staged through threadgroup memory measured 15-20% faster
            // than reading the cache as a device tensor.
            return d.dispatch_hybrid(
                name,
                if legacy {
                    &base
                } else if tree.is_some() {
                    &tree_inputs
                } else {
                    &sink_inputs
                },
                &[att.binding()],
                &args,
                [m.div_ceil(64), c.heads, 1],
                [128, 1, 1],
                64 * 64 * 4 + 64 * 64 * 2 + 3 * 64 * 4,
            );
        }
        let reduce = |args: &[u8]| {
            d.dispatch_hybrid(
                if wide {
                    "h_flash_reduce"
                } else {
                    "h_flash_reduce_d128"
                },
                &[
                    s.attn_partial.binding(),
                    s.attn_ml.binding(),
                    qg.binding(),
                    sinks_t.binding(),
                ],
                &[att.binding()],
                args,
                [m * c.heads, 1, 1],
                [c.head_dim, 1, 1],
                0,
            )
        };
        if m == 1 && group <= 8 && !flags().simd_attention && tree.is_none() {
            let target = DECODE_SPLIT_TARGET;
            let min_keys = DECODE_SPLIT_MIN_KEYS;
            let splits = (target / c.kv_heads)
                .max(1)
                .min(keys.div_ceil(min_keys))
                .clamp(1, MAX_SPLITS);
            let keys_per_split = keys.div_ceil(splits).next_multiple_of(32);
            let splits = keys.div_ceil(keys_per_split);
            let args = fa(keys_per_split, splits)?;
            let name = match (q8, wide) {
                (false, true) => "h_attn_decode",
                (true, true) => "h_attn_decode_q8",
                (false, false) => "h_attn_decode_d128",
                (true, false) => "h_attn_decode_q8_d128",
            };
            d.dispatch_hybrid(
                name,
                &sink_inputs,
                &[att.binding(), s.attn_partial.binding(), s.attn_ml.binding()],
                &args,
                [c.kv_heads, splits, 1],
                [128, 1, 1],
                8 * c.head_dim * 2 + 4 * 8 * 2 * 4 + 2 * 8 * c.head_dim * 4,
            )?;
            if splits > 1 {
                reduce(&args)?;
            }
            return Ok(());
        }
        if !legacy {
            // Speculative verify widths on TensorOps: queries regrouped so each
            // KV head's (token, head) rows form one matrix, taken 32 rows per
            // threadgroup, with keys split across threadgroups to fill the GPU.
            let qk = s.q_grouped.view(0, [m, c.heads * c.head_dim])?;
            let args = fa(0, 1)?;
            d.dispatch_hybrid(
                "h_q_by_kv_head",
                &[q.binding()],
                &[qk.binding()],
                &args,
                [(m * c.heads * c.head_dim).div_ceil(256), 1, 1],
                [256, 1, 1],
                0,
            )?;
            let br = 32;
            let row_blocks = rows.div_ceil(br);
            let capacity = (MAX_SPLITS * MAX_SPLIT_ROWS / (m * c.heads)).max(1);
            let splits = (VERIFY_SPLIT_TARGET / (row_blocks * c.kv_heads))
                .max(1)
                .min(keys.div_ceil(SPLIT_MIN_KEYS))
                .min(capacity)
                .clamp(1, MAX_SPLITS);
            let keys_per_split = keys.div_ceil(splits).next_multiple_of(64);
            let splits = keys.div_ceil(keys_per_split);
            let args = fa(keys_per_split, splits)?;
            let shared = if q8 {
                (64 * 128 * 2).max(br * 64 * 4) + br * 64 * 2 + 3 * br * 4
            } else {
                br * 64 * 4 + br * 64 * 2 + 3 * br * 4
            };
            let name = match (q8, wide, tree.is_some()) {
                (false, true, false) => "h_attn_verify_mpp_32",
                (true, true, false) => "h_attn_verify_mpp_32_q8",
                (false, false, false) => "h_attn_verify_mpp_32_d128",
                (true, false, false) => "h_attn_verify_mpp_32_q8_d128",
                (false, true, true) => "h_attn_verify_mpp_32_tree",
                (true, true, true) => "h_attn_verify_mpp_32_q8_tree",
                (false, false, true) => "h_attn_verify_mpp_32_d128_tree",
                (true, false, true) => "h_attn_verify_mpp_32_q8_d128_tree",
            };
            let qk_inputs = [
                qk.binding(),
                kc.binding(),
                vc.binding(),
                qg.binding(),
                sinks_t.binding(),
                tree_t.binding(),
            ];
            d.dispatch_hybrid(
                // F16 reads K/V directly: staging only pays off for prompt
                // chunks. Q8 must stage to dequantize.
                name,
                &qk_inputs[..if tree.is_some() { 6 } else { 5 }],
                &[att.binding(), s.attn_partial.binding(), s.attn_ml.binding()],
                &args,
                [row_blocks, c.kv_heads, splits],
                [128, 1, 1],
                shared,
            )?;
            if splits > 1 {
                reduce(&args)?;
            }
            return Ok(());
        }
        let simds = if rows <= 8 { 1 } else { 4 };
        let row_blocks = rows.div_ceil(8 * simds);
        // Split long key ranges across threadgroups when few query rows exist.
        let splits = if m * c.heads <= MAX_SPLIT_ROWS {
            let target = SPLIT_TARGET_GROUPS;
            let min_keys = SPLIT_MIN_KEYS;
            let wanted = (target / (row_blocks * c.kv_heads)).max(1);
            wanted.min(keys.div_ceil(min_keys)).clamp(1, MAX_SPLITS)
        } else {
            1
        };
        let keys_per_split = keys.div_ceil(splits).next_multiple_of(32);
        let splits = keys.div_ceil(keys_per_split);
        let args = fa(keys_per_split, splits)?;
        d.dispatch_hybrid(
            if simds == 1 {
                "h_flash_attn_1"
            } else {
                "h_flash_attn_4"
            },
            &base,
            &[att.binding(), s.attn_partial.binding(), s.attn_ml.binding()],
            &args,
            [row_blocks, c.kv_heads, splits],
            [32 * simds, 1, 1],
            simds * 5888,
        )?;
        if splits > 1 {
            reduce(&args)?;
        }
        Ok(())
    }

    /// Gated DeltaNet mixer. With `tape` (speculative verify) the pre-conv
    /// qkv, g and beta rows are written to the tape and the recurrent state
    /// is only read.
    #[allow(clippy::too_many_arguments)]
    fn delta(
        &self,
        d: &MetalDevice,
        s: &Scratch,
        w: &DeltaWeights,
        state: &HybridState,
        layer: usize,
        xn: &Tensor,
        out: &Tensor,
        m: usize,
        tape: Option<(&Tensor, &Tensor, &Tensor)>,
        tree: Option<&Tensor>,
    ) -> Result<()> {
        let c = &self.config;
        let channels = c.conv_channels();
        let vd = c.ssm_value_dim();
        let heads = c.ssm_v_heads;
        let (qkv, g, b) = match tape {
            Some((tq, tg, tb)) => (
                rows(tq, 0, m, channels)?,
                rows(tg, 0, m, heads)?,
                rows(tb, 0, m, heads)?,
            ),
            None => (
                rows(&s.qkv, 0, m, channels)?,
                rows(&s.g, 0, m, heads)?,
                rows(&s.b, 0, m, heads)?,
            ),
        };
        let write_state = tape.is_none();
        let z = rows(&s.z, 0, m, vd)?;
        let alpha = rows(&s.alpha, 0, m, heads)?;
        let beta = rows(&s.beta, 0, m, heads)?;
        let conv = rows(&s.conv, 0, m, channels)?;
        let delta = rows(&s.delta, 0, m, vd)?;
        let gated = rows(&s.gated, 0, m, vd)?;
        self.dump(d, "attn_norm", layer, xn)?;
        Self::project(d, &w.qkv, xn, m, &qkv)?;
        Self::project(d, &w.z, xn, m, &z)?;
        Self::project(d, &w.alpha, xn, m, &alpha)?;
        Self::project(d, &w.beta, xn, m, &beta)?;
        self.dump(d, "linear_attn_qkv_mixed", layer, &qkv)?;
        self.dump(d, "z", layer, &z)?;
        self.dump(d, "alpha", layer, &alpha)?;
        self.dump(d, "beta", layer, &beta)?;
        d.dispatch_hybrid(
            "h_gdn_gates",
            &[
                alpha.binding(),
                beta.binding(),
                w.dt_bias.binding(),
                w.a.binding(),
            ],
            &[g.binding(), b.binding()],
            &Params::default().u(heads)?.u(m)?.0,
            [(heads * m).div_ceil(64), 1, 1],
            [64, 1, 1],
            0,
        )?;
        // Tree rows only arise in a verify, which never writes the state.
        let tree = tree.filter(|_| tape.is_some());
        self.gdn_conv(d, &qkv, w, &conv, state, layer, m, write_state, tree)?;
        self.gdn_recurrent(d, &conv, &g, &b, &delta, state, layer, m, write_state, tree)?;
        self.dump(d, "attn_output", layer, &delta)?;
        d.dispatch_hybrid(
            "h_gated_rmsnorm",
            &[delta.binding(), z.binding(), w.norm.binding()],
            &[gated.binding()],
            &Params::default().u(heads)?.u(c.ssm_head_dim)?.f(c.eps).0,
            [m * heads, 1, 1],
            [c.ssm_head_dim, 1, 1],
            0,
        )?;
        self.dump(d, "final_output", layer, &gated)?;
        Self::project(d, &w.out, &gated, m, out)?;
        self.dump(d, "linear_attn_out", layer, out)
    }

    /// Causal conv + SiLU + Q/K L2 norm of `m` qkv rows into `conv`.
    #[allow(clippy::too_many_arguments)]
    fn gdn_conv(
        &self,
        d: &MetalDevice,
        qkv: &Tensor,
        w: &DeltaWeights,
        conv: &Tensor,
        state: &HybridState,
        layer: usize,
        m: usize,
        write_state: bool,
        tree: Option<&Tensor>,
    ) -> Result<()> {
        let c = &self.config;
        let channels = c.conv_channels();
        let conv_state = state.conv[layer]
            .as_ref()
            .expect("delta layer has conv state");
        let params = Params::default()
            .u(channels)?
            .u(m)?
            .u(c.ssm_key_dim())?
            .u(c.ssm_head_dim)?
            .f(c.eps)
            .u(usize::from(write_state))?
            .0;
        let groups = [channels / c.ssm_head_dim, 1, 1];
        let threads = [c.ssm_head_dim, 1, 1];
        match tree {
            // Draft-tree rows read their ancestors' rows; the state is only read.
            Some(tree) => d.dispatch_hybrid(
                "h_gdn_conv_tree",
                &[
                    qkv.binding(),
                    w.conv.binding(),
                    conv_state.binding(),
                    tree.binding(),
                ],
                &[conv.binding()],
                &params,
                groups,
                threads,
                0,
            ),
            None => d.dispatch_hybrid(
                "h_gdn_conv",
                &[qkv.binding(), w.conv.binding()],
                &[conv.binding(), conv_state.binding()],
                &params,
                groups,
                threads,
                0,
            ),
        }
    }

    /// Sequential gated delta rule over `m` conv rows into `delta`.
    #[allow(clippy::too_many_arguments)]
    fn gdn_recurrent(
        &self,
        d: &MetalDevice,
        conv: &Tensor,
        g: &Tensor,
        b: &Tensor,
        delta: &Tensor,
        state: &HybridState,
        layer: usize,
        m: usize,
        write_state: bool,
        tree: Option<&Tensor>,
    ) -> Result<()> {
        let c = &self.config;
        let ssm = state.ssm[layer]
            .as_ref()
            .expect("delta layer has recurrent state");
        const NSG: usize = 4;
        let params = Params::default()
            .u(m)?
            .u(c.ssm_v_heads)?
            .u(c.ssm_k_heads)?
            .u(c.conv_channels())?
            .u(c.ssm_key_dim())?
            .f(1. / (c.ssm_head_dim as f32).sqrt())
            .u(usize::from(write_state))?
            .u(usize::from(c.ssm_grouped_heads))?
            .0;
        let groups = [c.ssm_head_dim / NSG, c.ssm_v_heads, 1];
        match tree {
            // Draft-tree rows start from their parent's state; the state is only read.
            Some(tree) => d.dispatch_hybrid(
                "h_gdn_recurrent_tree",
                &[
                    conv.binding(),
                    g.binding(),
                    b.binding(),
                    ssm.binding(),
                    tree.binding(),
                ],
                &[delta.binding()],
                &params,
                groups,
                [32, NSG, 1],
                0,
            ),
            None => d.dispatch_hybrid(
                "h_gdn_recurrent",
                &[conv.binding(), g.binding(), b.binding()],
                &[delta.binding(), ssm.binding()],
                &params,
                groups,
                [32, NSG, 1],
                0,
            ),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn moe_ffn(
        &self,
        d: &MetalDevice,
        s: &Scratch,
        f: &MoeFfn,
        xn: &Tensor,
        out: &Tensor,
        m: usize,
    ) -> Result<()> {
        let c = &self.config;
        let moe = c.moe.expect("MoE layer has expert geometry");
        let ms = s.moe.as_ref().expect("MoE model has MoE scratch");
        let (experts, used, hidden, ef) = (moe.experts, moe.experts_used, c.hidden, moe.expert_ffn);
        let routes = m * used;
        let logits = rows(&ms.logits, 0, m, experts)?;
        let ids = ms.ids.view(0, [routes])?;
        let weights = ms.weights.view(0, [routes])?;
        let shared_gate = ms.shared_gate.view(0, [m])?;
        // Without a shared-expert gate tensor the kernel never reads this buffer.
        let shared_w = f.shared_gate_inp.as_ref().unwrap_or(xn);
        let shared = rows(&ms.shared, 0, m, hidden)?;
        let gate = rows(&ms.gate, 0, routes, ef)?;
        let up = rows(&ms.up, 0, routes, ef)?;
        let act = rows(&ms.act, 0, routes, ef)?;
        let down = rows(&ms.down, 0, routes, hidden)?;
        Self::project(d, &f.router, xn, m, &logits)?;
        d.dispatch_hybrid(
            "h_moe_route",
            &[logits.binding(), xn.binding(), shared_w.binding()],
            &[ids.binding(), weights.binding(), shared_gate.binding()],
            &Params::default()
                .u(experts)?
                .u(used)?
                .u(hidden)?
                .u(m)?
                .u(usize::from(moe.sigmoid))?
                .u(usize::from(moe.norm_weights))?
                .u(usize::from(moe.shared_gate))?
                .0,
            [m, 1, 1],
            [experts, 1, 1],
            0,
        )?;
        self.dense_ffn(d, s, &f.shared, xn, &shared, m)?;
        if flags().dump {
            d.synchronize()?;
            let picked: Vec<u32> = ids.to_f32().iter().map(|v| v.to_bits()).collect();
            eprintln!(
                "dump ffn_moe_topk: sum {} ids {:?}",
                picked.iter().map(|&v| v as u64).sum::<u64>(),
                picked
            );
        }
        self.dump(d, "ffn_moe_logits", 0, &logits)?;
        self.dump(d, "ffn_moe_weights", 0, &weights)?;
        self.dump(d, "ffn_shexp", 0, &shared)?;
        self.dump(d, "shared_expert_gate_sigmoid", 0, &shared_gate)?;
        let identity = routes <= MV_ID_MAX_ROUTES || exact_math();
        if identity {
            Self::project_id(d, &f.gate_exps, xn, &ids, routes, used, &gate)?;
            Self::project_id(d, &f.up_exps, xn, &ids, routes, used, &up)?;
            self.glu(d, &gate, &up, &act)?;
            Self::project_id(d, &f.down_exps, &act, &ids, routes, 1, &down)?;
        } else {
            let offsets = ms.offsets.view(0, [experts + 1])?;
            let sorted = ms.sorted.view(0, [routes])?;
            let position = ms.position.view(0, [routes])?;
            let xg = rows(&ms.xg, 0, routes, hidden)?;
            d.dispatch_hybrid(
                "h_moe_map",
                &[ids.binding()],
                &[offsets.binding(), sorted.binding(), position.binding()],
                &Params::default().u(experts)?.u(routes)?.0,
                [1, 1, 1],
                [experts.min(1024), 1, 1],
                0,
            )?;
            d.dispatch_hybrid(
                "h_moe_gather",
                &[xn.binding(), sorted.binding()],
                &[xg.binding()],
                &Params::default().u(hidden)?.u(used)?.u(routes)?.0,
                [hidden.div_ceil(256), routes, 1],
                [256, 1, 1],
                0,
            )?;
            self.project_mm_id(d, &f.gate_exps, &xg, &offsets, m, &gate)?;
            self.project_mm_id(d, &f.up_exps, &xg, &offsets, m, &up)?;
            self.glu(d, &gate, &up, &act)?;
            self.project_mm_id(d, &f.down_exps, &act, &offsets, m, &down)?;
        }
        let position = ms.position.view(0, [routes])?;
        d.dispatch_hybrid(
            "h_moe_combine",
            &[
                down.binding(),
                weights.binding(),
                if identity {
                    ids.binding()
                } else {
                    position.binding()
                },
                shared.binding(),
                shared_gate.binding(),
            ],
            &[out.binding()],
            &Params::default()
                .u(hidden)?
                .u(used)?
                .u(m)?
                .u(usize::from(identity))?
                .0,
            [(m * hidden).div_ceil(256), 1, 1],
            [256, 1, 1],
            0,
        )
    }

    /// `out = act(gate) * up`: SiLU, or GELU (tanh) for `gelu_ffn` models.
    fn glu(&self, d: &MetalDevice, gate: &Tensor, up: &Tensor, out: &Tensor) -> Result<()> {
        let n = gate.numel();
        d.dispatch_hybrid(
            if self.config.gelu_ffn {
                "h_geglu"
            } else {
                "h_swiglu"
            },
            &[gate.binding(), up.binding()],
            &[out.binding()],
            &Params::default().u(n)?.0,
            [n.div_ceil(256), 1, 1],
            [256, 1, 1],
            0,
        )
    }

    /// Expert GEMV: route r multiplies activation row r / x_div by expert ids[r].
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn project_id(
        d: &MetalDevice,
        w: &Matrix,
        x: &Tensor,
        ids: &Tensor,
        routes: usize,
        x_div: usize,
        y: &Tensor,
    ) -> Result<()> {
        let rows_e = w.expert_rows();
        if y.numel() != routes * rows_e || x.numel() * x_div != routes * w.cols {
            return Err(Error::Shape("expert projection geometry".into()));
        }
        let args = Params::default()
            .u(w.cols)?
            .u(rows_e)?
            .u(routes)?
            .u(w.row_bytes)?
            .u(w.cols)?
            .u(rows_e)?
            .u(rows_e)?
            .u(x_div)?
            .0;
        let (name, rows_per_group, simds, shared) = mv_kernel(w.format, true);
        d.dispatch_hybrid(
            name,
            &[w.binding(), x.binding(), ids.binding()],
            &[y.binding()],
            &args,
            [rows_e.div_ceil(rows_per_group), routes, 1],
            [32, simds, 1],
            shared,
        )
    }

    /// Expert GEMM over expert-sorted rows delimited by `offsets`.
    fn project_mm_id(
        &self,
        d: &MetalDevice,
        w: &Matrix,
        x: &Tensor,
        offsets: &Tensor,
        max_rows: usize,
        y: &Tensor,
    ) -> Result<()> {
        let rows_e = w.expert_rows();
        let args = Params::default()
            .u(w.cols)?
            .u(rows_e)?
            .u(0)?
            .u(w.row_bytes)?
            .u(w.cols)?
            .u(rows_e)?
            .u(rows_e)?
            .0;
        let name = match w.format {
            Format::Q4K => "h_mm_id_q4_k",
            Format::Q5K => "h_mm_id_q5_k",
            Format::Q6K => "h_mm_id_q6_k",
            Format::Q8_0 => "h_mm_id_q8_0",
            Format::Q4_0 => "h_mm_id_q4_0",
            Format::Iq4Xs => "h_mm_id_iq4_xs",
            Format::Iq3S => "h_mm_id_iq3_s",
            Format::Iq3Xxs => "h_mm_id_iq3_xxs",
            Format::Iq2S => "h_mm_id_iq2_s",
            Format::F32 | Format::Bf16 | Format::Mlx4 => {
                return Err(Error::Gguf(format!(
                    "{} experts are unsupported",
                    w.format.name()
                )));
            }
        };
        d.dispatch_hybrid(
            name,
            &[w.binding(), x.binding(), offsets.binding()],
            &[y.binding()],
            &args,
            [max_rows.div_ceil(32), rows_e.div_ceil(64), w.experts],
            [128, 1, 1],
            64 * 32 * 2,
        )
    }

    pub(crate) fn dense_ffn(
        &self,
        d: &MetalDevice,
        s: &Scratch,
        f: &DenseFfn,
        xn: &Tensor,
        out: &Tensor,
        m: usize,
    ) -> Result<()> {
        let width = f.gate.rows;
        let gate = rows(&s.ffn_gate, 0, m, width)?;
        let up = rows(&s.ffn_up, 0, m, width)?;
        let act = rows(&s.ffn_act, 0, m, width)?;
        Self::project(d, &f.gate, xn, m, &gate)?;
        Self::project(d, &f.up, xn, m, &up)?;
        self.glu(d, &gate, &up, &act)?;
        Self::project(d, &f.down, &act, m, out)
    }

    /// Debug aid (`FERRUM_HYBRID_DUMP=1`): complete queued work and print the
    /// sum of `t`, named like llama.cpp's eval-callback tensors.
    fn dump(&self, d: &MetalDevice, name: &str, layer: usize, t: &Tensor) -> Result<()> {
        if flags().dump {
            d.synchronize()?;
            let values = t.to_f32();
            let sum: f64 = values.iter().map(|&v| v as f64).sum();
            eprintln!(
                "dump {name}-{layer}: sum {sum:.6} first {:?}",
                &values[..values.len().min(4)]
            );
        }
        Ok(())
    }

    pub(crate) fn fill_zero(&self, d: &MetalDevice, t: &Tensor) -> Result<()> {
        // Zero through the GPU so the write is ordered with queued work.
        let n = t.numel();
        d.dispatch_hybrid(
            "h_zero",
            &[],
            &[t.binding()],
            &Params::default().u(n)?.0,
            [n.div_ceil(256), 1, 1],
            [256, 1, 1],
            0,
        )
    }

    pub(crate) fn get_rows(d: &MetalDevice, w: &Matrix, ids: &Tensor, y: &Tensor) -> Result<()> {
        let tokens = ids.numel();
        let name = match w.format {
            Format::Q4K => "h_get_rows_q4_k",
            Format::Q5K => "h_get_rows_q5_k",
            Format::Q6K => "h_get_rows_q6_k",
            Format::Q8_0 => "h_get_rows_q8_0",
            Format::Q4_0 => "h_get_rows_q4_0",
            Format::Iq4Xs => "h_get_rows_iq4_xs",
            Format::Bf16 => "h_get_rows_bf16",
            Format::Mlx4 => "h_get_rows_mlx4",
            f => return Err(Error::Gguf(format!("no embedding lookup for {}", f.name()))),
        };
        d.dispatch_hybrid(
            name,
            &[w.binding(), ids.binding()],
            &[y.binding()],
            &Params::default().u(w.cols)?.u(w.row_bytes)?.u(tokens)?.0,
            [(w.cols / 16).div_ceil(32), tokens, 1],
            [32, 1, 1],
            0,
        )
    }

    pub(crate) fn rmsnorm(
        &self,
        d: &MetalDevice,
        x: &Tensor,
        w: &Tensor,
        y: &Tensor,
        m: usize,
    ) -> Result<()> {
        let n = self.config.hidden;
        d.dispatch_hybrid(
            "h_rmsnorm",
            &[x.binding(), w.binding()],
            &[y.binding()],
            &Params::default().u(n)?.f(self.config.eps).u(m)?.0,
            [m, 1, 1],
            [256, 1, 1],
            0,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn add_rmsnorm(
        &self,
        d: &MetalDevice,
        x: &Tensor,
        r: &Tensor,
        w: &Tensor,
        h: &Tensor,
        y: &Tensor,
        m: usize,
    ) -> Result<()> {
        let n = self.config.hidden;
        d.dispatch_hybrid(
            "h_add_rmsnorm",
            &[x.binding(), r.binding(), w.binding()],
            &[h.binding(), y.binding()],
            &Params::default().u(n)?.f(self.config.eps).u(m)?.0,
            [m, 1, 1],
            [256, 1, 1],
            0,
        )
    }

    /// y[m][rows] = x[m][cols] · Wᵀ.
    /// An MLX 4-bit projection of 2..=32 rows on the matrix units: the rows
    /// as a T table (`h_mlx_ttable`), then Kapitsa's integer-code GEMM in
    /// 8-row tiles, or tile pairs past 8 rows (`h_mlx_qmm8t*`).
    fn project_mlx_tiles(
        d: &MetalDevice,
        w: &Matrix,
        x: &Tensor,
        m: usize,
        y: &Tensor,
    ) -> Result<()> {
        let k = w.cols;
        let groups = k / 64;
        let pair = m > 8;
        let trows = if pair { 16 } else { 8 };
        let tiles = m.div_ceil(trows);
        let rows = tiles * trows;
        let (table, sums) = mlx_tables(d, rows * k, rows / 8 * groups * 8)?;
        d.dispatch_hybrid(
            "h_mlx_ttable",
            &[x.binding()],
            &[table.binding(), sums.binding()],
            &Params::default().u(k)?.u(m)?.u(k)?.u(rows)?.0,
            [groups, rows, 1],
            [64, 1, 1],
            0,
        )?;
        // Kapitsa's tiling (qwen35_v8.cpp v8t_plan): 32 weight rows per SIMD
        // group for LM-head-sized grids, else 16; 4 k-slices for 33..=64
        // groups, else 8.
        let nt = if w.rows.div_ceil(16) > 8192 { 32 } else { 16 };
        let mut ks = if groups > 32 && groups <= 64 { 4 } else { 8 };
        while ks > 1 && groups < ks {
            ks /= 2;
        }
        let per = w.rows.div_ceil(nt);
        let name = match (pair, nt) {
            (false, 16) => "h_mlx_qmm8t_nt16",
            (false, _) => "h_mlx_qmm8t_nt32",
            (true, 16) => "h_mlx_qmm8t2_nt16",
            (true, _) => "h_mlx_qmm8t2_nt32",
        };
        d.dispatch_hybrid(
            name,
            &[w.binding(), table.binding(), sums.binding()],
            &[y.binding()],
            &Params::default()
                .u(k)?
                .u(m)?
                .u(ks)?
                .u(w.rows)?
                .u(w.rows)?
                .u(w.row_bytes)?
                .u(per)?
                .u(0)?
                .0,
            [per * tiles, 1, 1],
            [32 * ks, 1, 1],
            0,
        )
    }

    pub(crate) fn project(
        d: &MetalDevice,
        w: &Matrix,
        x: &Tensor,
        m: usize,
        y: &Tensor,
    ) -> Result<()> {
        if x.numel() != m * w.cols || y.numel() != m * w.rows {
            return Err(Error::Shape(format!(
                "projection [{m}, {}] x [{}, {}]ᵀ into {} values",
                w.cols,
                w.rows,
                w.cols,
                y.numel()
            )));
        }
        let args = Params::default()
            .u(w.cols)?
            .u(w.rows)?
            .u(m)?
            .u(w.row_bytes)?
            .u(w.cols)?
            .u(w.rows)?
            .0;
        let inputs: [Binding; 2] = [w.binding(), x.binding()];
        let outputs: [Binding; 1] = [y.binding()];
        // MLX from 3 rows: the integer-code tiles (2 rows: the batched GEMV,
        // 133 against 148 ms per 27B forward; 3 rows: 185 against 148).
        if w.format == Format::Mlx4 && (3..=MMS_MAX_ROWS).contains(&m) && !exact_math() {
            return Self::project_mlx_tiles(d, w, x, m, y);
        }
        if (2..=MVB_MAX_ROWS).contains(&m) && !exact_math() {
            const SIMDS: usize = 4;
            return d.dispatch_hybrid(
                mvb_kernel(w.format),
                &inputs,
                &outputs,
                &args,
                [w.rows.div_ceil(SIMDS * 2), m.div_ceil(4), 1],
                [32, SIMDS, 1],
                0,
            );
        }
        if (2..=MMS_MAX_ROWS).contains(&m) && !exact_math() {
            return d.dispatch_hybrid(
                mms_kernel(w.format),
                &inputs,
                &outputs,
                &args,
                [m.div_ceil(16), w.rows.div_ceil(64), 1],
                [128, 1, 1],
                64 * 64 * 2,
            );
        }
        if m > 1 && !exact_math() {
            let name = match w.format {
                Format::Q4K => "h_mm_q4_k",
                Format::Q5K => "h_mm_q5_k",
                Format::Q6K => "h_mm_q6_k",
                Format::Q8_0 => "h_mm_q8_0",
                Format::Q4_0 => "h_mm_q4_0",
                Format::Iq4Xs => "h_mm_iq4_xs",
                Format::F32 => "h_mm_f32",
                Format::Bf16 => "h_mm_bf16",
                Format::Iq3S => "h_mm_iq3_s",
                Format::Iq3Xxs => "h_mm_iq3_xxs",
                Format::Iq2S => "h_mm_iq2_s",
                Format::Mlx4 => "h_mm_mlx4",
            };
            return d.dispatch_hybrid(
                name,
                &inputs,
                &outputs,
                &args,
                [m.div_ceil(128), w.rows.div_ceil(64), 1],
                [128, 1, 1],
                64 * 32 * 2,
            );
        }
        let (name, rows_per_group, simds, shared) = mv_kernel(w.format, false);
        d.dispatch_hybrid(
            name,
            &inputs,
            &outputs,
            &args,
            [w.rows.div_ceil(rows_per_group), m, 1],
            [32, simds, 1],
            shared,
        )
    }
}

thread_local! {
    /// T table and group sums of the MLX verify-width GEMMs
    /// (`HybridModel::project_mlx_tiles`), grown to the widest use and reused:
    /// at most 32 rows of the widest projection input (about 1.1 MB on the
    /// 27B). Consecutive projections are ordered by the encoder's hazards.
    static MLX_TABLES: RefCell<Option<(Tensor, Tensor)>> = const { RefCell::new(None) };
}

fn mlx_tables(d: &MetalDevice, halves: usize, sums: usize) -> Result<(Tensor, Tensor)> {
    MLX_TABLES.with(|cell| {
        let mut cell = cell.borrow_mut();
        let fits = cell.as_ref().is_some_and(|(t, s)| {
            d.owns(t.binding().0) && t.numel() >= halves && s.numel() >= sums
        });
        if !fits {
            let (h, n) = match cell.as_ref().filter(|(t, _)| d.owns(t.binding().0)) {
                Some((t, s)) => (halves.max(t.numel()), sums.max(s.numel())),
                None => (halves, sums),
            };
            *cell = Some((
                Tensor::zeros_resident(d, [h], DType::F16)?,
                Tensor::zeros_resident(d, [n], DType::F32)?,
            ));
        }
        let (t, s) = cell.as_ref().expect("allocated above");
        Ok((t.clone(), s.clone()))
    })
}

/// Diagnostic switches, read once from the environment:
/// - `FERRUM_HYBRID_EXACT=1`: every projection uses the F32-activation GEMV
///   kernels (no F16 GEMM tiles). Slow; a high-precision evaluation reference.
/// - `FERRUM_HYBRID_REFERENCE_ATTENTION=1`: scalar online-softmax attention.
/// - `FERRUM_HYBRID_SIMD_ATTENTION=1`: simdgroup-matrix attention everywhere.
/// - `FERRUM_HYBRID_DUMP=1`: print sums of intermediate tensors.
struct Flags {
    exact: bool,
    reference_attention: bool,
    simd_attention: bool,
    legacy_attention: bool,
    dump: bool,
}

fn flags() -> &'static Flags {
    static FLAGS: std::sync::OnceLock<Flags> = std::sync::OnceLock::new();
    FLAGS.get_or_init(|| {
        let on = |name: &str| std::env::var_os(name).is_some_and(|v| v != "0");
        Flags {
            exact: on("FERRUM_HYBRID_EXACT"),
            reference_attention: on("FERRUM_HYBRID_REFERENCE_ATTENTION"),
            simd_attention: on("FERRUM_HYBRID_SIMD_ATTENTION"),
            // Pre-verify-kernel routing for 2..15 query tokens (F16 only).
            legacy_attention: on("FERRUM_HYBRID_LEGACY_ATTENTION"),
            dump: on("FERRUM_HYBRID_DUMP"),
        }
    })
}

fn exact_math() -> bool {
    flags().exact
}

/// Query tokens from which attention uses the TensorOps flash kernel.
const MPP_ATTENTION_MIN_TOKENS: usize = 16;

/// Most key splits for few-row attention, and (token, head) rows they may cover.
const MAX_SPLITS: usize = 256;
/// Few-row attention aims for this many threadgroups, each with at least
/// `SPLIT_MIN_KEYS` keys, so long-context decode fills the GPU.
const SPLIT_TARGET_GROUPS: usize = 64;
const SPLIT_MIN_KEYS: usize = 256;
/// Single-token decode attention: target threadgroups and keys per split.
const DECODE_SPLIT_TARGET: usize = 128;
const DECODE_SPLIT_MIN_KEYS: usize = 256;
const MAX_SPLIT_ROWS: usize = 64;
/// Speculative-verify attention: target threadgroups across row blocks,
/// KV heads and key splits (swept 16..1024 on Tiel at 4K-128K keys; 16-64
/// are within noise, larger loses up to 15%).
const VERIFY_SPLIT_TARGET: usize = 32;
/// Activation rows served by the small-batch GEMV, and by the narrow
/// TensorOps tiles (speculative verify widths; see the Phase 9 journal).
const MVB_MAX_ROWS: usize = 4;
const MMS_MAX_ROWS: usize = 32;
/// Expert routes at or below which experts use per-route GEMVs.
const MV_ID_MAX_ROUTES: usize = 32;

/// Small-batch GEMV (4 activation rows per threadgroup, 2 output rows per SIMD group).
fn mvb_kernel(format: Format) -> &'static str {
    match format {
        Format::Q4K => "h_mvb_q4_k",
        Format::Q5K => "h_mvb_q5_k",
        Format::Q6K => "h_mvb_q6_k",
        Format::Q8_0 => "h_mvb_q8_0",
        Format::Q4_0 => "h_mvb_q4_0",
        Format::Iq4Xs => "h_mvb_iq4_xs",
        Format::Iq3S => "h_mvb_iq3_s",
        Format::Iq3Xxs => "h_mvb_iq3_xxs",
        Format::Iq2S => "h_mvb_iq2_s",
        Format::Mlx4 => "h_mvb_mlx4",
        Format::F32 => "h_mvb_f32",
        Format::Bf16 => "h_mvb_bf16",
    }
}

/// Narrow-tile TensorOps GEMM (16 activation rows by 64 output rows).
fn mms_kernel(format: Format) -> &'static str {
    match format {
        Format::Q4K => "h_mms_q4_k",
        Format::Q5K => "h_mms_q5_k",
        Format::Q6K => "h_mms_q6_k",
        Format::Q8_0 => "h_mms_q8_0",
        Format::Q4_0 => "h_mms_q4_0",
        Format::Iq4Xs => "h_mms_iq4_xs",
        Format::Iq3S => "h_mms_iq3_s",
        Format::Iq3Xxs => "h_mms_iq3_xxs",
        Format::Iq2S => "h_mms_iq2_s",
        Format::Mlx4 => "h_mms_mlx4",
        Format::F32 => "h_mms_f32",
        Format::Bf16 => "h_mms_bf16",
    }
}

/// GEMV kernel, output rows per threadgroup, SIMD groups, threadgroup bytes.
fn mv_kernel(format: Format, expert: bool) -> (&'static str, usize, usize, usize) {
    let pick = |plain, id| if expert { id } else { plain };
    match format {
        Format::Q4K => (pick("h_mv_q4_k", "h_mv_id_q4_k"), 4, 2, 0),
        Format::Q5K => (pick("h_mv_q5_k", "h_mv_id_q5_k"), 2, 2, 0),
        Format::Q6K => (pick("h_mv_q6_k", "h_mv_id_q6_k"), 4, 2, 0),
        Format::Q4_0 => (pick("h_mv_q4_0", "h_mv_id_q4_0"), 8, 2, 0),
        Format::F32 => ("h_mv_f32", 8, 2, 0),
        Format::Bf16 => ("h_mv_bf16", 8, 2, 0),
        Format::Q8_0 => (pick("h_mv_q8_0", "h_mv_id_q8_0"), 2, 4, 32 * 2 * 4),
        Format::Iq4Xs => (pick("h_mv_iq4_xs", "h_mv_id_iq4_xs"), 4, 2, 32 * 4),
        Format::Iq3S => (pick("h_mv_iq3_s", "h_mv_id_iq3_s"), 8, 2, 512 * 4),
        Format::Iq3Xxs => (pick("h_mv_iq3_xxs", "h_mv_id_iq3_xxs"), 8, 2, 256 * 4 + 128),
        Format::Iq2S => (pick("h_mv_iq2_s", "h_mv_id_iq2_s"), 8, 2, 0),
        // Expert matrices are never MLX (`moe_ffn` refuses them first). One row
        // per SIMD group with 16-byte loads beat Kapitsa's 2 x 8 (its M4 Pro
        // tuning) on every 27B shape on the M5.
        Format::Mlx4 => ("h_mv_mlx4", 4, 4, 0),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn random(n: usize, seed: u64) -> Vec<f32> {
        let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        (0..n)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                ((x >> 40) as f32 / (1u64 << 24) as f32) * 2. - 1.
            })
            .collect()
    }

    #[test]
    fn rope_positions_compress_around_images() {
        let d = MetalDevice::new().unwrap();
        let span = |start, grid: (usize, usize)| ImageSpan {
            start,
            grid,
            embedding: std::rc::Rc::new(
                Tensor::zeros(&d, [grid.0 * grid.1, 4], DType::F32).unwrap(),
            ),
            deepstack: Vec::new(),
            hash: 0,
            linear: false,
        };
        // A 3x4 image at cache positions 5..17 advances rotary positions by 4.
        let media = [span(5, (3, 4)), span(30, (2, 2))];
        assert_eq!(rope_position(&media, 0), 0);
        assert_eq!(rope_position(&media, 5), 5);
        assert_eq!(rope_position(&media, 16), 16);
        assert_eq!(rope_position(&media, 17), 9);
        assert_eq!(rope_position(&media, 29), 21);
        assert_eq!(rope_position(&media, 34), 24);
        assert_eq!(rope_position(&media, 40), 30);
        let mut state = HybridState {
            capacity: 64,
            len: 0,
            valid: true,
            verified: 0,
            verified_tree: Vec::new(),
            media: media.to_vec(),
            k: Vec::new(),
            v: Vec::new(),
            conv: Vec::new(),
            ssm: Vec::new(),
            ring: Vec::new(),
            window: Vec::new(),
            max_rows: 1,
            written: 0,
        };
        // Text only: a plain offset.
        let plan = state.rope_plan(&d, 17, 4).unwrap();
        assert_eq!((plan.delta, plan.table.is_none()), (-8, true));
        // A chunk through the first image carries explicit (t, h, w) positions.
        let plan = state.rope_plan(&d, 4, 15).unwrap();
        let table = plan.table.expect("chunk touches an image").to_f32();
        let ints: Vec<i32> = table.iter().map(|v| v.to_bits() as i32).collect();
        assert_eq!(&ints[..3], &[4, 4, 4]);
        // Row for cache position 5: image origin (t = 5, h = 5, w = 5).
        assert_eq!(&ints[3..6], &[5, 5, 5]);
        // Cache position 6 is image column 1; position 10 is row 1, column 1.
        assert_eq!(&ints[6..9], &[5, 5, 6]);
        assert_eq!(&ints[18..21], &[5, 6, 6]);
        // The first text row after the image continues past the position grid.
        assert_eq!(&ints[36..39], &[5, 7, 8]);
        assert_eq!(&ints[39..42], &[9, 9, 9]);
        state.media.clear();
        assert_eq!(state.rope_plan(&d, 17, 4).unwrap().delta, 0);
        // LFM3: one position per image token, so the text after it is not pulled back.
        let linear = ImageSpan {
            linear: true,
            ..span(5, (3, 4))
        };
        assert_eq!(rope_position(std::slice::from_ref(&linear), 17), 17);
        state.media = vec![linear];
        let plan = state.rope_plan(&d, 5, 3).unwrap();
        let ints: Vec<i32> = plan
            .table
            .expect("linear image")
            .to_f32()
            .iter()
            .map(|v| v.to_bits() as i32)
            .collect();
        assert_eq!(&ints[..9], &[5, 5, 5, 6, 6, 6, 7, 7, 7]);
    }

    /// The attention-prep kernel rotates image rows by their (t, h, w) table and
    /// text rows by a scalar offset, exactly like llama.cpp's interleaved M-RoPE.
    #[test]
    fn attention_prep_follows_the_position_table() {
        let d = MetalDevice::new().unwrap();
        let (heads, kv_heads, head_dim, rope_dims, rows) =
            (2usize, 1usize, 256usize, 64usize, 5usize);
        let (theta, eps, scale) = (1.0e7f32, 1e-6f32, 0.0625f32);
        let qg = random(rows * heads * 2 * head_dim, 1);
        let kk = random(rows * kv_heads * head_dim, 2);
        let vv = random(rows * kv_heads * head_dim, 3);
        let norm = vec![1.; head_dim];
        let positions: [[i32; 3]; 5] = [[3, 3, 3], [9, 9, 9], [9, 9, 10], [9, 10, 9], [9, 10, 10]];
        let (sec_h, sec_w) = (11usize, 10usize);
        let run = |table: Option<&[[i32; 3]]>, delta: i32| -> (Vec<f32>, Vec<f32>) {
            let t = |data: &[f32], n: usize| Tensor::from_f32(&d, [n], DType::F32, data).unwrap();
            let (qg_t, k_t, v_t) = (t(&qg, qg.len()), t(&kk, kk.len()), t(&vv, vv.len()));
            let (qn, kn) = (t(&norm, head_dim), t(&norm, head_dim));
            let pos3 = Tensor::from_le_bytes(
                &d,
                [rows * 3],
                DType::F32,
                &table
                    .map(|tab| {
                        tab.iter()
                            .flatten()
                            .flat_map(|v| v.to_le_bytes())
                            .collect::<Vec<u8>>()
                    })
                    .unwrap_or_else(|| vec![0; rows * 12]),
            )
            .unwrap();
            let q_out = Tensor::zeros(&d, [rows * heads * head_dim], DType::F16).unwrap();
            let cache = Tensor::zeros(&d, [(rows + 2) * kv_heads * head_dim], DType::F16).unwrap();
            let params = Params::default()
                .u(rows)
                .unwrap()
                .u(heads)
                .unwrap()
                .u(kv_heads)
                .unwrap()
                .u(head_dim)
                .unwrap()
                .u(rope_dims)
                .unwrap()
                .u(2)
                .unwrap()
                .f(theta)
                .f(eps)
                .u(kv_heads * head_dim)
                .unwrap()
                .f(scale)
                .i(delta)
                .u(usize::from(table.is_some()))
                .unwrap()
                .u(sec_h)
                .unwrap()
                .u(sec_w)
                .unwrap()
                .u(rows + 2)
                .unwrap()
                .u(1)
                .unwrap()
                .u(0)
                .unwrap() // ring
                .u(1)
                .unwrap() // qk_norm
                .u(0)
                .unwrap(); // use_factors
            let e = d.execution_with_shared_encoder(true).unwrap();
            d.dispatch_hybrid(
                "h_attn_prep",
                &[
                    qg_t.binding(),
                    k_t.binding(),
                    v_t.binding(),
                    qn.binding(),
                    kn.binding(),
                    pos3.binding(),
                    qg_t.binding(),
                ],
                &[q_out.binding(), cache.binding()],
                &params.0,
                [heads + kv_heads, rows, 1],
                [head_dim, 1, 1],
                0,
            )
            .unwrap();
            e.finish().unwrap();
            (
                q_out.to_f32(),
                cache.to_f32()[2 * kv_heads * head_dim..].to_vec(),
            )
        };
        let reference = |pos_of: &dyn Fn(usize, usize) -> f32| -> Vec<f32> {
            // Query heads of every row.
            let mut out = Vec::new();
            for r in 0..rows {
                for h in 0..heads {
                    let base = (r * heads + h) * 2 * head_dim;
                    let x = &qg[base..base + head_dim];
                    let rms = (x.iter().map(|v| v * v).sum::<f32>() / head_dim as f32 + eps)
                        .sqrt()
                        .recip();
                    let n: Vec<f32> = x.iter().map(|v| v * rms).collect();
                    let mut y = n.clone();
                    for pair in 0..rope_dims / 2 {
                        let angle =
                            pos_of(r, pair) * theta.powf(-2. * pair as f32 / rope_dims as f32);
                        let (c, s) = (angle.cos(), angle.sin());
                        y[pair] = n[pair] * c - n[pair + rope_dims / 2] * s;
                        y[pair + rope_dims / 2] = n[pair] * s + n[pair + rope_dims / 2] * c;
                    }
                    out.extend(y.iter().map(|v| v * scale));
                }
            }
            out
        };
        let close = |got: &[f32], want: &[f32], what: &str| {
            let worst = got
                .iter()
                .zip(want)
                .map(|(a, b)| (a - b).abs())
                .fold(0., f32::max);
            assert!(worst < 2e-2, "{what}: worst error {worst}");
        };
        // Scalar path: rotary position = cache position (2 + row) + delta.
        let (q, _) = run(None, -3);
        close(&q, &reference(&|r, _| (2 + r) as f32 - 3.), "scalar offset");
        // Table path: pairs cycle time, height, width.
        let (q, _) = run(Some(&positions), 0);
        let axis_of = |pair: usize| {
            if pair % 3 == 1 && pair < 3 * sec_h {
                1
            } else if pair % 3 == 2 && pair < 3 * sec_w {
                2
            } else {
                0
            }
        };
        close(
            &q,
            &reference(&|r, pair| positions[r][axis_of(pair)] as f32),
            "position table",
        );
        // A table of equal axes is the plain rotary embedding at that position.
        let flat: Vec<[i32; 3]> = (0..rows as i32).map(|r| [r + 7; 3]).collect();
        let (q_table, k_table) = run(Some(&flat), 0);
        let (q_plain, k_plain) = run(None, 5);
        close(&q_table, &q_plain, "equal axes, queries");
        close(&k_table, &k_plain, "equal axes, keys");
    }

    /// Flash attention (both SIMD-group shapes, split and unsplit) against the
    /// reference online-softmax kernel for several GQA geometries.
    #[test]
    fn flash_attention_matches_reference() {
        let d = MetalDevice::new().unwrap();
        let cases: [(usize, usize, usize, usize); 8] = [
            (24, 4, 1, 0),
            (24, 4, 1, 700),
            (16, 2, 1, 300),
            (16, 2, 5, 40),
            (16, 2, 37, 0),
            (24, 4, 37, 90),
            (16, 2, 64, 128),
            (24, 4, 150, 70),
        ];
        for (heads, kv_heads, m, pos) in cases {
            let dim = 256;
            let stride = kv_heads * dim;
            let keys = pos + m;
            let cap = keys.next_multiple_of(32);
            let q = Tensor::from_f32(
                &d,
                [m, heads * dim],
                DType::F16,
                &random(m * heads * dim, 1)
                    .iter()
                    .map(|v| v / 16.)
                    .collect::<Vec<_>>(),
            )
            .unwrap();
            let kc =
                Tensor::from_f32(&d, [cap, stride], DType::F16, &random(cap * stride, 2)).unwrap();
            let vc =
                Tensor::from_f32(&d, [cap, stride], DType::F16, &random(cap * stride, 3)).unwrap();
            let qg = Tensor::from_f32(
                &d,
                [m, 2 * heads * dim],
                DType::F32,
                &random(m * 2 * heads * dim, 4),
            )
            .unwrap();
            let reference = Tensor::zeros(&d, [m, heads * dim], DType::F32).unwrap();
            let args = Params::default()
                .u(m)
                .unwrap()
                .u(heads)
                .unwrap()
                .u(kv_heads)
                .unwrap()
                .u(dim)
                .unwrap()
                .u(pos)
                .unwrap()
                .u(stride)
                .unwrap()
                .f(1.)
                .u(0)
                .unwrap()
                .u(1)
                .unwrap()
                .u(0)
                .unwrap()
                .u(0)
                .unwrap()
                .u(0)
                .unwrap(); // ring, tree
            {
                let e = d.execution_with_shared_encoder(true).unwrap();
                d.dispatch_hybrid(
                    "h_attention",
                    &[
                        q.binding(),
                        kc.binding(),
                        vc.binding(),
                        qg.binding(),
                        qg.binding(),
                        qg.binding(),
                    ],
                    &[reference.binding()],
                    &args.0,
                    [heads, m, 1],
                    [dim, 1, 1],
                    0,
                )
                .unwrap();
                e.finish().unwrap();
            }
            let expected = reference.to_f32();
            let group = heads / kv_heads;
            for simds in [1usize, 4] {
                for splits_wanted in [1usize, 3] {
                    let out = Tensor::zeros(&d, [m, heads * dim], DType::F32).unwrap();
                    let partial = Tensor::zeros(&d, [3 * m * heads * dim], DType::F32).unwrap();
                    let ml = Tensor::zeros(&d, [3 * m * heads * 2], DType::F32).unwrap();
                    let kps = keys.div_ceil(splits_wanted).next_multiple_of(32);
                    let splits = keys.div_ceil(kps);
                    let fa = Params::default()
                        .u(m)
                        .unwrap()
                        .u(heads)
                        .unwrap()
                        .u(kv_heads)
                        .unwrap()
                        .u(group)
                        .unwrap()
                        .u(pos)
                        .unwrap()
                        .u(stride)
                        .unwrap()
                        .u(kps)
                        .unwrap()
                        .u(splits)
                        .unwrap()
                        .u(cap)
                        .unwrap()
                        .u(dim)
                        .unwrap()
                        .u(0)
                        .unwrap()
                        .u(0)
                        .unwrap()
                        .u(1)
                        .unwrap()
                        .u(0)
                        .unwrap()
                        .u(0)
                        .unwrap(); // ring
                    let e = d.execution_with_shared_encoder(true).unwrap();
                    d.dispatch_hybrid(
                        if simds == 1 {
                            "h_flash_attn_1"
                        } else {
                            "h_flash_attn_4"
                        },
                        &[q.binding(), kc.binding(), vc.binding(), qg.binding()],
                        &[out.binding(), partial.binding(), ml.binding()],
                        &fa.0,
                        [(m * group).div_ceil(8 * simds), kv_heads, splits],
                        [32 * simds, 1, 1],
                        simds * 5888,
                    )
                    .unwrap();
                    if splits > 1 {
                        d.dispatch_hybrid(
                            "h_flash_reduce",
                            &[partial.binding(), ml.binding(), qg.binding(), qg.binding()],
                            &[out.binding()],
                            &fa.0,
                            [m * heads, 1, 1],
                            [dim, 1, 1],
                            0,
                        )
                        .unwrap();
                    }
                    e.finish().unwrap();
                    let got = out.to_f32();
                    let worst = got
                        .iter()
                        .zip(&expected)
                        .map(|(a, b)| (a - b).abs())
                        .fold(0f32, f32::max);
                    assert!(
                        worst < 2e-3,
                        "heads {heads}/{kv_heads} m {m} pos {pos} simds {simds} splits {splits}: max error {worst}"
                    );
                }
            }
            // Vector decode kernel (one query token).
            if m == 1 {
                for splits_wanted in [1usize, 5] {
                    let out = Tensor::zeros(&d, [m, heads * dim], DType::F32).unwrap();
                    let partial = Tensor::zeros(&d, [8 * heads * dim], DType::F32).unwrap();
                    let ml = Tensor::zeros(&d, [8 * heads * 2], DType::F32).unwrap();
                    let kps = keys.div_ceil(splits_wanted).next_multiple_of(32);
                    let splits = keys.div_ceil(kps);
                    let fa = Params::default()
                        .u(1)
                        .unwrap()
                        .u(heads)
                        .unwrap()
                        .u(kv_heads)
                        .unwrap()
                        .u(group)
                        .unwrap()
                        .u(pos)
                        .unwrap()
                        .u(stride)
                        .unwrap()
                        .u(kps)
                        .unwrap()
                        .u(splits)
                        .unwrap()
                        .u(cap)
                        .unwrap()
                        .u(dim)
                        .unwrap()
                        .u(0)
                        .unwrap()
                        .u(0)
                        .unwrap()
                        .u(1)
                        .unwrap()
                        .u(0)
                        .unwrap()
                        .u(0)
                        .unwrap(); // ring
                    let e = d.execution_with_shared_encoder(true).unwrap();
                    d.dispatch_hybrid(
                        "h_attn_decode",
                        &[
                            q.binding(),
                            kc.binding(),
                            vc.binding(),
                            qg.binding(),
                            qg.binding(),
                        ],
                        &[out.binding(), partial.binding(), ml.binding()],
                        &fa.0,
                        [kv_heads, splits, 1],
                        [128, 1, 1],
                        8 * 256 * 2 + 4 * 8 * 2 * 4 + 2 * 8 * 256 * 4,
                    )
                    .unwrap();
                    if splits > 1 {
                        d.dispatch_hybrid(
                            "h_flash_reduce",
                            &[partial.binding(), ml.binding(), qg.binding(), qg.binding()],
                            &[out.binding()],
                            &fa.0,
                            [heads, 1, 1],
                            [dim, 1, 1],
                            0,
                        )
                        .unwrap();
                    }
                    e.finish().unwrap();
                    let worst = out
                        .to_f32()
                        .iter()
                        .zip(&expected)
                        .map(|(a, b)| (a - b).abs())
                        .fold(0f32, f32::max);
                    assert!(
                        worst < 2e-3,
                        "decode heads {heads}/{kv_heads} pos {pos} splits {splits}: max error {worst}"
                    );
                }
            }
            // TensorOps kernel (prompt chunks).
            let out = Tensor::zeros(&d, [m, heads * dim], DType::F32).unwrap();
            let fa = Params::default()
                .u(m)
                .unwrap()
                .u(heads)
                .unwrap()
                .u(kv_heads)
                .unwrap()
                .u(group)
                .unwrap()
                .u(pos)
                .unwrap()
                .u(stride)
                .unwrap()
                .u(0)
                .unwrap()
                .u(1)
                .unwrap()
                .u(cap)
                .unwrap()
                .u(dim)
                .unwrap()
                .u(0)
                .unwrap()
                .u(0)
                .unwrap()
                .u(1)
                .unwrap()
                .u(0)
                .unwrap()
                .u(0)
                .unwrap(); // ring
            let e = d.execution_with_shared_encoder(true).unwrap();
            d.dispatch_hybrid(
                "h_flash_attn_mpp",
                &[q.binding(), kc.binding(), vc.binding(), qg.binding()],
                &[out.binding()],
                &fa.0,
                [m.div_ceil(64), heads, 1],
                [128, 1, 1],
                64 * 64 * 4 + 64 * 64 * 2 + 3 * 64 * 4,
            )
            .unwrap();
            e.finish().unwrap();
            let worst = out
                .to_f32()
                .iter()
                .zip(&expected)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            assert!(
                worst < 2e-3,
                "TensorOps heads {heads}/{kv_heads} m {m} pos {pos}: max error {worst}"
            );
        }
    }

    /// llama.cpp q8_0 rounding of `x` (rows of `width`) in the Q8 cache
    /// layout: every value, then every block scale. Returns (bytes, dequantized).
    fn quantize_q8(x: &[f32], width: usize) -> (Vec<u8>, Vec<f32>) {
        let mut values = Vec::with_capacity(x.len());
        let mut scales = Vec::with_capacity(x.len() / 16);
        let mut deq = Vec::with_capacity(x.len());
        assert_eq!(x.len() % width, 0);
        for block in x.chunks(32) {
            let amax = block.iter().fold(0f32, |a, v| a.max(v.abs()));
            let d = half::f16::from_f32(amax / 127.);
            let id = if d.to_f32() > 0. { 1. / d.to_f32() } else { 0. };
            for v in block {
                let q = (v * id).round_ties_even().clamp(-127., 127.) as i8;
                values.push(q as u8);
                deq.push(f32::from(q) * d.to_f32());
            }
            scales.extend_from_slice(&d.to_le_bytes());
        }
        values.extend_from_slice(&scales);
        (values, deq)
    }

    /// A ring layer can resume at a position only while its ring still holds
    /// that position's window (from its start aligned down to 64); linear
    /// layers can always resume.
    /// Spark's feed-forward activation, gelu(gate) * up with ggml's tanh
    /// approximation, and its per-head output gate, att *= sigmoid(gate[head]).
    #[test]
    fn spark_geglu_and_head_gate_match_the_cpu() {
        let d = MetalDevice::new().unwrap();
        let t = |v: &[f32]| Tensor::from_f32(&d, [v.len()], DType::F32, v).unwrap();
        let close = |got: &[f32], want: &[f32], what: &str| {
            for (i, (g, w)) in got.iter().zip(want).enumerate() {
                assert!(
                    (g - w).abs() <= 1e-5 * w.abs().max(1.),
                    "{what} {i}: {g} vs {w}"
                );
            }
        };
        let n = 1000;
        let gate: Vec<f32> = random(n, 3).iter().map(|v| v * 6.).collect();
        let up = random(n, 4);
        let (g, u, y) = (t(&gate), t(&up), t(&vec![0.; n]));
        d.dispatch_hybrid(
            "h_geglu",
            &[g.binding(), u.binding()],
            &[y.binding()],
            &Params::default().u(n).unwrap().0,
            [n.div_ceil(256), 1, 1],
            [256, 1, 1],
            0,
        )
        .unwrap();
        let want: Vec<f32> = gate
            .iter()
            .zip(&up)
            .map(|(&x, &u)| {
                let x = x as f64;
                let inner = (2. / std::f64::consts::PI).sqrt() * (x + 0.044715 * x * x * x);
                (0.5 * x * (1. + inner.tanh()) * u as f64) as f32
            })
            .collect();
        close(&y.to_f32(), &want, "geglu");

        let (tokens, heads, head_dim) = (3, 4, 128);
        let logits: Vec<f32> = random(tokens * heads, 5).iter().map(|v| v * 8.).collect();
        let att = random(tokens * heads * head_dim, 6);
        let (l, a) = (t(&logits), t(&att));
        let n = att.len();
        d.dispatch_hybrid(
            "h_head_gate",
            &[l.binding()],
            &[a.binding()],
            &Params::default().u(n).unwrap().u(head_dim).unwrap().0,
            [n.div_ceil(256), 1, 1],
            [256, 1, 1],
            0,
        )
        .unwrap();
        let want: Vec<f32> = att
            .iter()
            .enumerate()
            .map(|(i, &v)| v / (1. + (-logits[i / head_dim]).exp()))
            .collect();
        close(&a.to_f32(), &want, "head gate");
    }

    #[test]
    fn ring_states_resume_only_inside_the_ring() {
        let state = |written: usize| HybridState {
            capacity: 1 << 16,
            len: written,
            valid: true,
            verified: 0,
            verified_tree: Vec::new(),
            media: Vec::new(),
            k: Vec::new(),
            v: Vec::new(),
            conv: Vec::new(),
            ssm: Vec::new(),
            ring: vec![0, 1088],
            window: vec![0, 513],
            max_rows: 512,
            written,
        };
        // Nothing overwritten yet: any position resumes.
        assert!(state(1000).can_resume(10));
        // 3000 written: rows of positions below 3000 - 1088 = 1912 are gone.
        let s = state(3000);
        assert!(s.can_resume(3000));
        // Window of 2500 starts at 1988, aligned down to 1984 >= 1912.
        assert!(s.can_resume(2500));
        // Window of 2432 starts at 1920, already a multiple of 64.
        assert!(s.can_resume(2432));
        // Window of 2431 starts at 1919, aligned to 1856 < 1912.
        assert!(!s.can_resume(2431));
        assert!(!s.can_resume(100));
    }

    /// Ungated queries (Flint) read one head-width row per head, and with RoPE
    /// off the prep is the RMS-normalized query times the scale alone.
    #[test]
    fn ungated_prep_without_rope_is_the_normalized_query() {
        let d = MetalDevice::new().unwrap();
        let (heads, kv_heads, head_dim, rows) = (4usize, 1usize, 128usize, 3usize);
        let (eps, scale) = (1e-6f32, 1. / (head_dim as f32).sqrt());
        let qg = random(rows * heads * head_dim, 31);
        let kk = random(rows * kv_heads * head_dim, 32);
        let norm: Vec<f32> = random(head_dim, 33).iter().map(|v| 1. + v / 4.).collect();
        let t = |data: &[f32], n: usize| Tensor::from_f32(&d, [n], DType::F32, data).unwrap();
        let q_out = Tensor::zeros(&d, [rows * heads * head_dim], DType::F16).unwrap();
        let cache = Tensor::zeros(&d, [(rows + 2) * kv_heads * head_dim], DType::F16).unwrap();
        let params = Params::default()
            .u(rows)
            .unwrap()
            .u(heads)
            .unwrap()
            .u(kv_heads)
            .unwrap()
            .u(head_dim)
            .unwrap()
            .u(0)
            .unwrap() // rope_dims: no rotation
            .u(0)
            .unwrap()
            .f(1e7)
            .f(eps)
            .u(kv_heads * head_dim)
            .unwrap()
            .f(scale)
            .i(0)
            .u(0)
            .unwrap()
            .u(0)
            .unwrap()
            .u(0)
            .unwrap()
            .u(rows + 2)
            .unwrap()
            .u(0)
            .unwrap() // gated: qg rows hold queries only
            .u(0)
            .unwrap() // ring
            .u(1)
            .unwrap() // qk_norm
            .u(0)
            .unwrap(); // use_factors
        let (qg_t, k_t, zeros) = (t(&qg, qg.len()), t(&kk, kk.len()), t(&[0.], 1));
        let (qn, kn) = (t(&norm, head_dim), t(&norm, head_dim));
        let e = d.execution_with_shared_encoder(true).unwrap();
        d.dispatch_hybrid(
            "h_attn_prep",
            &[
                qg_t.binding(),
                k_t.binding(),
                k_t.binding(),
                qn.binding(),
                kn.binding(),
                zeros.binding(),
                zeros.binding(),
            ],
            &[q_out.binding(), cache.binding()],
            &params.0,
            [heads + kv_heads, rows, 1],
            [head_dim, 1, 1],
            0,
        )
        .unwrap();
        e.finish().unwrap();
        let got = q_out.to_f32();
        for r in 0..rows {
            for h in 0..heads {
                let row = &qg[(r * heads + h) * head_dim..][..head_dim];
                let ss: f32 = row.iter().map(|x| x * x).sum::<f32>() / head_dim as f32;
                let inv = 1. / (ss + eps).sqrt();
                for dd in 0..head_dim {
                    let want = row[dd] * inv * norm[dd] * scale;
                    let g = got[(r * heads + h) * head_dim + dd];
                    assert!(
                        (g - want).abs() <= 2e-3 * want.abs().max(1e-2),
                        "row {r} head {h} dim {dd}: {g} vs {want}"
                    );
                }
            }
        }
    }

    /// A factor of 1 matches plain RoPE. A factor of 1e30 is identity on that pair.
    #[test]
    fn rope_factors_divide_the_angle() {
        let d = MetalDevice::new().unwrap();
        let (heads, kv_heads, head_dim, rope_dims, rows) =
            (2usize, 1usize, 128usize, 128usize, 2usize);
        let (theta, eps, scale) = (1.0e6f32, 1e-5f32, 1. / (head_dim as f32).sqrt());
        let qg = random(rows * heads * head_dim, 7);
        let kk = random(rows * kv_heads * head_dim, 8);
        let norm = vec![1f32; head_dim];
        let mut factors = vec![1f32; rope_dims / 2];
        for f in &mut factors[rope_dims / 4..] {
            *f = 1e30;
        }
        let run = |use_factors: usize| -> Vec<f32> {
            let t = |data: &[f32]| Tensor::from_f32(&d, [data.len()], DType::F32, data).unwrap();
            let q_out = Tensor::zeros(&d, [rows * heads * head_dim], DType::F16).unwrap();
            let cache = Tensor::zeros(&d, [8 * kv_heads * head_dim], DType::F16).unwrap();
            let mut p = Params::default();
            for v in [rows, heads, kv_heads, head_dim, rope_dims, 3usize] {
                p = p.u(v).unwrap();
            }
            p = p
                .f(theta)
                .f(eps)
                .u(kv_heads * head_dim)
                .unwrap()
                .f(scale)
                .i(0);
            for v in [0usize, 0, 0, 8, 0, 0, 1, use_factors] {
                p = p.u(v).unwrap();
            }
            let (qg_t, k_t, n_t, f_t) = (t(&qg), t(&kk), t(&norm), t(&factors));
            let e = d.execution_with_shared_encoder(true).unwrap();
            d.dispatch_hybrid(
                "h_attn_prep",
                &[
                    qg_t.binding(),
                    k_t.binding(),
                    k_t.binding(),
                    n_t.binding(),
                    n_t.binding(),
                    qg_t.binding(),
                    f_t.binding(),
                ],
                &[q_out.binding(), cache.binding()],
                &p.0,
                [heads + kv_heads, rows, 1],
                [head_dim, 1, 1],
                0,
            )
            .unwrap();
            e.finish().unwrap();
            q_out.to_f32()
        };
        let plain = run(0);
        let factored = run(1);
        let half = rope_dims / 2;
        for r in 0..rows {
            for h in 0..heads {
                let base = (r * heads + h) * head_dim;
                let row = &qg[base..base + head_dim];
                let ss = row.iter().map(|x| x * x).sum::<f32>() / head_dim as f32;
                let inv = 1. / (ss + eps).sqrt();
                let n: Vec<f32> = row.iter().map(|v| v * inv).collect();
                for pair in 0..half {
                    let pos = (3 + r) as f32;
                    let freq = theta.powf(-2. * pair as f32 / rope_dims as f32);
                    let angle = pos * freq;
                    let (c, s) = (angle.cos(), angle.sin());
                    let rotated = n[pair] * c - n[pair + half] * s;
                    let plain_v = plain[base + pair];
                    let fact_v = factored[base + pair];
                    let tol = 2e-3 * rotated.abs().max(1e-2);
                    assert!(
                        (plain_v - rotated * scale).abs() <= tol,
                        "plain pair {pair}"
                    );
                    if pair < half / 2 {
                        assert!((fact_v - plain_v).abs() <= tol, "factor 1 pair {pair}");
                    } else {
                        let ident = n[pair] * scale;
                        assert!(
                            (fact_v - ident).abs() <= 2e-3 * ident.abs().max(1e-2),
                            "factor 1e30 pair {pair}: {fact_v} vs {ident}"
                        );
                    }
                }
            }
        }
    }

    /// Short conv matches prepend-state + ggml_ssm_conv, including a non-zero
    /// window and a single token.
    #[test]
    fn short_conv_matches_prepend_state() {
        let d = MetalDevice::new().unwrap();
        let (hidden, state_len) = (32usize, 6usize);
        let kernel = state_len + 1;
        let reference = |packed: &[f32], weight: &[f32], state0: &[f32], tokens: usize| {
            let mut hist = state0.to_vec();
            let mut y = vec![0f32; tokens * hidden];
            for t in 0..tokens {
                for c in 0..hidden {
                    let b = packed[t * 3 * hidden + c];
                    let gate = packed[t * 3 * hidden + hidden + c];
                    let x = packed[t * 3 * hidden + 2 * hidden + c];
                    let bx = b * x;
                    let mut acc = bx * weight[c * kernel + state_len];
                    for k in 0..state_len {
                        acc += hist[k * hidden + c] * weight[c * kernel + k];
                    }
                    y[t * hidden + c] = acc * gate;
                }
                if state_len > 0 {
                    hist.copy_within(hidden.., 0);
                    for c in 0..hidden {
                        let b = packed[t * 3 * hidden + c];
                        let x = packed[t * 3 * hidden + 2 * hidden + c];
                        hist[(state_len - 1) * hidden + c] = b * x;
                    }
                }
            }
            (y, hist)
        };
        for (tokens, seed, nonzero) in [(1usize, 1u64, false), (3, 2, true), (8, 3, true)] {
            let packed = random(tokens * 3 * hidden, seed);
            let weight = random(hidden * kernel, seed + 10);
            let mut state0 = vec![0f32; state_len * hidden];
            if nonzero {
                state0 = random(state_len * hidden, seed + 20);
            }
            let (want_y, want_state) = reference(&packed, &weight, &state0, tokens);
            let t = |data: &[f32]| Tensor::from_f32(&d, [data.len()], DType::F32, data).unwrap();
            let (packed_t, weight_t) = (t(&packed), t(&weight));
            let state_t = t(&state0);
            let y = Tensor::zeros(&d, [tokens * hidden], DType::F32).unwrap();
            let bx = Tensor::zeros(&d, [tokens * hidden], DType::F32).unwrap();
            let replay = t(&state0);
            let e = d.execution_with_shared_encoder(true).unwrap();
            d.dispatch_hybrid(
                "h_short_conv",
                &[packed_t.binding(), weight_t.binding()],
                &[y.binding(), state_t.binding(), bx.binding()],
                &Params::default()
                    .u(tokens)
                    .unwrap()
                    .u(hidden)
                    .unwrap()
                    .u(state_len)
                    .unwrap()
                    .u(1)
                    .unwrap()
                    .u(1)
                    .unwrap()
                    .0,
                [hidden.div_ceil(256), 1, 1],
                [256, 1, 1],
                0,
            )
            .unwrap();
            d.dispatch_hybrid(
                "h_bx_state",
                &[bx.binding()],
                &[replay.binding()],
                &Params::default()
                    .u(tokens)
                    .unwrap()
                    .u(hidden)
                    .unwrap()
                    .u(state_len)
                    .unwrap()
                    .0,
                [hidden.div_ceil(256), 1, 1],
                [256, 1, 1],
                0,
            )
            .unwrap();
            e.finish().unwrap();
            let close = |got: &[f32], want: &[f32], what: &str| {
                for (i, (g, w)) in got.iter().zip(want).enumerate() {
                    assert!(
                        (g - w).abs() <= 1e-5 * w.abs().max(1e-4),
                        "{what} [{i}] {g} vs {w} tokens {tokens}"
                    );
                }
            };
            close(&y.to_f32(), &want_y, "y");
            close(&state_t.to_f32(), &want_state, "state");
            close(&replay.to_f32(), &want_state, "replay");
        }
    }

    /// Head width 128 (Flint: GQA 32/4, learned sinks, sliding windows, ungated
    /// output) on every attention path against the reference kernel.
    #[test]
    fn narrow_heads_with_sinks_and_windows_match_reference() {
        let d = MetalDevice::new().unwrap();
        let (heads, kv_heads, dim) = (32usize, 4usize, 128usize);
        let group = heads / kv_heads;
        let stride = kv_heads * dim;
        // (m, pos, window, sinks, gated): decode, short and long prompt
        // chunks, and verify widths, with and without each feature.
        let cases: [(usize, usize, Option<usize>, bool, bool); 9] = [
            (1, 600, Some(513), true, false),
            (1, 40, None, true, false),
            (1, 1200, Some(513), false, true),
            (5, 300, Some(513), true, false),
            (9, 600, Some(513), true, false),
            (12, 1500, Some(513), false, true),
            (37, 700, Some(513), true, false),
            (37, 90, None, false, true),
            (64, 1000, Some(513), true, false),
        ];
        for (m, pos, window, sinks, gated) in cases {
            let keys = pos + m;
            let cap = keys.next_multiple_of(32);
            let first_key = window.map_or(0, |w| (pos + 1).saturating_sub(w));
            let window_arg = window.unwrap_or(0);
            let (sinks_on, gated_on) = (usize::from(sinks), usize::from(gated));
            let q = Tensor::from_f32(
                &d,
                [m, heads * dim],
                DType::F16,
                &random(m * heads * dim, 1)
                    .iter()
                    .map(|v| v / 16.)
                    .collect::<Vec<_>>(),
            )
            .unwrap();
            let (k_raw, v_raw) = (random(cap * stride, 2), random(cap * stride, 3));
            let qg = Tensor::from_f32(
                &d,
                [m, 2 * heads * dim],
                DType::F32,
                &random(m * 2 * heads * dim, 4),
            )
            .unwrap();
            let sink_logits = Tensor::from_f32(&d, [heads], DType::F32, &random(heads, 5)).unwrap();
            // The attention layer's sinks are real logits; the reference reads them too.
            let sink_arg = if sinks { &sink_logits } else { &qg };
            for kv in [KvCache::F16, KvCache::Q8] {
                let q8 = kv == KvCache::Q8;
                let (kc, vc, k_ref, v_ref) = match kv {
                    KvCache::F16 => {
                        let t =
                            |x: &[f32]| Tensor::from_f32(&d, [cap, stride], DType::F16, x).unwrap();
                        (t(&k_raw), t(&v_raw), k_raw.clone(), v_raw.clone())
                    }
                    KvCache::Q8 => {
                        let (kb, kd) = quantize_q8(&k_raw, stride);
                        let (vb, vd) = quantize_q8(&v_raw, stride);
                        let t = |b: &[u8]| {
                            Tensor::from_le_bytes(
                                &d,
                                [cap, kv.row_bytes(stride) / 2],
                                DType::F16,
                                b,
                            )
                            .unwrap()
                        };
                        (t(&kb), t(&vb), kd, vd)
                    }
                };
                let kr = Tensor::from_f32(&d, [cap, stride], DType::F16, &k_ref).unwrap();
                let vr = Tensor::from_f32(&d, [cap, stride], DType::F16, &v_ref).unwrap();
                let reference = Tensor::zeros(&d, [m, heads * dim], DType::F32).unwrap();
                let args = Params::default()
                    .u(m)
                    .unwrap()
                    .u(heads)
                    .unwrap()
                    .u(kv_heads)
                    .unwrap()
                    .u(dim)
                    .unwrap()
                    .u(pos)
                    .unwrap()
                    .u(stride)
                    .unwrap()
                    .f(1.)
                    .u(window_arg)
                    .unwrap()
                    .u(gated_on)
                    .unwrap()
                    .u(sinks_on)
                    .unwrap()
                    .u(0)
                    .unwrap()
                    .u(0)
                    .unwrap(); // ring, tree
                let e = d.execution_with_shared_encoder(true).unwrap();
                d.dispatch_hybrid(
                    "h_attention",
                    &[
                        q.binding(),
                        kr.binding(),
                        vr.binding(),
                        qg.binding(),
                        sink_arg.binding(),
                        qg.binding(),
                    ],
                    &[reference.binding()],
                    &args.0,
                    [heads, m, 1],
                    [dim, 1, 1],
                    0,
                )
                .unwrap();
                e.finish().unwrap();
                let expected = reference.to_f32();
                // Each kernel runs on the linear cache and, for windowed
                // cases whose keys wrap it, on a ring of `ring_rows` rows
                // holding key j in row j % ring (as `h_attn_prep` stores it).
                let ring = window.map_or(0, |w| (w + 62 + m).next_multiple_of(64));
                let mut layouts = vec![(0, kc, vc, cap, first_key)];
                if ring > 0 && ring < keys {
                    let wrap = |raw: &[f32]| {
                        let mut out = random(ring * stride, 9);
                        for j in 0..keys {
                            out[(j % ring) * stride..][..stride]
                                .copy_from_slice(&raw[j * stride..][..stride]);
                        }
                        out
                    };
                    let (kw, vw) = (wrap(&k_raw), wrap(&v_raw));
                    let (kc, vc) = match kv {
                        KvCache::F16 => {
                            let t = |x: &[f32]| {
                                Tensor::from_f32(&d, [ring, stride], DType::F16, x).unwrap()
                            };
                            (t(&kw), t(&vw))
                        }
                        KvCache::Q8 => {
                            let t = |x: &[f32]| {
                                let (b, _) = quantize_q8(x, stride);
                                Tensor::from_le_bytes(
                                    &d,
                                    [ring, kv.row_bytes(stride) / 2],
                                    DType::F16,
                                    &b,
                                )
                                .unwrap()
                            };
                            (t(&kw), t(&vw))
                        }
                    };
                    layouts.push((ring, kc, vc, ring, first_key / 64 * 64));
                }
                for (ring, kc, vc, kv_rows, first_key) in layouts {
                    // Keys the kernels split: everything from the first visible key.
                    let span = keys - first_key;
                    let fa = |keys_per_split: usize, splits: usize| {
                        let mut p = Params::default();
                        for x in [
                            m,
                            heads,
                            kv_heads,
                            group,
                            pos,
                            stride,
                            keys_per_split,
                            splits,
                            kv_rows,
                            dim,
                            first_key,
                            window_arg,
                            gated_on,
                            sinks_on,
                            ring,
                        ] {
                            p = p.u(x).unwrap();
                        }
                        p.0
                    };
                    let check = |name: &'static str,
                                 inputs_extra: bool,
                                 groups: [usize; 3],
                                 threads: usize,
                                 shared: usize,
                                 kps: usize,
                                 splits: usize| {
                        let out = Tensor::zeros(&d, [m, heads * dim], DType::F32).unwrap();
                        let partial =
                            Tensor::zeros(&d, [splits * m * heads * dim], DType::F32).unwrap();
                        let ml = Tensor::zeros(&d, [splits * m * heads * 2], DType::F32).unwrap();
                        let args = fa(kps, splits);
                        let qk = Tensor::zeros(&d, [m, heads * dim], DType::F16).unwrap();
                        let verify = name.starts_with("h_attn_verify");
                        let mpp = name.starts_with("h_flash_attn_mpp");
                        let e = d.execution_with_shared_encoder(true).unwrap();
                        if verify {
                            d.dispatch_hybrid(
                                "h_q_by_kv_head",
                                &[q.binding()],
                                &[qk.binding()],
                                &args,
                                [(m * heads * dim).div_ceil(256), 1, 1],
                                [256, 1, 1],
                                0,
                            )
                            .unwrap();
                        }
                        let outputs = if mpp {
                            vec![out.binding()]
                        } else {
                            vec![out.binding(), partial.binding(), ml.binding()]
                        };
                        let mut inputs = vec![
                            if verify { qk.binding() } else { q.binding() },
                            kc.binding(),
                            vc.binding(),
                            qg.binding(),
                        ];
                        if inputs_extra {
                            inputs.push(sink_arg.binding());
                        }
                        d.dispatch_hybrid(
                            name,
                            &inputs,
                            &outputs,
                            &args,
                            groups,
                            [threads, 1, 1],
                            shared,
                        )
                        .unwrap();
                        if splits > 1 {
                            d.dispatch_hybrid(
                                "h_flash_reduce_d128",
                                &[
                                    partial.binding(),
                                    ml.binding(),
                                    qg.binding(),
                                    sink_arg.binding(),
                                ],
                                &[out.binding()],
                                &args,
                                [m * heads, 1, 1],
                                [dim, 1, 1],
                                0,
                            )
                            .unwrap();
                        }
                        e.finish().unwrap();
                        let worst = out
                            .to_f32()
                            .iter()
                            .zip(&expected)
                            .map(|(a, b)| (a - b).abs())
                            .fold(0f32, f32::max);
                        assert!(
                            worst < 2e-3,
                            "{name} m {m} pos {pos} window {window:?} sinks {sinks} gated {gated} ring {ring} splits {splits}: max error {worst}"
                        );
                    };
                    if m == 1 {
                        for splits_wanted in [1usize, 5] {
                            let kps = span.div_ceil(splits_wanted).next_multiple_of(32);
                            let splits = span.div_ceil(kps);
                            check(
                                if q8 {
                                    "h_attn_decode_q8_d128"
                                } else {
                                    "h_attn_decode_d128"
                                },
                                true,
                                [kv_heads, splits, 1],
                                128,
                                8 * dim * 2 + 4 * 8 * 2 * 4 + 2 * 8 * dim * 4,
                                kps,
                                splits,
                            );
                        }
                    }
                    if m > 1 && m < 16 {
                        for splits_wanted in [1usize, 3] {
                            let rows = m * group;
                            let row_blocks = rows.div_ceil(32);
                            let kps = span.div_ceil(splits_wanted).next_multiple_of(64);
                            let splits = span.div_ceil(kps);
                            check(
                                if q8 {
                                    "h_attn_verify_mpp_32_q8_d128"
                                } else {
                                    "h_attn_verify_mpp_32_d128"
                                },
                                true,
                                [row_blocks, kv_heads, splits],
                                128,
                                if q8 {
                                    // The K/V staging tile outgrows the 32-row score tile.
                                    64 * 128 * 2 + 32 * 64 * 2 + 3 * 32 * 4
                                } else {
                                    32 * 64 * 4 + 32 * 64 * 2 + 3 * 32 * 4
                                },
                                kps,
                                splits,
                            );
                        }
                    }
                    if m >= 16 {
                        check(
                            if q8 {
                                "h_flash_attn_mpp_q8_d128"
                            } else {
                                "h_flash_attn_mpp_staged_d128"
                            },
                            true,
                            [m.div_ceil(64), heads, 1],
                            128,
                            64 * 64 * 4 + 64 * 64 * 2 + 3 * 64 * 4,
                            0,
                            1,
                        );
                    }
                }
            }
        }
    }

    /// Expert routing: softmax or sigmoid scores, top-k with ties to the lower
    /// index, optional renormalization, and the shared-expert gate (or 1).
    #[test]
    fn moe_routing_matches_the_reference_selection() {
        let d = MetalDevice::new().unwrap();
        let (experts, used, hidden, m) = (256usize, 6usize, 2048usize, 3usize);
        let logits = random(m * experts, 21)
            .iter()
            .map(|v| v * 3.)
            .collect::<Vec<_>>();
        let x = random(m * hidden, 22);
        let shared_w = random(hidden, 23);
        // (sigmoid, norm, shared_gate): Flint, Qwen3.5-MoE, and sigmoid-only routing.
        for (sigmoid, norm, shared_gate) in [
            (true, true, false),
            (false, true, true),
            (true, false, true),
        ] {
            let lt = Tensor::from_f32(&d, [m, experts], DType::F32, &logits).unwrap();
            let xt = Tensor::from_f32(&d, [m, hidden], DType::F32, &x).unwrap();
            let wt = Tensor::from_f32(&d, [hidden], DType::F32, &shared_w).unwrap();
            let ids = Tensor::zeros(&d, [m * used], DType::F32).unwrap();
            let weights = Tensor::zeros(&d, [m * used], DType::F32).unwrap();
            let gate = Tensor::zeros(&d, [m], DType::F32).unwrap();
            let args = Params::default()
                .u(experts)
                .unwrap()
                .u(used)
                .unwrap()
                .u(hidden)
                .unwrap()
                .u(m)
                .unwrap()
                .u(usize::from(sigmoid))
                .unwrap()
                .u(usize::from(norm))
                .unwrap()
                .u(usize::from(shared_gate))
                .unwrap();
            let e = d.execution_with_shared_encoder(true).unwrap();
            d.dispatch_hybrid(
                "h_moe_route",
                &[lt.binding(), xt.binding(), wt.binding()],
                &[ids.binding(), weights.binding(), gate.binding()],
                &args.0,
                [m, 1, 1],
                [experts, 1, 1],
                0,
            )
            .unwrap();
            e.finish().unwrap();
            let got_ids: Vec<u32> = ids.to_f32().iter().map(|v| v.to_bits()).collect();
            let got_w = weights.to_f32();
            let got_gate = gate.to_f32();
            for t in 0..m {
                let row = &logits[t * experts..][..experts];
                let scores: Vec<f32> = if sigmoid {
                    row.iter().map(|l| 1. / (1. + (-l).exp())).collect()
                } else {
                    let mx = row.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                    let e: Vec<f32> = row.iter().map(|l| (l - mx).exp()).collect();
                    let total: f32 = e.iter().sum();
                    e.iter().map(|v| v / total).collect()
                };
                let mut order: Vec<usize> = (0..experts).collect();
                order.sort_by(|&a, &b| scores[b].total_cmp(&scores[a]).then(a.cmp(&b)));
                let picked = &order[..used];
                let sum: f32 = picked.iter().map(|&i| scores[i]).sum();
                for (k, &i) in picked.iter().enumerate() {
                    assert_eq!(got_ids[t * used + k] as usize, i, "route {k} of token {t}");
                    let want = if norm { scores[i] / sum } else { scores[i] };
                    let got = got_w[t * used + k];
                    assert!(
                        (got - want).abs() < 1e-5,
                        "weight {k} of token {t}: {got} vs {want}"
                    );
                }
                let want_gate = if shared_gate {
                    let dot: f32 = x[t * hidden..][..hidden]
                        .iter()
                        .zip(&shared_w)
                        .map(|(a, b)| a * b)
                        .sum();
                    1. / (1. + (-dot).exp())
                } else {
                    1.
                };
                assert!(
                    (got_gate[t] - want_gate).abs() < 1e-4,
                    "shared gate of token {t}: {} vs {want_gate}",
                    got_gate[t]
                );
            }
        }
    }

    #[test]
    fn q8_cache_stores_match_cpu_quantization() {
        let d = MetalDevice::new().unwrap();
        let (kv_heads, head_dim, rows, pos) = (2usize, 256usize, 3usize, 5usize);
        let stride = kv_heads * head_dim;
        let cap = 32usize;
        let v: Vec<f32> = random(rows * stride, 9).iter().map(|x| x * 3.).collect();
        let v_t = Tensor::from_f32(&d, [rows * stride], DType::F32, &v).unwrap();
        let cache = KvCache::Q8.alloc(&d, cap, stride).unwrap();
        let mut params = Params::default();
        for x in [rows, 1, kv_heads, head_dim, 64, pos] {
            params = params.u(x).unwrap();
        }
        let params = params
            .f(1e7)
            .f(1e-6)
            .u(stride)
            .unwrap()
            .f(1.)
            .i(0)
            .u(0)
            .unwrap()
            .u(0)
            .unwrap()
            .u(0)
            .unwrap()
            .u(cap)
            .unwrap()
            .u(1)
            .unwrap()
            .u(0)
            .unwrap() // ring
            .u(1)
            .unwrap(); // qk_norm
        let e = d.execution_with_shared_encoder(true).unwrap();
        d.dispatch_hybrid(
            "h_store_v_q8",
            &[v_t.binding()],
            &[cache.binding()],
            &params.0,
            [(rows * stride).div_ceil(256), 1, 1],
            [256, 1, 1],
            0,
        )
        .unwrap();
        e.finish().unwrap();
        let (buffer, offset, len) = cache.binding();
        let bytes = buffer.with_bytes(offset, len, <[u8]>::to_vec);
        let (want, _) = quantize_q8(&v, stride);
        let n = rows * stride;
        let got_values = &bytes[pos * stride..pos * stride + n];
        assert_eq!(got_values, &want[..n]);
        let scale_base = cap * stride + pos * stride / 16;
        assert_eq!(&bytes[scale_base..scale_base + n / 16], &want[n..]);
    }

    #[test]
    fn verify_and_q8_attention_match_reference() {
        let d = MetalDevice::new().unwrap();
        // (heads, kv_heads, m, pos): verify widths over both GQA shapes, plus
        // decode and prompt chunks for the Q8 kernels.
        let cases: [(usize, usize, usize, usize); 9] = [
            (16, 2, 1, 700),
            (24, 4, 1, 90),
            (16, 2, 2, 300),
            (16, 2, 3, 0),
            (24, 4, 3, 517),
            (16, 2, 9, 1000),
            (16, 2, 15, 64),
            (24, 4, 37, 90),
            (16, 2, 64, 128),
        ];
        let dim = 256;
        for (heads, kv_heads, m, pos) in cases {
            let stride = kv_heads * dim;
            let keys = pos + m;
            let cap = keys.next_multiple_of(32);
            let group = heads / kv_heads;
            let q = Tensor::from_f32(
                &d,
                [m, heads * dim],
                DType::F16,
                &random(m * heads * dim, 1)
                    .iter()
                    .map(|v| v / 16.)
                    .collect::<Vec<_>>(),
            )
            .unwrap();
            let qg = Tensor::from_f32(
                &d,
                [m, 2 * heads * dim],
                DType::F32,
                &random(m * 2 * heads * dim, 4),
            )
            .unwrap();
            let (k_raw, v_raw) = (random(cap * stride, 2), random(cap * stride, 3));
            for kv in [KvCache::F16, KvCache::Q8] {
                // The reference reads an F16 cache holding exactly the values
                // the kernel under test sees.
                let (kc, vc, k_ref, v_ref) = match kv {
                    KvCache::F16 => {
                        let t = |x: &[f32]| Tensor::from_f32(&d, [cap, stride], DType::F16, x);
                        (
                            t(&k_raw).unwrap(),
                            t(&v_raw).unwrap(),
                            k_raw.clone(),
                            v_raw.clone(),
                        )
                    }
                    KvCache::Q8 => {
                        let (kb, kd) = quantize_q8(&k_raw, stride);
                        let (vb, vd) = quantize_q8(&v_raw, stride);
                        let t = |b: &[u8]| {
                            Tensor::from_le_bytes(
                                &d,
                                [cap, kv.row_bytes(stride) / 2],
                                DType::F16,
                                b,
                            )
                        };
                        (t(&kb).unwrap(), t(&vb).unwrap(), kd, vd)
                    }
                };
                let kr = Tensor::from_f32(&d, [cap, stride], DType::F16, &k_ref).unwrap();
                let vr = Tensor::from_f32(&d, [cap, stride], DType::F16, &v_ref).unwrap();
                let reference = Tensor::zeros(&d, [m, heads * dim], DType::F32).unwrap();
                let mut args = Params::default();
                for x in [m, heads, kv_heads, dim, pos, stride] {
                    args = args.u(x).unwrap();
                }
                let args = args
                    .f(1.)
                    .u(0)
                    .unwrap()
                    .u(1)
                    .unwrap()
                    .u(0)
                    .unwrap()
                    .u(0)
                    .unwrap()
                    .u(0)
                    .unwrap(); // ring, tree
                let e = d.execution_with_shared_encoder(true).unwrap();
                d.dispatch_hybrid(
                    "h_attention",
                    &[
                        q.binding(),
                        kr.binding(),
                        vr.binding(),
                        qg.binding(),
                        qg.binding(),
                        qg.binding(),
                    ],
                    &[reference.binding()],
                    &args.0,
                    [heads, m, 1],
                    [dim, 1, 1],
                    0,
                )
                .unwrap();
                e.finish().unwrap();
                let expected = reference.to_f32();
                let q8 = kv == KvCache::Q8;
                let check = |name: &'static str,
                             groups: [usize; 3],
                             threads: usize,
                             shared: usize,
                             kps: usize,
                             splits: usize| {
                    let out = Tensor::zeros(&d, [m, heads * dim], DType::F32).unwrap();
                    let partial =
                        Tensor::zeros(&d, [splits * m * heads * dim], DType::F32).unwrap();
                    let ml = Tensor::zeros(&d, [splits * m * heads * 2], DType::F32).unwrap();
                    let mut fa = Params::default();
                    for x in [m, heads, kv_heads, group, pos, stride, kps, splits, cap] {
                        fa = fa.u(x).unwrap();
                    }
                    for x in [dim, 0, 0, 1, 0, 0] {
                        fa = fa.u(x).unwrap();
                    }
                    let mpp = name.starts_with("h_flash_attn_mpp");
                    let verify = name.starts_with("h_attn_verify_mpp");
                    let qk = Tensor::zeros(&d, [m, heads * dim], DType::F16).unwrap();
                    let e = d.execution_with_shared_encoder(true).unwrap();
                    if verify {
                        d.dispatch_hybrid(
                            "h_q_by_kv_head",
                            &[q.binding()],
                            &[qk.binding()],
                            &fa.0,
                            [(m * heads * dim).div_ceil(256), 1, 1],
                            [256, 1, 1],
                            0,
                        )
                        .unwrap();
                    }
                    let outputs = if mpp {
                        vec![out.binding()]
                    } else {
                        vec![out.binding(), partial.binding(), ml.binding()]
                    };
                    let mut inputs = vec![
                        if verify { qk.binding() } else { q.binding() },
                        kc.binding(),
                        vc.binding(),
                        qg.binding(),
                    ];
                    // The legacy non-staged prefill kernel predates sinks.
                    if name != "h_flash_attn_mpp" {
                        inputs.push(qg.binding());
                    }
                    d.dispatch_hybrid(
                        name,
                        &inputs,
                        &outputs,
                        &fa.0,
                        groups,
                        [threads, 1, 1],
                        shared,
                    )
                    .unwrap();
                    if splits > 1 {
                        d.dispatch_hybrid(
                            "h_flash_reduce",
                            &[partial.binding(), ml.binding(), qg.binding(), qg.binding()],
                            &[out.binding()],
                            &fa.0,
                            [m * heads, 1, 1],
                            [dim, 1, 1],
                            0,
                        )
                        .unwrap();
                    }
                    e.finish().unwrap();
                    let worst = out
                        .to_f32()
                        .iter()
                        .zip(&expected)
                        .map(|(a, b)| (a - b).abs())
                        .fold(0f32, f32::max);
                    assert!(
                        worst < 2e-3,
                        "{name} heads {heads}/{kv_heads} m {m} pos {pos} splits {splits}: max error {worst}"
                    );
                };
                for splits_wanted in [1usize, 5] {
                    let kps = keys.div_ceil(splits_wanted).next_multiple_of(32);
                    let splits = keys.div_ceil(kps);
                    if m == 1 {
                        let name = if q8 {
                            "h_attn_decode_q8"
                        } else {
                            "h_attn_decode"
                        };
                        check(
                            name,
                            [kv_heads, splits, 1],
                            128,
                            8 * 256 * 2 + 4 * 8 * 2 * 4 + 2 * 8 * 256 * 4,
                            kps,
                            splits,
                        );
                    }
                    if m > 1 && m < 16 {
                        let name = if q8 {
                            "h_attn_verify_mpp_32_q8"
                        } else {
                            "h_attn_verify_mpp_32"
                        };
                        let shared = if q8 {
                            64 * 128 * 2 + 32 * 64 * 2 + 3 * 32 * 4
                        } else {
                            32 * 64 * 4 + 32 * 64 * 2 + 3 * 32 * 4
                        };
                        let kps = keys.div_ceil(splits_wanted).next_multiple_of(64);
                        let splits = keys.div_ceil(kps);
                        check(
                            name,
                            [(m * group).div_ceil(32), kv_heads, splits],
                            128,
                            shared,
                            kps,
                            splits,
                        );
                    }
                }
                if m >= 16 {
                    let names: &[&'static str] = if q8 {
                        &["h_flash_attn_mpp_q8"]
                    } else {
                        &["h_flash_attn_mpp", "h_flash_attn_mpp_staged"]
                    };
                    for &name in names {
                        check(
                            name,
                            [m.div_ceil(64), heads, 1],
                            128,
                            64 * 64 * 4 + 64 * 64 * 2 + 3 * 64 * 4,
                            0,
                            1,
                        );
                    }
                }
            }
        }
    }
}
