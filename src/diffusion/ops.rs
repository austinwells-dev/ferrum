//! Dispatch of the image transformer's kernels (`diffusion.metal`). Matrix
//! products go through `HybridModel::project`, which serves every weight format.
use crate::{
    Error, MetalDevice, Result, Tensor,
    hybrid::{
        engine::Params,
        weights::{Format, Matrix},
    },
};

const ELEMENTS_PER_GROUP: usize = 256;
/// K values per GEMM step (`IMM_NK` in diffusion.metal): sets the tile's
/// threadgroup memory.
const GEMM_STEP_K: usize = 64;
/// The kernels are written for 128-wide heads.
pub(crate) const HEAD_DIM: usize = 128;

fn flat(
    d: &MetalDevice,
    name: &'static str,
    inputs: &[&Tensor],
    out: &Tensor,
    p: Params,
    n: usize,
) -> Result<()> {
    let inputs: Vec<_> = inputs.iter().map(|t| t.binding()).collect();
    d.dispatch_hybrid(
        name,
        &inputs,
        &[out.binding()],
        &p.0,
        [n.div_ceil(ELEMENTS_PER_GROUP), 1, 1],
        [ELEMENTS_PER_GROUP, 1, 1],
        0,
    )
}

pub(crate) fn silu(d: &MetalDevice, x: &Tensor, y: &Tensor) -> Result<()> {
    let n = x.numel();
    flat(d, "i_silu", &[x], y, Params::default().u(n)?, n)
}

/// `y = silu(gate) * up`, F32 in and F16 out.
pub(crate) fn swiglu(d: &MetalDevice, gate: &Tensor, up: &Tensor, y: &Tensor) -> Result<()> {
    let n = y.numel();
    flat(d, "i_swiglu", &[gate, up], y, Params::default().u(n)?, n)
}

pub(crate) fn gelu_tanh(d: &MetalDevice, x: &Tensor, y: &Tensor) -> Result<()> {
    let n = x.numel();
    flat(d, "i_gelu_tanh", &[x], y, Params::default().u(n)?, n)
}

/// `y = layernorm(x) * (offset + scale)` for each of `rows` rows, as F16.
pub(crate) fn layernorm_scale(
    d: &MetalDevice,
    x: &Tensor,
    scale: &Tensor,
    y: &Tensor,
    rows: usize,
    eps: f32,
    offset: f32,
) -> Result<()> {
    let n = scale.numel();
    d.dispatch_hybrid(
        "i_ln_mod",
        &[x.binding(), scale.binding()],
        &[y.binding()],
        &Params::default().u(n)?.f(eps).f(offset).0,
        [rows, 1, 1],
        [256, 1, 1],
        0,
    )
}

/// `x += tanh(gate) * y`, with `gate` one row broadcast over `x`'s rows.
pub(crate) fn gate_add(d: &MetalDevice, y: &Tensor, gate: &Tensor, x: &Tensor) -> Result<()> {
    let n = x.numel();
    let cols = gate.numel();
    flat(
        d,
        "i_gate_add",
        &[y, gate],
        x,
        Params::default().u(n)?.u(cols)?,
        n,
    )
}

/// `x += dt * v`.
pub(crate) fn euler(d: &MetalDevice, v: &Tensor, x: &Tensor, dt: f32) -> Result<()> {
    let n = x.numel();
    flat(d, "i_euler", &[v], x, Params::default().u(n)?.f(dt), n)
}

/// One layer's F32 projections, the per-head norm weights and the rotary rows.
pub(crate) struct QkvIn<'a> {
    pub q: &'a Tensor,
    pub k: &'a Tensor,
    pub v: &'a Tensor,
    pub q_norm: &'a Tensor,
    pub k_norm: &'a Tensor,
    pub cos: &'a Tensor,
    pub sin: &'a Tensor,
}

/// Per-head RMSNorm and rotary embedding of Q and K (Q also scaled by
/// `1 / sqrt(head width)`), written with V as F16 into `out` = Q, K, V.
pub(crate) fn qkv_prep(
    d: &MetalDevice,
    i: &QkvIn,
    out: [&Tensor; 3],
    rows: usize,
    heads: usize,
    eps: f32,
) -> Result<()> {
    let args = Params::default()
        .u(rows)?
        .u(heads)?
        .u(HEAD_DIM / 2)?
        .f(eps)
        .f(1. / (HEAD_DIM as f32).sqrt());
    d.dispatch_hybrid(
        "i_qkv_prep",
        &[
            i.q.binding(),
            i.k.binding(),
            i.v.binding(),
            i.q_norm.binding(),
            i.k_norm.binding(),
            i.cos.binding(),
            i.sin.binding(),
        ],
        &[out[0].binding(), out[1].binding(), out[2].binding()],
        &args.0,
        [rows, heads, 1],
        [HEAD_DIM, 1, 1],
        0,
    )
}

/// K and V of one layer as attention reads them: the cached text rows, then
/// the current rows.
pub(crate) struct KeyValues<'a> {
    pub prefix_k: &'a Tensor,
    pub prefix_v: &'a Tensor,
    pub prefix: usize,
    pub k: &'a Tensor,
    pub v: &'a Tensor,
}

/// Flash attention of `tokens` query rows over `keys` keys, the first
/// `kv.prefix` from the cache and the rest from the current rows. `causal`
/// masks keys after the row (the text prefill); image rows see every key. The
/// output is F16.
#[allow(clippy::too_many_arguments)]
pub(crate) fn attention(
    d: &MetalDevice,
    q: &Tensor,
    kv: &KeyValues,
    out: &Tensor,
    tokens: usize,
    heads: usize,
    keys: usize,
    causal: bool,
) -> Result<()> {
    let args = Params::default()
        .u(tokens)?
        .u(heads)?
        .u(kv.prefix)?
        .u(keys)?
        .u(usize::from(causal))?;
    d.dispatch_hybrid(
        "i_attn",
        &[
            q.binding(),
            kv.prefix_k.binding(),
            kv.prefix_v.binding(),
            kv.k.binding(),
            kv.v.binding(),
        ],
        &[out.binding()],
        &args.0,
        [tokens.div_ceil(64), heads, 1],
        [128, 1, 1],
        64 * 64 * 4 + 64 * 64 * 2 + 3 * 64 * 4,
    )
}

/// `y = x · wᵀ` for `m` rows of F16 activations, with F32 accumulation and
/// output: the transformer's matrix products, which are all wide enough for
/// the TensorOps tiles. The hybrid engine's `project` reads F32 activations.
pub(crate) fn project(d: &MetalDevice, w: &Matrix, x: &Tensor, m: usize, y: &Tensor) -> Result<()> {
    let name = match w.format {
        Format::Bf16 => "i_mm_bf16",
        Format::Q8_0 => "i_mm_q8_0",
        Format::Q4K => "i_mm_q4_k",
        Format::Q5K => "i_mm_q5_k",
        Format::Q6K => "i_mm_q6_k",
        Format::Q4_0 => "i_mm_q4_0",
        f => {
            return Err(Error::Parameter(format!(
                "the image transformer has no GEMM for {} weights",
                f.name()
            )));
        }
    };
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
    d.dispatch_hybrid(
        name,
        &[w.binding(), x.binding()],
        &[y.binding()],
        &args,
        [m.div_ceil(128), w.rows.div_ceil(64), 1],
        [128, 1, 1],
        64 * GEMM_STEP_K * 2,
    )
}
