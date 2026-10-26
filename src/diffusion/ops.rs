//! Dispatch of the image transformer's kernels (`diffusion.metal`). Matrix
//! products go through `HybridModel::project`, which serves every weight format.
use crate::{MetalDevice, Result, Tensor, hybrid::engine::Params};

const ELEMENTS_PER_GROUP: usize = 256;
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

/// `y = silu(gate) * up`.
pub(crate) fn swiglu(d: &MetalDevice, gate: &Tensor, up: &Tensor, y: &Tensor) -> Result<()> {
    let n = y.numel();
    flat(d, "h_swiglu", &[gate, up], y, Params::default().u(n)?, n)
}

pub(crate) fn gelu_tanh(d: &MetalDevice, x: &Tensor, y: &Tensor) -> Result<()> {
    let n = x.numel();
    flat(d, "i_gelu_tanh", &[x], y, Params::default().u(n)?, n)
}

/// `y = layernorm(x) * (offset + scale)` for each of `rows` rows.
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
/// masks keys after the row (the text prefill); image rows see every key.
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
