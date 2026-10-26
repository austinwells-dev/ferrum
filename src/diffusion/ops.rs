//! Dispatch of the image transformer's kernels (`diffusion.metal`). Matrix
//! products go through `HybridModel::project`, which serves every weight format.
use crate::{
    DType, Error, MetalDevice, Result, Tensor,
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

/// The kernels index their bindings from the dispatch parameters, which the
/// device's binding checks cannot see: each wrapper first checks that every
/// tensor holds at least what its parameters make the kernel touch.
pub(crate) fn expect(t: &Tensor, dtype: DType, values: usize, what: &str) -> Result<()> {
    if t.dtype() != dtype || t.numel() < values {
        return Err(Error::Shape(format!(
            "{what}: needs {values} {dtype:?} values, got {} {:?}",
            t.numel(),
            t.dtype()
        )));
    }
    Ok(())
}

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
    expect(x, DType::F32, n, "silu input")?;
    expect(y, DType::F32, n, "silu output")?;
    flat(d, "i_silu", &[x], y, Params::default().u(n)?, n)
}

/// `y = silu(gate) * up`, F32 in and F16 out.
pub(crate) fn swiglu(d: &MetalDevice, gate: &Tensor, up: &Tensor, y: &Tensor) -> Result<()> {
    let n = y.numel();
    expect(gate, DType::F32, n, "swiglu gate")?;
    expect(up, DType::F32, n, "swiglu up")?;
    expect(y, DType::F16, n, "swiglu output")?;
    flat(d, "i_swiglu", &[gate, up], y, Params::default().u(n)?, n)
}

pub(crate) fn gelu_tanh(d: &MetalDevice, x: &Tensor, y: &Tensor) -> Result<()> {
    let n = x.numel();
    expect(x, DType::F32, n, "gelu input")?;
    expect(y, DType::F32, n, "gelu output")?;
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
    expect(scale, DType::F32, n, "layernorm scale")?;
    expect(x, DType::F32, rows * n, "layernorm input")?;
    expect(y, DType::F16, rows * n, "layernorm output")?;
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
    if cols == 0 || !n.is_multiple_of(cols) {
        return Err(Error::Shape(format!("gate of {cols} over {n} values")));
    }
    expect(x, DType::F32, n, "gated residual")?;
    expect(y, DType::F32, n, "gated input")?;
    expect(gate, DType::F32, cols, "gate")?;
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
    expect(x, DType::F32, n, "latents")?;
    expect(v, DType::F32, n, "velocity")?;
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
    let n = rows * heads * HEAD_DIM;
    let pairs = rows * (HEAD_DIM / 2);
    for (t, what) in [(i.q, "Q"), (i.k, "K"), (i.v, "V")] {
        expect(t, DType::F32, n, what)?;
    }
    for (t, what) in [
        (out[0], "prepared Q"),
        (out[1], "prepared K"),
        (out[2], "prepared V"),
    ] {
        expect(t, DType::F16, n, what)?;
    }
    expect(i.q_norm, DType::F32, HEAD_DIM, "Q norm")?;
    expect(i.k_norm, DType::F32, HEAD_DIM, "K norm")?;
    expect(i.cos, DType::F32, pairs, "rotary cosines")?;
    expect(i.sin, DType::F32, pairs, "rotary sines")?;
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
/// `kv.prefix` from the cache and the rest from the current rows. `limits`
/// (F32, one per row) masks the prefill block-causally: row r sees keys
/// `[0, limits[r])`. Without it every row sees every key, as the image rows
/// of a denoising step do. The output is F16.
#[allow(clippy::too_many_arguments)]
pub(crate) fn attention(
    d: &MetalDevice,
    q: &Tensor,
    kv: &KeyValues,
    out: &Tensor,
    tokens: usize,
    heads: usize,
    keys: usize,
    limits: Option<&Tensor>,
) -> Result<()> {
    let width = heads * HEAD_DIM;
    if keys < kv.prefix || keys == 0 {
        return Err(Error::Shape(format!(
            "attention over {keys} keys with a prefix of {}",
            kv.prefix
        )));
    }
    expect(q, DType::F16, tokens * width, "attention queries")?;
    expect(out, DType::F16, tokens * width, "attention output")?;
    expect(kv.prefix_k, DType::F16, kv.prefix * width, "cached keys")?;
    expect(kv.prefix_v, DType::F16, kv.prefix * width, "cached values")?;
    expect(kv.k, DType::F16, (keys - kv.prefix) * width, "keys")?;
    expect(kv.v, DType::F16, (keys - kv.prefix) * width, "values")?;
    if let Some(l) = limits {
        expect(l, DType::F32, tokens, "attention limits")?;
    }
    let args = Params::default()
        .u(tokens)?
        .u(heads)?
        .u(kv.prefix)?
        .u(keys)?
        .u(usize::from(limits.is_some()))?;
    d.dispatch_hybrid(
        "i_attn",
        &[
            q.binding(),
            kv.prefix_k.binding(),
            kv.prefix_v.binding(),
            kv.k.binding(),
            kv.v.binding(),
            // Unread without a mask.
            limits.unwrap_or(q).binding(),
        ],
        &[out.binding()],
        &args.0,
        [tokens.div_ceil(64), heads, 1],
        [128, 1, 1],
        64 * 64 * 4 + 64 * 64 * 2 + 4 * 64 * 4,
    )
}

/// The GEMM kernel for a weight encoding, if the transformer has one.
fn gemm_kernel(format: Format) -> Option<&'static str> {
    Some(match format {
        Format::Bf16 => "i_mm_bf16",
        Format::Q8_0 => "i_mm_q8_0",
        Format::Q4K => "i_mm_q4_k",
        Format::Q5K => "i_mm_q5_k",
        Format::Q6K => "i_mm_q6_k",
        Format::Q4_0 => "i_mm_q4_0",
        _ => return None,
    })
}

/// Whether `project` can multiply by `w`: a supported encoding and a width
/// that is a whole number of the kernel's K steps.
pub(crate) fn check_projectable(w: &Matrix, name: &str) -> Result<()> {
    if gemm_kernel(w.format).is_none() {
        return Err(Error::Weight {
            name: name.into(),
            message: format!(
                "the image transformer has no GEMM for {} weights (supported: bf16, q8_0, q4_0, q4_k, q5_k, q6_k)",
                w.format.name()
            ),
        });
    }
    if !w.cols.is_multiple_of(GEMM_STEP_K) {
        return Err(Error::Weight {
            name: name.into(),
            message: format!("width {} is not a multiple of {GEMM_STEP_K}", w.cols),
        });
    }
    Ok(())
}

/// `y = x · wᵀ` for `m` rows of F16 activations, with F32 accumulation and
/// output: the transformer's matrix products, which are all wide enough for
/// the TensorOps tiles. The hybrid engine's `project` reads F32 activations.
pub(crate) fn project(d: &MetalDevice, w: &Matrix, x: &Tensor, m: usize, y: &Tensor) -> Result<()> {
    check_projectable(w, "projection")?;
    let name = gemm_kernel(w.format).expect("checked above");
    if x.dtype() != DType::F16 || y.dtype() != DType::F32 {
        return Err(Error::DType);
    }
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
