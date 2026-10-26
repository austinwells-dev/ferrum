//! Convolution weights and dispatch for the image decoder (`diffusion_vae.metal`).
//! Activations are F16 and NHWC: `[pixels, channels]` rows, pixels in raster order.
use super::checkpoint::Checkpoint;
use crate::{DType, Error, MetalDevice, Result, Tensor, hybrid::engine::Params};

/// A 1x1 or 3x3 convolution: F16 weights `[cout, taps * cin]` (tap-major,
/// `tap = ky * 3 + kx`) and an F16 bias.
pub(crate) struct Conv {
    w: Tensor,
    b: Tensor,
    pub cin: usize,
    pub cout: usize,
    taps: usize,
}

impl Conv {
    /// `{name}.weight` is `[cout, cin, k, k]` and `{name}.bias` `[cout]`.
    pub fn load(
        device: &MetalDevice,
        ckpt: &Checkpoint,
        name: &str,
        cin: usize,
        cout: usize,
        k: usize,
    ) -> Result<Self> {
        let (wn, bn) = (format!("{name}.weight"), format!("{name}.bias"));
        let e = ckpt.entry(&wn)?;
        if e.shape != [cout, cin, k, k] || !(k == 1 || k == 3) {
            return Err(Error::Weight {
                name: wn,
                message: format!("expected [{cout}, {cin}, {k}, {k}], found {:?}", e.shape),
            });
        }
        let src = ckpt.read_f32(&wn)?;
        let taps = k * k;
        let mut packed = vec![0f32; src.len()];
        for o in 0..cout {
            for c in 0..cin {
                for t in 0..taps {
                    packed[o * taps * cin + t * cin + c] = src[(o * cin + c) * taps + t];
                }
            }
        }
        let bias = ckpt.read_f32(&bn)?;
        if bias.len() != cout {
            return Err(Error::Weight {
                name: bn,
                message: format!("expected {cout} values, found {}", bias.len()),
            });
        }
        Ok(Self {
            w: Tensor::from_f32(device, [cout, taps * cin], DType::F16, &packed)?,
            b: Tensor::from_f32(device, [cout], DType::F16, &bias)?,
            cin,
            cout,
            taps,
        })
    }

    /// A convolution from host values (`weight` in `[cout, cin, k, k]` order), for tests.
    #[cfg(test)]
    pub fn from_values(
        device: &MetalDevice,
        weight: &[f32],
        bias: &[f32],
        cin: usize,
        cout: usize,
        k: usize,
    ) -> Result<Self> {
        let taps = k * k;
        let mut packed = vec![0f32; weight.len()];
        for o in 0..cout {
            for c in 0..cin {
                for t in 0..taps {
                    packed[o * taps * cin + t * cin + c] = weight[(o * cin + c) * taps + t];
                }
            }
        }
        Ok(Self {
            w: Tensor::from_f32(device, [cout, taps * cin], DType::F16, &packed)?,
            b: Tensor::from_f32(device, [cout], DType::F16, bias)?,
            cin,
            cout,
            taps,
        })
    }

    pub fn byte_size(&self) -> usize {
        self.w.byte_size() + self.b.byte_size()
    }
}

/// What the convolution adds to its output.
pub(crate) enum Residual<'a> {
    None,
    /// A tensor of the output's shape.
    Same(&'a Tensor),
    /// A residual from half the resolution: output channel `o` at `(2y + a,
    /// 2x + b)` reads channel `(((o * frames + frames - 1) * 2 + a) * 2 + b) /
    /// repeats` of `src` at `(y, x)` (the decoder's pixel-shuffled shortcut,
    /// for the last of `frames` frames).
    Shuffled {
        src: &'a Tensor,
        channels: usize,
        frames: usize,
        repeats: usize,
    },
}

/// `y = conv(x) + residual`, `x` read at `(h / up, w / up)` with nearest upsampling.
pub(crate) fn conv(
    d: &MetalDevice,
    c: &Conv,
    x: &Tensor,
    y: &Tensor,
    (h, w): (usize, usize),
    up: usize,
    res: Residual,
) -> Result<()> {
    if !matches!(up, 1 | 2) || !h.is_multiple_of(up) || !w.is_multiple_of(up) {
        return Err(Error::Parameter(format!("upsample {up} of {h} x {w}")));
    }
    if x.numel() != h / up * (w / up) * c.cin || y.numel() != h * w * c.cout {
        return Err(Error::Shape(format!(
            "conv {} -> {} over {h} x {w} (up {up}): {} input and {} output values",
            c.cin,
            c.cout,
            x.numel(),
            y.numel()
        )));
    }
    let (mode, res_t, res_cin, frames, repeats) = match res {
        Residual::None => (0, &c.b, 0, 1, 1),
        Residual::Same(t) => (1, t, 0, 1, 1),
        Residual::Shuffled {
            src,
            channels,
            frames,
            repeats,
        } => (2, src, channels, frames, repeats),
    };
    let args = Params::default()
        .u(h)?
        .u(w)?
        .u(c.cin)?
        .u(c.cout)?
        .u(c.taps)?
        .u(up)?
        .u(mode)?
        .u(res_cin)?
        .u(frames)?
        .u(repeats)?;
    let (name, nk) = if c.cin.is_multiple_of(32) {
        ("v_conv_k32", 32)
    } else if c.cin.is_multiple_of(16) {
        ("v_conv_k16", 16)
    } else {
        return Err(Error::Shape(format!("{} input channels", c.cin)));
    };
    d.dispatch_hybrid(
        name,
        &[x.binding(), c.w.binding(), c.b.binding(), res_t.binding()],
        &[y.binding()],
        &args.0,
        [(h * w).div_ceil(128), c.cout.div_ceil(64), 1],
        [128, 1, 1],
        128 * nk * 2,
    )
}

/// `y = x / rms(x over channels) * gamma`, then SiLU if asked.
pub(crate) fn channel_rms(
    d: &MetalDevice,
    x: &Tensor,
    gamma: &Tensor,
    y: &Tensor,
    pixels: usize,
    silu: bool,
) -> Result<()> {
    let channels = gamma.numel();
    d.dispatch_hybrid(
        "v_chan_rms",
        &[x.binding(), gamma.binding()],
        &[y.binding()],
        &Params::default()
            .u(pixels)?
            .u(channels)?
            .u(usize::from(silu))?
            .0,
        [pixels.div_ceil(4), 1, 1],
        [128, 1, 1],
        0,
    )
}

/// `z = x * std + mean` per channel, F32 latents to F16.
pub(crate) fn denormalize(
    d: &MetalDevice,
    x: &Tensor,
    mean: &Tensor,
    std: &Tensor,
    z: &Tensor,
) -> Result<()> {
    let n = x.numel();
    d.dispatch_hybrid(
        "v_denorm",
        &[x.binding(), mean.binding(), std.binding()],
        &[z.binding()],
        &Params::default().u(n)?.u(mean.numel())?.0,
        [n.div_ceil(256), 1, 1],
        [256, 1, 1],
        0,
    )
}

/// A matrix operand: `[rows, k]` F16 elements `row * stride_row + k * stride_k`.
pub(crate) struct Operand<'a> {
    pub t: &'a Tensor,
    pub rows: usize,
    pub stride_row: usize,
    pub stride_k: usize,
}

/// `C[m][n] = sum_k A[m][k] * B[n][k]` into F32 (`wide`) or F16 `c`, row stride `ldc`.
pub(crate) fn gemm(
    d: &MetalDevice,
    a: &Operand,
    b: &Operand,
    c: &Tensor,
    k: usize,
    ldc: usize,
) -> Result<()> {
    let wide = c.dtype() == DType::F32;
    let args = Params::default()
        .u(a.rows)?
        .u(b.rows)?
        .u(k)?
        .u(a.stride_row)?
        .u(b.stride_k)?
        .u(b.stride_row)?
        .u(ldc)?;
    d.dispatch_hybrid(
        if wide { "v_gemm_f32" } else { "v_gemm_f16" },
        &[a.t.binding(), b.t.binding()],
        &[c.binding()],
        &args.0,
        [a.rows.div_ceil(64), b.rows.div_ceil(64), 1],
        [128, 1, 1],
        0,
    )
}

/// `y[c][r] = x[r][c]` for the `rows x cols` block of F16 `x` with row stride `ld`.
pub(crate) fn transpose(
    d: &MetalDevice,
    x: &Tensor,
    y: &Tensor,
    rows: usize,
    cols: usize,
    ld: usize,
) -> Result<()> {
    d.dispatch_hybrid(
        "v_transpose",
        &[x.binding()],
        &[y.binding()],
        &Params::default().u(rows)?.u(cols)?.u(ld)?.0,
        [cols.div_ceil(16), rows.div_ceil(16), 1],
        [16, 16, 1],
        0,
    )
}

/// Row softmax of `scale * s` (F32 `[rows, cols]`) into F16.
pub(crate) fn softmax_rows(
    d: &MetalDevice,
    s: &Tensor,
    p: &Tensor,
    rows: usize,
    cols: usize,
    scale: f32,
) -> Result<()> {
    d.dispatch_hybrid(
        "v_softmax_rows",
        &[s.binding()],
        &[p.binding()],
        &Params::default().u(rows)?.u(cols)?.f(scale).0,
        [rows, 1, 1],
        [256, 1, 1],
        0,
    )
}
