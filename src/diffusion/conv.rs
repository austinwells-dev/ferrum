//! Convolution weights and dispatch for the image decoder (`diffusion_vae.metal`).
//! Activations are F16 and NHWC: `[pixels, channels]` rows, pixels in raster order.
use super::{checkpoint::Checkpoint, ops::expect};
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
        Self::load_padded(device, ckpt, name, cin, cin, cout, k)
    }

    /// `load` with the input channels zero-padded to `width` (the kernels
    /// read 16 channels at a time; the encoder's input has four).
    pub fn load_padded(
        device: &MetalDevice,
        ckpt: &Checkpoint,
        name: &str,
        cin: usize,
        width: usize,
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
        let mut packed = vec![0f32; cout * taps * width];
        for o in 0..cout {
            for c in 0..cin {
                for t in 0..taps {
                    packed[o * taps * width + t * width + c] = src[(o * cin + c) * taps + t];
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
            w: Tensor::from_f32(device, [cout, taps * width], DType::F16, &packed)?,
            b: Tensor::from_f32(device, [cout], DType::F16, &bias)?,
            cin: width,
            cout,
            taps,
        })
    }

    /// Device bytes of a `[cout, cin, k, k]` convolution (as `load` stores it).
    pub fn bytes(cin: usize, cout: usize, k: usize) -> usize {
        crate::hybrid::engine::page(cout * cin * k * k * 2) + crate::hybrid::engine::page(cout * 2)
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

/// How a convolution's output grid relates to its input's.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Sample {
    /// Same size, padding 1.
    Same,
    /// Input at half the size, read nearest (the decoder's upsampling).
    Up,
    /// Input at twice the size, stride 2, padded after the last row and
    /// column only (the encoder's downsampling).
    Down,
}

/// `y = conv(x) + residual` over an `h x w` output grid.
pub(crate) fn conv(
    d: &MetalDevice,
    c: &Conv,
    x: &Tensor,
    y: &Tensor,
    (h, w): (usize, usize),
    sample: Sample,
    res: Residual,
) -> Result<()> {
    let (up, down) = match sample {
        Sample::Same => (1, 1),
        Sample::Up => (2, 1),
        Sample::Down => (1, 2),
    };
    if !h.is_multiple_of(up) || !w.is_multiple_of(up) || (down == 2 && c.taps != 9) {
        return Err(Error::Parameter(format!(
            "{sample:?} convolution of {h} x {w}"
        )));
    }
    let input_pixels = h * down / up * (w * down / up);
    if x.numel() != input_pixels * c.cin || y.numel() != h * w * c.cout {
        return Err(Error::Shape(format!(
            "conv {} -> {} over {h} x {w} ({sample:?}): {} input and {} output values",
            c.cin,
            c.cout,
            x.numel(),
            y.numel()
        )));
    }
    expect(x, DType::F16, x.numel(), "convolution input")?;
    expect(y, DType::F16, y.numel(), "convolution output")?;
    let (mode, res_t, res_cin, frames, repeats) = match res {
        Residual::None => (0, &c.b, 0, 1, 1),
        Residual::Same(t) => {
            expect(t, DType::F16, h * w * c.cout, "convolution residual")?;
            (1, t, 0, 1, 1)
        }
        Residual::Shuffled {
            src,
            channels,
            frames,
            repeats,
        } => {
            // The highest source channel the epilogue reads.
            let last = (((c.cout - 1) * frames + frames - 1) * 2 + 1) * 2 + 1;
            if repeats == 0
                || last / repeats >= channels
                || !h.is_multiple_of(2)
                || !w.is_multiple_of(2)
            {
                return Err(Error::Shape(format!(
                    "shuffled shortcut of {channels} channels (frames {frames}, repeats {repeats}) into {}",
                    c.cout
                )));
            }
            expect(
                src,
                DType::F16,
                h / 2 * (w / 2) * channels,
                "shuffled shortcut",
            )?;
            (2, src, channels, frames, repeats)
        }
    };
    let args = Params::default()
        .u(h)?
        .u(w)?
        .u(c.cin)?
        .u(c.cout)?
        .u(c.taps)?
        .u(up)?
        .u(down)?
        .u(mode)?
        .u(res_cin)?
        .u(frames)?
        .u(repeats)?;
    let (name, nk) = if c.cin.is_multiple_of(32) {
        ("vae_conv_k32", 32)
    } else if c.cin.is_multiple_of(16) {
        ("vae_conv_k16", 16)
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

/// The encoder's averaged shortcut: `x` (`side*h x side*w` pixels of `cin`
/// channels) to `h x w` pixels of `cout`, each the mean of a group of the
/// space-to-depth stack; `frames` 2 counts a zero frame before the image.
#[allow(clippy::too_many_arguments)]
pub(crate) fn avg_down(
    d: &MetalDevice,
    x: &Tensor,
    y: &Tensor,
    (h, w): (usize, usize),
    cin: usize,
    cout: usize,
    frames: usize,
    side: usize,
) -> Result<()> {
    if !matches!(frames, 1 | 2)
        || !matches!(side, 1 | 2)
        || cout == 0
        || !(cin * frames * side * side).is_multiple_of(cout)
    {
        return Err(Error::Shape(format!(
            "averaged shortcut {cin} -> {cout} (frames {frames}, side {side})"
        )));
    }
    expect(x, DType::F16, h * side * w * side * cin, "shortcut input")?;
    expect(y, DType::F16, h * w * cout, "shortcut output")?;
    let n = h * w * cout;
    d.dispatch_hybrid(
        "vae_avg_down",
        &[x.binding()],
        &[y.binding()],
        &Params::default()
            .u(h)?
            .u(w)?
            .u(cin)?
            .u(cout)?
            .u(frames)?
            .u(side)?
            .0,
        [n.div_ceil(256), 1, 1],
        [256, 1, 1],
        0,
    )
}

/// `y += x` for F16 tensors of the same size.
pub(crate) fn add(d: &MetalDevice, x: &Tensor, y: &Tensor) -> Result<()> {
    let n = y.numel();
    expect(x, DType::F16, n, "addend")?;
    expect(y, DType::F16, n, "sum")?;
    d.dispatch_hybrid(
        "vae_add",
        &[x.binding()],
        &[y.binding()],
        &Params::default().u(n)?.0,
        [n.div_ceil(256), 1, 1],
        [256, 1, 1],
        0,
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
    expect(gamma, DType::F32, channels, "channel norm weight")?;
    expect(x, DType::F16, pixels * channels, "channel norm input")?;
    expect(y, DType::F16, pixels * channels, "channel norm output")?;
    d.dispatch_hybrid(
        "vae_chan_rms",
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
    let channels = mean.numel();
    if channels == 0 || !n.is_multiple_of(channels) {
        return Err(Error::Shape(format!("{n} latents of {channels} channels")));
    }
    expect(x, DType::F32, n, "latents")?;
    expect(std, DType::F32, channels, "latent std")?;
    expect(z, DType::F16, n, "denormalised latents")?;
    d.dispatch_hybrid(
        "vae_denorm",
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
    // The last element each operand's (row, k) addressing reaches.
    let reach = |o: &Operand| (o.rows - 1) * o.stride_row + (k - 1) * o.stride_k + 1;
    if a.rows == 0 || b.rows == 0 || k == 0 || ldc < b.rows {
        return Err(Error::Shape(format!(
            "gemm [{}, {k}] x [{}, {k}] into rows of {ldc}",
            a.rows, b.rows
        )));
    }
    expect(a.t, DType::F16, reach(a), "gemm A")?;
    expect(b.t, DType::F16, reach(b), "gemm B")?;
    expect(c, c.dtype(), (a.rows - 1) * ldc + b.rows, "gemm C")?;
    if !matches!(c.dtype(), DType::F32 | DType::F16) {
        return Err(Error::DType);
    }
    let args = Params::default()
        .u(a.rows)?
        .u(b.rows)?
        .u(k)?
        .u(a.stride_row)?
        .u(b.stride_k)?
        .u(b.stride_row)?
        .u(ldc)?;
    d.dispatch_hybrid(
        if wide { "vae_gemm_f32" } else { "vae_gemm_f16" },
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
    if rows == 0 || cols == 0 || ld < cols {
        return Err(Error::Shape(format!(
            "transpose of {rows} x {cols} rows of {ld}"
        )));
    }
    expect(x, DType::F16, (rows - 1) * ld + cols, "transpose input")?;
    expect(y, DType::F16, rows * cols, "transpose output")?;
    d.dispatch_hybrid(
        "vae_transpose",
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
    expect(s, DType::F32, rows * cols, "scores")?;
    expect(p, DType::F16, rows * cols, "softmax output")?;
    d.dispatch_hybrid(
        "vae_softmax_rows",
        &[s.binding()],
        &[p.binding()],
        &Params::default().u(rows)?.u(cols)?.f(scale).0,
        [rows, 1, 1],
        [256, 1, 1],
        0,
    )
}
