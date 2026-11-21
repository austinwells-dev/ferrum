//! Inkling 2's vision tower (`clip.vision.projector_type = inkling2v`): a ViT
//! with a class token, a learned position table resized (bilinear, half-pixel
//! centres) to the patch grid, 2-D rotary attention, a pre-norm and per-branch
//! layer scales, then a 2x2 merge through a two-layer GELU projector. The
//! class token's output is dropped.
//!
//! Images are not stretched: each side is padded to whole merged tokens (40
//! pixels), the padding normalised as a pixel value of -1, as llama.cpp's
//! reference does. Only an image over the token cap is first scaled down.
//!
//! The reference rotates adjacent pairs, x on the first half of each head and
//! y on the second. The Q and K rows of each head are permuted at load so the
//! half-split rotary kernel (`v_rope`) computes the same scores.
use super::*;

pub(super) struct Tower {
    layers: Vec<Layer>,
    patch: (Tensor, Tensor),
    /// `[table * table, hidden]`.
    position: Vec<f32>,
    /// Class embedding plus its position embedding.
    class: Vec<f32>,
    pre_ln: (Tensor, Tensor),
    fc1: (Tensor, Tensor),
    fc2: (Tensor, Tensor),
    pub weight_bytes: usize,
}

/// Head-dimension order the half-split rotary kernel needs: new index `i`
/// takes old index `perm[i]`.
fn rotary_permutation(head_dim: usize) -> Vec<usize> {
    let (half, quarter) = (head_dim / 2, head_dim / 4);
    let mut perm = vec![0; head_dim];
    for i in 0..quarter {
        // x pairs (0..half), then y pairs (half..head_dim).
        perm[i] = 2 * i;
        perm[i + half] = 2 * i + 1;
        perm[quarter + i] = half + 2 * i;
        perm[quarter + i + half] = half + 2 * i + 1;
    }
    perm
}

pub(super) fn load(
    d: &MetalDevice,
    config: &VisionConfig,
    source: &mut dyn Source,
) -> Result<Tower> {
    let bad = |what: String| Error::Config(format!("unsupported inkling2v tower: {what}"));
    let (h, ffn) = (config.hidden, config.ffn);
    let hd = config.head_dim();
    if config.merge != 2 || !hd.is_multiple_of(4) {
        return Err(bad(format!("merge {} with {hd}-wide heads", config.merge)));
    }
    let mut load = Loader {
        d,
        source,
        bytes: 0,
    };
    let f16 = |load: &mut Loader, rows: usize, cols: usize, values: &[f32]| {
        load.bytes += rows * cols * 2;
        Tensor::from_f32(load.d, [rows, cols], DType::F16, values)
    };
    let zeros = |load: &mut Loader, len: usize| {
        load.bytes += len * 4;
        Tensor::zeros(load.d, [len], DType::F32)
    };
    let values = |load: &mut Loader, name: &str, len: usize| -> Result<Vec<f32>> {
        let v = load.source.values(name)?;
        match v.len() == len {
            true => Ok(v),
            false => Err(bad(format!(
                "{name} has {} values, expected {len}",
                v.len()
            ))),
        }
    };
    let patch_w = values(&mut load, "v.patch_embd.weight", h * config.patch_len())?;
    let patch = (
        f16(&mut load, h, config.patch_len(), &patch_w)?,
        zeros(&mut load, h)?,
    );
    let position = values(
        &mut load,
        "v.position_embd.weight",
        config.table * config.table * h,
    )?;
    let class: Vec<f32> = values(&mut load, "v.class_embd", h)?
        .iter()
        .zip(values(&mut load, "v.class_position_embd.weight", h)?)
        .map(|(a, b)| a + b)
        .collect();
    let perm = rotary_permutation(hd);
    let mut layers = Vec::with_capacity(config.layers);
    for i in 0..config.layers {
        let n = |s: &str| format!("v.blk.{i}.{s}");
        // Q, K (rows permuted per head) and V fused into one projection.
        let (mut qkv, mut qkv_b) = (Vec::with_capacity(3 * h * h), Vec::with_capacity(3 * h));
        for part in ["attn_q", "attn_k", "attn_v"] {
            let w = values(&mut load, &n(&format!("{part}.weight")), h * h)?;
            let b = values(&mut load, &n(&format!("{part}.bias")), h)?;
            for row in 0..h {
                let src = match part {
                    "attn_v" => row,
                    _ => row / hd * hd + perm[row % hd],
                };
                qkv.extend_from_slice(&w[src * h..(src + 1) * h]);
                qkv_b.push(b[src]);
            }
        }
        // Layer scales fold into the projection that ends each branch.
        let scaled = |load: &mut Loader, name: &str, scale: &str, rows: usize, cols: usize| {
            let ls = values(load, &n(&format!("{scale}.weight")), rows)?;
            let mut w = values(load, &n(&format!("{name}.weight")), rows * cols)?;
            let mut b = values(load, &n(&format!("{name}.bias")), rows)?;
            for r in 0..rows {
                w[r * cols..(r + 1) * cols]
                    .iter_mut()
                    .for_each(|x| *x *= ls[r]);
                b[r] *= ls[r];
            }
            load.bytes += rows * 4;
            Ok::<_, Error>((
                f16(load, rows, cols, &w)?,
                Tensor::from_f32(load.d, [rows], DType::F32, &b)?,
            ))
        };
        layers.push(Layer {
            ln1: load.affine(&n("ln1"), h, None)?,
            ln2: load.affine(&n("ln2"), h, None)?,
            qkv: (
                f16(&mut load, 3 * h, h, &qkv)?,
                Tensor::from_f32(d, [3 * h], DType::F32, &qkv_b)?,
            ),
            out: scaled(&mut load, "attn_out", "ls1", h, h)?,
            up: load.affine(&n("ffn_up"), ffn, Some(h))?,
            down: scaled(&mut load, "ffn_down", "ls2", h, ffn)?,
        });
    }
    let merged = h * config.merge * config.merge;
    let fc1 = (
        load.source.matrix(d, "mm.1.weight", merged, merged)?,
        zeros(&mut load, merged)?,
    );
    let fc2 = (
        load.source
            .matrix(d, "mm.2.weight", config.out_dim, merged)?,
        zeros(&mut load, config.out_dim)?,
    );
    load.bytes += (merged + config.out_dim) * merged * 2;
    Ok(Tower {
        layers,
        patch,
        position,
        class,
        pre_ln: load.affine("v.pre_ln", h, None)?,
        fc1,
        fc2,
        weight_bytes: load.bytes,
    })
}

/// The size an image is resized to before padding (its own size unless its
/// padded area exceeds `max_tokens` merged tokens), and the padded size.
pub(super) fn sizes(
    width: usize,
    height: usize,
    align: usize,
    max_tokens: usize,
) -> ((usize, usize), (usize, usize)) {
    let up = |x: usize| x.div_ceil(align) * align;
    let max_pixels = max_tokens * align * align;
    let size = if up(width) * up(height) <= max_pixels {
        (width, height)
    } else {
        // llama.cpp's `calc_size_preserved_ratio` with only a pixel cap.
        let a = align as f64;
        let (w, h) = (width as f64, height as f64);
        let round = |x: f64| ((x / a).round() * a).max(a) as usize;
        let (w_bar, h_bar) = (round(w), round(h));
        if w_bar * h_bar > max_pixels {
            let beta = (h * w / max_pixels as f64).sqrt();
            let floor = |x: f64| (((x / beta) / a).floor() * a).max(a) as usize;
            (floor(w), floor(h))
        } else {
            (w_bar, h_bar)
        }
    };
    (size, (up(size.0), up(size.1)))
}

impl Tower {
    /// Normalised, padded pixels of `rgb`, at most `max_tokens` tokens.
    pub(super) fn prepare(
        &self,
        model: &VisionModel,
        rgb: &image::RgbImage,
        hash: u64,
        max_tokens: usize,
    ) -> Prepared {
        let c = &model.config;
        let (w, h) = (rgb.width() as usize, rgb.height() as usize);
        let ((rw, rh), (width, height)) = sizes(w, h, c.factor(), max_tokens);
        let resized;
        let rgb = if (rw, rh) == (w, h) {
            rgb
        } else {
            resized = image::imageops::resize(rgb, rw as u32, rh as u32, model.filter);
            &resized
        };
        let mut pixels = Vec::with_capacity(3 * width * height);
        for ch in 0..3 {
            let pad = (-1. / 255. - c.mean[ch]) / c.std[ch];
            for y in 0..height {
                for x in 0..width {
                    pixels.push(if x < rw && y < rh {
                        (f32::from(rgb.get_pixel(x as u32, y as u32)[ch]) / 255. - c.mean[ch])
                            / c.std[ch]
                    } else {
                        pad
                    });
                }
            }
        }
        Prepared {
            width,
            height,
            pixels,
            hash,
        }
    }

    /// The learned table resized to `gh x gw` (bilinear, half-pixel centres,
    /// edges clamped, as `ggml_interpolate`), in merge-block order.
    fn position_rows(&self, c: &VisionConfig, gh: usize, gw: usize) -> Vec<f32> {
        let (t, h) = (c.table, c.hidden);
        if (gh, gw) == (t, t) {
            return block_order(gh, gw, c.merge)
                .flat_map(|(r, k)| {
                    self.position[(r * t + k) * h..(r * t + k + 1) * h]
                        .iter()
                        .copied()
                })
                .collect();
        }
        let axis = |len: usize| -> Vec<(usize, usize, f32)> {
            (0..len)
                .map(|i| {
                    let at = ((i as f32 + 0.5) * t as f32 / len as f32 - 0.5).max(0.);
                    let lo = (at.floor() as usize).min(t - 1);
                    (lo, (lo + 1).min(t - 1), at - lo as f32)
                })
                .collect()
        };
        let (rows, cols) = (axis(gh), axis(gw));
        let mut out = Vec::with_capacity(gh * gw * h);
        for (row, col) in block_order(gh, gw, c.merge) {
            let (r0, r1, dr) = rows[row];
            let (c0, c1, dc) = cols[col];
            let at = |r: usize, k: usize| &self.position[(r * t + k) * h..(r * t + k + 1) * h];
            let (a, b, cc, e) = (at(r0, c0), at(r0, c1), at(r1, c0), at(r1, c1));
            for i in 0..h {
                out.push(
                    a[i] * (1. - dr) * (1. - dc)
                        + b[i] * (1. - dr) * dc
                        + cc[i] * dr * (1. - dc)
                        + e[i] * dr * dc,
                );
            }
        }
        out
    }

    pub(super) fn encode(
        &self,
        model: &VisionModel,
        d: &MetalDevice,
        image: &Prepared,
    ) -> Result<ImageEmbedding> {
        let c = &model.config;
        let (gh, gw) = (image.height / c.patch, image.width / c.patch);
        if gh == 0 || gw == 0 || !gh.is_multiple_of(c.merge) || !gw.is_multiple_of(c.merge) {
            return Err(Error::Parameter(format!(
                "a {}x{} image is not whole {}-pixel tokens",
                image.width,
                image.height,
                c.factor()
            )));
        }
        let n = gh * gw;
        // The class token is row 0.
        let rows = n + 1;
        let (h, hd) = (c.hidden, c.head_dim());
        if rows.saturating_mul(rows).saturating_mul(4) > model.score_bytes {
            return Err(Error::Parameter(format!(
                "a {gh}x{gw} patch image needs one attention matrix larger than the encoder allows; lower --image-tokens"
            )));
        }
        let tensor = |dims: &[usize], data: &[f32]| Tensor::from_f32(d, dims, DType::F32, data);
        let patches = tensor(&[n, c.patch_len()], &model.patchify(image))?;
        let (cos, sin) = rope_tables(gh, gw, c.merge, hd);
        let cos = tensor(&[rows, hd / 2], &cos)?;
        let sin = tensor(&[rows, hd / 2], &sin)?;
        let mut first = vec![0f32; rows * h];
        first[..h].copy_from_slice(&self.class);
        let x0 = tensor(&[rows, h], &first)?;
        let x = Tensor::zeros(d, [rows, h], DType::F32)?;
        let xn = Tensor::zeros(d, [rows, h], DType::F32)?;
        let qkv = Tensor::zeros(d, [rows, 3 * h], DType::F32)?;
        let att = Tensor::zeros(d, [rows, h], DType::F32)?;
        let hid = Tensor::zeros(d, [rows, c.ffn], DType::F32)?;
        let tmp = Tensor::zeros(d, [rows, h], DType::F32)?;
        let group = (model.score_bytes / (rows * rows * 4)).clamp(1, c.heads);
        let scores = Tensor::zeros(d, [group * rows * rows], DType::F32)?;
        let scale = 1. / (hd as f32).sqrt();
        let execution = d.execution_with_shared_encoder(true)?;
        let patch_rows = x0.view(h, [n, h])?;
        model.linear(
            d,
            &patches,
            &self.patch,
            &patch_rows,
            n,
            h,
            c.patch_len(),
            0,
        )?;
        let pos = tensor(&[n, h], &self.position_rows(c, gh, gw))?;
        model.add(d, &pos, &patch_rows, n * h)?;
        model.norm(d, &x0, &self.pre_ln, &x, rows, h)?;
        for layer in &self.layers {
            model.norm(d, &x, &layer.ln1, &xn, rows, h)?;
            model.linear(d, &xn, &layer.qkv, &qkv, rows, 3 * h, h, 0)?;
            d.dispatch_hybrid(
                "v_rope",
                &[cos.binding(), sin.binding()],
                &[qkv.binding()],
                &Params::default()
                    .u(rows)?
                    .u(c.heads)?
                    .u(hd)?
                    .u(3 * h)?
                    .u(h)?
                    .0,
                [(rows * 2 * c.heads * (hd / 2)).div_ceil(256), 1, 1],
                [256, 1, 1],
                0,
            )?;
            let mut head = 0;
            while head < c.heads {
                let g = group.min(c.heads - head);
                model.gemm(
                    d,
                    &Gemm {
                        m: rows,
                        n: rows,
                        k: hd,
                        lda: 3 * h,
                        ldb_n: 3 * h,
                        ldb_k: 1,
                        ldc: rows,
                        batch: g,
                        strides: (hd, hd, rows * rows),
                        offsets: (head * hd, h + head * hd, 0),
                        alpha: scale,
                        ..Gemm::default()
                    },
                    &qkv,
                    &qkv,
                    None,
                    &scores,
                )?;
                d.dispatch_hybrid(
                    "v_softmax",
                    &[],
                    &[scores.binding()],
                    &Params::default().u(g * rows)?.u(rows)?.0,
                    [g * rows, 1, 1],
                    [256, 1, 1],
                    0,
                )?;
                model.gemm(
                    d,
                    &Gemm {
                        m: rows,
                        n: hd,
                        k: rows,
                        lda: rows,
                        ldb_n: 1,
                        ldb_k: 3 * h,
                        ldc: h,
                        batch: g,
                        strides: (rows * rows, hd, hd),
                        offsets: (0, 2 * h + head * hd, head * hd),
                        ..Gemm::default()
                    },
                    &scores,
                    &qkv,
                    None,
                    &att,
                )?;
                head += g;
            }
            model.linear(d, &att, &layer.out, &tmp, rows, h, h, 0)?;
            model.add(d, &tmp, &x, rows * h)?;
            model.norm(d, &x, &layer.ln2, &xn, rows, h)?;
            model.linear(d, &xn, &layer.up, &hid, rows, c.ffn, h, 2)?;
            model.linear(d, &hid, &layer.down, &tmp, rows, h, c.ffn, 0)?;
            model.add(d, &tmp, &x, rows * h)?;
        }
        // Drop the class token; four consecutive patch rows are one merge block.
        let tokens = n / (c.merge * c.merge);
        let merged = h * c.merge * c.merge;
        let grouped = x.view(h, [tokens, merged])?;
        let mid = Tensor::zeros(d, [tokens, merged], DType::F32)?;
        let out = Tensor::zeros(d, [tokens, c.out_dim], DType::F32)?;
        model.linear(d, &grouped, &self.fc1, &mid, tokens, merged, merged, 2)?;
        model.linear(d, &mid, &self.fc2, &out, tokens, c.out_dim, merged, 0)?;
        execution.finish()?;
        Ok(ImageEmbedding {
            hash: image.hash,
            grid: (gh / c.merge, gw / c.merge),
            rows: std::rc::Rc::new(out),
            deepstack: Vec::new(),
        })
    }
}

/// Rotary tables for the class token (position 0, 0) then the patches in
/// merge-block order at (column + 1, row + 1): the first half of the pairs
/// rotates with the column, the second with the row.
fn rope_tables(gh: usize, gw: usize, merge: usize, head_dim: usize) -> (Vec<f32>, Vec<f32>) {
    let half = head_dim / 2;
    let axis = half / 2;
    let freq: Vec<f32> = (0..axis)
        .map(|j| 10000f32.powf(-2. * j as f32 / half as f32))
        .collect();
    let positions =
        std::iter::once((0, 0)).chain(block_order(gh, gw, merge).map(|(r, k)| (k + 1, r + 1)));
    let (mut cos, mut sin) = (Vec::new(), Vec::new());
    for (x, y) in positions {
        for i in 0..half {
            let angle = if i < axis {
                x as f32 * freq[i]
            } else {
                y as f32 * freq[i - axis]
            };
            cos.push(angle.cos());
            sin.push(angle.sin());
        }
    }
    (cos, sin)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The permuted half-split rotation equals the reference's adjacent-pair
    /// rotation (x on the first half of the head, y on the second) applied
    /// before the permutation.
    #[test]
    fn permuted_rotation_matches_adjacent_pairs() {
        let hd = 16;
        let perm = rotary_permutation(hd);
        let v: Vec<f32> = (0..hd).map(|i| (i as f32 * 0.37).sin()).collect();
        let (x, y) = (3usize, 5usize);
        // Reference: rope over [0, hd/2) with x and over [hd/2, hd) with y,
        // pairs (2i, 2i+1), theta_i = base^(-2i / (hd/2)).
        let half = hd / 2;
        let mut reference = v.clone();
        for (offset, pos) in [(0, x), (half, y)] {
            for i in 0..half / 2 {
                let theta = pos as f32 * 10000f32.powf(-2. * i as f32 / half as f32);
                let (a, b) = (v[offset + 2 * i], v[offset + 2 * i + 1]);
                reference[offset + 2 * i] = a * theta.cos() - b * theta.sin();
                reference[offset + 2 * i + 1] = b * theta.cos() + a * theta.sin();
            }
        }
        // Ferrum: permute, then rotate pairs (j, j + half) with the tables.
        let permuted: Vec<f32> = perm.iter().map(|&p| v[p]).collect();
        let (cos, sin) = rope_tables(2, 2, 2, hd);
        // Row of patch (row y - 1, column x - 1) is not in a 2x2 grid; use the
        // formula directly with the table layout of one row.
        let row = |x: usize, y: usize| -> (Vec<f32>, Vec<f32>) {
            let freq = |j: usize| 10000f32.powf(-2. * j as f32 / half as f32);
            (0..half)
                .map(|i| {
                    let a = if i < half / 2 {
                        x as f32 * freq(i)
                    } else {
                        y as f32 * freq(i - half / 2)
                    };
                    (a.cos(), a.sin())
                })
                .unzip()
        };
        assert_eq!(cos.len(), 5 * half);
        assert_eq!(&cos[..half], &vec![1.; half][..]);
        assert_eq!(&sin[..half], &vec![0.; half][..]);
        let (c, s) = row(x, y);
        let mut rotated = permuted.clone();
        for j in 0..half {
            let (a, b) = (permuted[j], permuted[j + half]);
            rotated[j] = a * c[j] - b * s[j];
            rotated[j + half] = b * c[j] + a * s[j];
        }
        let back: Vec<f32> = (0..hd)
            .map(|old| rotated[perm.iter().position(|&p| p == old).unwrap()])
            .collect();
        for (a, b) in back.iter().zip(&reference) {
            assert!((a - b).abs() < 1e-5, "{back:?} vs {reference:?}");
        }
    }

    #[test]
    fn images_pad_to_whole_tokens_and_shrink_only_past_the_cap() {
        assert_eq!(sizes(100, 50, 40, 1024), ((100, 50), (120, 80)));
        assert_eq!(sizes(40, 40, 40, 1), ((40, 40), (40, 40)));
        let ((w, h), (pw, ph)) = sizes(4000, 3000, 40, 1024);
        assert!(pw * ph <= 1024 * 1600 && w == pw && h == ph, "{w}x{h}");
    }
}
