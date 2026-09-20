//! Narrow contiguous copies and transformer operations; no strided views.
use super::*;
impl MetalDevice {
    /// Integer IDs never pass through floating-point conversion.
    pub fn embedding_gather(&self, weight: &Tensor, tokens: &[u32]) -> Result<Output> {
        let dims = weight.shape().dimensions();
        if dims.len() != 2 || dims.contains(&0) {
            return Err(Error::Shape("embedding requires [vocab,hidden]".into()));
        }
        if !self.owns(weight.buffer()) {
            return Err(Error::DeviceMismatch);
        }
        for &id in tokens {
            if id as usize >= dims[0] {
                return Err(Error::Token { id, vocab: dims[0] });
            }
        }
        let shape = crate::tensor::Shape::new([tokens.len(), dims[1]])?;
        index(shape.numel())?;
        let bytes_len = tokens
            .len()
            .checked_mul(4)
            .ok_or_else(|| Error::Shape("token bytes overflow".into()))?;
        let bytes: Vec<u8> = tokens.iter().flat_map(|id| id.to_le_bytes()).collect();
        let ids = Tensor::from_le_bytes(
            self,
            [bytes_len / weight.dtype().size_bytes()],
            weight.dtype(),
            &bytes,
        )?;
        if ids.storage_info().alignment < 4 {
            return Err(Error::Range(
                "token ID carrier requires u32 alignment".into(),
            ));
        }
        let mut p = [0; 9];
        p[1] = index(dims[1])?;
        self.run(
            "embedding_gather",
            weight,
            Some(&ids),
            shape.dimensions(),
            p,
            [shape.numel(), 1],
        )
    }

    pub fn copy_range(&self, a: &Tensor, start: usize, dims: &[usize]) -> Result<Output> {
        let n = crate::tensor::Shape::new(dims)?.numel();
        if start.checked_add(n).is_none_or(|end| end > a.numel()) {
            return Err(Error::Range("copy exceeds input".into()));
        }
        let mut p = [0; 9];
        p[1] = index(start)?;
        self.run("copy_range", a, None, dims, p, [n, 1])
    }
    pub fn concat_first(&self, a: &Tensor, b: &Tensor) -> Result<Output> {
        let ad = a.shape().dimensions();
        let bd = b.shape().dimensions();
        if ad.is_empty() || ad.len() != bd.len() || ad[1..] != bd[1..] {
            return Err(Error::Shape(
                "concat requires matching trailing axes".into(),
            ));
        }
        let mut dims = ad.to_vec();
        dims[0] = ad[0]
            .checked_add(bd[0])
            .ok_or_else(|| Error::Shape("concat overflow".into()))?;
        let n = crate::tensor::Shape::new(&dims)?.numel();
        self.run("concat_flat", a, Some(b), &dims, [0; 9], [n, 1])
    }
    pub fn transpose2(&self, a: &Tensor) -> Result<Output> {
        let d = a.shape().dimensions();
        if d.len() != 2 {
            return Err(Error::Shape("transpose requires rank two".into()));
        }
        let mut p = [0; 9];
        p[1] = index(d[0])?;
        p[2] = index(d[1])?;
        self.run("transpose2", a, None, &[d[1], d[0]], p, [a.numel(), 1])
    }
    pub fn swap01(&self, a: &Tensor) -> Result<Output> {
        let d = a.shape().dimensions();
        if d.len() != 3 {
            return Err(Error::Shape("swap01 requires rank three".into()));
        }
        let mut p = [0; 9];
        for j in 0..3 {
            p[j + 1] = index(d[j])?;
        }
        self.run("swap01", a, None, &[d[1], d[0], d[2]], p, [a.numel(), 1])
    }
    pub fn select_head(&self, a: &Tensor, head: usize) -> Result<Output> {
        let d = a.shape().dimensions();
        if d.len() != 3 || head >= d[1] {
            return Err(Error::Shape("head index out of range for [S,H,D]".into()));
        }
        let mut p = [0; 9];
        p[1] = index(head)?;
        p[2] = index(d[1])?;
        p[3] = index(d[2])?;
        self.run("select_head", a, None, &[d[0], d[2]], p, [d[0] * d[2], 1])
    }
    pub fn bias_add(&self, a: &Tensor, bias: &Tensor) -> Result<Output> {
        let w = width(a)?;
        if bias.shape().dimensions() != [w] {
            return Err(Error::Shape("bias must match last dimension".into()));
        }
        let mut p = [0; 9];
        p[1] = index(w)?;
        self.run(
            "bias_add",
            a,
            Some(bias),
            a.shape().dimensions(),
            p,
            [a.numel(), 1],
        )
    }
    pub fn scale(&self, a: &Tensor, value: f32) -> Result<Output> {
        if !value.is_finite() {
            return Err(Error::Parameter("scale must be finite".into()));
        }
        let mut p = [0; 9];
        p[7] = value.to_bits();
        self.run("scale", a, None, a.shape().dimensions(), p, [a.numel(), 1])
    }
    pub fn causal_mask(&self, a: &Tensor, offset: usize) -> Result<Output> {
        let d = a.shape().dimensions();
        if d.len() != 2 || d[1] == 0 || offset.checked_add(d[0]) != Some(d[1]) {
            return Err(Error::Parameter(
                "causal scores require [M,offset+M]".into(),
            ));
        }
        let mut p = [0; 9];
        p[1] = index(d[1])?;
        p[5] = index(offset)?;
        self.run("causal_mask", a, None, d, p, [a.numel(), 1])
    }
    pub fn rope_split(&self, a: &Tensor, offset: usize, theta: f32) -> Result<Output> {
        let d = a.shape().dimensions();
        if d.len() != 3
            || d[1] == 0
            || d[2] == 0
            || !d[2].is_multiple_of(2)
            || !theta.is_finite()
            || theta <= 0.
        {
            return Err(Error::Parameter(
                "split RoPE requires [S,H,even D], positive heads and theta".into(),
            ));
        }
        index(
            offset
                .checked_add(d[0])
                .ok_or_else(|| Error::Parameter("position overflow".into()))?,
        )?;
        let mut p = [0; 9];
        p[1] = index(d[1])?;
        p[5] = index(offset)?;
        p[6] = index(d[2])?;
        p[8] = theta.to_bits();
        self.run("rope_split", a, None, d, p, [a.numel() / 2, 1])
    }
}

impl MetalDevice {
    /// Internal append-only KV write. Caller reserves an unpublished suffix before
    /// calling; no published tensor may cover the destination range.
    pub(crate) fn write_kv(&self, source: &Tensor, suffix: &Tensor) -> Result<()> {
        if source.dtype() != suffix.dtype() || source.numel() != suffix.numel() {
            return Err(Error::Cache("KV write shape/dtype mismatch".into()));
        }
        let mut p = [0; 9];
        p[0] = index(source.numel())?;
        p[4] = source.dtype() as u32;
        self.dispatch(
            "copy_range",
            &[source.binding(), source.binding(), suffix.binding()],
            &p,
            [source.numel(), 1],
            false,
        )?;
        Ok(())
    }
}
