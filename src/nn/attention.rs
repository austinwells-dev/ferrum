use super::{Linear, RmsNorm, kv_cache::KvCache};
use crate::model::architecture::QkNormLayout;
use crate::{Error, MetalDevice, Result, Tensor};
/// Optional diagnostic snapshots; retaining them is explicitly opt-in.
pub type Trace = std::collections::BTreeMap<String, Tensor>;
pub(crate) fn record(trace: &mut Option<&mut Trace>, name: impl Into<String>, t: &Tensor) {
    if let Some(trace) = trace {
        trace.insert(name.into(), t.clone());
    }
}
pub struct Attention {
    pub q: Linear,
    pub k: Linear,
    pub v: Linear,
    pub output: Linear,
    pub q_norm: Option<RmsNorm>,
    pub k_norm: Option<RmsNorm>,
    pub q_heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub theta: f32,
    pub qk_norm_layout: QkNormLayout,
    pub scale: f32,
}
impl Attention {
    pub fn forward(
        &self,
        d: &MetalDevice,
        x: &Tensor,
        cache: &mut KvCache,
        layer: usize,
        mut trace: Option<&mut Trace>,
    ) -> Result<Tensor> {
        if self.kv_heads == 0 || self.q_heads == 0 || !self.q_heads.is_multiple_of(self.kv_heads) {
            return Err(Error::Config("invalid GQA ratio".into()));
        }
        let s = *x
            .shape()
            .dimensions()
            .first()
            .ok_or_else(|| Error::Shape("attention needs [S,hidden]".into()))?;
        if x.shape().rank() != 2 {
            return Err(Error::Shape("attention needs [S,hidden]".into()));
        }
        let offset = cache.layer_len(layer)?;
        let prefix = format!("layer.{layer}");
        // Fused path: RoPE writes K straight into its cache slot and V lands in
        // its slot directly; projection biases fold into those kernels when no
        // q/k norm sits in between. Bit-identical to the unfused sequence.
        let fused = trace.is_none() && !d.reference_math() && d.fuse_rope_cache();
        let q = if fused {
            let fold = self.q_norm.is_none() && self.k_norm.is_none();
            let project = |linear: &crate::nn::Linear, name: &'static str| {
                d.profile_projection(name, || {
                    if fold {
                        linear.forward_unbiased(d, x)
                    } else {
                        linear.forward(d, x)
                    }
                })
            };
            let (q_bias, k_bias) = if fold {
                (self.q.bias(), self.k.bias())
            } else {
                (None, None)
            };
            let mut q = project(&self.q, "q_proj")?;
            let mut k = project(&self.k, "k_proj")?;
            if self.qk_norm_layout == QkNormLayout::Projection {
                if let Some(norm) = &self.q_norm {
                    q = norm.forward(d, &q)?;
                }
                if let Some(norm) = &self.k_norm {
                    k = norm.forward(d, &k)?;
                }
            }
            let mut q = q.reshape([s, self.q_heads, self.head_dim])?;
            let mut k = k.reshape([s, self.kv_heads, self.head_dim])?;
            let v = d.profile_projection("v_proj", || self.v.forward_unbiased(d, x))?;
            if self.qk_norm_layout == QkNormLayout::PerHead {
                if let Some(norm) = &self.q_norm {
                    q = norm.forward(d, &q)?;
                }
                if let Some(norm) = &self.k_norm {
                    k = norm.forward(d, &k)?;
                }
            }
            let rotated_q = crate::Tensor::output(d, q.shape().dimensions(), q.dtype())?;
            d.rope_split_into(&q, q_bias, offset, self.theta, &rotated_q)?;
            cache.append_with(d, layer, s, |k_slot, v_slot| {
                d.rope_split_into(&k, k_bias, offset, self.theta, k_slot)?;
                match self.v.bias() {
                    Some(bias) => d.bias_add_into(&v, bias, v_slot),
                    None => d.write_kv(&v, v_slot),
                }
            })?;
            rotated_q
        } else {
            let offset = cache.layer_len(layer)?;
            let mut q = d.profile_projection("q_proj", || self.q.forward(d, x))?;
            let mut k = d.profile_projection("k_proj", || self.k.forward(d, x))?;
            if self.qk_norm_layout == QkNormLayout::Projection {
                if let Some(norm) = &self.q_norm {
                    q = norm.forward(d, &q)?;
                }
                if let Some(norm) = &self.k_norm {
                    k = norm.forward(d, &k)?;
                }
            }
            let mut q = q.reshape([s, self.q_heads, self.head_dim])?;
            let mut k = k.reshape([s, self.kv_heads, self.head_dim])?;
            let v = d
                .profile_projection("v_proj", || self.v.forward(d, x))?
                .reshape([s, self.kv_heads, self.head_dim])?;
            if self.qk_norm_layout == QkNormLayout::PerHead {
                if let Some(norm) = &self.q_norm {
                    q = norm.forward(d, &q)?;
                }
                if let Some(norm) = &self.k_norm {
                    k = norm.forward(d, &k)?;
                }
            }
            for (name, t) in [("q", &q), ("k", &k), ("v", &v)] {
                record(&mut trace, format!("{prefix}.{name}"), t);
            }
            let q = d.rope_split(&q, offset, self.theta)?.tensor;
            let k = d.rope_split(&k, offset, self.theta)?.tensor;
            record(&mut trace, format!("{prefix}.rope_q"), &q);
            record(&mut trace, format!("{prefix}.rope_k"), &k);
            cache.append(d, layer, &k, &v)?;
            q
        };
        let (k, v) = cache
            .active(layer)?
            .ok_or_else(|| Error::Cache("missing active K/V".into()))?;
        if trace.is_none() && d.can_flash_decode(&q, k, v) {
            let merged = d
                .attention_decode(&q, k, v, self.scale)?
                .reshape([s, self.q_heads * self.head_dim])?;
            return d.profile_projection("o_proj", || self.output.forward(d, &merged));
        }
        let scores = d.attention_scores(&q, k)?.tensor;
        let t = k.shape().dimensions()[0];
        let scale = self.scale;
        let probs = if trace.is_none() && !d.reference_math() {
            d.attention_softmax(&scores, offset, scale)?.tensor
        } else {
            let scores = d.scale(&scores, scale)?.tensor;
            if trace.is_some() {
                for h in 0..self.q_heads {
                    record(
                        &mut trace,
                        format!("{prefix}.head.{h}.scores"),
                        &scores.view(h * s * t, [s, t])?,
                    );
                }
            }
            let masked = d.attention_mask(&scores, offset)?.tensor;
            d.softmax(&masked)?.tensor
        };
        if trace.is_some() {
            for h in 0..self.q_heads {
                record(
                    &mut trace,
                    format!("{prefix}.head.{h}.probs"),
                    &probs.view(h * s * t, [s, t])?,
                );
            }
        }
        let merged = d
            .attention_context(&probs, v)?
            .tensor
            .reshape([s, self.q_heads * self.head_dim])?;
        record(&mut trace, format!("{prefix}.context"), &merged);
        let output = d.profile_projection("o_proj", || self.output.forward(d, &merged))?;
        record(&mut trace, format!("{prefix}.attention"), &output);
        Ok(output)
    }
}
