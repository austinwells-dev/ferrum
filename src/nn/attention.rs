use super::{Linear, kv_cache::KvCache};
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
    pub q_heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub theta: f32,
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
        let q = self
            .q
            .forward(d, x)?
            .reshape([s, self.q_heads, self.head_dim])?;
        let k = self
            .k
            .forward(d, x)?
            .reshape([s, self.kv_heads, self.head_dim])?;
        let v = self
            .v
            .forward(d, x)?
            .reshape([s, self.kv_heads, self.head_dim])?;
        let prefix = format!("layer.{layer}");
        for (name, t) in [("q", &q), ("k", &k), ("v", &v)] {
            record(&mut trace, format!("{prefix}.{name}"), t);
        }
        let q = d.rope_split(&q, offset, self.theta)?.tensor;
        let k = d.rope_split(&k, offset, self.theta)?.tensor;
        record(&mut trace, format!("{prefix}.rope_q"), &q);
        record(&mut trace, format!("{prefix}.rope_k"), &k);
        cache.append(d, layer, &k, &v)?;
        let (k, v) = cache
            .active(layer)?
            .ok_or_else(|| Error::Cache("missing active K/V".into()))?;
        let scores = d.attention_scores(&q, k)?.tensor;
        let scores = d
            .scale(&scores, (self.head_dim as f32).sqrt().recip())?
            .tensor;
        let t = k.shape().dimensions()[0];
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
        let probs = d.softmax(&masked)?.tensor;
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
        let output = self.output.forward(d, &merged)?;
        record(&mut trace, format!("{prefix}.attention"), &output);
        Ok(output)
    }
}
