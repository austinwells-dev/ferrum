//! Independent scalar full-sequence oracle. No tensors or GPU operations.
use crate::{
    DType, Result,
    model::{ModelConfig, tiny::CpuWeights},
};
use std::collections::BTreeMap;
pub type Trace = BTreeMap<String, Vec<f32>>;
fn rounded(ty: DType, v: Vec<f32>) -> Vec<f32> {
    v.into_iter().map(|x| ty.round(x)).collect()
}
struct Oracle<'a> {
    c: &'a ModelConfig,
    w: &'a CpuWeights,
}
impl Oracle<'_> {
    fn weight(&self, name: &str) -> &[f32] {
        &self.w[name].1
    }
    fn round(&self, v: Vec<f32>) -> Vec<f32> {
        rounded(self.c.dtype, v)
    }
    fn linear(&self, x: &[f32], name: &str) -> Vec<f32> {
        let (shape, w) = &self.w[&format!("{name}.weight")];
        let (o, k) = (shape[0], shape[1]);
        let mut y = self.round(
            x.chunks(k)
                .flat_map(|row| {
                    (0..o).map(move |j| {
                        (0..k)
                            .map(|i| row[i] as f64 * w[j * k + i] as f64)
                            .sum::<f64>() as f32
                    })
                })
                .collect(),
        );
        if let Some((_, b)) = self.w.get(&format!("{name}.bias")) {
            for (i, v) in y.iter_mut().enumerate() {
                *v = self.c.dtype.round(*v + b[i % o]);
            }
        }
        y
    }
    fn norm(&self, x: &[f32], name: &str) -> Vec<f32> {
        self.round(super::rmsnorm(
            x,
            self.weight(name),
            self.c.rms_norm_epsilon,
        ))
    }
    fn rope(&self, x: &[f32], heads: usize) -> Vec<f32> {
        let d = self.c.head_dim;
        let mut y = x.to_vec();
        for (h, row) in x.chunks(d).enumerate() {
            for j in 0..d / 2 {
                let angle = (h / heads) as f64
                    * (self.c.rope_theta as f64).powf(-((2 * j) as f64) / d as f64);
                y[h * d + j] =
                    (row[j] as f64 * angle.cos() - row[j + d / 2] as f64 * angle.sin()) as f32;
                y[h * d + j + d / 2] =
                    (row[j] as f64 * angle.sin() + row[j + d / 2] as f64 * angle.cos()) as f32;
            }
        }
        self.round(y)
    }
}
/// Fixture oracle: requires complete shape-validated weights and nonempty in-vocabulary tokens.
pub fn forward(c: &ModelConfig, w: &CpuWeights, tokens: &[u32]) -> Result<Trace> {
    c.validate()?;
    let oracle = Oracle { c, w };
    let s = tokens.len();
    let mut trace = Trace::new();
    let embedding = oracle.weight("embedding.weight");
    let mut x: Vec<_> = tokens
        .iter()
        .flat_map(|&id| {
            embedding[id as usize * c.hidden_size..(id as usize + 1) * c.hidden_size]
                .iter()
                .copied()
        })
        .collect();
    trace.insert("embedding".into(), x.clone());
    for l in 0..c.num_layers {
        let p = format!("layers.{l}");
        let t = format!("layer.{l}");
        let norm = oracle.norm(&x, &format!("{p}.input_norm.weight"));
        trace.insert(format!("{t}.norm"), norm.clone());
        let q = oracle.linear(&norm, &format!("{p}.q"));
        let k = oracle.linear(&norm, &format!("{p}.k"));
        let v = oracle.linear(&norm, &format!("{p}.v"));
        for (name, v) in [("q", &q), ("k", &k), ("v", &v)] {
            trace.insert(format!("{t}.{name}"), v.clone());
        }
        let q = oracle.rope(&q, c.num_attention_heads);
        let k = oracle.rope(&k, c.num_key_value_heads);
        trace.insert(format!("{t}.rope_q"), q.clone());
        trace.insert(format!("{t}.rope_k"), k.clone());
        let mut context = vec![0.; s * c.num_attention_heads * c.head_dim];
        for h in 0..c.num_attention_heads {
            let kv = h / (c.num_attention_heads / c.num_key_value_heads);
            let mut scores = vec![0.; s * s];
            for i in 0..s {
                for j in 0..s {
                    let dot = (0..c.head_dim)
                        .map(|z| {
                            q[(i * c.num_attention_heads + h) * c.head_dim + z] as f64
                                * k[(j * c.num_key_value_heads + kv) * c.head_dim + z] as f64
                        })
                        .sum::<f64>() as f32;
                    scores[i * s + j] = c
                        .dtype
                        .round(c.dtype.round(dot) * (c.head_dim as f32).sqrt().recip());
                }
            }
            trace.insert(format!("{t}.head.{h}.scores"), scores.clone());
            for i in 0..s {
                for j in i + 1..s {
                    scores[i * s + j] = f32::NEG_INFINITY;
                }
            }
            let probs = oracle.round(super::softmax(&scores, s));
            trace.insert(format!("{t}.head.{h}.probs"), probs.clone());
            for i in 0..s {
                for z in 0..c.head_dim {
                    let dot = (0..s)
                        .map(|j| {
                            probs[i * s + j] as f64
                                * v[(j * c.num_key_value_heads + kv) * c.head_dim + z] as f64
                        })
                        .sum::<f64>() as f32;
                    context[(i * c.num_attention_heads + h) * c.head_dim + z] = c.dtype.round(dot);
                }
            }
        }
        trace.insert(format!("{t}.context"), context.clone());
        let attention = oracle.linear(&context, &format!("{p}.o"));
        trace.insert(format!("{t}.attention"), attention.clone());
        x = oracle.round(super::add(&x, &attention));
        let norm = oracle.norm(&x, &format!("{p}.post_norm.weight"));
        trace.insert(format!("{t}.post_norm"), norm.clone());
        let gate = oracle.linear(&norm, &format!("{p}.gate"));
        let up = oracle.linear(&norm, &format!("{p}.up"));
        let gate = oracle.round(super::silu(&gate));
        let hidden = oracle.round(super::mul(&gate, &up));
        let mlp = oracle.linear(&hidden, &format!("{p}.down"));
        trace.insert(format!("{t}.mlp"), mlp.clone());
        x = oracle.round(super::add(&x, &mlp));
        trace.insert(format!("{t}.output"), x.clone());
    }
    let x = oracle.norm(&x, "final_norm.weight");
    trace.insert("final_hidden".into(), x.clone());
    // Tied LM projection uses the exact same matrix as embedding.
    let name = if c.tie_word_embeddings {
        "embedding"
    } else {
        "lm_head"
    };
    let mut logits = oracle.linear(&x, name);
    if c.tie_word_embeddings
        && let Some((_, bias)) = w.get("lm_head.bias")
    {
        for (i, v) in logits.iter_mut().enumerate() {
            *v = c.dtype.round(*v + bias[i % c.vocab_size]);
        }
    }
    trace.insert("logits".into(), logits);
    Ok(trace)
}
