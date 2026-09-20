use crate::{DType, Error, MetalDevice, Result, Tensor};
/// Immutable active K/V tensors [length, KV heads, head dim]. Appends copy.
#[derive(Clone)]
pub struct KvCache {
    layers: Vec<Option<(Tensor, Tensor)>>,
    capacity: usize,
    heads: usize,
    dim: usize,
    dtype: DType,
}
impl KvCache {
    pub fn new(
        layers: usize,
        capacity: usize,
        heads: usize,
        dim: usize,
        dtype: DType,
    ) -> Result<Self> {
        if [layers, capacity, heads, dim].contains(&0) {
            return Err(Error::Cache("dimensions must be positive".into()));
        }
        let shape = crate::tensor::Shape::new([capacity, heads, dim])?;
        if shape.numel() > u32::MAX as usize {
            return Err(Error::Cache("index limit exceeded".into()));
        }
        Ok(Self {
            layers: vec![None; layers],
            capacity,
            heads,
            dim,
            dtype,
        })
    }
    pub fn layer_count(&self) -> usize {
        self.layers.len()
    }
    pub fn len(&self) -> Result<usize> {
        let first = self.layer_len(0)?;
        for i in 1..self.layers.len() {
            if self.layer_len(i)? != first {
                return Err(Error::Cache("layer length mismatch".into()));
            }
        }
        Ok(first)
    }
    pub fn is_empty(&self) -> Result<bool> {
        Ok(self.len()? == 0)
    }
    pub fn layer_len(&self, layer: usize) -> Result<usize> {
        Ok(self
            .layers
            .get(layer)
            .ok_or_else(|| Error::Cache("layer index out of range".into()))?
            .as_ref()
            .map_or(0, |(k, _)| k.shape().dimensions()[0]))
    }
    pub fn active(&self, layer: usize) -> Result<Option<(&Tensor, &Tensor)>> {
        Ok(self
            .layers
            .get(layer)
            .ok_or_else(|| Error::Cache("layer index out of range".into()))?
            .as_ref()
            .map(|(k, v)| (k, v)))
    }
    pub fn validate_for(&self, c: &crate::model::ModelConfig, extra: usize) -> Result<usize> {
        if self.layers.len() != c.num_layers
            || self.heads != c.num_key_value_heads
            || self.dim != c.head_dim
            || self.dtype != c.dtype
        {
            return Err(Error::Cache("model/cache layout mismatch".into()));
        }
        let offset = self.len()?;
        if offset
            .checked_add(extra)
            .is_none_or(|n| n > self.capacity || n > c.max_context_length)
        {
            return Err(Error::Cache("capacity/context overflow".into()));
        }
        Ok(offset)
    }
    pub fn append(&mut self, d: &MetalDevice, layer: usize, k: &Tensor, v: &Tensor) -> Result<()> {
        let offset = self.layer_len(layer)?;
        let dims = k.shape().dimensions();
        if dims.len() != 3 || dims[1..] != [self.heads, self.dim] || k.shape() != v.shape() {
            return Err(Error::Cache(
                "K/V require matching [S,KV heads,head dim]".into(),
            ));
        }
        if k.dtype() != self.dtype || v.dtype() != self.dtype {
            return Err(Error::DType);
        }
        if !d.owns(k.buffer()) || !d.owns(v.buffer()) {
            return Err(Error::DeviceMismatch);
        }
        if offset
            .checked_add(dims[0])
            .is_none_or(|n| n > self.capacity)
        {
            return Err(Error::Cache("capacity overflow".into()));
        }
        let pair = match self.active(layer)? {
            None => (k.clone(), v.clone()),
            Some((oldk, oldv)) => (
                d.concat_first(oldk, k)?.tensor,
                d.concat_first(oldv, v)?.tensor,
            ),
        };
        self.layers[layer] = Some(pair);
        Ok(())
    }
    pub fn reset(&mut self) {
        self.layers.fill(None);
    }
    pub fn bytes(&self) -> usize {
        self.layers
            .iter()
            .flatten()
            .map(|(k, v)| k.byte_size() + v.byte_size())
            .sum()
    }
}
