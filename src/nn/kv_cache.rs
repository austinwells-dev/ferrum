use crate::{DType, Error, MetalDevice, Result, Tensor};
use std::{cell::Cell, rc::Rc};
#[derive(Clone)]
struct LayerCache {
    active: (Tensor, Tensor),
    storage: (Tensor, Tensor),
    // Monotonic reservation, shared with snapshots. A fork must allocate its own
    // storage if it would overwrite any previously published/reserved suffix.
    reserved: Rc<Cell<usize>>,
}
/// Immutable active prefix views over append-only, capacity-backed K/V storage.
#[derive(Clone)]
pub struct KvCache {
    layers: Vec<Option<LayerCache>>,
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
            .map_or(0, |entry| entry.active.0.shape().dimensions()[0]))
    }
    pub fn active(&self, layer: usize) -> Result<Option<(&Tensor, &Tensor)>> {
        Ok(self
            .layers
            .get(layer)
            .ok_or_else(|| Error::Cache("layer index out of range".into()))?
            .as_ref()
            .map(|entry| (&entry.active.0, &entry.active.1)))
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
        let end = offset + dims[0];
        let row = self.heads * self.dim;
        let existing = self.layers[layer].as_ref();
        let entry = if let Some(entry) = existing
            .filter(|e| e.reserved.get() == offset && end <= e.storage.0.shape().dimensions()[0])
        {
            entry.clone()
        } else {
            // Reserve a small initial block, then grow geometrically. The logical
            // context bound is unchanged; only growth/branching copies old history.
            let physical_capacity = end.max(256).next_power_of_two().min(self.capacity);
            let storage = (
                Tensor::zeros(d, [physical_capacity, self.heads, self.dim], self.dtype)?,
                Tensor::zeros(d, [physical_capacity, self.heads, self.dim], self.dtype)?,
            );
            if let Some(old) = existing {
                // Copy only on geometric growth or explicit cache branching/rollback.
                d.write_kv(
                    &old.active.0,
                    &storage.0.view(0, old.active.0.shape().dimensions())?,
                )?;
                d.write_kv(
                    &old.active.1,
                    &storage.1.view(0, old.active.1.shape().dimensions())?,
                )?;
            }
            LayerCache {
                active: (
                    storage.0.view(0, [offset, self.heads, self.dim])?,
                    storage.1.view(0, [offset, self.heads, self.dim])?,
                ),
                storage,
                reserved: Rc::new(Cell::new(offset)),
            }
        };
        // Reserve before any fallible dispatch; failures can never authorize reuse
        // of a range that an already encoded command might still write.
        entry.reserved.set(end);
        d.write_kv(k, &entry.storage.0.view(offset * row, dims)?)?;
        d.write_kv(v, &entry.storage.1.view(offset * row, dims)?)?;
        self.layers[layer] = Some(LayerCache {
            active: (
                entry.storage.0.view(0, [end, self.heads, self.dim])?,
                entry.storage.1.view(0, [end, self.heads, self.dim])?,
            ),
            ..entry
        });
        Ok(())
    }
    pub fn reset(&mut self) {
        self.layers.fill(None);
    }
    pub fn reserved_bytes(&self) -> usize {
        self.layers
            .iter()
            .flatten()
            .map(|entry| entry.storage.0.byte_size() + entry.storage.1.byte_size())
            .sum()
    }
    pub fn bytes(&self) -> usize {
        self.layers
            .iter()
            .flatten()
            .map(|entry| entry.active.0.byte_size() + entry.active.1.byte_size())
            .sum()
    }
}
