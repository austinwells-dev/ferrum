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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheLayerKind {
    KeyValue,
    Convolution { kernel_size: usize },
}

/// Immutable active prefix views over append-only, capacity-backed K/V storage.
#[derive(Clone)]
pub struct KvCache {
    layers: Vec<Option<LayerCache>>,
    conv_states: Vec<Option<Tensor>>,
    kinds: Vec<CacheLayerKind>,
    capacity: usize,
    heads: usize,
    dim: usize,
    state_width: usize,
    sequence_len: usize,
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
        let kinds = vec![CacheLayerKind::KeyValue; layers];
        Self::with_layout(layers, capacity, heads, dim, heads * dim, dtype, kinds)
    }

    pub fn with_layout(
        layers: usize,
        capacity: usize,
        heads: usize,
        dim: usize,
        state_width: usize,
        dtype: DType,
        kinds: Vec<CacheLayerKind>,
    ) -> Result<Self> {
        if [layers, capacity, heads, dim].contains(&0) {
            return Err(Error::Cache("dimensions must be positive".into()));
        }
        if state_width == 0 || kinds.len() != layers {
            return Err(Error::Cache("invalid per-layer state layout".into()));
        }
        if kinds
            .iter()
            .any(|kind| matches!(kind, CacheLayerKind::Convolution { kernel_size: 0 }))
        {
            return Err(Error::Cache(
                "convolution state length must be positive".into(),
            ));
        }
        let shape = crate::tensor::Shape::new([capacity, heads, dim])?;
        if shape.numel() > u32::MAX as usize {
            return Err(Error::Cache("index limit exceeded".into()));
        }
        Ok(Self {
            layers: vec![None; layers],
            conv_states: vec![None; layers],
            kinds,
            capacity,
            heads,
            dim,
            state_width,
            sequence_len: 0,
            dtype,
        })
    }
    pub fn layer_count(&self) -> usize {
        self.layers.len()
    }
    pub fn len(&self) -> Result<usize> {
        let mut attention_len = None;
        for (index, kind) in self.kinds.iter().enumerate() {
            if *kind == CacheLayerKind::KeyValue {
                let length = self.layers[index]
                    .as_ref()
                    .map_or(0, |entry| entry.active.0.shape().dimensions()[0]);
                if attention_len.is_some_and(|first| first != length) {
                    return Err(Error::Cache("layer length mismatch".into()));
                }
                attention_len = Some(length);
            }
        }
        let length = attention_len.unwrap_or(self.sequence_len);
        if self
            .kinds
            .iter()
            .any(|kind| matches!(kind, CacheLayerKind::Convolution { .. }))
            && length != self.sequence_len
        {
            return Err(Error::Cache("hybrid state length mismatch".into()));
        }
        Ok(length)
    }
    pub fn is_empty(&self) -> Result<bool> {
        Ok(self.len()? == 0)
    }
    pub fn layer_len(&self, layer: usize) -> Result<usize> {
        let kind = *self
            .kinds
            .get(layer)
            .ok_or_else(|| Error::Cache("layer index out of range".into()))?;
        Ok(match kind {
            CacheLayerKind::KeyValue => self.layers[layer]
                .as_ref()
                .map_or(0, |entry| entry.active.0.shape().dimensions()[0]),
            CacheLayerKind::Convolution { .. } => self.sequence_len,
        })
    }
    pub fn active(&self, layer: usize) -> Result<Option<(&Tensor, &Tensor)>> {
        if self.kinds.get(layer) != Some(&CacheLayerKind::KeyValue) {
            return Err(Error::Cache(
                "K/V requested for a non-attention layer".into(),
            ));
        }
        Ok(self
            .layers
            .get(layer)
            .ok_or_else(|| Error::Cache("layer index out of range".into()))?
            .as_ref()
            .map(|entry| (&entry.active.0, &entry.active.1)))
    }

    pub fn conv_state(&self, layer: usize) -> Result<Option<&Tensor>> {
        if !matches!(
            self.kinds.get(layer),
            Some(CacheLayerKind::Convolution { .. })
        ) {
            return Err(Error::Cache(
                "convolution state requested for a non-conv layer".into(),
            ));
        }
        Ok(self.conv_states[layer].as_ref())
    }

    pub fn set_conv_state(&mut self, d: &MetalDevice, layer: usize, state: Tensor) -> Result<()> {
        let kind = *self
            .kinds
            .get(layer)
            .ok_or_else(|| Error::Cache("layer index out of range".into()))?;
        let CacheLayerKind::Convolution { kernel_size } = kind else {
            return Err(Error::Cache("state update targets a non-conv layer".into()));
        };
        if state.shape().dimensions() != [kernel_size, self.state_width]
            || state.dtype() != self.dtype
        {
            return Err(Error::Cache(
                "convolution state shape/dtype mismatch".into(),
            ));
        }
        if !d.owns(state.buffer()) {
            return Err(Error::DeviceMismatch);
        }
        self.conv_states[layer] = Some(state);
        Ok(())
    }

    pub fn validate_layout(&self, expected: &[CacheLayerKind]) -> Result<()> {
        if self.kinds != expected {
            return Err(Error::Cache(
                "model/cache layer-state layout mismatch".into(),
            ));
        }
        Ok(())
    }

    pub fn set_sequence_len(&mut self, sequence_len: usize) -> Result<()> {
        if sequence_len > self.capacity {
            return Err(Error::Cache("sequence length exceeds capacity".into()));
        }
        self.sequence_len = sequence_len;
        self.len()?;
        if sequence_len > 0 {
            for (index, kind) in self.kinds.iter().enumerate() {
                if matches!(kind, CacheLayerKind::Convolution { .. })
                    && self.conv_states[index].is_none()
                {
                    return Err(Error::Cache(
                        "missing convolution state after update".into(),
                    ));
                }
            }
        }
        Ok(())
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
        self.append_with(d, layer, dims[0], |k_slot, v_slot| {
            d.write_kv(k, k_slot)?;
            d.write_kv(v, v_slot)
        })
    }
    /// Reserve `rows` new K/V rows and let `write` fill the two `[rows, KV
    /// heads, head dim]` slots directly (for example RoPE writing rotated K).
    /// The slots are unpublished until `write` succeeds.
    pub(crate) fn append_with(
        &mut self,
        d: &MetalDevice,
        layer: usize,
        rows: usize,
        write: impl FnOnce(&Tensor, &Tensor) -> Result<()>,
    ) -> Result<()> {
        if self.kinds.get(layer) != Some(&CacheLayerKind::KeyValue) {
            return Err(Error::Cache(
                "K/V append targets a non-attention layer".into(),
            ));
        }
        let offset = self.layer_len(layer)?;
        let dims = [rows, self.heads, self.dim];
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
        write(
            &entry.storage.0.view(offset * row, dims)?,
            &entry.storage.1.view(offset * row, dims)?,
        )?;
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
        self.conv_states.fill(None);
        self.sequence_len = 0;
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

    pub fn state_bytes(&self) -> usize {
        self.conv_states
            .iter()
            .flatten()
            .map(Tensor::byte_size)
            .sum()
    }
}
