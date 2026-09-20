#![forbid(unsafe_code)]
use crate::{Error, MetalDevice, Result, metal::MetalBuffer};
use smallvec::{SmallVec, smallvec};
use std::rc::Rc;
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum DType {
    F32 = 0,
    F16 = 1,
    BF16 = 2,
}
impl DType {
    pub fn size_bytes(self) -> usize {
        match self {
            Self::F32 => 4,
            _ => 2,
        }
    }
    pub fn round(self, x: f32) -> f32 {
        match self {
            Self::F32 => x,
            Self::F16 => half::f16::from_f32(x).to_f32(),
            Self::BF16 => half::bf16::from_f32(x).to_f32(),
        }
    }
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Shape {
    dims: SmallVec<[usize; 4]>,
    elements: usize,
}
impl Shape {
    pub fn new(dims: impl AsRef<[usize]>) -> Result<Self> {
        let dims = SmallVec::from_slice(dims.as_ref());
        let elements = dims
            .iter()
            .try_fold(1usize, |n, &d| n.checked_mul(d))
            .ok_or_else(|| Error::Shape("element count overflow".into()))?;
        Ok(Self { dims, elements })
    }
    pub fn dimensions(&self) -> &[usize] {
        &self.dims
    }
    pub fn rank(&self) -> usize {
        self.dims.len()
    }
    pub fn numel(&self) -> usize {
        self.elements
    }
    pub fn byte_size(&self, dtype: DType) -> Result<usize> {
        self.elements
            .checked_mul(dtype.size_bytes())
            .ok_or_else(|| Error::Shape("byte size overflow".into()))
    }
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Layout {
    strides: SmallVec<[usize; 4]>,
}
impl Layout {
    pub fn contiguous(shape: &Shape) -> Result<Self> {
        let mut strides = smallvec![0; shape.rank()];
        let mut stride = 1usize;
        for (i, &d) in shape.dimensions().iter().enumerate().rev() {
            strides[i] = stride;
            stride = stride
                .checked_mul(d)
                .ok_or_else(|| Error::Shape("stride overflow".into()))?;
        }
        Ok(Self { strides })
    }
    pub fn strides(&self) -> &[usize] {
        &self.strides
    }
    pub fn is_contiguous(&self) -> bool {
        true
    }
}
#[derive(Debug, Clone)]
pub struct StorageInfo {
    pub allocation_bytes: usize,
    pub offset_bytes: usize,
    pub length_bytes: usize,
    pub alignment: usize,
    pub mode: &'static str,
    pub owners: usize,
}
#[derive(Clone)]
pub struct Tensor {
    storage: Rc<MetalBuffer>,
    offset: usize,
    shape: Shape,
    layout: Layout,
    dtype: DType,
}
impl Tensor {
    pub fn zeros(device: &MetalDevice, dims: impl AsRef<[usize]>, dtype: DType) -> Result<Self> {
        let shape = Shape::new(dims)?;
        let layout = Layout::contiguous(&shape)?;
        let storage = Rc::new(device.allocate(shape.byte_size(dtype)?)?);
        Ok(Self {
            storage,
            offset: 0,
            shape,
            layout,
            dtype,
        })
    }
    pub(crate) fn output(device: &MetalDevice, dims: &[usize], dtype: DType) -> Result<Self> {
        let shape = Shape::new(dims)?;
        let layout = Layout::contiguous(&shape)?;
        let storage = Rc::new(device.allocate_output(shape.byte_size(dtype)?)?);
        Ok(Self {
            storage,
            offset: 0,
            shape,
            layout,
            dtype,
        })
    }
    pub fn from_f32(
        device: &MetalDevice,
        dims: impl AsRef<[usize]>,
        dtype: DType,
        data: &[f32],
    ) -> Result<Self> {
        let shape = Shape::new(dims)?;
        if shape.numel() != data.len() {
            return Err(Error::Shape("input data length differs from shape".into()));
        }
        let layout = Layout::contiguous(&shape)?;
        let mut storage = device.allocate(shape.byte_size(dtype)?)?;
        // Convert directly into unified memory; there is no staging Vec or GPU upload copy.
        storage.with_bytes_mut(|bytes| {
            for (dst, &x) in bytes.chunks_exact_mut(dtype.size_bytes()).zip(data) {
                match dtype {
                    DType::F32 => dst.copy_from_slice(&x.to_ne_bytes()),
                    DType::F16 => {
                        dst.copy_from_slice(&half::f16::from_f32(x).to_bits().to_ne_bytes())
                    }
                    DType::BF16 => {
                        dst.copy_from_slice(&half::bf16::from_f32(x).to_bits().to_ne_bytes())
                    }
                }
            }
        });
        Ok(Self {
            storage: Rc::new(storage),
            offset: 0,
            shape,
            layout,
            dtype,
        })
    }
    /// Copy validated little-endian file bytes directly into fresh shared storage.
    pub fn from_le_bytes(
        device: &MetalDevice,
        dims: impl AsRef<[usize]>,
        dtype: DType,
        data: &[u8],
    ) -> Result<Self> {
        let shape = Shape::new(dims)?;
        let layout = Layout::contiguous(&shape)?;
        if shape.byte_size(dtype)? != data.len() {
            return Err(Error::Shape("byte length differs from shape/dtype".into()));
        }
        let mut storage = device.allocate(data.len())?;
        storage.with_bytes_mut(|dst| dst.copy_from_slice(data)); // Apple Silicon is little-endian.
        Ok(Self {
            storage: Rc::new(storage),
            offset: 0,
            shape,
            layout,
            dtype,
        })
    }
    pub fn to_f32(&self) -> Vec<f32> {
        self.storage.with_bytes(|bytes| {
            bytes[self.offset..self.offset + self.byte_size()]
                .chunks_exact(self.dtype.size_bytes())
                .map(|b| match self.dtype {
                    DType::F32 => f32::from_ne_bytes([b[0], b[1], b[2], b[3]]),
                    DType::F16 => half::f16::from_bits(u16::from_ne_bytes([b[0], b[1]])).to_f32(),
                    DType::BF16 => half::bf16::from_bits(u16::from_ne_bytes([b[0], b[1]])).to_f32(),
                })
                .collect()
        })
    }
    pub fn shape(&self) -> &Shape {
        &self.shape
    }
    pub fn layout(&self) -> &Layout {
        &self.layout
    }
    pub fn dtype(&self) -> DType {
        self.dtype
    }
    pub fn numel(&self) -> usize {
        self.shape.numel()
    }
    pub fn byte_size(&self) -> usize {
        self.numel() * self.dtype.size_bytes()
    }
    pub fn storage_info(&self) -> StorageInfo {
        StorageInfo {
            allocation_bytes: self.storage.allocation_bytes(),
            offset_bytes: self.offset,
            length_bytes: self.byte_size(),
            alignment: self.storage.alignment(),
            mode: "shared",
            owners: Rc::strong_count(&self.storage),
        }
    }
    pub fn reshape(&self, dims: impl AsRef<[usize]>) -> Result<Self> {
        let shape = Shape::new(dims).map_err(|e| Error::Reshape(e.to_string()))?;
        if shape.numel() != self.numel() {
            return Err(Error::Reshape("element count must remain unchanged".into()));
        }
        let layout = Layout::contiguous(&shape)?;
        Ok(Self {
            storage: self.storage.clone(),
            offset: self.offset,
            shape,
            layout,
            dtype: self.dtype,
        })
    }
    /// Restricted contiguous view. No arbitrary strides or mutable mapping.
    pub fn view(&self, start: usize, dims: impl AsRef<[usize]>) -> Result<Self> {
        let shape = Shape::new(dims)?;
        if start
            .checked_add(shape.numel())
            .is_none_or(|end| end > self.numel())
        {
            return Err(Error::Range("view exceeds logical input range".into()));
        }
        let offset = start
            .checked_mul(self.dtype.size_bytes())
            .and_then(|n| self.offset.checked_add(n))
            .ok_or_else(|| Error::Range("view offset overflow".into()))?;
        let end = offset
            .checked_add(shape.byte_size(self.dtype)?)
            .ok_or_else(|| Error::Range("view range overflow".into()))?;
        if end > self.storage.len_bytes() {
            return Err(Error::Range("view exceeds storage".into()));
        }
        let layout = Layout::contiguous(&shape)?;
        Ok(Self {
            storage: self.storage.clone(),
            offset,
            shape,
            layout,
            dtype: self.dtype,
        })
    }
    pub(crate) fn binding(&self) -> (&MetalBuffer, usize) {
        (&self.storage, self.offset)
    }
    pub(crate) fn buffer(&self) -> &MetalBuffer {
        &self.storage
    }
}
