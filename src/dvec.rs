use std::{
    alloc::Allocator,
    ops::{Deref, DerefMut},
    slice::{ChunksExact, ChunksExactMut, ChunksMut},
};

use boojum::field::U64RawRepresentable;

use super::*;

#[derive(Debug)]
pub struct DVec<T, A: StaticAllocator = StaticDeviceAllocator> {
    pub(crate) data: Vec<T, A>,
}

impl<T, A: StaticAllocator> Default for DVec<T, A> {
    fn default() -> Self {
        todo!()
    }
}

impl<T> Clone for DVec<T, StaticDeviceAllocator> {
    fn clone(&self) -> Self {
        let mut new = dvec!(self.len());
        new.copy_from_device_slice(&self).unwrap();

        new
    }
}

impl<T, A: StaticAllocator> DVec<T, A> {
    pub fn chunks<'a>(&'a self, chunk_size: usize) -> Chunks<'a, T> {
        self.data.chunks(chunk_size)
    }

    pub fn chunks_mut<'a>(&'a mut self, chunk_size: usize) -> ChunksMut<'a, T> {
        self.data.chunks_mut(chunk_size)
    }

    pub fn chunks_exact<'a>(&'a self, chunk_size: usize) -> ChunksExact<'a, T> {
        self.data.chunks_exact(chunk_size)
    }

    pub fn chunks_exact_mut<'a>(&'a mut self, chunk_size: usize) -> ChunksExactMut<'a, T> {
        self.data.chunks_exact_mut(chunk_size)
    }

    pub fn copy_from_slice(&mut self, other: &[T]) -> CudaResult<()> {
        mem::h2d(other, self)
    }

    pub fn copy_from_device_slice(&mut self, other: &Self) -> CudaResult<()> {
        mem::d2d(other, self)
    }

    pub fn split_at(&self, mid: usize) -> (&[T], &[T]) {
        self.data.split_at(mid)
    }

    pub fn split_at_mut(&mut self, mid: usize) -> (&mut [T], &mut [T]) {
        self.data.split_at_mut(mid)
    }

    pub fn as_ptr(&self) -> *const T {
        self.data.as_ptr()
    }

    // pub fn new_in(alloc: A) -> Self {
    //     let data = Vec::new_in(alloc.clone());
    //     Self { data }
    // }

    // FIXME
    #[allow(clippy::uninit_vec)]
    pub fn to_vec(&self) -> CudaResult<Vec<T>> {
        let mut other = Vec::with_capacity(self.len());
        unsafe {
            other.set_len(self.len());
        }
        mem::d2h(&self.data[..], &mut other)?;

        Ok(other)
    }

    pub fn to_vec_in<B: Allocator>(&self, alloc: B) -> CudaResult<Vec<T, B>> {
        let mut other = Vec::with_capacity_in(self.len(), alloc);
        unsafe {
            other.set_len(self.len());
        }
        mem::d2h(&self.data[..], &mut other)?;

        Ok(other)
    }

    pub fn allocator(&self) -> A {
        self.data.allocator().clone()
    }

    pub fn into_adjacent_chunks(mut self, chunk_size: usize) -> Vec<DVec<T, A>> {
        assert_eq!(self.len() % chunk_size, 0);
        let num_chunks = self.len() / chunk_size;
        let (original_ptr, _len, _cap, alloc) = self.data.into_raw_parts_with_alloc();
        let mut chunks = Vec::with_capacity(num_chunks);
        for chunk_idx in 0..num_chunks {
            unsafe {
                let ptr = original_ptr.add(chunk_idx * chunk_size);
                let len = chunk_size;
                let chunk = Vec::from_raw_parts_in(ptr, len, len, alloc.clone());
                let chunk = Self::from(chunk);
                chunks.push(chunk);
            }
        }

        chunks
    }

    // FIXME
    #[allow(clippy::uninit_vec)]
    pub fn clone_range_to_host(&self, range: std::ops::Range<usize>) -> CudaResult<Vec<T>> {
        assert!(!range.is_empty());
        let mut h_values = Vec::with_capacity(range.len());
        unsafe {
            h_values.set_len(range.len());
        }
        mem::d2h(&self.data[range], &mut h_values[..])?;

        Ok(h_values)
    }

    pub fn clone_range_into_device(
        &self,
        range: std::ops::Range<usize>,
        result: &mut Self,
    ) -> CudaResult<()> {
        assert_eq!(range.len(), result.len());
        mem::d2d(&self[range], result)
    }

    pub fn clone_el_to_host(&self, pos: usize) -> CudaResult<T> {
        let mut result = self.clone_range_to_host(pos..pos + 1)?;
        Ok(result.pop().unwrap())
    }

    pub fn into_raw_parts_with_alloc(self) -> (*mut T, usize, usize, A) {
        self.data.into_raw_parts_with_alloc()
    }

    pub fn from_raw_parts_in(ptr: *mut T, length: usize, capacity: usize, alloc: A) -> Self {
        unsafe {
            Self {
                data: Vec::from_raw_parts_in(ptr, length, capacity, alloc),
            }
        }
    }
}

impl DVec<F> {
    pub fn get(&self, pos: usize) -> CudaResult<DF> {
        let mut el = DF::zero()?;
        mem::d2d(&self.data[pos..pos + 1], &mut el.inner[..])?;
        Ok(el)
    }
}

impl<T> DVec<T, StaticDeviceAllocator> {
    pub fn from_vec(data: Vec<T>) -> CudaResult<Self> {
        let size = data.len();
        assert!(size.is_power_of_two());

        let mut this = dvec!(size);
        mem::h2d(&data, &mut this)?;

        Ok(this)
    }

    pub fn with_capacity_in(cap: usize, alloc: StaticDeviceAllocator) -> Self {
        if cap == 0 {
            return Self {
                data: Vec::with_capacity_in(0, alloc),
            };
        }
        // Allocator itself can handle padding but it is okey to do padding here,
        // since DVec is the entrypoint of the all allocations in gpu memory
        let cap_in_bytes = cap * std::mem::size_of::<T>();
        let block_size_in_bytes = _alloc().block_size_in_bytes();
        let padded_cap_in_bytes = calculate_padded_capacity(cap_in_bytes, block_size_in_bytes);
        assert_eq!(padded_cap_in_bytes % block_size_in_bytes, 0);
        assert_eq!(padded_cap_in_bytes % std::mem::size_of::<T>(), 0);
        let mut padded_cap = padded_cap_in_bytes / std::mem::size_of::<T>();
        if padded_cap_in_bytes % std::mem::size_of::<T>() != 0 {
            padded_cap += 1;
        }

        let mut data = Vec::with_capacity_in(padded_cap, alloc);
        unsafe {
            data.set_len(cap);
            helpers::set_zero_generic(&mut data).expect("zeroize");
        }
        Self { data }
    }
}

impl<T, A: StaticAllocator> From<Vec<T, A>> for DVec<T, A> {
    fn from(data: Vec<T, A>) -> Self {
        Self { data }
    }
}

impl<T, A: StaticAllocator> AsRef<[T]> for DVec<T, A> {
    fn as_ref(&self) -> &[T] {
        self.data.as_ref()
    }
}

impl<T, A: StaticAllocator> AsMut<[T]> for DVec<T, A> {
    fn as_mut(&mut self) -> &mut [T] {
        self.data.as_mut()
    }
}

impl<T, A: StaticAllocator> Deref for DVec<T, A> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        self.data.deref()
    }
}

impl<T, A: StaticAllocator> DerefMut for DVec<T, A> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.data.deref_mut()
    }
}

pub struct DVecIterator<'a, T, A: StaticAllocator> {
    inner: &'a DVec<T, A>,
    index: usize,
}

impl<'a, T, A: StaticAllocator> Iterator for DVecIterator<'a, T, A> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index < self.inner.data.len() {
            let el = &self.inner.data[self.index];
            self.index += 1;
            Some(el)
        } else {
            None
        }
    }
}

impl<'a, T, A: StaticAllocator> IntoIterator for &'a DVec<T, A> {
    type Item = &'a T;

    type IntoIter = DVecIterator<'a, T, A>;

    fn into_iter(self) -> Self::IntoIter {
        DVecIterator {
            inner: self,
            index: 0,
        }
    }
}

#[macro_export]
macro_rules! dvec {
    () => {
        DVec::new_in(_alloc().clone())
    };
    ($capacity:expr) => {
        DVec::<_, StaticDeviceAllocator>::with_capacity_in($capacity, _alloc().clone())
    };
}
#[macro_export]
macro_rules! svec {
    () => {
        SVec::new_in(_small_alloc().clone())
    };
    ($capacity:expr) => {
        SVec::with_capacity_in($capacity, _small_alloc().clone())
    };
}

pub type SVec<T> = DVec<T, SmallStaticDeviceAllocator>;

impl<T> SVec<T> {
    pub fn with_capacity_in(cap: usize, alloc: SmallStaticDeviceAllocator) -> Self {
        if cap == 0 {
            return Self {
                data: Vec::with_capacity_in(0, alloc),
            };
        }
        let cap_in_bytes = cap * std::mem::size_of::<T>();
        let block_size_in_bytes = _small_alloc().block_size_in_bytes();
        let padded_cap_in_bytes = calculate_padded_capacity(cap_in_bytes, block_size_in_bytes);
        assert_eq!(padded_cap_in_bytes % block_size_in_bytes, 0);
        assert_eq!(padded_cap_in_bytes % std::mem::size_of::<T>(), 0);
        let mut padded_cap = padded_cap_in_bytes / std::mem::size_of::<T>();
        if padded_cap_in_bytes % std::mem::size_of::<T>() != 0 {
            padded_cap += 1;
        }

        let mut data = Vec::with_capacity_in(padded_cap, alloc);
        unsafe {
            helpers::set_zero_generic(&mut data).expect("zeroize");
            data.set_len(cap);
        }
        Self { data }
    }
}

fn calculate_padded_capacity(actual_cap_in_bytes: usize, block_size_in_bytes: usize) -> usize {
    assert!(actual_cap_in_bytes > 0);
    assert!(block_size_in_bytes > 0);
    assert_eq!(block_size_in_bytes % 8, 0);
    let mut num_blocks = actual_cap_in_bytes / block_size_in_bytes;
    if actual_cap_in_bytes % block_size_in_bytes != 0 {
        num_blocks += 1;
    }
    let padded_cap_in_bytes = num_blocks * block_size_in_bytes;

    padded_cap_in_bytes
}

pub struct DF {
    pub inner: DVec<F, SmallStaticDeviceAllocator>,
}

impl Clone for DF {
    fn clone(&self) -> Self {
        let mut new = Self::zero().unwrap();
        new.inner
            .copy_from_device_slice(&self.inner)
            .expect("copy device value");
        new
    }
}

impl std::fmt::Debug for DF {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let values = self.inner.to_vec().unwrap();
        assert_eq!(values.len(), 1);
        write!(f, "{}", values[0]);

        Ok(())
    }
}

impl DF {
    pub fn allocator(&self) -> SmallStaticDeviceAllocator {
        _small_alloc().clone()
    }

    pub fn zero() -> CudaResult<Self> {
        let mut storage = svec!(1);
        storage.copy_from_slice(&[F::ZERO])?;

        Ok(Self { inner: storage })
    }

    pub fn one() -> CudaResult<Self> {
        let mut this = Self::zero()?;
        this.copy_from_host_value(&F::ONE)?;

        Ok(this)
    }

    pub fn non_residue() -> CudaResult<Self> {
        let non_residue = F::from_raw_u64_unchecked(7);
        let mut this = Self::zero()?;
        this.copy_from_host_value(&non_residue)?;

        Ok(this)
    }

    pub fn copy_from_host_value(&mut self, value: &F) -> CudaResult<()> {
        self.inner.copy_from_slice(&[*value])?;

        Ok(())
    }

    pub fn from_host_value(value: &F) -> CudaResult<Self> {
        let mut this = Self::zero()?;
        this.inner.copy_from_slice(&[*value])?;

        Ok(this)
    }

    pub fn as_mut_ptr(&mut self) -> *mut F {
        self as *mut DF as *mut _
    }
}

impl Into<F> for DF {
    fn into(self) -> F {
        let mut value = self.inner.to_vec().expect("to host vector");
        value.pop().unwrap()
    }
}

impl From<F> for DF {
    fn from(value: F) -> Self {
        let mut this = Self::zero().expect("");
        this.copy_from_host_value(&value).expect("");
        this
    }
}

impl From<&F> for DF {
    fn from(value: &F) -> Self {
        let mut this = Self::zero().expect("");
        this.copy_from_host_value(value).expect("");
        this
    }
}
#[derive(Debug)]
pub struct DExt {
    pub c0: DF,
    pub c1: DF,
}

impl Clone for DExt {
    fn clone(&self) -> Self {
        Self {
            c0: self.c0.clone(),
            c1: self.c1.clone(),
        }
    }
}

impl DExt {
    pub fn new(c0: DF, c1: DF) -> Self {
        Self { c0, c1 }
    }

    pub fn zero() -> CudaResult<Self> {
        let c0 = DF::zero()?;
        let c1 = DF::zero()?;

        Ok(Self { c0, c1 })
    }

    pub fn one() -> CudaResult<Self> {
        let c0 = DF::one()?;
        let c1 = DF::zero()?;

        Ok(Self { c0, c1 })
    }

    pub fn copy_from_host_value(&mut self, value: &EF) -> CudaResult<()> {
        let [c0, c1] = value.into_coeffs_in_base();
        self.c0.copy_from_host_value(&c0)?;
        self.c1.copy_from_host_value(&c1)?;

        Ok(())
    }

    #[allow(dead_code)]
    pub fn into_coeffs(self) -> [DF; 2] {
        [self.c0, self.c1]
    }
}

impl Into<EF> for DExt {
    fn into(self) -> EF {
        let c0: F = self.c0.into();
        let c1: F = self.c1.into();

        EF::from_coeff_in_base([c0, c1])
    }
}

impl From<EF> for DExt {
    fn from(value: EF) -> Self {
        let mut this = Self::zero().expect("");
        this.copy_from_host_value(&value).expect("");
        this
    }
}

impl From<&EF> for DExt {
    fn from(value: &EF) -> Self {
        let mut this = Self::zero().expect("");
        this.copy_from_host_value(value).expect("");
        this
    }
}
