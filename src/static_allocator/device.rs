use cudart::memory::DeviceAllocation;

use super::*;
use derivative::*;
use std::alloc::{Allocator, Global, Layout};
use std::mem::{self, ManuallyDrop, MaybeUninit};
use std::ptr::NonNull;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;

#[derive(Derivative)]
#[derivative(Clone, Debug)]
pub struct StaticDeviceAllocator {
    memory: Arc<DeviceAllocation<u8>>,
    memory_size: usize,
    block_size_in_bytes: usize,
    // TODO: Can we use deque
    bitmap: Arc<Vec<AtomicBool>>,
}

impl Default for StaticDeviceAllocator {
    fn default() -> Self {
        let domain_size = 1 << ZKSYNC_DEFAULT_TRACE_LOG_LENGTH;
        Self::init_all(domain_size).unwrap()
    }
}

impl StaticAllocator for StaticDeviceAllocator {}
impl GoodAllocator for StaticDeviceAllocator {}

impl StaticDeviceAllocator {
    fn init_bitmap(num_blocks: usize) -> Vec<AtomicBool> {
        let mut bitmap = vec![];
        for _ in 0..num_blocks {
            bitmap.push(AtomicBool::new(false))
        }

        bitmap
    }

    pub fn as_ptr(&self) -> *const u8 {
        use cudart::slice::CudaSlice;
        self.memory.as_ptr()
    }

    pub fn block_size_in_bytes(&self) -> usize {
        self.block_size_in_bytes
    }

    pub fn init(num_blocks: usize, block_size: usize) -> CudaResult<Self> {
        assert!(num_blocks > 32);
        assert!(block_size.is_power_of_two());
        use cudart::{
            memory::DeviceAllocation,
            slice::{CudaSlice, CudaSliceMut},
        };
        let memory_size = num_blocks * block_size;
        let memory_size_in_bytes = memory_size * std::mem::size_of::<F>();
        let block_size_in_bytes = block_size * std::mem::size_of::<F>();

        let memory = DeviceAllocation::alloc(memory_size_in_bytes).expect(&format!(
            "failed to allocate {} bytes",
            memory_size_in_bytes
        ));        

        let alloc = StaticDeviceAllocator {
            memory: Arc::new(memory),
            memory_size: memory_size_in_bytes,
            block_size_in_bytes,
            bitmap: Arc::new(Self::init_bitmap(num_blocks)),
        };

        Ok(alloc)
    }

    pub fn init_all(block_size: usize) -> CudaResult<Self> {
        use cudart::{
            memory::{memory_get_info, DeviceAllocation},
            slice::CudaSliceMut,
        };
        use std::mem::ManuallyDrop;

        let block_size_in_bytes = block_size * std::mem::size_of::<F>();
        let (memory_size_in_bytes, total) = memory_get_info().expect("get memory info");
        let precomputed_data_in_bytes = 256 * 1024 * 1024; // precomputed data is <=256mb
        let free_memory_size_in_bytes = memory_size_in_bytes - precomputed_data_in_bytes;
        assert!(free_memory_size_in_bytes >= block_size);
        let num_blocks = free_memory_size_in_bytes / block_size_in_bytes;
        Self::init(num_blocks, block_size)
    }

    pub fn init_14gb(block_size: usize) -> CudaResult<Self> {
        use cudart::{
            memory::{memory_get_info, DeviceAllocation},
            slice::CudaSliceMut,
        };
        use std::mem::ManuallyDrop;

        let block_size_in_bytes = block_size * std::mem::size_of::<F>();
        let (memory_size_in_bytes, total) = memory_get_info().expect("get memory info");
        let precomputed_data_in_bytes = 256 * 1024 * 1024; // precomputed data is <=256mb
        let free_memory_size_in_bytes = memory_size_in_bytes - precomputed_data_in_bytes;
        assert!(free_memory_size_in_bytes >= block_size);
        let requested_memory_size_in_bytes = 14usize * 0x40000000; // 16gb        
        assert!(
            requested_memory_size_in_bytes <= free_memory_size_in_bytes,
            "requested memory {}bytes, free memory {} bytes",
            requested_memory_size_in_bytes,
            free_memory_size_in_bytes
        );
        let num_blocks = requested_memory_size_in_bytes / block_size_in_bytes;
        Self::init(num_blocks, block_size)
    }

    fn find_free_block(&self) -> Option<usize> {
        for (idx, entry) in self.bitmap.iter().enumerate() {
            if entry
                .compare_exchange(false, true, Ordering::SeqCst, Ordering::Relaxed)
                .is_ok()
            {
                return Some(idx);
            }
        }
        None
    }

    // TODO: handle thread-safety
    fn find_adjacent_free_blocks(
        &self,
        requested_num_blocks: usize,
    ) -> Option<std::ops::Range<usize>> {
        if requested_num_blocks > self.bitmap.len() {
            return None;
        }
        let mut range_of_blocks_found = false;
        let mut found_range = 0..0;

        let mut start = 0;
        let mut end = requested_num_blocks;
        let mut busy_block_idx = 0;
        loop {
            let mut has_busy_block = false;
            for (idx, sub_entry) in self.bitmap[start..end].iter().enumerate() {
                if sub_entry.load(Ordering::SeqCst) {
                    has_busy_block = true;
                    busy_block_idx = start + idx;
                }
            }
            if has_busy_block == false {
                for entry in self.bitmap[start..end].iter() {
                    entry.store(true, Ordering::SeqCst);
                }
                return Some(start..end);
            } else {
                start = busy_block_idx + 1;
                end = start + requested_num_blocks;
                if end > self.bitmap.len() {
                    break;
                }
            }
        }
        panic!("not found block {} {} {}", start, end, self.bitmap.len());
        None
    }

    fn free_block(&self, index: usize) {
        // assert!(self.bitmap[index].load(Ordering::SeqCst));
        self.bitmap[index]
            .compare_exchange(true, false, Ordering::SeqCst, Ordering::SeqCst)
            .is_ok();
    }

    pub fn free(self) -> CudaResult<()> {
        println!("freeing static cuda allocation");
        assert_eq!(Arc::weak_count(&self.memory), 0);
        assert_eq!(Arc::strong_count(&self.memory), 1);
        let Self { memory, .. } = self;
        let memory = Arc::try_unwrap(memory).expect("exclusive access");
        memory.free()?;
        Ok(())
    }
}

unsafe impl Send for StaticDeviceAllocator {}
unsafe impl Sync for StaticDeviceAllocator {}

unsafe impl Allocator for StaticDeviceAllocator {
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, std::alloc::AllocError> {
        let size = layout.size();
        assert!(size > 0);
        assert_eq!(size % self.block_size_in_bytes, 0);
        let alignment = layout.align();
        if size > self.block_size_in_bytes {
            let num_blocks = size / self.block_size_in_bytes;
            if let Some(range) = self.find_adjacent_free_blocks(num_blocks) {
                let index = range.start;
                let offset = index * self.block_size_in_bytes;
                let ptr = unsafe { self.as_ptr().add(offset) };
                let ptr = unsafe { NonNull::new_unchecked(ptr as _) };
                return Ok(NonNull::slice_from_raw_parts(ptr, size));
            }
            panic!("allocation of {} blocks has failed", num_blocks);
            return Err(std::alloc::AllocError);
        }

        if let Some(index) = self.find_free_block() {
            let offset = index * self.block_size_in_bytes;
            let ptr = unsafe { self.as_ptr().add(offset) };
            let ptr = unsafe { NonNull::new_unchecked(ptr as _) };
            Ok(NonNull::slice_from_raw_parts(ptr, size))
        } else {
            panic!("allocation of 1 block has failed");
            Err(std::alloc::AllocError)
        }
    }

    fn allocate_zeroed(&self, layout: Layout) -> Result<NonNull<[u8]>, std::alloc::AllocError> {
        todo!()
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        let size = layout.size();
        assert!(size > 0);
        // assert_eq!(size % self.block_size_in_bytes, 0);
        let offset = unsafe { ptr.as_ptr().offset_from(self.as_ptr()) } as usize;
        assert_eq!(offset % self.block_size_in_bytes, 0);
        let index = offset / self.block_size_in_bytes;
        let num_blocks = size / self.block_size_in_bytes;
        assert!(num_blocks > 0);
        for actual_idx in index..index + num_blocks {
            self.free_block(actual_idx);
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct SmallStaticDeviceAllocator {
    inner: StaticDeviceAllocator,
}

impl SmallStaticDeviceAllocator {
    pub fn init() -> CudaResult<Self> {
        // cuda requires alignment to be  multiple of 32 goldilocks elems
        let block_size = 32;
        let num_blocks = 1 << 20; // <1gb
        let inner = StaticDeviceAllocator::init(num_blocks, block_size)?;
        Ok(Self { inner })
    }

    pub fn free(self) -> CudaResult<()> {
        self.inner.free()
    }

    pub fn block_size_in_bytes(&self) -> usize {
        self.inner.block_size_in_bytes
    }
}

unsafe impl Allocator for SmallStaticDeviceAllocator {
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, std::alloc::AllocError> {
        self.inner.allocate(layout)
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        self.inner.deallocate(ptr, layout)
    }
}

impl StaticAllocator for SmallStaticDeviceAllocator {}
impl SmallStaticAllocator for SmallStaticDeviceAllocator {}
impl GoodAllocator for SmallStaticDeviceAllocator {}
