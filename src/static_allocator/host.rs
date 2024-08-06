use super::*;
use derivative::*;
use era_cudart::memory::{CudaHostAllocFlags, HostAllocation};
use std::alloc::{Allocator, Layout};

use std::ptr::NonNull;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

#[derive(Derivative)]
#[derivative(Clone, Debug)]
pub struct StaticHostAllocator {
    memory: Arc<HostAllocation<u8>>,
    memory_size: usize,
    block_size_in_bytes: usize,
    // TODO: Can we use deque
    bitmap: Arc<Vec<AtomicBool>>,
}

impl Default for StaticHostAllocator {
    fn default() -> Self {
        let _domain_size = 1 << ZKSYNC_DEFAULT_TRACE_LOG_LENGTH;
        Self::init(0, 0).unwrap() // TODO
    }
}

impl StaticAllocator for StaticHostAllocator {}
impl GoodAllocator for StaticHostAllocator {}

impl StaticHostAllocator {
    fn init_bitmap(num_blocks: usize) -> Vec<AtomicBool> {
        let mut bitmap = vec![];
        for _ in 0..num_blocks {
            bitmap.push(AtomicBool::new(false))
        }

        bitmap
    }

    pub fn as_ptr(&self) -> *const u8 {
        use era_cudart::slice::CudaSlice;
        self.memory.as_ptr()
    }

    #[allow(dead_code)]
    pub fn block_size_in_bytes(&self) -> usize {
        self.block_size_in_bytes
    }

    pub fn init(num_blocks: usize, block_size: usize) -> CudaResult<Self> {
        assert_ne!(num_blocks, 0);
        assert!(block_size.is_power_of_two());
        let memory_size = num_blocks * block_size;
        let memory_size_in_bytes = memory_size * std::mem::size_of::<F>();
        let block_size_in_bytes = block_size * std::mem::size_of::<F>();

        let memory =
            HostAllocation::alloc(memory_size_in_bytes, CudaHostAllocFlags::DEFAULT).expect(
                &format!("failed to allocate {} bytes", memory_size_in_bytes),
            );

        println!("allocated {memory_size_in_bytes} bytes on host");

        let alloc = StaticHostAllocator {
            memory: Arc::new(memory),
            memory_size: memory_size_in_bytes,
            block_size_in_bytes,
            bitmap: Arc::new(Self::init_bitmap(num_blocks)),
        };

        Ok(alloc)
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
    #[allow(unreachable_code)]
    fn find_adjacent_free_blocks(
        &self,
        requested_num_blocks: usize,
    ) -> Option<std::ops::Range<usize>> {
        if requested_num_blocks > self.bitmap.len() {
            return None;
        }
        let _range_of_blocks_found = false;
        let _found_range = 0..0;

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
        assert!(self.bitmap[index].load(Ordering::SeqCst));
        let _ = self.bitmap[index]
            .compare_exchange(true, false, Ordering::SeqCst, Ordering::SeqCst)
            .is_ok();
    }

    #[allow(dead_code)]
    pub fn free(self) -> CudaResult<()> {
        println!("freeing static host allocation");
        assert_eq!(Arc::weak_count(&self.memory), 0);
        assert_eq!(Arc::strong_count(&self.memory), 1);
        let Self { memory, .. } = self;
        let memory = Arc::try_unwrap(memory).expect("exclusive access");
        memory.free()?;
        Ok(())
    }
}

unsafe impl Send for StaticHostAllocator {}
unsafe impl Sync for StaticHostAllocator {}

unsafe impl Allocator for StaticHostAllocator {
    #[allow(unreachable_code)]
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, std::alloc::AllocError> {
        let size = layout.size();
        assert!(size > 0);
        assert_eq!(size % self.block_size_in_bytes, 0);
        let _alignment = layout.align();
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

    fn allocate_zeroed(&self, _layout: Layout) -> Result<NonNull<[u8]>, std::alloc::AllocError> {
        todo!()
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        let size = layout.size();
        assert!(size > 0);
        assert_eq!(size % self.block_size_in_bytes, 0);
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
pub struct SmallStaticHostAllocator {
    inner: StaticHostAllocator,
}

impl SmallStaticHostAllocator {
    pub fn init() -> CudaResult<Self> {
        const NUM_BLOCKS: usize = 1 << 8;
        // cuda requires alignment to be  multiple of 32 goldilocks elems
        const BLOCK_SIZE: usize = 32;
        let inner = StaticHostAllocator::init(NUM_BLOCKS, BLOCK_SIZE)?;
        Ok(Self { inner })
    }

    #[allow(dead_code)]
    pub fn free(self) -> CudaResult<()> {
        self.inner.free()
    }

    #[allow(dead_code)]
    pub fn block_size_in_bytes(&self) -> usize {
        self.inner.block_size_in_bytes
    }
}

unsafe impl Allocator for SmallStaticHostAllocator {
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, std::alloc::AllocError> {
        self.inner.allocate(layout)
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        self.inner.deallocate(ptr, layout)
    }
}

impl StaticAllocator for SmallStaticHostAllocator {}
impl SmallStaticAllocator for SmallStaticHostAllocator {}
impl GoodAllocator for SmallStaticHostAllocator {}
