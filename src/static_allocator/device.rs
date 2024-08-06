use era_cudart::memory::{memory_get_info, DeviceAllocation};

use super::*;
use derivative::*;
use std::alloc::{Allocator, Layout};
use std::ops::Deref;

use era_cudart_sys::CudaError;
use std::ptr::NonNull;
use std::sync::{Arc, Mutex};

pub const FREE_MEMORY_SLACK: usize = 1 << 23; // 8 MB
pub const MIN_NUM_BLOCKS: usize = 512;
pub const SMALL_ALLOCATOR_BLOCKS_COUNT: usize = 1 << 10; // 256 KB

#[derive(Derivative)]
#[derivative(Clone, Debug)]
pub struct StaticDeviceAllocator {
    memory: Arc<DeviceAllocation<u8>>,
    memory_size: usize,
    block_size_in_bytes: usize,
    // TODO: Can we use deque
    bitmap: Arc<Mutex<Vec<bool>>>,
    #[cfg(feature = "allocator_stats")]
    pub(crate) stats: Arc<Mutex<stats::AllocationStats>>,
}

#[cfg(feature = "allocator_stats")]
#[allow(dead_code)]
mod stats {
    use derivative::Derivative;
    use std::collections::BTreeMap;

    #[derive(Derivative)]
    #[derivative(Clone, Debug, Default)]
    pub struct Allocations(BTreeMap<usize, (usize, String)>);

    impl Allocations {
        fn insert(&mut self, index: usize, size: usize, backtrace: String) {
            self.0.insert(index, (size, backtrace));
        }

        fn remove(&mut self, index: &usize) {
            self.0.remove(index);
        }

        fn block_count(&self) -> usize {
            self.0.values().map(|&(size, _)| size).sum()
        }

        pub fn tail_index(&self) -> usize {
            self.0
                .last_key_value()
                .map_or(0, |(&index, &(size, _))| index + size)
        }

        pub(crate) fn print(&self, detailed: bool, with_backtrace: bool) {
            assert!(detailed || !with_backtrace);
            if self.0.is_empty() {
                println!("no allocations");
                return;
            }
            println!("block_count: {}", self.block_count());
            println!("tail_index: {}", self.tail_index());
            const SEPARATOR: &str = "================================";
            println!("{SEPARATOR}");
            if detailed {
                let mut last_index = 0;
                for (index, (length, trace)) in &self.0 {
                    let gap = index - last_index;
                    last_index = index + length;
                    if gap != 0 {
                        println!("gap: {gap}");
                        println!("{SEPARATOR}");
                    }
                    println!("index: {index}");
                    println!("length: {length}");
                    if with_backtrace {
                        println!("backtrace: \n{trace}");
                    }
                    println!("{SEPARATOR}");
                }
            }
        }
    }

    #[derive(Derivative)]
    #[derivative(Clone, Debug, Default)]
    pub(crate) struct AllocationStats {
        pub allocations: Allocations,
        pub allocations_at_maximum_block_count: Allocations,
        pub allocations_at_maximum_block_count_at_maximum_tail_index: Allocations,
    }

    impl AllocationStats {
        pub fn alloc(&mut self, index: usize, size: usize, backtrace: String) {
            self.allocations.insert(index, size, backtrace);
            let current_block_count = self.allocations.block_count();
            let current_tail_index = self.allocations.tail_index();
            let previous_maximum_block_count =
                self.allocations_at_maximum_block_count.block_count();
            if current_block_count > previous_maximum_block_count {
                self.allocations_at_maximum_block_count = self.allocations.clone();
            }
            let previous_maximum_tail_index = self
                .allocations_at_maximum_block_count_at_maximum_tail_index
                .tail_index();
            if current_tail_index > previous_maximum_tail_index {
                self.allocations_at_maximum_block_count_at_maximum_tail_index =
                    self.allocations.clone();
            } else if current_tail_index == previous_maximum_tail_index {
                let previous_maximum_block_count_at_maximum_tail_index = self
                    .allocations_at_maximum_block_count_at_maximum_tail_index
                    .block_count();
                if current_block_count > previous_maximum_block_count_at_maximum_tail_index {
                    self.allocations_at_maximum_block_count_at_maximum_tail_index =
                        self.allocations.clone();
                }
            }
        }

        pub fn free(&mut self, index: usize) {
            self.allocations.remove(&index);
        }

        pub fn print(&self, detailed: bool, with_backtrace: bool) {
            println!("allocations:");
            self.allocations.print(detailed, with_backtrace);
            println!("allocations_at_maximum_block_count:");
            self.allocations_at_maximum_block_count
                .print(detailed, with_backtrace);
            println!("allocations_at_maximum_block_count_at_maximum_tail_index:");
            self.allocations_at_maximum_block_count_at_maximum_tail_index
                .print(detailed, with_backtrace);
        }

        pub fn reset(&mut self) {
            self.allocations_at_maximum_block_count = self.allocations.clone();
            self.allocations_at_maximum_block_count_at_maximum_tail_index =
                self.allocations.clone();
        }
    }
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
    fn init_bitmap(num_blocks: usize) -> Vec<bool> {
        vec![false; num_blocks]
    }

    pub fn as_ptr(&self) -> *const u8 {
        era_cudart::slice::CudaSlice::as_ptr(self.memory.deref())
    }

    pub fn block_size_in_bytes(&self) -> usize {
        self.block_size_in_bytes
    }

    pub fn init(
        max_num_blocks: usize,
        min_num_blocks: usize,
        block_size: usize,
    ) -> CudaResult<Self> {
        assert_ne!(min_num_blocks, 0);
        assert!(max_num_blocks >= min_num_blocks);
        assert!(block_size.is_power_of_two());
        let mut num_blocks = max_num_blocks;
        while num_blocks >= min_num_blocks {
            let memory_size = num_blocks * block_size;
            let memory_size_in_bytes = memory_size * std::mem::size_of::<F>();
            let block_size_in_bytes = block_size * std::mem::size_of::<F>();

            let result = DeviceAllocation::alloc(memory_size_in_bytes);
            let memory = match result {
                Ok(memory) => memory,
                Err(CudaError::ErrorMemoryAllocation) => {
                    num_blocks -= 1;
                    continue;
                }
                Err(e) => return Err(e),
            };

            println!("allocated {memory_size_in_bytes} bytes on device");

            let alloc = StaticDeviceAllocator {
                memory: Arc::new(memory),
                memory_size: memory_size_in_bytes,
                block_size_in_bytes,
                bitmap: Arc::new(Mutex::new(Self::init_bitmap(num_blocks))),
                #[cfg(feature = "allocator_stats")]
                stats: Default::default(),
            };

            return Ok(alloc);
        }
        Err(CudaError::ErrorMemoryAllocation)
    }

    pub fn init_all(block_size: usize) -> CudaResult<Self> {
        let block_size_in_bytes = block_size * std::mem::size_of::<F>();
        let (memory_size_in_bytes, _total) = memory_get_info().expect("get memory info");
        assert!(memory_size_in_bytes >= FREE_MEMORY_SLACK);
        let free_memory_size_in_bytes = memory_size_in_bytes - FREE_MEMORY_SLACK;
        assert!(free_memory_size_in_bytes >= block_size);
        let max_num_blocks = free_memory_size_in_bytes / block_size_in_bytes;
        Self::init(max_num_blocks, MIN_NUM_BLOCKS, block_size)
    }

    fn find_free_block(&self) -> Option<usize> {
        for (idx, entry) in self.bitmap.lock().unwrap().iter_mut().enumerate() {
            if !*entry {
                *entry = true;
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
        let mut bitmap = self.bitmap.lock().unwrap();
        if requested_num_blocks > bitmap.len() {
            return None;
        }
        let _range_of_blocks_found = false;
        let _found_range = 0..0;

        let mut start = 0;
        let mut end = requested_num_blocks;
        let mut busy_block_idx = 0;
        loop {
            let mut has_busy_block = false;
            for (idx, sub_entry) in bitmap[start..end].iter().copied().enumerate() {
                if sub_entry {
                    has_busy_block = true;
                    busy_block_idx = start + idx;
                }
            }
            if has_busy_block == false {
                for entry in bitmap[start..end].iter_mut() {
                    *entry = true;
                }
                return Some(start..end);
            } else {
                start = busy_block_idx + 1;
                end = start + requested_num_blocks;
                if end > bitmap.len() {
                    break;
                }
            }
        }
        // panic!("not found block {} {} {}", start, end, self.bitmap.len());
        None
    }

    fn free_blocks(&self, index: usize, num_blocks: usize) {
        assert!(num_blocks > 0);
        let mut guard = self.bitmap.lock().unwrap();
        for i in index..index + num_blocks {
            guard[i] = false;
        }
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

#[cfg(feature = "allocator_stats")]
macro_rules! backtrace {
    () => {{
        let backtrace = std::backtrace::Backtrace::force_capture().to_string();
        let mut x: Vec<&str> = backtrace
            .split('\n')
            .rev()
            .skip_while(|&s| !s.contains("shivini"))
            .take_while(|&s| !s.contains("backtrace"))
            .collect();
        x.reverse();
        let backtrace: String = x.join("\n");
        backtrace
    }};
}

unsafe impl Allocator for StaticDeviceAllocator {
    #[allow(unreachable_code)]
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, std::alloc::AllocError> {
        let size = layout.size();
        assert!(size > 0);
        assert_eq!(size % self.block_size_in_bytes, 0);
        let _alignment = layout.align();
        if size > self.block_size_in_bytes {
            let num_blocks = size / self.block_size_in_bytes;
            if let Some(range) = self.find_adjacent_free_blocks(num_blocks) {
                #[cfg(feature = "allocator_stats")]
                self.stats
                    .lock()
                    .unwrap()
                    .alloc(range.start, num_blocks, backtrace!());
                let index = range.start;
                let offset = index * self.block_size_in_bytes;
                let ptr = unsafe { self.as_ptr().add(offset) };
                let ptr = unsafe { NonNull::new_unchecked(ptr as _) };
                return Ok(NonNull::slice_from_raw_parts(ptr, size));
            }
            if is_dry_run().unwrap_or(true) {
                dry_run_fail(CudaError::ErrorMemoryAllocation);
                let ptr =
                    unsafe { NonNull::new_unchecked(self.as_ptr().add(self.memory_size) as _) };
                return Ok(NonNull::slice_from_raw_parts(ptr, size));
            };
            panic!("allocation of {} blocks has failed", num_blocks);
            return Err(std::alloc::AllocError);
        }

        if let Some(index) = self.find_free_block() {
            #[cfg(feature = "allocator_stats")]
            self.stats.lock().unwrap().alloc(index, 1, backtrace!());
            let offset = index * self.block_size_in_bytes;
            let ptr = unsafe { self.as_ptr().add(offset) };
            let ptr = unsafe { NonNull::new_unchecked(ptr as _) };
            Ok(NonNull::slice_from_raw_parts(ptr, size))
        } else {
            if is_dry_run().unwrap_or(true) {
                dry_run_fail(CudaError::ErrorMemoryAllocation);
                let ptr =
                    unsafe { NonNull::new_unchecked(self.as_ptr().add(self.memory_size) as _) };
                return Ok(NonNull::slice_from_raw_parts(ptr, size));
            };
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
        // assert_eq!(size % self.block_size_in_bytes, 0);
        let offset = unsafe { ptr.as_ptr().offset_from(self.as_ptr()) } as usize;
        if offset >= self.memory_size {
            assert_eq!(is_dry_run().err(), Some(CudaError::ErrorMemoryAllocation));
            return;
        }
        assert_eq!(offset % self.block_size_in_bytes, 0);
        let index = offset / self.block_size_in_bytes;
        let num_blocks = size / self.block_size_in_bytes;
        self.free_blocks(index, num_blocks);
        #[cfg(feature = "allocator_stats")]
        self.stats.lock().unwrap().free(index);
    }
}

#[derive(Clone, Debug, Default)]
pub struct SmallStaticDeviceAllocator {
    inner: StaticDeviceAllocator,
}

impl SmallStaticDeviceAllocator {
    pub fn init() -> CudaResult<Self> {
        // cuda requires alignment to be  multiple of 32 goldilocks elems
        const BLOCK_SIZE: usize = 32;
        let inner = StaticDeviceAllocator::init(
            SMALL_ALLOCATOR_BLOCKS_COUNT,
            SMALL_ALLOCATOR_BLOCKS_COUNT,
            BLOCK_SIZE,
        )?;
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
