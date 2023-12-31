use cudart::memory::{memory_get_info, DeviceAllocation};

use super::*;
use derivative::*;
use std::alloc::{Allocator, Layout};

use std::ptr::NonNull;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

pub const FREE_MEMORY_SLACK: usize = 1 << 23; // 8 MB
#[cfg(feature = "recompute")]
pub const ALLOCATOR_LIMITED_BLOCKS_COUNT: usize = 1340 + 32;
#[cfg(not(feature = "recompute"))]
pub const ALLOCATOR_LIMITED_BLOCKS_COUNT: usize = 1695 + 64;
pub const SMALL_ALLOCATOR_LIMITED_BLOCKS_COUNT: usize = 27 + 16;

#[derive(Derivative)]
#[derivative(Clone, Debug)]
pub struct StaticDeviceAllocator {
    memory: Arc<DeviceAllocation<u8>>,
    memory_size: usize,
    block_size_in_bytes: usize,
    // TODO: Can we use deque
    bitmap: Arc<Vec<AtomicBool>>,
    #[cfg(feature = "allocator_stats")]
    pub stats: Arc<std::sync::Mutex<stats::AllocationStats>>,
}

#[cfg(feature = "allocator_stats")]
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

        fn print(&self, detailed: bool, with_backtrace: bool) {
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
    pub struct AllocationStats {
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
        assert_ne!(num_blocks, 0);
        assert!(block_size.is_power_of_two());
        let memory_size = num_blocks * block_size;
        let memory_size_in_bytes = memory_size * std::mem::size_of::<F>();
        let block_size_in_bytes = block_size * std::mem::size_of::<F>();

        let memory = DeviceAllocation::alloc(memory_size_in_bytes).expect(&format!(
            "failed to allocate {} bytes",
            memory_size_in_bytes
        ));

        println!("allocated {memory_size_in_bytes} bytes on device");

        let alloc = StaticDeviceAllocator {
            memory: Arc::new(memory),
            memory_size: memory_size_in_bytes,
            block_size_in_bytes,
            bitmap: Arc::new(Self::init_bitmap(num_blocks)),
            #[cfg(feature = "allocator_stats")]
            stats: Arc::new(std::sync::Mutex::new(Default::default())),
        };

        Ok(alloc)
    }

    pub fn init_all(block_size: usize) -> CudaResult<Self> {
        let block_size_in_bytes = block_size * std::mem::size_of::<F>();
        let (memory_size_in_bytes, _total) = memory_get_info().expect("get memory info");
        assert!(memory_size_in_bytes >= FREE_MEMORY_SLACK);
        let free_memory_size_in_bytes = memory_size_in_bytes - FREE_MEMORY_SLACK;
        assert!(free_memory_size_in_bytes >= block_size);
        let num_blocks = free_memory_size_in_bytes / block_size_in_bytes;
        Self::init(num_blocks, block_size)
    }

    pub fn init_limited(block_size: usize) -> CudaResult<Self> {
        let (memory_size_in_bytes, _total) = memory_get_info().expect("get memory info");
        assert!(memory_size_in_bytes >= FREE_MEMORY_SLACK);
        let free_memory_size_in_bytes = memory_size_in_bytes - FREE_MEMORY_SLACK;
        let block_size_in_bytes = block_size * std::mem::size_of::<F>();
        let requested_memory_size_in_bytes = ALLOCATOR_LIMITED_BLOCKS_COUNT * block_size_in_bytes;
        assert!(
            requested_memory_size_in_bytes <= free_memory_size_in_bytes,
            "requested memory {}bytes, free memory {} bytes",
            requested_memory_size_in_bytes,
            free_memory_size_in_bytes
        );
        Self::init(ALLOCATOR_LIMITED_BLOCKS_COUNT, block_size)
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
        // assert!(self.bitmap[index].load(Ordering::SeqCst));
        let _ = self.bitmap[index]
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
        assert_eq!(offset % self.block_size_in_bytes, 0);
        let index = offset / self.block_size_in_bytes;
        let num_blocks = size / self.block_size_in_bytes;
        assert!(num_blocks > 0);
        for actual_idx in index..index + num_blocks {
            self.free_block(actual_idx);
        }
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
        let inner = StaticDeviceAllocator::init(SMALL_ALLOCATOR_LIMITED_BLOCKS_COUNT, BLOCK_SIZE)?;
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
