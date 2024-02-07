use super::*;
use boojum_cuda::context::Context;
use cudart::device::{device_get_attribute, get_device};
use cudart::event::{CudaEvent, CudaEventCreateFlags};
use cudart::stream::CudaStreamCreateFlags;
use cudart_sys::CudaDeviceAttr;
use std::collections::HashMap;

pub(crate) const NUM_AUX_STREAMS_AND_EVENTS: usize = 4;

#[allow(dead_code)]
struct ProverContextSingleton {
    cuda_context: CudaContext,
    exec_stream: Stream,
    h2d_stream: Stream,
    d2h_stream: Stream,
    device_allocator: StaticDeviceAllocator,
    small_device_allocator: SmallStaticDeviceAllocator,
    host_allocator: StaticHostAllocator,
    small_host_allocator: SmallStaticHostAllocator,
    setup_cache: Option<SetupCache>,
    strategy_cache: HashMap<Vec<[F; 4]>, CacheStrategy>,
    l2_cache_size: usize,
    compute_capability_major: u32,
    aux_streams: [CudaStream; NUM_AUX_STREAMS_AND_EVENTS],
    aux_events: [CudaEvent; NUM_AUX_STREAMS_AND_EVENTS],
}

static mut CONTEXT: Option<ProverContextSingleton> = None;

pub struct ProverContext;

pub const ZKSYNC_DEFAULT_TRACE_LOG_LENGTH: usize = 20;

impl ProverContext {
    fn create_internal(
        cuda_context: Context,
        small_device_allocator: SmallStaticDeviceAllocator,
        device_allocator: StaticDeviceAllocator,
        small_host_allocator: SmallStaticHostAllocator,
        host_allocator: StaticHostAllocator,
    ) -> CudaResult<Self> {
        unsafe {
            assert!(CONTEXT.is_none());
            let device_id = get_device()?;
            let l2_cache_size =
                device_get_attribute(CudaDeviceAttr::L2CacheSize, device_id)? as usize;
            let compute_capability_major =
                device_get_attribute(CudaDeviceAttr::ComputeCapabilityMajor, device_id)? as u32;
            let aux_streams = (0..NUM_AUX_STREAMS_AND_EVENTS)
                .map(|_| CudaStream::create_with_flags(CudaStreamCreateFlags::NON_BLOCKING))
                .collect::<CudaResult<Vec<CudaStream>>>()?
                .try_into()
                .unwrap();
            let aux_events = (0..NUM_AUX_STREAMS_AND_EVENTS)
                .map(|_| CudaEvent::create_with_flags(CudaEventCreateFlags::DISABLE_TIMING))
                .collect::<CudaResult<Vec<CudaEvent>>>()?
                .try_into()
                .unwrap();
            CONTEXT = Some(ProverContextSingleton {
                cuda_context,
                exec_stream: Stream::create()?,
                h2d_stream: Stream::create()?,
                d2h_stream: Stream::create()?,
                device_allocator,
                small_device_allocator,
                host_allocator,
                small_host_allocator,
                setup_cache: None,
                strategy_cache: HashMap::new(),
                l2_cache_size,
                compute_capability_major,
                aux_streams,
                aux_events,
            });
        };
        Ok(Self {})
    }

    pub fn create() -> CudaResult<Self> {
        // size counts in field elements
        let block_size = 1 << ZKSYNC_DEFAULT_TRACE_LOG_LENGTH;
        let cuda_ctx = CudaContext::create(12, 12)?;
        // grab small slice then consume everything
        let small_device_alloc = SmallStaticDeviceAllocator::init()?;
        let device_alloc = StaticDeviceAllocator::init_all(block_size)?;
        let small_host_alloc = SmallStaticHostAllocator::init()?;
        let host_alloc = StaticHostAllocator::init(1 << 8, block_size)?;
        Self::create_internal(
            cuda_ctx,
            small_device_alloc,
            device_alloc,
            small_host_alloc,
            host_alloc,
        )
    }

    #[cfg(test)]
    pub(crate) fn create_limited(num_blocks: usize) -> CudaResult<Self> {
        // size counts in field elements
        let block_size = 1 << ZKSYNC_DEFAULT_TRACE_LOG_LENGTH;
        let cuda_ctx = CudaContext::create(12, 12)?;
        // grab small slice then consume everything
        let small_device_alloc = SmallStaticDeviceAllocator::init()?;
        let device_alloc = StaticDeviceAllocator::init(num_blocks, block_size)?;
        let small_host_alloc = SmallStaticHostAllocator::init()?;
        let host_alloc = StaticHostAllocator::init(1 << 8, block_size)?;
        Self::create_internal(
            cuda_ctx,
            small_device_alloc,
            device_alloc,
            small_host_alloc,
            host_alloc,
        )
    }

    #[cfg(test)]
    pub(crate) fn dev(domain_size: usize) -> CudaResult<Self> {
        assert!(domain_size.is_power_of_two());
        // size counts in field elements
        let block_size = domain_size;
        let cuda_ctx = CudaContext::create(12, 12)?;
        let small_device_alloc = SmallStaticDeviceAllocator::init()?;
        let device_alloc = StaticDeviceAllocator::init_all(block_size)?;
        let small_host_alloc = SmallStaticHostAllocator::init()?;
        let host_alloc = StaticHostAllocator::init(1 << 8, block_size)?;
        Self::create_internal(
            cuda_ctx,
            small_device_alloc,
            device_alloc,
            small_host_alloc,
            host_alloc,
        )
    }
}

impl Drop for ProverContext {
    fn drop(&mut self) {
        _strategy_cache_reset();
        unsafe {
            CONTEXT = None;
        }
    }
}

fn get_context() -> &'static ProverContextSingleton {
    unsafe { CONTEXT.as_ref().expect("prover context") }
}

fn get_context_mut() -> &'static mut ProverContextSingleton {
    unsafe { CONTEXT.as_mut().expect("prover context") }
}

pub(crate) fn get_stream() -> &'static CudaStream {
    &get_context().exec_stream.inner
}

pub(crate) fn get_h2d_stream() -> &'static CudaStream {
    // &get_context().h2d_stream.inner
    get_stream()
}

pub(crate) fn get_d2h_stream() -> &'static CudaStream {
    // &get_context().d2h_stream.inner
    get_stream()
}

pub fn synchronize_streams() -> CudaResult<()> {
    if_not_dry_run! {
        get_stream().synchronize()?;
        get_h2d_stream().synchronize()?;
        get_d2h_stream().synchronize()
    }
}

// use custom wrapper to work around send + sync requirement of static var
pub struct Stream {
    inner: CudaStream,
}

impl Stream {
    pub fn create() -> CudaResult<Self> {
        Ok(Self {
            inner: CudaStream::create()?,
        })
    }
}

unsafe impl Send for Stream {}
unsafe impl Sync for Stream {}

pub(crate) fn _alloc() -> &'static StaticDeviceAllocator {
    &get_context().device_allocator
}

pub(crate) fn _small_alloc() -> &'static SmallStaticDeviceAllocator {
    &get_context().small_device_allocator
}
pub(crate) fn _host_alloc() -> &'static StaticHostAllocator {
    &get_context().host_allocator
}

pub(crate) fn _small_host_alloc() -> &'static SmallStaticHostAllocator {
    &get_context().small_host_allocator
}

pub(crate) fn _setup_cache_get() -> Option<&'static mut SetupCache> {
    get_context_mut().setup_cache.as_mut()
}

pub(crate) fn _setup_cache_set(value: SetupCache) {
    assert!(_setup_cache_get().is_none());
    get_context_mut().setup_cache = Some(value);
}

pub(crate) fn _setup_cache_reset() {
    get_context_mut().setup_cache = None;
}

pub(crate) fn _strategy_cache_get() -> &'static mut HashMap<Vec<[F; 4]>, CacheStrategy> {
    &mut get_context_mut().strategy_cache
}
pub(crate) fn _strategy_cache_reset() {
    get_context_mut().strategy_cache.clear();
}

pub(crate) fn is_prover_context_initialized() -> bool {
    unsafe { CONTEXT.is_some() }
}

pub(crate) fn _l2_cache_size() -> usize {
    get_context().l2_cache_size
}

pub(crate) fn _compute_capability_major() -> u32 {
    get_context().compute_capability_major
}

pub(crate) fn _aux_streams() -> &'static [CudaStream; NUM_AUX_STREAMS_AND_EVENTS] {
    &get_context().aux_streams
}

pub(crate) fn _aux_events() -> &'static [CudaEvent; NUM_AUX_STREAMS_AND_EVENTS] {
    &get_context().aux_events
}
