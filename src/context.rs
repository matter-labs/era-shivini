use super::*;
use boojum_cuda::context::Context;
use std::collections::HashMap;

pub struct ProverContext;

pub const ZKSYNC_DEFAULT_TRACE_LOG_LENGTH: usize = 20;

impl ProverContext {
    fn create_internal(
        cuda_ctx: Context,
        small_device_alloc: SmallStaticDeviceAllocator,
        device_alloc: StaticDeviceAllocator,
        small_host_alloc: SmallStaticHostAllocator,
        host_alloc: StaticHostAllocator,
    ) -> CudaResult<Self> {
        unsafe {
            assert!(_CUDA_CONTEXT.is_none());
            _CUDA_CONTEXT = Some(cuda_ctx);
            assert!(_DEVICE_ALLOCATOR.is_none());
            _DEVICE_ALLOCATOR = Some(device_alloc);
            assert!(_SMALL_DEVICE_ALLOCATOR.is_none());
            _SMALL_DEVICE_ALLOCATOR = Some(small_device_alloc);
            assert!(_HOST_ALLOCATOR.is_none());
            _HOST_ALLOCATOR = Some(host_alloc);
            assert!(_SMALL_HOST_ALLOCATOR.is_none());
            _SMALL_HOST_ALLOCATOR = Some(small_host_alloc);
            assert!(_EXEC_STREAM.is_none());
            _EXEC_STREAM = Some(Stream::create()?);
            assert!(_H2D_STREAM.is_none());
            _H2D_STREAM = Some(Stream::create()?);
            assert!(_D2H_STREAM.is_none());
            _D2H_STREAM = Some(Stream::create()?);
            assert!(_SETUP_CACHE.is_none());
            assert!(_STRATEGY_CACHE.is_none());
            _STRATEGY_CACHE = Some(HashMap::new());
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

    pub fn create_limited(num_blocks: usize) -> CudaResult<Self> {
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
        unsafe {
            _setup_cache_reset();

            let cuda_ctx = _CUDA_CONTEXT.take().expect("cuda ctx");
            cuda_ctx.destroy().expect("destroy cuda ctx");

            _DEVICE_ALLOCATOR
                .take()
                .unwrap()
                .free()
                .expect("free allocator");
            _SMALL_DEVICE_ALLOCATOR
                .take()
                .unwrap()
                .free()
                .expect("free small allocator");
            _HOST_ALLOCATOR
                .take()
                .unwrap()
                .free()
                .expect("free allocator");
            _SMALL_HOST_ALLOCATOR
                .take()
                .unwrap()
                .free()
                .expect("free small allocator");
            _EXEC_STREAM
                .take()
                .unwrap()
                .inner
                .destroy()
                .expect("destroy stream");
            _H2D_STREAM
                .take()
                .unwrap()
                .inner
                .destroy()
                .expect("destroy h2d stream");
            _D2H_STREAM
                .take()
                .unwrap()
                .inner
                .destroy()
                .expect("destroy d2h stream");

            drop(_STRATEGY_CACHE.take());
        }
    }
}

static mut _CUDA_CONTEXT: Option<CudaContext> = None;
static mut _EXEC_STREAM: Option<Stream> = None;
static mut _H2D_STREAM: Option<Stream> = None;
static mut _D2H_STREAM: Option<Stream> = None;

pub(crate) fn get_stream() -> &'static CudaStream {
    unsafe { &_EXEC_STREAM.as_ref().expect("execution stream").inner }
}

pub(crate) fn get_h2d_stream() -> &'static CudaStream {
    // unsafe { &_H2D_STREAM.as_ref().expect("host to device stream").inner }
    get_stream()
}

pub(crate) fn get_d2h_stream() -> &'static CudaStream {
    // unsafe { &_D2H_STREAM.as_ref().expect("device to host stream").inner }
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

static mut _DEVICE_ALLOCATOR: Option<StaticDeviceAllocator> = None;
static mut _SMALL_DEVICE_ALLOCATOR: Option<SmallStaticDeviceAllocator> = None;
static mut _HOST_ALLOCATOR: Option<StaticHostAllocator> = None;
static mut _SMALL_HOST_ALLOCATOR: Option<SmallStaticHostAllocator> = None;

pub(crate) fn _alloc() -> &'static StaticDeviceAllocator {
    unsafe {
        _DEVICE_ALLOCATOR
            .as_ref()
            .expect("device allocator should be initialized")
    }
}

pub(crate) fn _small_alloc() -> &'static SmallStaticDeviceAllocator {
    unsafe {
        _SMALL_DEVICE_ALLOCATOR
            .as_ref()
            .expect("small device allocator should be initialized")
    }
}
pub(crate) fn _host_alloc() -> &'static StaticHostAllocator {
    unsafe {
        _HOST_ALLOCATOR
            .as_ref()
            .expect("host allocator should be initialized")
    }
}

pub(crate) fn _small_host_alloc() -> &'static SmallStaticHostAllocator {
    unsafe {
        _SMALL_HOST_ALLOCATOR
            .as_ref()
            .expect("small host allocator should be initialized")
    }
}

static mut _SETUP_CACHE: Option<SetupCache> = None;

pub(crate) fn _setup_cache_get() -> Option<&'static mut SetupCache> {
    unsafe { _SETUP_CACHE.as_mut() }
}

pub(crate) fn _setup_cache_set(value: SetupCache) {
    unsafe {
        assert!(_SETUP_CACHE.is_none());
        _SETUP_CACHE = Some(value)
    }
}

pub(crate) fn _setup_cache_reset() {
    unsafe { _SETUP_CACHE = None }
}

static mut _STRATEGY_CACHE: Option<HashMap<Vec<[F; 4]>, CacheStrategy>> = None;

pub(crate) fn _strategy_cache_get() -> &'static mut HashMap<Vec<[F; 4]>, CacheStrategy> {
    unsafe {
        _STRATEGY_CACHE
            .as_mut()
            .expect("strategy cache should be initialized")
    }
}
pub(crate) fn _strategy_cache_reset() {
    unsafe { _STRATEGY_CACHE = Some(HashMap::new()) }
}

pub(crate) fn is_prover_context_initialized() -> bool {
    unsafe {
        _CUDA_CONTEXT.is_some()
            & _EXEC_STREAM.is_some()
            & _H2D_STREAM.is_some()
            & _D2H_STREAM.is_some()
            & _DEVICE_ALLOCATOR.is_some()
            & _SMALL_DEVICE_ALLOCATOR.is_some()
            & _HOST_ALLOCATOR.is_some()
            & _SMALL_HOST_ALLOCATOR.is_some()
            & _STRATEGY_CACHE.is_some()
    }
}
