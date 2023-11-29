use super::*;

pub struct ProverContext;

pub const ZKSYNC_DEFAULT_TRACE_LOG_LENGTH: usize = 20;

impl ProverContext {
    pub fn create() -> CudaResult<Self> {
        unsafe {
            assert!(_CUDA_CONTEXT.is_none());
            assert!(_DEVICE_ALLOCATOR.is_none());
            assert!(_SMALL_DEVICE_ALLOCATOR.is_none());
            assert!(_HOST_ALLOCATOR.is_none());
            assert!(_SMALL_HOST_ALLOCATOR.is_none());
            assert!(_EXEC_STREAM.is_none());
            assert!(_H2D_STREAM.is_none());
            assert!(_D2H_STREAM.is_none());
        }
        // size counts in field elements
        let block_size = 1 << ZKSYNC_DEFAULT_TRACE_LOG_LENGTH;
        let cuda_ctx = CudaContext::create(12, 12)?;

        // grab small slice then consume everything
        let small_device_alloc = SmallStaticDeviceAllocator::init()?;
        let device_alloc = StaticDeviceAllocator::init_all(block_size)?;

        let small_host_alloc = SmallStaticHostAllocator::init()?;
        let host_alloc = StaticHostAllocator::init(1 << 8, block_size)?;

        unsafe {
            _CUDA_CONTEXT = Some(cuda_ctx);
            _DEVICE_ALLOCATOR = Some(device_alloc);
            _SMALL_DEVICE_ALLOCATOR = Some(small_device_alloc);
            _HOST_ALLOCATOR = Some(host_alloc);
            _SMALL_HOST_ALLOCATOR = Some(small_host_alloc);
            _EXEC_STREAM = Some(Stream::create()?);
            _H2D_STREAM = Some(Stream::create()?);
            _D2H_STREAM = Some(Stream::create()?);
        }

        Ok(Self {})
    }

    #[allow(dead_code)]
    pub(crate) fn create_limited_dev(block_size: usize) -> CudaResult<Self> {
        unsafe {
            assert!(_CUDA_CONTEXT.is_none());
            assert!(_DEVICE_ALLOCATOR.is_none());
            assert!(_SMALL_DEVICE_ALLOCATOR.is_none());
            assert!(_HOST_ALLOCATOR.is_none());
            assert!(_SMALL_HOST_ALLOCATOR.is_none());
            assert!(_EXEC_STREAM.is_none());
            assert!(_H2D_STREAM.is_none());
            assert!(_D2H_STREAM.is_none());
        }
        // size counts in field elements
        let cuda_ctx = CudaContext::create(12, 12)?;

        // grab small slice then consume everything
        let small_device_alloc = SmallStaticDeviceAllocator::init()?;
        let device_alloc = StaticDeviceAllocator::init_limited(block_size)?;
        let small_host_alloc = SmallStaticHostAllocator::init()?;
        let host_alloc = StaticHostAllocator::init(1 << 8, block_size)?;

        unsafe {
            _CUDA_CONTEXT = Some(cuda_ctx);
            _DEVICE_ALLOCATOR = Some(device_alloc);
            _SMALL_DEVICE_ALLOCATOR = Some(small_device_alloc);
            _HOST_ALLOCATOR = Some(host_alloc);
            _SMALL_HOST_ALLOCATOR = Some(small_host_alloc);
            _EXEC_STREAM = Some(Stream::create()?);
            _H2D_STREAM = Some(Stream::create()?);
            _D2H_STREAM = Some(Stream::create()?);
        }

        Ok(Self {})
    }

    pub fn create_limited() -> CudaResult<Self> {
        unsafe {
            assert!(_CUDA_CONTEXT.is_none());
            assert!(_DEVICE_ALLOCATOR.is_none());
            assert!(_SMALL_DEVICE_ALLOCATOR.is_none());
            assert!(_HOST_ALLOCATOR.is_none());
            assert!(_SMALL_HOST_ALLOCATOR.is_none());
            assert!(_EXEC_STREAM.is_none());
            assert!(_H2D_STREAM.is_none());
            assert!(_D2H_STREAM.is_none());
        }
        // size counts in field elements
        let block_size = 1 << ZKSYNC_DEFAULT_TRACE_LOG_LENGTH;
        let cuda_ctx = CudaContext::create(12, 12)?;

        // grab small slice then consume everything
        let small_device_alloc = SmallStaticDeviceAllocator::init()?;
        let device_alloc = StaticDeviceAllocator::init_limited(block_size)?;
        let small_host_alloc = SmallStaticHostAllocator::init()?;
        let host_alloc = StaticHostAllocator::init(1 << 8, block_size)?;

        unsafe {
            _CUDA_CONTEXT = Some(cuda_ctx);
            _DEVICE_ALLOCATOR = Some(device_alloc);
            _SMALL_DEVICE_ALLOCATOR = Some(small_device_alloc);
            _HOST_ALLOCATOR = Some(host_alloc);
            _SMALL_HOST_ALLOCATOR = Some(small_host_alloc);
            _EXEC_STREAM = Some(Stream::create()?);
            _H2D_STREAM = Some(Stream::create()?);
            _D2H_STREAM = Some(Stream::create()?);
        }

        Ok(Self {})
    }

    #[allow(dead_code)]
    pub(crate) fn dev(domain_size: usize) -> CudaResult<Self> {
        assert!(domain_size.is_power_of_two());
        // size counts in field elements
        let block_size = domain_size;
        let cuda_ctx = CudaContext::create(12, 12)?;

        let small_device_alloc = SmallStaticDeviceAllocator::init()?;
        let device_alloc = StaticDeviceAllocator::init_all(block_size)?;

        let small_host_alloc = SmallStaticHostAllocator::init()?;
        let host_alloc = StaticHostAllocator::init(1 << 8, block_size)?;

        unsafe {
            _CUDA_CONTEXT = Some(cuda_ctx);
            _DEVICE_ALLOCATOR = Some(device_alloc);
            _SMALL_DEVICE_ALLOCATOR = Some(small_device_alloc);
            _HOST_ALLOCATOR = Some(host_alloc);
            _SMALL_HOST_ALLOCATOR = Some(small_host_alloc);
            _EXEC_STREAM = Some(Stream::create()?);
            _H2D_STREAM = Some(Stream::create()?);
            _D2H_STREAM = Some(Stream::create()?);
        }

        Ok(Self {})
    }
}

impl Drop for ProverContext {
    fn drop(&mut self) {
        unsafe {
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
        }
    }
}

pub(crate) static mut _CUDA_CONTEXT: Option<CudaContext> = None;
pub(crate) static mut _EXEC_STREAM: Option<Stream> = None;
pub(crate) static mut _H2D_STREAM: Option<Stream> = None;
pub(crate) static mut _D2H_STREAM: Option<Stream> = None;

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
    get_stream().synchronize()?;
    get_h2d_stream().synchronize()?;
    get_d2h_stream().synchronize()?;

    Ok(())
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

pub(crate) static mut _DEVICE_ALLOCATOR: Option<StaticDeviceAllocator> = None;
pub(crate) static mut _SMALL_DEVICE_ALLOCATOR: Option<SmallStaticDeviceAllocator> = None;
pub(crate) static mut _HOST_ALLOCATOR: Option<StaticHostAllocator> = None;
pub(crate) static mut _SMALL_HOST_ALLOCATOR: Option<SmallStaticHostAllocator> = None;

pub(crate) fn _alloc() -> &'static StaticDeviceAllocator {
    unsafe {
        &_DEVICE_ALLOCATOR
            .as_ref()
            .expect("device allocator should be initialized")
    }
}

pub(crate) fn _small_alloc() -> &'static SmallStaticDeviceAllocator {
    unsafe {
        &_SMALL_DEVICE_ALLOCATOR
            .as_ref()
            .expect("small device allocator should be initialized")
    }
}
pub(crate) fn _host_alloc() -> &'static StaticHostAllocator {
    unsafe {
        &_HOST_ALLOCATOR
            .as_ref()
            .expect("host allocator should be initialized")
    }
}

pub(crate) fn _small_host_alloc() -> &'static SmallStaticHostAllocator {
    unsafe {
        &_SMALL_HOST_ALLOCATOR
            .as_ref()
            .expect("small host allocator should be initialized")
    }
}
