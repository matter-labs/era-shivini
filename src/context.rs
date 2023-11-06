use super::*;

pub struct ProverContext;

pub const ZKSYNC_DEFAULT_TRACE_LOG_LENGTH: usize = 19;

impl ProverContext {
    pub fn create() -> CudaResult<Self> {
        unsafe {
            assert!(_cuda_context.is_none());
            assert!(_DEVICE_ALLOCATOR.is_none());
            assert!(_SMALL_DEVICE_ALLOCATOR.is_none());
            assert!(_HOST_ALLOCATOR.is_none());
            assert!(_SMALL_HOST_ALLOCATOR.is_none());
            assert!(_exec_stream.is_none());
            assert!(_h2d_stream.is_none());
            assert!(_d2h_stream.is_none());
        }
        // size counts in field elements
        let block_size = 1 << ZKSYNC_DEFAULT_TRACE_LOG_LENGTH;
        let cuda_ctx = CudaContext::create(12, 12)?;

        // grab small slice then consume everything
        let small_device_alloc = SmallStaticDeviceAllocator::init()?;
        let device_alloc = StaticDeviceAllocator::init_all(block_size)?;

        let small_host_alloc = SmallStaticHostAllocator::init()?;
        let host_alloc = StaticHostAllocator::init(block_size, 1 << 8)?;

        unsafe {
            _cuda_context = Some(cuda_ctx);
            _DEVICE_ALLOCATOR = Some(device_alloc);
            _SMALL_DEVICE_ALLOCATOR = Some(small_device_alloc);
            _HOST_ALLOCATOR = Some(host_alloc);
            _SMALL_HOST_ALLOCATOR = Some(small_host_alloc);
            _exec_stream = Some(Stream::create()?);
            _h2d_stream = Some(Stream::create()?);
            _d2h_stream = Some(Stream::create()?);
        }

        Ok(Self {})
    }

    pub(crate) fn create_14gb_dev(block_size: usize) -> CudaResult<Self> {
        unsafe {
            assert!(_cuda_context.is_none());
            assert!(_DEVICE_ALLOCATOR.is_none());
            assert!(_SMALL_DEVICE_ALLOCATOR.is_none());
            assert!(_HOST_ALLOCATOR.is_none());
            assert!(_SMALL_HOST_ALLOCATOR.is_none());
            assert!(_exec_stream.is_none());
            assert!(_h2d_stream.is_none());
            assert!(_d2h_stream.is_none());
        }
        // size counts in field elements
        let cuda_ctx = CudaContext::create(12, 12)?;

        // grab small slice then consume everything
        let small_device_alloc = SmallStaticDeviceAllocator::init()?;
        let device_alloc = StaticDeviceAllocator::init_14gb(block_size)?;
        println!("allocated 14gb on device");
        let small_host_alloc = SmallStaticHostAllocator::init()?;
        let host_alloc = StaticHostAllocator::init(block_size, 1 << 8)?;

        unsafe {
            _cuda_context = Some(cuda_ctx);
            _DEVICE_ALLOCATOR = Some(device_alloc);
            _SMALL_DEVICE_ALLOCATOR = Some(small_device_alloc);
            _HOST_ALLOCATOR = Some(host_alloc);
            _SMALL_HOST_ALLOCATOR = Some(small_host_alloc);
            _exec_stream = Some(Stream::create()?);
            _h2d_stream = Some(Stream::create()?);
            _d2h_stream = Some(Stream::create()?);
        }

        Ok(Self {})
    }

    pub fn create_14gb() -> CudaResult<Self> {
        unsafe {
            assert!(_cuda_context.is_none());
            assert!(_DEVICE_ALLOCATOR.is_none());
            assert!(_SMALL_DEVICE_ALLOCATOR.is_none());
            assert!(_HOST_ALLOCATOR.is_none());
            assert!(_SMALL_HOST_ALLOCATOR.is_none());
            assert!(_exec_stream.is_none());
            assert!(_h2d_stream.is_none());
            assert!(_d2h_stream.is_none());
        }
        // size counts in field elements
        let block_size = 1 << ZKSYNC_DEFAULT_TRACE_LOG_LENGTH;
        let cuda_ctx = CudaContext::create(12, 12)?;

        // grab small slice then consume everything
        let small_device_alloc = SmallStaticDeviceAllocator::init()?;
        let device_alloc = StaticDeviceAllocator::init_14gb(block_size)?;
        println!("allocated 14gb on device");
        let small_host_alloc = SmallStaticHostAllocator::init()?;
        let host_alloc = StaticHostAllocator::init(block_size, 1 << 8)?;

        unsafe {
            _cuda_context = Some(cuda_ctx);
            _DEVICE_ALLOCATOR = Some(device_alloc);
            _SMALL_DEVICE_ALLOCATOR = Some(small_device_alloc);
            _HOST_ALLOCATOR = Some(host_alloc);
            _SMALL_HOST_ALLOCATOR = Some(small_host_alloc);
            _exec_stream = Some(Stream::create()?);
            _h2d_stream = Some(Stream::create()?);
            _d2h_stream = Some(Stream::create()?);
        }

        Ok(Self {})
    }

    pub(crate) fn dev(domain_size: usize) -> CudaResult<Self> {
        assert!(domain_size.is_power_of_two());
        // size counts in field elements
        let block_size = domain_size;
        let cuda_ctx = CudaContext::create(12, 12)?;

        let small_device_alloc = SmallStaticDeviceAllocator::init()?;
        let device_alloc = StaticDeviceAllocator::init_all(block_size)?;

        let small_host_alloc = SmallStaticHostAllocator::init()?;
        let host_alloc = StaticHostAllocator::init(block_size, 1 << 8)?;

        unsafe {
            _cuda_context = Some(cuda_ctx);
            _DEVICE_ALLOCATOR = Some(device_alloc);
            _SMALL_DEVICE_ALLOCATOR = Some(small_device_alloc);
            _HOST_ALLOCATOR = Some(host_alloc);
            _SMALL_HOST_ALLOCATOR = Some(small_host_alloc);
            _exec_stream = Some(Stream::create()?);
            _h2d_stream = Some(Stream::create()?);
            _d2h_stream = Some(Stream::create()?);
        }

        Ok(Self {})
    }
}

impl Drop for ProverContext {
    fn drop(&mut self) {
        unsafe {
            let cuda_ctx = _cuda_context.take().expect("cuda ctx");
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
            _exec_stream
                .take()
                .unwrap()
                .inner
                .destroy()
                .expect("destroy stream");
            _h2d_stream
                .take()
                .unwrap()
                .inner
                .destroy()
                .expect("destroy h2d stream");
            _d2h_stream
                .take()
                .unwrap()
                .inner
                .destroy()
                .expect("destroy d2h stream");
        }
    }
}

pub(crate) static mut _cuda_context: Option<CudaContext> = None;
pub(crate) static mut _exec_stream: Option<Stream> = None;
pub(crate) static mut _h2d_stream: Option<Stream> = None;
pub(crate) static mut _d2h_stream: Option<Stream> = None;

pub(crate) fn get_stream() -> &'static CudaStream {
    unsafe { &_exec_stream.as_ref().expect("execution stream").inner }
}

pub(crate) fn get_h2d_stream() -> &'static CudaStream {
    // unsafe { &_h2d_stream.as_ref().expect("host to device stream").inner }
    get_stream()
}

pub(crate) fn get_d2h_stream() -> &'static CudaStream {
    // unsafe { &_d2h_stream.as_ref().expect("device to host stream").inner }
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
