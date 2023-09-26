use super::*;

pub struct ProverContext;

impl ProverContext {
    pub fn create() -> CudaResult<Self> {
        unsafe {
            assert!(_CUDA_CONTEXT.is_none());
            assert!(_ALLOCATOR.is_none());
            assert!(_SMALL_ALLOCATOR.is_none());
            assert!(_EXEC_STREAM.is_none());
            assert!(_H2D_STREAM.is_none());
            assert!(_D2H_STREAM.is_none());
        }
        // size counts in field elements
        let block_size = 1 << 20; // TODO:
                                  // grab small slice then consume everything
        let cuda_ctx = CudaContext::create(12, 12)?;
        let small_alloc = SmallVirtualMemoryManager::init()?;
        let alloc = VirtualMemoryManager::init_all(block_size)?;

        unsafe {
            _CUDA_CONTEXT = Some(cuda_ctx);
            _ALLOCATOR = Some(alloc);
            _SMALL_ALLOCATOR = Some(small_alloc);
            _EXEC_STREAM = Some(Stream::create()?);
            _H2D_STREAM = Some(Stream::create()?);
            _D2H_STREAM = Some(Stream::create()?);
        }

        Ok(Self {})
    }

    #[cfg(test)]
    pub(crate) fn dev(domain_size: usize) -> CudaResult<Self> {
        assert!(domain_size.is_power_of_two());
        // size counts in field elements
        let block_size = domain_size;
        let cuda_ctx = CudaContext::create(12, 12)?;
        let small_alloc = SmallVirtualMemoryManager::init()?;
        let alloc = VirtualMemoryManager::init_all(block_size)?;

        unsafe {
            _CUDA_CONTEXT = Some(cuda_ctx);
            _ALLOCATOR = Some(alloc);
            _SMALL_ALLOCATOR = Some(small_alloc);
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

            _ALLOCATOR.take().unwrap().free().expect("free allocator");
            _SMALL_ALLOCATOR
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

pub(crate) static mut _ALLOCATOR: Option<VirtualMemoryManager> = None;
pub(crate) static mut _SMALL_ALLOCATOR: Option<SmallVirtualMemoryManager> = None;

pub(crate) fn _alloc() -> &'static VirtualMemoryManager {
    unsafe {
        &_ALLOCATOR
            .as_ref()
            .expect("allocator should be initialized")
    }
}

pub(crate) fn _small_alloc() -> &'static SmallVirtualMemoryManager {
    unsafe {
        &_SMALL_ALLOCATOR
            .as_ref()
            .expect("small allocator should be initialized")
    }
}
