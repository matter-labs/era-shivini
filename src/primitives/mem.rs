use super::*;
use boojum::worker::Worker;
use era_cudart::execution::{launch_host_fn, HostFn};
pub use era_cudart::memory::memory_copy_async;
use era_cudart::stream::CudaStreamWaitEventFlags;
use std::intrinsics::copy_nonoverlapping;
use std::mem::size_of;
use std::ops::DerefMut;
use std::slice;

pub fn h2d<T>(host: &[T], device: &mut [T]) -> CudaResult<()> {
    assert!(!host.is_empty());
    assert_eq!(host.len(), device.len());
    if_not_dry_run! {
        memory_copy_async(&mut device[..], host, get_h2d_stream())
    }
}

pub fn h2d_buffered<'a, T: Send + Sync>(
    host: &'a [T],
    device: &'a mut [T],
    chunk_size: usize,
    worker: &'a Worker,
) -> CudaResult<Vec<HostFn<'a>>> {
    assert!(!host.is_empty());
    assert_eq!(host.len(), device.len());
    assert_ne!(chunk_size, 0);
    if is_dry_run()? {
        return Ok(vec![]);
    } else {
        const STREAMS_COUNT: usize = 2;
        assert!(STREAMS_COUNT <= NUM_AUX_STREAMS_AND_EVENTS);
        assert!(chunk_size * STREAMS_COUNT * size_of::<T>() <= AUX_H2D_BUFFER_SIZE);
        let events = &_aux_events()[0..STREAMS_COUNT];
        let streams = &_aux_streams()[0..STREAMS_COUNT];
        let buffer: &mut [T] = unsafe { std::mem::transmute(_aux_h2d_buffer().deref_mut()) };
        let main_stream = get_h2d_stream();
        let copy = |src: &[T], dst: &mut [T]| {
            worker.scope(src.len(), |scope, chunk_size| {
                for (src_chunk, dst_chunk) in src.chunks(chunk_size).zip(dst.chunks_mut(chunk_size))
                {
                    scope.spawn(|_| unsafe {
                        copy_nonoverlapping(
                            src_chunk.as_ptr(),
                            dst_chunk.as_mut_ptr(),
                            src_chunk.len(),
                        )
                    })
                }
            });
        };
        events[0].record(main_stream)?;
        for stream in streams.iter() {
            stream.wait_event(&events[0], CudaStreamWaitEventFlags::DEFAULT)?;
        }
        let mut pending_callbacks = vec![];
        for (i, (src, dst)) in host
            .chunks(chunk_size)
            .zip(device.chunks_mut(chunk_size))
            .enumerate()
        {
            let idx = i % STREAMS_COUNT;
            let stream = &streams[idx];
            let buffer_offset = idx * chunk_size;
            let buffer = &buffer[buffer_offset..buffer_offset + src.len()];
            let callback = HostFn::new(move || {
                let dst =
                    unsafe { slice::from_raw_parts_mut(buffer.as_ptr() as *mut T, buffer.len()) };
                copy(src, dst);
            });
            launch_host_fn(stream, &callback)?;
            pending_callbacks.push(callback);
            let dst = unsafe { DeviceSlice::from_mut_slice(dst) };
            memory_copy_async(dst, buffer, stream)?;
        }

        for (event, stream) in events.iter().zip(streams.iter()) {
            event.record(stream)?;
            main_stream.wait_event(event, CudaStreamWaitEventFlags::DEFAULT)?;
        }
        Ok(pending_callbacks)
    }
}

#[allow(dead_code)]
pub fn h2d_on_stream<T>(host: &[T], device: &mut [T], stream: &CudaStream) -> CudaResult<()> {
    assert!(!host.is_empty());
    assert_eq!(host.len(), device.len());
    if_not_dry_run! {
        memory_copy_async(&mut device[..], host, stream)
    }
}

pub fn d2h<T>(device: &[T], host: &mut [T]) -> CudaResult<()> {
    assert!(!host.is_empty());
    assert_eq!(host.len(), device.len());
    if_not_dry_run! {
        memory_copy_async(host, &device[..], get_d2h_stream())
    }
}

pub fn d2d<T>(src: &[T], dst: &mut [T]) -> CudaResult<()> {
    assert!(!src.is_empty());
    assert_eq!(src.len(), dst.len());
    if_not_dry_run! {
        memory_copy_async(&mut dst[..], &src[..], get_stream())
    }
}
