use super::*;

pub fn h2d<T>(host: &[T], device: &mut [T]) -> CudaResult<()> {
    assert!(!host.is_empty());
    assert_eq!(host.len(), device.len());
    memory_copy_async(&mut device[..], host, get_h2d_stream())?;
    Ok(())
}

pub fn h2d_on_stream<T>(host: &[T], device: &mut [T], stream: &CudaStream) -> CudaResult<()> {
    assert!(!host.is_empty());
    assert_eq!(host.len(), device.len());
    memory_copy_async(&mut device[..], host, stream)?;
    Ok(())
}

pub fn d2h<T>(device: &[T], host: &mut [T]) -> CudaResult<()> {
    assert!(!host.is_empty());
    assert_eq!(host.len(), device.len());
    memory_copy_async(host, &device[..], get_d2h_stream())?;
    Ok(())
}

pub fn d2d<T>(src: &[T], dst: &mut [T]) -> CudaResult<()> {
    assert!(!src.is_empty());
    assert_eq!(src.len(), dst.len());
    memory_copy_async(&mut dst[..], &src[..], get_stream())?;
    Ok(())
}
