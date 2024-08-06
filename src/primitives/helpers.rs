use boojum_cuda::{
    ops_complex::{get_powers_of_g, get_powers_of_w},
    ops_cub::device_scan::get_scan_temp_storage_bytes,
    ops_simple::{set_by_ref, set_by_val, set_to_zero, SetByRef, SetByVal},
};
use era_cudart::slice::DeviceVariable;

use super::*;

// helper functions
pub fn compute_domain_elems(buffer: &mut [F], size: usize) -> CudaResult<()> {
    let log_n = size.trailing_zeros();
    let buffer = unsafe { DeviceSlice::from_mut_slice(buffer) };
    if_not_dry_run! {
        get_powers_of_w(log_n, 0, false, false, buffer, get_stream())
    }
}

#[allow(dead_code)]
pub fn compute_twiddles(buffer: &mut [F], size: usize, inverse: bool) -> CudaResult<()> {
    let fft_size = size >> 1;
    assert_eq!(buffer.len(), fft_size);
    let log_n = size.trailing_zeros();
    let tmp_buffer = unsafe { DeviceSlice::from_mut_slice(buffer) };
    if_not_dry_run! {
        get_powers_of_w(log_n, 0, inverse, false, tmp_buffer, get_stream())?;
        ntt::bitreverse(buffer)
    }
}

#[allow(dead_code)]
pub fn compute_coset_elems(buffer: &mut [F], size: usize) -> CudaResult<()> {
    let log_n = size.trailing_zeros();
    let buffer = unsafe { DeviceSlice::from_mut_slice(buffer) };
    if_not_dry_run! {
        get_powers_of_g(log_n, 0, false, false, buffer, get_stream())
    }
}

pub fn calculate_tmp_buffer_size_for_grand_product(buffer_size: usize) -> CudaResult<usize> {
    let tmp_size_in_bytes = get_scan_temp_storage_bytes::<F>(
        boojum_cuda::ops_cub::device_scan::ScanOperation::Product,
        false,
        false,
        buffer_size as i32,
    )?;
    // TODO: DVec already handles padding
    let alloc = _alloc();
    let block_size_in_bytes = alloc.block_size_in_bytes();
    let tmp_size = if tmp_size_in_bytes <= block_size_in_bytes {
        block_size_in_bytes
    } else {
        let mut num_blocks = tmp_size_in_bytes / block_size_in_bytes;
        if tmp_size_in_bytes % block_size_in_bytes != 0 {
            num_blocks += 1;
        }
        num_blocks * block_size_in_bytes
    };
    let tmp_size_in_field_elements = tmp_size / std::mem::size_of::<F>();

    Ok(tmp_size_in_field_elements)
}

pub fn calculate_tmp_buffer_size_for_grand_sum(domain_size: usize) -> CudaResult<usize> {
    let tmp_size_in_bytes = get_scan_temp_storage_bytes::<F>(
        boojum_cuda::ops_cub::device_scan::ScanOperation::Sum,
        true,
        false,
        domain_size as i32,
    )?;

    // excessively use multiple of block size to prevent fragmentation if needed
    let alloc = _alloc();
    let block_size_in_bytes = alloc.block_size_in_bytes();
    let tmp_size = if tmp_size_in_bytes <= block_size_in_bytes {
        block_size_in_bytes
    } else {
        let mut num_blocks = tmp_size_in_bytes / block_size_in_bytes;
        if tmp_size_in_bytes % block_size_in_bytes != 0 {
            num_blocks += 1;
        }
        num_blocks * block_size_in_bytes
    };
    let tmp_size_in_field_elements = tmp_size / std::mem::size_of::<F>();

    Ok(tmp_size_in_field_elements)
}

pub fn set_value(buffer: &mut [F], value: &DF) -> CudaResult<()> {
    let (buffer, value) = unsafe {
        let d_var = DeviceVariable::from_ref(&value.inner[0]);
        (DeviceSlice::from_mut_slice(buffer), d_var)
    };
    if_not_dry_run! {
        set_by_ref(value, buffer, get_stream())
    }
}

// value is a device value
#[allow(dead_code)]
pub fn set_value_generic<T: SetByRef>(buffer: &mut [T], value: &T) -> CudaResult<()> {
    assert_eq!(buffer.is_empty(), false);
    let (buffer, value) = unsafe {
        let h_var = DeviceVariable::from_ref(value);
        (DeviceSlice::from_mut_slice(buffer), h_var)
    };
    if_not_dry_run! {
        set_by_ref(value, buffer, get_stream())
    }
}

pub fn set_by_value<T: SetByVal>(
    buffer: &mut [T],
    value: T,
    stream: &CudaStream,
) -> CudaResult<()> {
    assert_eq!(buffer.is_empty(), false);
    let buffer = unsafe { DeviceSlice::from_mut_slice(buffer) };
    if_not_dry_run! {
        set_by_val(value, buffer, stream)
    }
}

pub fn set_zero(buffer: &mut [F]) -> CudaResult<()> {
    let buffer = unsafe { DeviceSlice::from_mut_slice(buffer) };
    if_not_dry_run! {
        set_to_zero(buffer, get_stream())
    }
}

#[allow(dead_code)]
pub fn set_zero_generic<T>(buffer: &mut [T]) -> CudaResult<()> {
    let buffer = unsafe { DeviceSlice::from_mut_slice(buffer) };
    if_not_dry_run! {
        set_to_zero(buffer, get_stream())
    }
}

pub fn rotate_left(values: &mut [F]) -> CudaResult<()> {
    let mut tmp = dvec!(values.len());
    mem::d2d(values, &mut tmp)?;
    let offset = values.len() - 1;
    mem::d2d(&tmp[1..], &mut values[..offset])?;
    let first = tmp.get(0)?;
    if_not_dry_run! {
        set_value(&mut values[offset..(offset + 1)], &first)
    }
}

#[allow(dead_code)]
pub fn set_zero_static(buffer: &mut [u8]) -> CudaResult<()> {
    use era_cudart::memory::memory_set;
    let buffer = unsafe { DeviceSlice::from_mut_slice(buffer) };
    if_not_dry_run! {
        memory_set(buffer, 0)
    }
}
