use super::*;

// ntt operations
pub fn batch_ntt(
    input: &mut [F],
    bitreversed_input: bool,
    inverse: bool,
    domain_size: usize,
    num_polys: usize,
) -> CudaResult<()> {
    assert!(!input.is_empty());
    assert!(domain_size.is_power_of_two());
    assert_eq!(input.len(), domain_size * num_polys);
    let log_n = domain_size.trailing_zeros();
    let stride = 1 << log_n;
    let input = unsafe { DeviceSlice::from_mut_slice(input) };
    boojum_cuda::ntt::batch_ntt_in_place(
        input,
        log_n,
        num_polys as u32,
        0,
        stride,
        bitreversed_input,
        inverse,
        0,
        0,
        get_stream(),
    )
}

pub fn batch_ntt_into(
    input: &[F],
    output: &mut [F],
    bitreversed_input: bool,
    inverse: bool,
    domain_size: usize,
    num_polys: usize,
) -> CudaResult<()> {
    assert!(!input.is_empty());
    assert!(domain_size.is_power_of_two());
    assert_eq!(input.len(), domain_size * num_polys);
    let log_n = domain_size.trailing_zeros();
    let stride = 1 << log_n;
    let input = unsafe { DeviceSlice::from_slice(input) };
    let output = unsafe { DeviceSlice::from_mut_slice(output) };
    boojum_cuda::ntt::batch_ntt_out_of_place(
        input,
        output,
        log_n,
        num_polys as u32,
        0,
        0,
        stride,
        stride,
        bitreversed_input,
        inverse,
        0,
        0,
        get_stream(),
    )
}

fn ntt(input: &mut [F], inverse: bool) -> CudaResult<()> {
    assert!(!input.is_empty());
    assert!(input.len().is_power_of_two());
    let log_n = input.len().trailing_zeros();
    let stride = 1 << log_n;
    let input = unsafe { DeviceSlice::from_mut_slice(input) };
    boojum_cuda::ntt::batch_ntt_in_place(
        input,
        log_n,
        1,
        0,
        stride,
        false,
        inverse,
        0,
        0,
        get_stream(),
    )
}

#[allow(dead_code)]
fn ntt_into(input: &[F], output: &mut [F], inverse: bool) -> CudaResult<()> {
    assert!(!input.is_empty());
    assert!(input.len().is_power_of_two());
    let log_n = input.len().trailing_zeros();
    let stride = 1 << log_n;
    let input = unsafe { DeviceSlice::from_slice(input) };
    let output = unsafe { DeviceSlice::from_mut_slice(output) };
    boojum_cuda::ntt::batch_ntt_out_of_place(
        input,
        output,
        log_n,
        1,
        0,
        0,
        stride,
        stride,
        false,
        inverse,
        0,
        0,
        get_stream(),
    )
}

pub fn ifft(input: &mut [F], coset: &DF) -> CudaResult<()> {
    ntt(input, true)?;
    bitreverse(input)?;
    let d_coset: DF = coset.clone();
    let h_coset: F = d_coset.into();

    let h_coset_inv = h_coset.inverse().unwrap();
    let d_coset_inv = h_coset_inv.into();
    arith::distribute_powers(input, &d_coset_inv)?;

    Ok(())
}

#[allow(dead_code)]
pub fn ifft_into(input: &[F], output: &mut [F]) -> CudaResult<()> {
    ntt_into(input, output, true)?;
    bitreverse(output)?;
    Ok(())
}

pub fn fft(input: &mut [F], coset: &DF) -> CudaResult<()> {
    ntt(input, false)?;
    arith::distribute_powers(input, &coset)?;
    Ok(())
}

pub fn lde(coeffs: &[F], result: &mut [F], lde_degree: usize) -> CudaResult<()> {
    assert!(coeffs.len().is_power_of_two());
    assert!(result.len().is_power_of_two());
    let domain_size = coeffs.len();
    mem::d2d(coeffs, &mut result[..domain_size])?;

    for (coset_idx, current_coset) in result.chunks_mut(domain_size).enumerate() {
        mem::d2d(coeffs, current_coset)?;
        coset_fft(current_coset, coset_idx, lde_degree)?;
    }

    Ok(())
}

pub fn lde_from_lagrange_basis(
    result: &mut [F],
    domain_size: usize,
    lde_degree: usize,
) -> CudaResult<()> {
    assert!(result.len().is_power_of_two());
    let lde_size = lde_degree * domain_size;
    assert_eq!(result.len(), lde_size);

    let (first_coset, other_cosets) = result.split_at_mut(domain_size);
    let coset = DF::one()?;

    ifft(first_coset, &coset)?;

    for (prev_coset_idx, current_coset) in other_cosets.chunks_mut(domain_size).enumerate() {
        mem::d2d(first_coset, current_coset)?;
        let coset_idx = prev_coset_idx + 1;
        coset_fft(current_coset, coset_idx, lde_degree)?;
    }
    coset_fft(first_coset, 0, lde_degree)?;
    Ok(())
}

pub fn coset_fft(coeffs: &mut [F], coset_idx: usize, lde_degree: usize) -> CudaResult<()> {
    assert!(lde_degree > 1);
    debug_assert!(coeffs.len().is_power_of_two());
    let log_n = coeffs.len().trailing_zeros();
    let log_lde_factor = lde_degree.trailing_zeros();
    let stride = 1 << log_n;
    let coset_idx = bitreverse_index(coset_idx, log_lde_factor as usize);
    let d_coeffs = unsafe { DeviceSlice::from_mut_slice(coeffs) };
    let stream = get_stream();
    boojum_cuda::ntt::batch_ntt_in_place(
        d_coeffs,
        log_n,
        1,
        0,
        stride,
        false,
        false,
        log_lde_factor,
        coset_idx as u32,
        stream,
    )
}
pub fn coset_fft_into(
    coeffs: &[F],
    result: &mut [F],
    coset_idx: usize,
    lde_degree: usize,
) -> CudaResult<()> {
    assert!(lde_degree > 1);
    debug_assert!(coeffs.len().is_power_of_two());
    let log_n = coeffs.len().trailing_zeros();
    let log_lde_factor = lde_degree.trailing_zeros();
    let stride = 1 << log_n;
    let coset_idx = bitreverse_index(coset_idx, log_lde_factor as usize);
    let d_coeffs = unsafe { DeviceSlice::from_slice(coeffs) };
    let d_result = unsafe { DeviceSlice::from_mut_slice(result) };
    let stream = get_stream();
    boojum_cuda::ntt::batch_ntt_out_of_place(
        d_coeffs,
        d_result,
        log_n,
        1,
        0,
        0,
        stride,
        stride,
        false,
        false,
        log_lde_factor,
        coset_idx as u32,
        stream,
    )
}
pub fn batch_coset_fft(
    coeffs: &mut [F],
    coset_idx: usize,
    domain_size: usize,
    lde_degree: usize,
    num_polys: usize,
) -> CudaResult<()> {
    assert!(lde_degree > 1);
    assert!(coset_idx < lde_degree);
    assert_eq!(coeffs.len(), num_polys * domain_size);
    assert!(domain_size.is_power_of_two());
    assert!(lde_degree.is_power_of_two());
    let log_n = domain_size.trailing_zeros();
    let log_lde_factor = lde_degree.trailing_zeros();
    let stride = 1 << log_n;
    let coset_idx = bitreverse_index(coset_idx, log_lde_factor as usize);
    let d_coeffs = unsafe { DeviceSlice::from_mut_slice(coeffs) };
    let stream = get_stream();
    boojum_cuda::ntt::batch_ntt_in_place(
        d_coeffs,
        log_n,
        num_polys as u32,
        0,
        stride,
        false,
        false,
        log_lde_factor,
        coset_idx as u32,
        stream,
    )
}

pub fn batch_coset_fft_into(
    coeffs: &[F],
    result: &mut [F],
    coset_idx: usize,
    domain_size: usize,
    lde_degree: usize,
    num_polys: usize,
) -> CudaResult<()> {
    assert!(lde_degree > 1);
    assert!(coset_idx < lde_degree);
    assert_eq!(coeffs.len(), num_polys * domain_size);
    assert!(domain_size.is_power_of_two());
    assert!(lde_degree.is_power_of_two());
    let log_n = domain_size.trailing_zeros();
    let log_lde_factor = lde_degree.trailing_zeros();
    let stride = 1 << log_n;
    let coset_idx = bitreverse_index(coset_idx, log_lde_factor as usize);
    let d_coeffs = unsafe { DeviceSlice::from_slice(coeffs) };
    let d_result = unsafe { DeviceSlice::from_mut_slice(result) };
    let stream = get_stream();
    boojum_cuda::ntt::batch_ntt_out_of_place(
        d_coeffs,
        d_result,
        log_n,
        num_polys as u32,
        0,
        0,
        stride,
        stride,
        false,
        false,
        log_lde_factor,
        coset_idx as u32,
        stream,
    )
}

pub fn bitreverse(input: &mut [F]) -> CudaResult<()> {
    let stream = get_stream();
    let input = unsafe { DeviceSlice::from_mut_slice(input) };
    boojum_cuda::ops_complex::bit_reverse_in_place(input, stream)
}

pub fn batch_bitreverse(input: &mut [F], num_rows: usize) -> CudaResult<()> {
    use boojum_cuda::device_structures::DeviceMatrixMut;
    let stream = get_stream();
    let mut input = unsafe {
        let input = DeviceSlice::from_mut_slice(input);
        DeviceMatrixMut::new(input, num_rows)
    };
    boojum_cuda::ops_complex::bit_reverse_in_place(&mut input, stream)
}
