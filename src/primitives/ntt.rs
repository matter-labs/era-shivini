use super::*;

// ntt operations

// Raw boojum bindings

fn batch_coset_ntt_raw(
    inputs: &mut [F],
    bitreversed_input: bool,
    inverse: bool,
    coset_idx: usize,
    domain_size: usize,
    lde_degree: usize,
    num_polys: usize,
) -> CudaResult<()> {
    assert_eq!(inputs.len(), num_polys * domain_size);
    assert!(domain_size.is_power_of_two());
    let log_n = domain_size.trailing_zeros();
    let log_lde_factor = lde_degree.trailing_zeros();
    let stride = 1 << log_n;
    let coset_idx = bitreverse_index(coset_idx, log_lde_factor as usize);
    let d_inputs = unsafe { DeviceSlice::from_mut_slice(inputs) };
    let stream = get_stream();
    let inputs_offset = 0; // currently unused, but explicit for readability.
    boojum_cuda::ntt::batch_ntt_in_place(
        d_inputs,
        log_n,
        num_polys as u32,
        inputs_offset,
        stride,
        bitreversed_input,
        inverse,
        log_lde_factor,
        coset_idx as u32,
        stream,
    )
}

fn batch_coset_ntt_raw_into(
    inputs: &[F],
    outputs: &mut [F],
    bitreversed_input: bool,
    inverse: bool,
    coset_idx: usize,
    domain_size: usize,
    lde_degree: usize,
    num_polys: usize,
) -> CudaResult<()> {
    assert_eq!(inputs.len(), num_polys * domain_size);
    // The following is not required in general.
    // boojum-cuda's kernels can use a different stride for inputs and outputs.
    // But it's true for our current use cases, so we enforce it for now.
    assert_eq!(inputs.len(), outputs.len());
    assert!(domain_size.is_power_of_two());
    let log_n = domain_size.trailing_zeros();
    let log_lde_factor = lde_degree.trailing_zeros();
    let stride = 1 << log_n;
    let coset_idx = bitreverse_index(coset_idx, log_lde_factor as usize);
    let d_inputs = unsafe { DeviceSlice::from_slice(inputs) };
    let d_outputs = unsafe { DeviceSlice::from_mut_slice(outputs) };
    let stream = get_stream();
    let inputs_offset = 0; // currently unused, but explicit for readability.
    let outputs_offset = 0; // currently unused, but explicit for readability.
    boojum_cuda::ntt::batch_ntt_out_of_place(
        d_inputs,
        d_outputs,
        log_n,
        num_polys as u32,
        inputs_offset,
        outputs_offset,
        stride,
        stride,
        bitreversed_input,
        inverse,
        log_lde_factor,
        coset_idx as u32,
        stream,
    )
}

// Convenience wrappers for our use cases

pub(crate) fn batch_ntt(
    input: &mut [F],
    bitreversed_input: bool,
    inverse: bool,
    domain_size: usize,
    num_polys: usize,
) -> CudaResult<()> {
    batch_coset_ntt_raw(
        input,
        bitreversed_input,
        inverse,
        0,
        domain_size,
        1,
        num_polys,
    )
}

pub(crate) fn batch_ntt_into(
    inputs: &[F],
    outputs: &mut [F],
    bitreversed_input: bool,
    inverse: bool,
    domain_size: usize,
    num_polys: usize,
) -> CudaResult<()> {
    batch_coset_ntt_raw_into(
        inputs,
        outputs,
        bitreversed_input,
        inverse,
        0,
        domain_size,
        1,
        num_polys,
    )
}

#[allow(dead_code)]
pub(crate) fn coset_ntt_into(
    input: &[F],
    output: &mut [F],
    coset_idx: usize,
    lde_degree: usize,
) -> CudaResult<()> {
    assert!(lde_degree > 1);
    assert!(lde_degree.is_power_of_two());
    assert!(coset_idx < lde_degree);
    batch_coset_ntt_raw_into(
        input,
        output,
        false,
        false,
        coset_idx,
        input.len(),
        lde_degree,
        1,
    )
}

pub(crate) fn lde_intt(input: &mut [F]) -> CudaResult<()> {
    // Any power of two > 1 would work for lde_degree, it just signals to the kernel
    // that we're inverting an LDE and it should multiply x_i by g_inv^i
    let dummy_lde_degree = 2;
    let coset_idx = 0;
    batch_coset_ntt_raw(
        input,
        true,
        true,
        coset_idx,
        input.len(),
        dummy_lde_degree,
        1,
    )
}

pub(crate) fn intt_into(input: &[F], output: &mut [F]) -> CudaResult<()> {
    batch_coset_ntt_raw_into(input, output, false, true, 0, input.len(), 1, 1)
}

pub(crate) fn batch_coset_ntt_into(
    inputs: &[F],
    outputs: &mut [F],
    coset_idx: usize,
    domain_size: usize,
    lde_degree: usize,
    num_polys: usize,
) -> CudaResult<()> {
    assert!(lde_degree > 1);
    assert!(lde_degree.is_power_of_two());
    assert!(coset_idx < lde_degree);
    batch_coset_ntt_raw_into(
        inputs,
        outputs,
        false,
        false,
        coset_idx,
        domain_size,
        lde_degree,
        num_polys,
    )
}

pub(crate) fn bitreverse(input: &mut [F]) -> CudaResult<()> {
    let stream = get_stream();
    let input = unsafe { DeviceSlice::from_mut_slice(input) };
    boojum_cuda::ops_complex::bit_reverse_in_place(input, stream)
}

pub(crate) fn batch_bitreverse(input: &mut [F], num_rows: usize) -> CudaResult<()> {
    use boojum_cuda::device_structures::DeviceMatrixMut;
    let stream = get_stream();
    let mut input = unsafe {
        let input = DeviceSlice::from_mut_slice(input);
        DeviceMatrixMut::new(input, num_rows)
    };
    boojum_cuda::ops_complex::bit_reverse_in_place(&mut input, stream)
}
