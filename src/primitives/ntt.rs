use super::*;

use era_cudart::stream::CudaStreamWaitEventFlags;

// ntt operations

// Raw boojum bindings

fn raw_batch_coset_ntt(
    inputs: &mut [F],
    bitreversed_input: bool,
    inverse: bool,
    coset_idx: usize,
    domain_size: usize,
    lde_degree: usize,
    num_polys: usize,
    stream: &CudaStream,
) -> CudaResult<()> {
    assert_eq!(inputs.len(), num_polys * domain_size);
    assert!(domain_size.is_power_of_two());
    let log_n = domain_size.trailing_zeros();
    let log_lde_factor = lde_degree.trailing_zeros();
    let stride = 1 << log_n;
    let coset_idx = bitreverse_index(coset_idx, log_lde_factor as usize);
    let d_inputs = unsafe { DeviceSlice::from_mut_slice(inputs) };
    let inputs_offset = 0; // currently unused, but explicit for readability.
    if_not_dry_run! {
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
}

fn raw_batch_coset_ntt_into(
    inputs: &[F],
    outputs: &mut [F],
    bitreversed_input: bool,
    inverse: bool,
    coset_idx: usize,
    domain_size: usize,
    lde_degree: usize,
    num_polys: usize,
    stream: &CudaStream,
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
    let inputs_offset = 0; // currently unused, but explicit for readability.
    let outputs_offset = 0; // currently unused, but explicit for readability.
    if_not_dry_run! {
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
}

// Convenience wrappers for our use cases

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
    raw_batch_coset_ntt_into(
        input,
        output,
        false,
        false,
        coset_idx,
        input.len(),
        lde_degree,
        1,
        get_stream(),
    )
}

pub(crate) fn lde_intt(input: &mut [F]) -> CudaResult<()> {
    // Any power of two > 1 would work for lde_degree, it just signals to the kernel
    // that we're inverting an LDE and it should multiply x_i by g_inv^i
    let dummy_lde_degree = 2;
    let coset_idx = 0;
    raw_batch_coset_ntt(
        input,
        true,
        true,
        coset_idx,
        input.len(),
        dummy_lde_degree,
        1,
        get_stream(),
    )
}

#[allow(dead_code)]
pub(crate) fn intt_into(input: &[F], output: &mut [F]) -> CudaResult<()> {
    raw_batch_coset_ntt_into(
        input,
        output,
        false,
        true,
        0,
        input.len(),
        1,
        1,
        get_stream(),
    )
}

fn get_l2_chunk_elems(domain_size: usize) -> usize {
    let l2_cache_size_bytes = _l2_cache_size();
    // Targeting 3/8 of L2 capacity seems to yield good performance on L4
    let l2_cache_size_with_safety_margin = (l2_cache_size_bytes * 3) / 8;
    let bytes_per_col = 8 * domain_size;
    let cols_in_l2 = l2_cache_size_with_safety_margin / bytes_per_col;
    return domain_size * cols_in_l2;
}

fn l2_chunked_with_epilogue<E>(
    inputs: &mut [F],
    bitreversed_input: bool,
    inverse: bool,
    coset_idx: usize,
    domain_size: usize,
    lde_degree: usize,
    num_polys: usize,
    mut epilogue: E,
) -> CudaResult<()>
where
    E: FnMut(&mut [F], &CudaStream) -> CudaResult<()>,
{
    let l2_chunk_elems = get_l2_chunk_elems(domain_size);
    if l2_chunk_elems == 0 {
        // L2 cache is too small to fit even one chunk, so don't bother.
        let stream = get_stream();
        raw_batch_coset_ntt(
            inputs,
            bitreversed_input,
            inverse,
            coset_idx,
            domain_size,
            lde_degree,
            num_polys,
            stream,
        )?;
        epilogue(inputs, stream)?;
    } else {
        let mut num_cols_processed = 0;
        let main_stream = get_stream();
        let stream0 = &_aux_streams()[0];
        let stream1 = &_aux_streams()[1];
        let start_event = &_aux_events()[0];
        let end_event0 = &_aux_events()[1];
        let end_event1 = &_aux_events()[2];
        if_not_dry_run! {
            start_event.record(&main_stream)?;
            stream0.wait_event(start_event, CudaStreamWaitEventFlags::DEFAULT)?;
            stream1.wait_event(start_event, CudaStreamWaitEventFlags::DEFAULT)
        }?;
        for input_chunk in inputs.chunks_mut(l2_chunk_elems) {
            let len = input_chunk.len();
            let num_cols_this_chunk = len / domain_size;
            let num_cols0 = num_cols_this_chunk / 2;
            let num_cols1 = num_cols_this_chunk - num_cols0;
            let elems0 = num_cols0 * domain_size;
            // breadth first
            for ((stream, num_cols), range) in [stream0, stream1]
                .iter()
                .zip([num_cols0, num_cols1])
                .zip([0..elems0, elems0..len])
            {
                if num_cols > 0 {
                    raw_batch_coset_ntt(
                        &mut input_chunk[range.clone()],
                        bitreversed_input,
                        inverse,
                        coset_idx,
                        domain_size,
                        lde_degree,
                        num_cols,
                        stream,
                    )?;
                }
            }
            for ((stream, num_cols), range) in [stream0, stream1]
                .iter()
                .zip([num_cols0, num_cols1])
                .zip([0..elems0, elems0..len])
            {
                if num_cols > 0 {
                    epilogue(&mut input_chunk[range], stream)?;
                }
                num_cols_processed += num_cols;
            }
        }
        if_not_dry_run! {
            end_event0.record(stream0)?;
            end_event1.record(stream1)?;
            main_stream.wait_event(end_event0, CudaStreamWaitEventFlags::DEFAULT)?;
            main_stream.wait_event(end_event1, CudaStreamWaitEventFlags::DEFAULT)
        }?;

        assert_eq!(num_cols_processed, num_polys);
    }

    Ok(())
}

fn l2_chunked(
    inputs: &mut [F],
    bitreversed_input: bool,
    inverse: bool,
    coset_idx: usize,
    domain_size: usize,
    lde_degree: usize,
    num_polys: usize,
) -> CudaResult<()> {
    l2_chunked_with_epilogue(
        inputs,
        bitreversed_input,
        inverse,
        coset_idx,
        domain_size,
        lde_degree,
        num_polys,
        |_, _| Ok(()),
    )
}

fn l2_chunked_with_epilogue_into<E>(
    inputs: &[F],
    outputs: &mut [F],
    bitreversed_input: bool,
    inverse: bool,
    coset_idx: usize,
    domain_size: usize,
    lde_degree: usize,
    num_polys: usize,
    mut epilogue: E,
) -> CudaResult<()>
where
    E: FnMut(&mut [F], &CudaStream) -> CudaResult<()>,
{
    let l2_chunk_elems = get_l2_chunk_elems(domain_size);
    if l2_chunk_elems == 0 {
        let stream = get_stream();
        raw_batch_coset_ntt_into(
            inputs,
            outputs,
            bitreversed_input,
            inverse,
            coset_idx,
            domain_size,
            lde_degree,
            num_polys,
            stream,
        )?;
        epilogue(outputs, stream)?;
    } else {
        let mut num_cols_processed = 0;
        let main_stream = get_stream();
        let stream0 = &_aux_streams()[0];
        let stream1 = &_aux_streams()[1];
        let start_event = &_aux_events()[0];
        let end_event0 = &_aux_events()[1];
        let end_event1 = &_aux_events()[2];
        if_not_dry_run! {
            start_event.record(&main_stream)?;
            stream0.wait_event(start_event, CudaStreamWaitEventFlags::DEFAULT)?;
            stream1.wait_event(start_event, CudaStreamWaitEventFlags::DEFAULT)
        }?;
        for (input_chunk, output_chunk) in inputs
            .chunks(l2_chunk_elems)
            .zip(outputs.chunks_mut(l2_chunk_elems))
        {
            let len = input_chunk.len();
            assert_eq!(len, output_chunk.len());
            let num_cols_this_chunk = len / domain_size;
            let num_cols0 = num_cols_this_chunk / 2;
            let num_cols1 = num_cols_this_chunk - num_cols0;
            let elems0 = num_cols0 * domain_size;
            // breadth first
            for ((stream, num_cols), range) in [stream0, stream1]
                .iter()
                .zip([num_cols0, num_cols1])
                .zip([0..elems0, elems0..len])
            {
                if num_cols > 0 {
                    raw_batch_coset_ntt_into(
                        &input_chunk[range.clone()],
                        &mut output_chunk[range.clone()],
                        bitreversed_input,
                        inverse,
                        coset_idx,
                        domain_size,
                        lde_degree,
                        num_cols,
                        stream,
                    )?;
                }
            }
            for ((stream, num_cols), range) in [stream0, stream1]
                .iter()
                .zip([num_cols0, num_cols1])
                .zip([0..elems0, elems0..len])
            {
                if num_cols > 0 {
                    epilogue(&mut output_chunk[range], stream)?;
                }
                num_cols_processed += num_cols;
            }
        }
        if_not_dry_run! {
            end_event0.record(stream0)?;
            end_event1.record(stream1)?;
            main_stream.wait_event(end_event0, CudaStreamWaitEventFlags::DEFAULT)?;
            main_stream.wait_event(end_event1, CudaStreamWaitEventFlags::DEFAULT)
        }?;

        assert_eq!(num_cols_processed, num_polys);
    }

    Ok(())
}

fn l2_chunked_into(
    inputs: &[F],
    outputs: &mut [F],
    bitreversed_input: bool,
    inverse: bool,
    coset_idx: usize,
    domain_size: usize,
    lde_degree: usize,
    num_polys: usize,
) -> CudaResult<()> {
    l2_chunked_with_epilogue_into(
        inputs,
        outputs,
        bitreversed_input,
        inverse,
        coset_idx,
        domain_size,
        lde_degree,
        num_polys,
        |_, _| Ok(()),
    )
}

#[allow(dead_code)]
pub(crate) fn batch_ntt(
    input: &mut [F],
    bitreversed_input: bool,
    inverse: bool,
    domain_size: usize,
    num_polys: usize,
) -> CudaResult<()> {
    l2_chunked(
        input,
        bitreversed_input,
        inverse,
        0,
        domain_size,
        1,
        num_polys,
    )
}

pub(crate) fn batch_ntt_with_epilogue<E>(
    input: &mut [F],
    bitreversed_input: bool,
    inverse: bool,
    domain_size: usize,
    num_polys: usize,
    epilogue: E,
) -> CudaResult<()>
where
    E: FnMut(&mut [F], &CudaStream) -> CudaResult<()>,
{
    l2_chunked_with_epilogue(
        input,
        bitreversed_input,
        inverse,
        0,
        domain_size,
        1,
        num_polys,
        epilogue,
    )
}

#[allow(dead_code)]
pub(crate) fn batch_ntt_into(
    inputs: &[F],
    outputs: &mut [F],
    bitreversed_input: bool,
    inverse: bool,
    domain_size: usize,
    num_polys: usize,
) -> CudaResult<()> {
    l2_chunked_into(
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

pub(crate) fn batch_ntt_with_epilogue_into<E>(
    inputs: &[F],
    outputs: &mut [F],
    bitreversed_input: bool,
    inverse: bool,
    domain_size: usize,
    num_polys: usize,
    epilogue: E,
) -> CudaResult<()>
where
    E: FnMut(&mut [F], &CudaStream) -> CudaResult<()>,
{
    l2_chunked_with_epilogue_into(
        inputs,
        outputs,
        bitreversed_input,
        inverse,
        0,
        domain_size,
        1,
        num_polys,
        epilogue,
    )
}

pub(crate) fn batch_coset_ntt(
    inputs: &mut [F],
    coset_idx: usize,
    domain_size: usize,
    lde_degree: usize,
    num_polys: usize,
) -> CudaResult<()> {
    assert!(lde_degree > 1);
    assert!(lde_degree.is_power_of_two());
    assert!(coset_idx < lde_degree);
    l2_chunked(
        inputs,
        false,
        false,
        coset_idx,
        domain_size,
        lde_degree,
        num_polys,
    )
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
    l2_chunked_into(
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

pub(crate) fn batch_inverse_coset_ntt(
    inputs: &mut [F],
    coset_idx: usize,
    domain_size: usize,
    lde_degree: usize,
    num_polys: usize,
) -> CudaResult<()> {
    assert!(lde_degree > 1);
    assert!(lde_degree.is_power_of_two());
    assert!(coset_idx < lde_degree);
    l2_chunked(
        inputs,
        true,
        true,
        coset_idx,
        domain_size,
        lde_degree,
        num_polys,
    )
}

pub(crate) fn batch_inverse_coset_ntt_into(
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
    l2_chunked_into(
        inputs,
        outputs,
        true,
        true,
        coset_idx,
        domain_size,
        lde_degree,
        num_polys,
    )
}

pub(crate) fn bitreverse(input: &mut [F]) -> CudaResult<()> {
    let stream = get_stream();
    let input = unsafe { DeviceSlice::from_mut_slice(input) };
    if_not_dry_run! {
        boojum_cuda::ops_complex::bit_reverse_in_place(input, stream)
    }
}

pub(crate) fn batch_bitreverse_on_stream(
    input: &mut [F],
    num_rows: usize,
    stream: &CudaStream,
) -> CudaResult<()> {
    use boojum_cuda::device_structures::DeviceMatrixMut;
    let mut input = unsafe {
        let input = DeviceSlice::from_mut_slice(input);
        DeviceMatrixMut::new(input, num_rows)
    };
    if_not_dry_run! {
        boojum_cuda::ops_complex::bit_reverse_in_place(&mut input, stream)
    }
}

#[allow(dead_code)]
pub(crate) fn batch_bitreverse(input: &mut [F], num_rows: usize) -> CudaResult<()> {
    batch_bitreverse_on_stream(input, num_rows, get_stream())
}
