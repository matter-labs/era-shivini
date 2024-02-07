use super::*;

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

fn l2_chunked<E>(
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
    E: FnMut(&mut [F], &CudaStream) -> CudaResult<()>
{
    let l2_chunk_elems = get_l2_chunk_elems(domain_size)?;
    let mut num_cols_processed = 0;
    let main_stream = get_stream();
    let chunk_streams = [get_stream0(), get_stream1()];
    let stream1 = get_stream1();
    let start_event = CudaEvent::create_with_flags(CudaEventCreateFlags::DISABLE_TIMING)?;
    start_event.record(&main_stream)?;
    stream0.wait_event(&start_event, CudaStreamWaitEventFlags::DEFAULT)?;
    stream1.wait_event(&start_event, CudaStreamWaitEventFlags::DEFAULT)?;
    for input_chunk in inputs.chunks_mut(l2_chunk_elems) {
        let num_cols_this_chunk = input_chunk.len() / domain_size;
        let num_cols_stream0 = num_cols_this_chunk / 2;
        let num_cols_stream1 = num_cols_this_chunk - num_cols_stream0;
        let elems_stream0 = num_cols_stream0 * domain_size;
        if num_cols_stream0 > 0 {
            raw_batch_coset_ntt(
                &mut input_chunk[..elems_stream0],
                bitreversed_input,
                inverse,
                coset_idx,
                domain_size,
                lde_degree,
                num_cols_stream0,
                &stream0,
            )?;
        }
        if num_cols_stream1 > 0 {
            raw_batch_coset_ntt(
                &mut input_chunk[elems_stream0..],
                bitreversed_input,
                inverse,
                coset_idx,
                domain_size,
                lde_degree,
                num_cols_stream1,
                &stream1,
            )?;
        }
        if num_cols_stream0 > 0 {
            epilogue(&mut input_chunk[..elems_stream0], &stream0)?;
        }
        if num_cols_stream1 > 0 {
            epilogue(&mut input_chunk[elems_stream0..], &stream1)?;
        }
        num_cols_processed += num_cols_this_chunk;
    }
    let end_event0 = CudaEvent::create_with_flags(CudaEventCreateFlags::DISABLE_TIMING)?;
    let end_event1 = CudaEvent::create_with_flags(CudaEventCreateFlags::DISABLE_TIMING)?;
    end_event0.record(&stream0)?;
    end_event1.record(&stream1)?;
    main_stream.wait_event(&end_event0, CudaStreamWaitEventFlags::DEFAULT)?;
    main_stream.wait_event(&end_event1, CudaStreamWaitEventFlags::DEFAULT)?;
    
    end_event0.destroy()?;
    end_event1.destroy()?;
    
    assert_eq!(num_cols_processed, num_polys);

    Ok(())
}

fn l2_chunked_into<E>(
    inputs: &[F],
    outputs: &mut [F],
    bitreversed_input: bool,
    inverse: bool,
    coset_idx: usize,
    domain_size: usize,
    lde_degree: usize,
    num_polys: usize,
    stream: &CudaStream,
    mut epilogue: E,
) -> CudaResult<()>
where
    E: FnMut(&mut [F], &CudaStream) -> CudaResult<()>
{
    let l2_chunk_elems = get_l2_chunk_elems(domain_size)?;
    let mut num_cols_processed = 0;
    let main_stream = get_stream();
    let stream0 = get_stream0();;
    let stream1 = get_stream1();
    let start_event = CudaEvent::create_with_flags(CudaEventCreateFlags::DISABLE_TIMING)?;
    start_event.record(&main_stream)?;
    stream0.wait_event(&start_event, CudaStreamWaitEventFlags::DEFAULT)?;
    stream1.wait_event(&start_event, CudaStreamWaitEventFlags::DEFAULT)?;
    for input_chunk, output_chunk in inputs.chunks(l2_chunk_elems)
        .zip(outputs.chunks_mut(l2_chunk_elems))
    {
        assert_eq!(input_chunk.len(), output_chunk.len());
        let num_cols_this_chunk = input_chunk.len() / domain_size;
        let num_cols_stream0 = num_cols_this_chunk / 2;
        let num_cols_stream1 = num_cols_this_chunk - num_cols_stream0;
        let elems_stream0 = num_cols_stream0 * domain_size;
        if num_cols_stream0 > 0 {
            raw_batch_coset_ntt_into(
                &input_chunk[..elems_stream0],
                &mut output_chunk[..elems_stream0],
                bitreversed_input,
                inverse,
                coset_idx,
                domain_size,
                lde_degree,
                num_cols_stream0,
                &stream0,
            )?;
        }
        if num_cols_stream1 > 0 {
            raw_batch_coset_ntt_into(
                &input_chunk[elems_stream0..],
                &mut output_chunk[elems_stream0..],
                bitreversed_input,
                inverse,
                coset_idx,
                domain_size,
                lde_degree,
                num_cols_stream1,
                &stream1,
            )?;
        }
        if num_cols_stream0 > 0 {
            epilogue(&mut input_chunk[..elems_stream0], &stream0)?;
        }
        if num_cols_stream1 > 0 {
            epilogue(&mut input_chunk[elems_stream0..], &stream1)?;
        }
        num_cols_processed += num_cols_this_chunk;
    }
    let end_event0 = CudaEvent::create_with_flags(CudaEventCreateFlags::DISABLE_TIMING)?;
    let end_event1 = CudaEvent::create_with_flags(CudaEventCreateFlags::DISABLE_TIMING)?;
    end_event0.record(&stream0)?;
    end_event1.record(&stream1)?;
    main_stream.wait_event(&end_event0, CudaStreamWaitEventFlags::DEFAULT)?;
    main_stream.wait_event(&end_event1, CudaStreamWaitEventFlags::DEFAULT)?;
    
    end_event0.destroy()?;
    end_event1.destroy()?;
    
    assert_eq!(num_cols_processed, num_polys);

    Ok(())
}

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
        |_, _| Ok(()),
    )
}

pub(crate) fn batch_ntt_with_epilogue<E>(
    input: &mut [F],
    bitreversed_input: bool,
    inverse: bool,
    domain_size: usize,
    num_polys: usize,
    mut epilogue: E,
) -> CudaResult<()>
where
    E: FnMut(&mut [F], &CudaStream) -> CudaResult<()>
{
    l2_chunked(
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
        |_, _| Ok(()),
    )
}

pub(crate) fn batch_ntt_with_epilogue_into(
    inputs: &[F],
    outputs: &mut [F],
    bitreversed_input: bool,
    inverse: bool,
    domain_size: usize,
    num_polys: usize,
    mut epilogue: E,
) -> CudaResult<()>
where
    E: FnMut(&mut [F], &CudaStream) -> CudaResult<()>
{
    l2_chunked_into(
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
        |_, _| Ok(()),
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
        |_, _| Ok(()),
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
        |_, _| Ok(()),
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
        |_, _| Ok(()),
    )
}

pub(crate) fn bitreverse(input: &mut [F]) -> CudaResult<()> {
    let stream = get_stream();
    let input = unsafe { DeviceSlice::from_mut_slice(input) };
    if_not_dry_run! {
        boojum_cuda::ops_complex::bit_reverse_in_place(input, stream)
    }
}

pub(crate) fn batch_bitreverse(input: &mut [F], num_rows: usize) -> CudaResult<()> {
    use boojum_cuda::device_structures::DeviceMatrixMut;
    let stream = get_stream();
    let mut input = unsafe {
        let input = DeviceSlice::from_mut_slice(input);
        DeviceMatrixMut::new(input, num_rows)
    };
    if_not_dry_run! {
        boojum_cuda::ops_complex::bit_reverse_in_place(&mut input, stream)
    }
}
