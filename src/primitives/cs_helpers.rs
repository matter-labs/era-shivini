use super::*;
use boojum_cuda::device_structures::{DeviceMatrixChunk, DeviceMatrixChunkMut};
pub use boojum_cuda::gates::GateEvaluationParams;
use boojum_cuda::{
    device_structures::{DeviceMatrix, DeviceMatrixMut},
    extension_field::VectorizedExtensionField,
};
use era_cudart::stream::CudaStreamWaitEventFlags;
use std::mem::size_of;

#[allow(dead_code)]
pub fn assign_gate_selectors(
    _indexes: &[F],
    _gate_constants: &[F],
    _result: &mut [F],
    _stream: &CudaStream,
) -> CudaResult<()> {
    todo!()
}

pub fn constraint_evaluation(
    gates: &[GateEvaluationParams],
    variable_columns: &[F],
    witness_columns: &[F],
    constant_columns: &[F],
    challenge: &DExt,
    challenge_power_offset: usize,
    quotient: &mut [F],
    domain_size: usize,
    is_specialized: bool,
) -> CudaResult<()> {
    assert_eq!(quotient.len(), 2 * domain_size);
    assert!(!gates.is_empty());

    let mut d_challenge = svec!(2);
    mem::d2d(&challenge.c0.inner[..], &mut d_challenge[..1])?;
    mem::d2d(&challenge.c1.inner[..], &mut d_challenge[1..])?;
    let challenge = unsafe { DeviceSlice::from_slice(&d_challenge[..]) };
    let challenge = unsafe { challenge.transmute::<VectorizedExtensionField>() };

    let variables_slice = unsafe { DeviceSlice::from_slice(variable_columns.as_ref()) };
    let witnesses_slice = unsafe { DeviceSlice::from_slice(witness_columns.as_ref()) };
    let constants_slice = unsafe { DeviceSlice::from_slice(constant_columns.as_ref()) };
    let quotient_slice = unsafe {
        DeviceSlice::from_mut_slice(quotient.as_mut()).transmute_mut::<VectorizedExtensionField>()
    };
    const STREAMS_COUNT: usize = 4;
    assert!(STREAMS_COUNT <= NUM_AUX_STREAMS_AND_EVENTS);
    const BLOCK_SIZE: usize = 128;
    let l2_size = _l2_cache_size();
    let (cc_major, cc_minor) = _compute_capability();
    let cols_count =
        (variables_slice.len() + witnesses_slice.len() + constants_slice.len()) / domain_size + 2;
    let chunk_rows =
        l2_size / (STREAMS_COUNT * size_of::<F>() * cols_count) / BLOCK_SIZE * BLOCK_SIZE;
    let split = if chunk_rows == 0 {
        1
    } else {
        (domain_size + chunk_rows - 1) / chunk_rows
    };
    if is_specialized || split == 1 || cc_major < 8 || (cc_major == 8 && cc_minor == 6) {
        let variable_columns_matrix = DeviceMatrix::new(variables_slice, domain_size);
        let witness_columns_matrix = DeviceMatrix::new(witnesses_slice, domain_size);
        let constant_columns_matrix = DeviceMatrix::new(constants_slice, domain_size);
        let mut quotient_matrix = DeviceMatrixMut::new(quotient_slice, domain_size);
        if_not_dry_run! {
            boojum_cuda::gates::evaluate_gates(
                &gates,
                &variable_columns_matrix,
                &witness_columns_matrix,
                &constant_columns_matrix,
                challenge,
                &mut quotient_matrix,
                challenge_power_offset as u32,
                get_stream(),
            ).map(|_| ())
        }
    } else {
        if !is_dry_run()? {
            let events = &_aux_events()[0..STREAMS_COUNT];
            let streams = &_aux_streams()[0..STREAMS_COUNT];
            let main_stream = get_stream();
            events[0].record(main_stream)?;
            for stream in streams.iter() {
                stream.wait_event(&events[0], CudaStreamWaitEventFlags::DEFAULT)?;
            }
            for i in 0..split {
                let offset = i * chunk_rows;
                let rows = if i == split - 1 {
                    domain_size - offset
                } else {
                    chunk_rows
                };
                let variable_columns_matrix =
                    DeviceMatrixChunk::new(variables_slice, domain_size, offset, rows);
                let witness_columns_matrix =
                    DeviceMatrixChunk::new(witnesses_slice, domain_size, offset, rows);
                let constant_columns_matrix =
                    DeviceMatrixChunk::new(constants_slice, domain_size, offset, rows);
                let mut quotient_matrix =
                    DeviceMatrixChunkMut::new(quotient_slice, domain_size, offset, rows);
                let stream = &streams[i % STREAMS_COUNT];
                boojum_cuda::gates::evaluate_gates(
                    &gates,
                    &variable_columns_matrix,
                    &witness_columns_matrix,
                    &constant_columns_matrix,
                    challenge,
                    &mut quotient_matrix,
                    challenge_power_offset as u32,
                    stream,
                )
                .map(|_| ())?;
            }
            for (event, stream) in events.iter().zip(streams.iter()) {
                event.record(stream)?;
                main_stream.wait_event(event, CudaStreamWaitEventFlags::DEFAULT)?;
            }
        }
        Ok(())
    }
}

#[allow(dead_code)]
pub fn constraint_evaluation_over_lde(
    gates: Vec<GateEvaluationParams>,
    variable_columns: &DVec<F>,
    witness_columns: &DVec<F>,
    constant_columns: &DVec<F>,
    challenge: &DExt,
    quotient: &mut DVec<F>,
    lde_size: usize,
) -> CudaResult<()> {
    assert_eq!(quotient.len(), 2 * lde_size);
    assert!(!gates.is_empty());

    let variable_columns_matrix = DeviceMatrix::new(
        unsafe { DeviceSlice::from_slice(variable_columns.as_ref()) },
        lde_size,
    );
    let witness_columns_matrix = DeviceMatrix::new(
        unsafe { DeviceSlice::from_slice(witness_columns.as_ref()) },
        lde_size,
    );
    let constant_columns_matrix = DeviceMatrix::new(
        unsafe { DeviceSlice::from_slice(constant_columns.as_ref()) },
        lde_size,
    );

    let mut d_challenge = svec!(2);
    mem::d2d(&challenge.c0.inner[..], &mut d_challenge[..1])?;
    mem::d2d(&challenge.c1.inner[..], &mut d_challenge[1..])?;
    let challenge = unsafe { DeviceSlice::from_slice(&d_challenge[..]) };
    let challenge = unsafe { challenge.transmute::<VectorizedExtensionField>() };

    let quotient = unsafe { DeviceSlice::from_mut_slice(quotient.as_mut()) };
    let mut quotient_matrix = DeviceMatrixMut::new(
        unsafe { quotient.transmute_mut::<VectorizedExtensionField>() },
        lde_size,
    );

    if_not_dry_run! {
        boojum_cuda::gates::evaluate_gates(
            &gates,
            &variable_columns_matrix,
            &witness_columns_matrix,
            &constant_columns_matrix,
            challenge,
            &mut quotient_matrix,
            0,
            get_stream(),
        ).map(|_| ())
    }
}
