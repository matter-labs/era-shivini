use super::*;

#[allow(dead_code)]
pub fn assign_gate_selectors(
    _indexes: &[F],
    _gate_constants: &[F],
    _result: &mut [F],
    _stream: &CudaStream,
) -> CudaResult<()> {
    todo!()
}

pub use boojum_cuda::gates::GateEvaluationParams;
use boojum_cuda::{
    device_structures::{DeviceMatrix, DeviceMatrixMut},
    extension_field::VectorizedExtensionField,
};

pub fn constraint_evaluation(
    gates: &[GateEvaluationParams],
    variable_columns: &[F],
    witness_columns: &[F],
    constant_columns: &[F],
    challenge: &DExt,
    challenge_power_offset: usize,
    quotient: &mut [F],
    domain_size: usize,
) -> CudaResult<()> {
    assert_eq!(quotient.len(), 2 * domain_size);
    assert!(gates.is_empty() == false);

    let variable_columns_matrix = DeviceMatrix::new(
        unsafe { DeviceSlice::from_slice(variable_columns.as_ref()) },
        domain_size,
    );
    let witness_columns_matrix = DeviceMatrix::new(
        unsafe { DeviceSlice::from_slice(witness_columns.as_ref()) },
        domain_size,
    );
    let constant_columns_matrix = DeviceMatrix::new(
        unsafe { DeviceSlice::from_slice(constant_columns.as_ref()) },
        domain_size,
    );

    let mut d_challenge = svec!(2);
    mem::d2d(&challenge.c0.inner[..], &mut d_challenge[..1])?;
    mem::d2d(&challenge.c1.inner[..], &mut d_challenge[1..])?;
    let challenge = unsafe { DeviceSlice::from_slice(&d_challenge[..]) };
    let challenge = unsafe { challenge.transmute::<VectorizedExtensionField>() };

    let quotient = unsafe { DeviceSlice::from_mut_slice(quotient.as_mut()) };
    let mut quotient_matrix = DeviceMatrixMut::new(
        unsafe { quotient.transmute_mut::<VectorizedExtensionField>() },
        domain_size,
    );

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
    assert!(gates.is_empty() == false);

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
