use super::*;

use boojum_cuda::ops_complex::select;
use cudart::slice::DeviceSlice;

pub fn variable_assignment(
    d_variable_indexes: &DVec<u32>,
    d_variable_values: &DVec<F>,
    d_result: &mut [F],
) -> CudaResult<()> {
    if d_variable_indexes.is_empty() {
        return Ok(());
    }
    assert!(d_variable_values.len() > 0);
    assert!(d_variable_indexes.len() <= d_result.len());
    assert_eq!(
        d_variable_indexes.len() as u32 & PACKED_PLACEHOLDER_BITMASK,
        0
    );

    let (d_result, padding) = d_result.split_at_mut(d_variable_indexes.len());
    if !padding.is_empty() {
        helpers::set_zero(padding)?;
    }

    let (d_variable_indexes_ref, d_variable_values_ref, d_result) = unsafe {
        (
            DeviceSlice::from_slice(&d_variable_indexes),
            DeviceSlice::from_slice(&d_variable_values),
            DeviceSlice::from_mut_slice(d_result),
        )
    };

    select(
        d_variable_indexes_ref,
        d_variable_values_ref,
        d_result,
        get_stream(),
    )?;

    Ok(())
}
