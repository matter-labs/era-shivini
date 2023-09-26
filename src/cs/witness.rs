use super::*;

use boojum::cs::traits::GoodAllocator;
use boojum_cuda::ops_complex::select;
use cudart::slice::DeviceSlice;

pub fn variable_assignment<A: GoodAllocator>(
    variable_indexes: &Vec<Vec<u32, A>>,
    variable_values: &Vec<F>,
    result: &mut [F],
) -> CudaResult<DVec<F>> {
    assert!(variable_indexes.len() > 0);
    let domain_size = variable_indexes[0].len();
    assert!(domain_size.is_power_of_two());
    assert!(variable_values.len() > 0);

    let num_cols = variable_indexes.len();
    let num_all_variable_indexes = num_cols * domain_size;
    assert_eq!(result.len(), num_all_variable_indexes);

    let mut d_variable_indexes = dvec!(num_all_variable_indexes);

    for (src, dst) in variable_indexes
        .iter()
        .zip(d_variable_indexes.chunks_mut(domain_size))
    {
        mem::h2d(src, dst)?;
    }
    let mut d_variable_values = dvec!(variable_values.len());
    mem::h2d(variable_values, &mut d_variable_values)?;

    let (d_variable_indexes_ref, d_variable_values_ref, d_result) = unsafe {
        (
            DeviceSlice::from_slice(&d_variable_indexes),
            DeviceSlice::from_slice(&d_variable_values),
            DeviceSlice::from_mut_slice(result),
        )
    };

    select(
        d_variable_indexes_ref,
        d_variable_values_ref,
        d_result,
        get_stream(),
    )?;

    Ok(d_variable_values)
}

pub fn multiplicity_assignment<A: GoodAllocator>(
    multiplicity_indexes: &Vec<u32, A>,
    d_variable_values: &DVec<F>,
    result: &mut [F],
) -> CudaResult<()> {
    assert!(multiplicity_indexes.len() > 0);
    assert!(d_variable_values.len() > 0);
    assert_eq!(result.len(), multiplicity_indexes.len());

    let mut d_multiplicity_indexes = dvec!(multiplicity_indexes.len());
    mem::h2d(&multiplicity_indexes, &mut d_multiplicity_indexes)?;

    let (d_multiplicity_indexes_ref, d_variable_values_ref, d_result) = unsafe {
        (
            DeviceSlice::from_slice(&d_multiplicity_indexes),
            DeviceSlice::from_slice(&d_variable_values[..]),
            DeviceSlice::from_mut_slice(result),
        )
    };

    select(
        d_multiplicity_indexes_ref,
        d_variable_values_ref,
        d_result,
        get_stream(),
    )?;

    Ok(())
}
