use super::*;

use boojum_cuda::device_structures::{DeviceMatrix, DeviceMatrixMut, Vectorized};
use boojum_cuda::extension_field::VectorizedExtensionField;
// arithmetic operations
use boojum_cuda::ops_cub::device_scan::*;
use boojum_cuda::ops_simple::*;
use cudart::slice::DeviceVariable;

pub fn add_assign(this: &mut [F], other: &[F]) -> CudaResult<()> {
    assert_eq!(this.len(), other.len());
    let (this, other) = unsafe {
        let this = DeviceSlice::from_mut_slice(this);
        let other = DeviceSlice::from_slice(other);
        (this, other)
    };

    add_into_x(this, other, get_stream())
}

#[allow(dead_code)]
pub fn sub_assign(this: &mut [F], other: &[F]) -> CudaResult<()> {
    assert_eq!(this.len(), other.len());
    let (this, other) = unsafe {
        let this = DeviceSlice::from_mut_slice(this);
        let other = DeviceSlice::from_slice(other);
        (this, other)
    };
    sub_into_x(this, other, get_stream())
}

pub fn mul_assign(this: &mut [F], other: &[F]) -> CudaResult<()> {
    assert_eq!(this.len(), other.len());
    let (this, other) = unsafe {
        let this = DeviceSlice::from_mut_slice(this);
        let other = DeviceSlice::from_slice(other);
        (this, other)
    };

    mul_into_x(this, other, get_stream())
}

pub fn add_constant(this: &mut [F], value: &DF) -> CudaResult<()> {
    let (this, value) = unsafe {
        let this = DeviceSlice::from_mut_slice(this);
        let d_var = DeviceVariable::from_ref(&value.inner[0]);

        (this, d_var)
    };
    add_into_x(this, value, get_stream())
}

pub fn sub_constant(this: &mut [F], value: &DF) -> CudaResult<()> {
    let (this, value) = unsafe {
        let this = DeviceSlice::from_mut_slice(this);
        let d_var = DeviceVariable::from_ref(&value.inner[0]);

        (this, d_var)
    };
    sub_into_x(this, value, get_stream())
}

pub fn scale(this: &mut [F], value: &DF) -> CudaResult<()> {
    let (this, value) = unsafe {
        let this = DeviceSlice::from_mut_slice(this);
        let d_var = DeviceVariable::from_ref(&value.inner[0]);

        (this, d_var)
    };
    mul_into_x(this, value, get_stream())
}

pub fn inverse(values: &mut [F]) -> CudaResult<()> {
    let values_vector = unsafe { DeviceSlice::from_mut_slice(values) };
    boojum_cuda::ops_complex::batch_inv_in_place(values_vector, get_stream())
}

pub fn inverse_ef(c0: &mut [F], c1: &mut [F]) -> CudaResult<()> {
    use std::slice;
    type VEF = VectorizedExtensionField;

    let domain_size = c0.len();
    let c0_ptr = c0.as_ptr();
    unsafe {
        assert_eq!(c0_ptr.add(domain_size), c1.as_ptr());
    }
    let values_ptr = c0_ptr as *mut VEF;
    let mut values_slice: &mut [VEF] =
        unsafe { slice::from_raw_parts_mut(values_ptr, domain_size) };
    let values_vector = unsafe { DeviceSlice::from_mut_slice(&mut values_slice) };

    boojum_cuda::ops_complex::batch_inv_in_place(values_vector, get_stream())
}

pub fn negate(values: &mut [F]) -> CudaResult<()> {
    let values = unsafe { DeviceSlice::from_mut_slice(values) };
    neg_in_place(values, get_stream())
}

pub fn shifted_grand_product(values: &mut [F], tmp: &mut [F]) -> CudaResult<()> {
    let (values, tmp) = unsafe {
        let values = DeviceSlice::from_mut_slice(values);
        let ptr = tmp.as_mut_ptr() as *mut u8;
        let len = tmp.len() * 8;
        let tmp = std::slice::from_raw_parts_mut(ptr, len);
        let tmp = DeviceSlice::from_mut_slice(tmp);

        (values, tmp)
    };
    scan_in_place(
        ScanOperation::Product,
        false,
        false,
        tmp,
        values,
        get_stream(),
    )
}

pub fn complex_shifted_grand_product(c0: &mut [F], c1: &mut [F], tmp: &mut [F]) -> CudaResult<()> {
    let domain_size = c0.len();
    assert!(domain_size.is_power_of_two());
    assert_eq!(c0.len(), c1.len());

    let mut values_vectorized = dvec!(2 * domain_size);
    mem::d2d(&c0, &mut values_vectorized[..domain_size])?;
    mem::d2d(&c1, &mut values_vectorized[domain_size..])?;

    let mut values_tuple: DVec<F> = dvec!(2 * domain_size);

    unsafe {
        let tmp = DeviceSlice::from_mut_slice(tmp);
        let tmp = tmp.transmute_mut::<u8>();
        // cuda requires values in [(c00, c01), (c10, c11).. ] format
        // so we transform representation first then do the reverse once
        // values are computed
        let values_vectorized = DeviceSlice::from_mut_slice(&mut values_vectorized[..]);
        let values_vectorized = values_vectorized.transmute_mut::<VectorizedExtensionField>();

        let values_tuple = DeviceSlice::from_mut_slice(&mut values_tuple[..]);
        let values_tuple = values_tuple.transmute_mut::<EF>();

        boojum_cuda::extension_field::convert(values_vectorized, values_tuple, get_stream())?;

        scan_in_place(
            ScanOperation::Product,
            false,
            false,
            tmp,
            values_tuple,
            get_stream(),
        )?;

        boojum_cuda::extension_field::convert(values_tuple, values_vectorized, get_stream())?;
        let values_vectorized = values_vectorized.transmute();

        mem::d2d(&values_vectorized.as_slice()[..domain_size], c0)?;
        mem::d2d(&values_vectorized.as_slice()[domain_size..], c1)?;
    };

    Ok(())
}

pub fn grand_sum(values: &[F], tmp: &mut [F]) -> CudaResult<DF> {
    let domain_size = values.len();
    assert!(domain_size.is_power_of_two());

    let mut tmp_values = dvec!(domain_size);
    mem::d2d(values, &mut tmp_values)?;

    let (tmp_values_slice, tmp) = unsafe {
        let values = DeviceSlice::from_mut_slice(&mut tmp_values[..]);
        let tmp = DeviceSlice::from_mut_slice(tmp);
        let tmp = tmp.transmute_mut::<u8>();

        (values, tmp)
    };
    scan_in_place(
        ScanOperation::Sum,
        true,
        false,
        tmp,
        tmp_values_slice,
        get_stream(),
    )?;

    // last element is the accumulated value
    let result = tmp_values.get(domain_size - 1)?;

    Ok(result)
}

pub fn evaluate_base_at_ext(values: &[F], point: &DExt) -> CudaResult<DExt> {
    assert!(values.is_empty() == false);
    let domain_size = values.len();
    assert!(domain_size.is_power_of_two());

    let mut c0_values = dvec!(domain_size);
    let mut c1_values = dvec!(domain_size);
    helpers::set_value(&mut c0_values, &point.c0)?;
    helpers::set_value(&mut c1_values, &point.c1)?;

    let tmp_size = helpers::calculate_tmp_buffer_size_for_grand_product(2 * domain_size)?;
    let mut tmp = dvec!(tmp_size);

    arith::complex_shifted_grand_product(&mut c0_values, &mut c1_values, &mut tmp)?;

    arith::mul_assign(&mut c0_values, values)?;
    arith::mul_assign(&mut c1_values, values)?;

    let tmp_size2 = helpers::calculate_tmp_buffer_size_for_grand_sum(domain_size)?;
    let mut tmp = if tmp_size2 > tmp_size {
        dvec!(tmp_size2)
    } else {
        tmp
    };

    let c0 = arith::grand_sum(&c0_values, &mut tmp)?;
    let c1 = arith::grand_sum(&c1_values, &mut tmp)?;

    let result = DExt::new(c0, c1);

    Ok(result)
}

pub fn evaluate_ext_at_ext(values_c0: &[F], values_c1: &[F], point: &DExt) -> CudaResult<DExt> {
    assert!(values_c0.is_empty() == false);
    assert_eq!(values_c0.len(), values_c1.len());

    let domain_size = values_c0.len();
    assert!(domain_size.is_power_of_two());

    let mut tmp_c0_values = dvec!(domain_size);
    let mut tmp_c1_values = dvec!(domain_size);
    helpers::set_value(&mut tmp_c0_values, &point.c0)?;
    helpers::set_value(&mut tmp_c1_values, &point.c1)?;

    let tmp_size = helpers::calculate_tmp_buffer_size_for_grand_product(2 * domain_size)?;
    let mut tmp = dvec!(tmp_size);

    arith::complex_shifted_grand_product(&mut tmp_c0_values, &mut tmp_c1_values, &mut tmp)?;

    let non_residue = DF::non_residue()?;

    let mut t0 = dvec!(domain_size);
    mem::d2d(&values_c0, &mut t0)?;
    let mut t1 = dvec!(domain_size);
    mem::d2d(&values_c1, &mut t1)?;

    arith::mul_assign(&mut t0, &tmp_c0_values)?;
    arith::mul_assign(&mut t1, &tmp_c1_values)?;
    arith::scale(&mut t1, &non_residue)?;
    arith::add_assign(&mut t0, &t1)?;

    let tmp_size2 = helpers::calculate_tmp_buffer_size_for_grand_sum(domain_size)?;
    let mut tmp = if tmp_size2 > tmp_size {
        dvec!(tmp_size2)
    } else {
        tmp
    };

    let c0 = arith::grand_sum(&t0, &mut tmp)?;

    mem::d2d(&values_c0, &mut t0)?;
    mem::d2d(&values_c1, &mut t1)?;

    arith::mul_assign(&mut t0, &tmp_c1_values)?;
    arith::mul_assign(&mut t1, &tmp_c0_values)?;
    arith::add_assign(&mut t0, &mut t1)?;
    let c1 = arith::grand_sum(&t0, &mut tmp)?;

    let result = DExt::new(c0, c1);

    Ok(result)
}

#[allow(dead_code)]
pub fn fold(
    c0: &[F],
    c1: &[F],
    dst_c0: &mut [F],
    dst_c1: &mut [F],
    coset_inv: DF,
    challenge: DExt,
) -> CudaResult<()> {
    let domain_size = c0.len();
    let fold_size = domain_size >> 1;
    assert!(domain_size.is_power_of_two());
    assert!(fold_size.is_power_of_two());
    assert_eq!(c0.len(), c1.len());
    assert_eq!(dst_c0.len(), dst_c1.len());
    assert_eq!(dst_c0.len(), fold_size);

    let mut values = dvec!(2 * domain_size);
    mem::d2d(c0, &mut values[..domain_size])?;
    mem::d2d(c1, &mut values[domain_size..])?;

    let values = unsafe {
        let values = DeviceSlice::from_slice(&values[..]);
        values.transmute::<VectorizedExtensionField>()
    };

    let mut d_challenge: SVec<EF> = svec!(1);
    let d_challenge = unsafe {
        mem::d2d(
            &challenge.c0.inner[..],
            &mut d_challenge.data[0].coeffs[..1],
        )?;
        mem::d2d(
            &challenge.c1.inner[..],
            &mut d_challenge.data[0].coeffs[1..],
        )?;
        DeviceVariable::from_ref(&d_challenge[0])
    };

    let mut result: DVec<F> = dvec!(2 * fold_size);
    let result_slice = unsafe {
        let result = DeviceSlice::from_mut_slice(&mut result);
        result.transmute_mut()
    };

    let coset_inv = coset_inv.inner.to_vec()?;
    boojum_cuda::ops_complex::fold(
        coset_inv[0], // host data
        &d_challenge[0],
        values,
        result_slice,
        get_stream(),
    )?;

    mem::d2d(&result[..fold_size], dst_c0)?;
    mem::d2d(&result[fold_size..], dst_c1)?;

    Ok(())
}

pub fn fold_flattened(src: &[F], dst: &mut [F], coset_inv: F, challenge: &DExt) -> CudaResult<()> {
    let domain_size = src.len();
    let fold_size = domain_size >> 1;
    assert!(domain_size.is_power_of_two());
    assert!(fold_size.is_power_of_two());
    assert_eq!(dst.len(), fold_size);

    let values = unsafe {
        let values = DeviceSlice::from_slice(&src[..]);
        values.transmute::<VectorizedExtensionField>()
    };

    // TODO
    let mut d_challenge: SVec<EF> = svec!(1);
    let d_challenge = unsafe {
        mem::d2d(
            &challenge.c0.inner[..],
            &mut d_challenge.data[0].coeffs[..1],
        )?;
        mem::d2d(
            &challenge.c1.inner[..],
            &mut d_challenge.data[0].coeffs[1..],
        )?;
        DeviceVariable::from_ref(&d_challenge[0])
    };

    let _result: DVec<F> = dvec!(2 * fold_size);
    let result_ref = unsafe {
        let result = DeviceSlice::from_mut_slice(dst);
        result.transmute_mut()
    };

    boojum_cuda::ops_complex::fold(coset_inv, &d_challenge[0], values, result_ref, get_stream())?;

    Ok(())
}

pub fn distribute_powers(values: &mut [F], base: &DF) -> CudaResult<()> {
    assert!(values.len().is_power_of_two());
    let powers = compute_powers(base, values.len())?;
    arith::mul_assign(values, &powers)?;

    Ok(())
}

pub fn compute_powers(base: &DF, size: usize) -> CudaResult<DVec<F>> {
    let mut powers = dvec!(size);
    helpers::set_value(&mut powers, base)?;
    let tmp_size = helpers::calculate_tmp_buffer_size_for_grand_product(size)?;
    let mut tmp = dvec!(tmp_size);
    arith::shifted_grand_product(&mut powers, &mut tmp)?;

    Ok(powers)
}

#[allow(dead_code)]
pub fn compute_powers_ext(base: &DExt, size: usize) -> CudaResult<[DVec<F>; 2]> {
    let mut powers_c0 = dvec!(size);
    helpers::set_value(&mut powers_c0, &base.c0)?;
    let mut powers_c1 = dvec!(size);
    helpers::set_value(&mut powers_c1, &base.c1)?;

    let tmp_size = helpers::calculate_tmp_buffer_size_for_grand_product(2 * size)?;
    let mut tmp = dvec!(tmp_size);
    arith::complex_shifted_grand_product(&mut powers_c0, &mut powers_c1, &mut tmp)?;

    Ok([powers_c0, powers_c1])
}

#[allow(dead_code)]
pub fn dext_as_dvec(input: &DExt) -> CudaResult<DVec<EF, SmallStaticDeviceAllocator>> {
    let mut out: DVec<EF, _> = svec!(1);
    mem::d2d(&input.c0.inner[..], &mut out.data[0].coeffs[..1])?;
    mem::d2d(&input.c1.inner[..], &mut out.data[0].coeffs[1..])?;

    Ok(out)
}

pub fn precompute_barycentric_bases(
    bases: &mut [F],
    domain_size: usize,
    coset: F,
    point: EF,
) -> CudaResult<()> {
    // (X^N - 1)/ N
    // evaluations are elems of first coset of the lde
    // shift is k*w^0=k where k is multiplicative generator
    let mut d_point = svec!(1);
    d_point.copy_from_slice(&[point])?;

    let mut d_tmp: SVec<EF> = svec!(1);

    let (bases, point, common_factor_storage) = unsafe {
        let point = &d_point[0];
        let tmp_point = &mut d_tmp[0];
        let v_bases = std::slice::from_raw_parts_mut(bases.as_ptr() as *mut _, domain_size);
        (
            DeviceSlice::from_mut_slice(v_bases),
            DeviceVariable::from_ref(point),
            DeviceVariable::from_mut(tmp_point),
        )
    };
    use boojum_cuda::barycentric::PrecomputeAtExt;
    boojum_cuda::barycentric::precompute_lagrange_coeffs::<PrecomputeAtExt>(
        point,
        common_factor_storage,
        coset,
        bases,
        true,
        get_stream(),
    )?;

    Ok(())
}

pub fn barycentric_evaluate_base_at_ext<A: GoodAllocator>(
    values: &[F],
    bases: &[F],
    domain_size: usize,
    num_polys: usize,
) -> CudaResult<Vec<EF, A>> {
    assert_eq!(values.len(), num_polys * domain_size);
    barycentric_evaluate::<boojum_cuda::barycentric::EvalBaseAtExt, A>(
        values,
        bases,
        domain_size,
        num_polys,
    )
}

pub fn barycentric_evaluate_ext_at_ext<A: GoodAllocator>(
    values: &[F],
    bases: &[F],
    domain_size: usize,
    num_polys: usize,
) -> CudaResult<Vec<EF, A>> {
    assert_eq!(values.len(), 2 * num_polys * domain_size);
    let h_values = unsafe {
        std::slice::from_raw_parts_mut(values.as_ptr() as *mut _, num_polys * domain_size)
    };
    barycentric_evaluate::<boojum_cuda::barycentric::EvalExtAtExt, _>(
        h_values,
        bases,
        domain_size,
        num_polys,
    )
}

fn barycentric_evaluate<E: boojum_cuda::barycentric::EvalImpl, A: GoodAllocator>(
    values: &[<E::Y as Vectorized>::Type],
    bases: &[F],
    domain_size: usize,
    num_polys: usize,
) -> CudaResult<Vec<E::X, A>> {
    assert_eq!(values.len(), num_polys * domain_size);
    assert_eq!(bases.len(), 2 * domain_size);
    assert!(domain_size.is_power_of_two());

    // transmute bases into vectorized form
    let (values_matrix, bases) = unsafe {
        let values = DeviceSlice::from_slice(values);
        let values_matrix = DeviceMatrix::new(values, domain_size);

        let v_bases = std::slice::from_raw_parts(bases.as_ptr() as *const _, domain_size);
        let bases = DeviceSlice::from_slice(&v_bases);
        (values_matrix, bases)
    };

    // allocate necessary tmp buffers
    let (partial_reduce_temp_elems, final_cub_reduce_temp_bytes) =
        boojum_cuda::barycentric::get_batch_eval_temp_storage_sizes::<E>(&values_matrix).unwrap();

    let temp_storage_partial_reduce: DVec<EF, StaticDeviceAllocator> =
        dvec!(partial_reduce_temp_elems);

    let mut temp_storage_partial_reduce = unsafe {
        let buf = std::slice::from_raw_parts_mut(
            temp_storage_partial_reduce.data.as_ptr() as *mut _,
            partial_reduce_temp_elems,
        );
        assert_eq!(partial_reduce_temp_elems % num_polys, 0);
        let num_elems = partial_reduce_temp_elems / num_polys;
        DeviceMatrixMut::new(DeviceSlice::from_mut_slice(buf), num_elems)
    };

    let mut temp_storage_final_cub_reduce = dvec!(final_cub_reduce_temp_bytes);
    let temp_storage_final_cub_reduce = unsafe {
        DeviceSlice::from_mut_slice(
            &mut temp_storage_final_cub_reduce.data[..final_cub_reduce_temp_bytes],
        )
    };
    let mut evals: SVec<E::X> = svec!(num_polys);
    let evals_ref = unsafe { DeviceSlice::from_mut_slice(&mut evals) };

    boojum_cuda::barycentric::batch_eval::<E>(
        &values_matrix,
        bases,
        &mut temp_storage_partial_reduce,
        temp_storage_final_cub_reduce,
        evals_ref,
        get_stream(),
    )?;

    let result = evals.to_vec_in(A::default())?;

    Ok(result)
}

// TODO: Rework to accept slices
pub fn partial_products_num_denom_chunk<'a>(
    num: &mut ComplexPoly<'a, LagrangeBasis>,
    denom: &mut ComplexPoly<'a, LagrangeBasis>,
    variable_cols_chunk: &[Poly<'a, LagrangeBasis>],
    sigma_cols_chunk: &[Poly<'a, LagrangeBasis>],
    omega_values: &[F],
    non_residues_by_beta_chunk: &[EF],
    beta: &DExt,
    gamma: &DExt,
    num_cols_per_product: usize,
    domain_size: usize,
) -> CudaResult<()> {
    use std::slice;
    type VEF = VectorizedExtensionField;

    let num_polys = variable_cols_chunk.len();

    assert_eq!(num.c0.storage.len(), domain_size);
    let num_c0_ptr = num.c0.storage.as_ref().as_ptr();
    unsafe {
        assert_eq!(
            num_c0_ptr.add(domain_size),
            num.c1.storage.as_ref().as_ptr()
        );
    }
    let num_ptr = num_c0_ptr as *mut VEF;
    let mut num_slice: &mut [VEF] = unsafe { slice::from_raw_parts_mut(num_ptr, domain_size) };
    let num_vector = unsafe { DeviceSlice::from_mut_slice(&mut num_slice) };

    assert_eq!(denom.c0.storage.len(), domain_size);
    let denom_c0_ptr = denom.c0.storage.as_ref().as_ptr();
    unsafe {
        assert_eq!(
            denom_c0_ptr.add(domain_size),
            denom.c1.storage.as_ref().as_ptr()
        );
    }
    let denom_ptr = denom_c0_ptr as *mut VEF;
    let mut denom_slice: &mut [VEF] = unsafe { slice::from_raw_parts_mut(denom_ptr, domain_size) };
    let denom_vector = unsafe { DeviceSlice::from_mut_slice(&mut denom_slice) };

    assert_eq!(variable_cols_chunk.len(), num_polys);
    let variable_cols_ptr = variable_cols_chunk[0].storage.as_ref().as_ptr() as *const F;
    let variable_cols_slice =
        unsafe { slice::from_raw_parts(variable_cols_ptr, num_polys * domain_size) };
    let variable_cols_device_slice = unsafe { DeviceSlice::from_slice(variable_cols_slice) };
    let variable_cols_matrix = DeviceMatrix::new(variable_cols_device_slice, domain_size);

    assert_eq!(sigma_cols_chunk.len(), num_polys);
    let sigma_cols_ptr = sigma_cols_chunk[0].storage.as_ref().as_ptr() as *const F;
    let sigma_cols_slice =
        unsafe { slice::from_raw_parts(sigma_cols_ptr, num_polys * domain_size) };
    let sigma_cols_device_slice = unsafe { DeviceSlice::from_slice(sigma_cols_slice) };
    let sigma_cols_matrix = DeviceMatrix::new(sigma_cols_device_slice, domain_size);

    assert_eq!(omega_values.len(), domain_size);
    let omega_values_vector = unsafe { DeviceSlice::from_slice(&omega_values) };

    assert_eq!(non_residues_by_beta_chunk.len(), num_polys);
    let non_residues_by_beta_vector =
        unsafe { DeviceSlice::from_slice(&non_residues_by_beta_chunk) };

    let beta_c0 = unsafe { DeviceVariable::from_ref(&beta.c0.inner[0]) };
    let beta_c1 = unsafe { DeviceVariable::from_ref(&beta.c1.inner[0]) };
    let gamma_c0 = unsafe { DeviceVariable::from_ref(&gamma.c0.inner[0]) };
    let gamma_c1 = unsafe { DeviceVariable::from_ref(&gamma.c1.inner[0]) };

    boojum_cuda::ops_complex::partial_products_f_g_chunk(
        num_vector,
        denom_vector,
        &variable_cols_matrix,
        &sigma_cols_matrix,
        omega_values_vector,
        non_residues_by_beta_vector,
        beta_c0,
        beta_c1,
        gamma_c0,
        gamma_c1,
        num_cols_per_product,
        get_stream(),
    )
}

// TODO: Rework to accept slices
pub fn partial_products_quotient_terms<'a, 'b>(
    partial_products: &'a [ComplexPoly<'a, CosetEvaluations>],
    z_poly: &'a ComplexPoly<'a, CosetEvaluations>,
    variable_cols: &Vec<Poly<'a, CosetEvaluations>>,
    sigma_cols: &Vec<Poly<'a, CosetEvaluations>>,
    omega_values: &'a Poly<'a, CosetEvaluations>,
    powers_of_alpha: &[EF],
    non_residues_by_beta: &[EF],
    beta: &DExt,
    gamma: &DExt,
    quotient: &mut ComplexPoly<'b, CosetEvaluations>,
    num_cols_per_product: usize,
    num_polys: usize,
    domain_size: usize,
) -> CudaResult<()> {
    use std::slice;
    type VEF = VectorizedExtensionField;

    // Handling empty partial products would require special-case logic.
    // For now we don't need it. Assert as a reminder.
    assert!(partial_products.len() > 0);

    let num_partial_products = ((num_polys + num_cols_per_product - 1) / num_cols_per_product) - 1;

    assert_eq!(partial_products.len(), num_partial_products);
    let partial_products_ptr = partial_products[0].c0.storage.as_ref().as_ptr() as *const VEF;
    let partial_products_slice =
        unsafe { slice::from_raw_parts(partial_products_ptr, num_partial_products * domain_size) };
    let partial_products_device_slice = unsafe { DeviceSlice::from_slice(partial_products_slice) };
    let partial_products_matrix = DeviceMatrix::new(partial_products_device_slice, domain_size);

    assert_eq!(z_poly.c0.storage.len(), domain_size);
    let z_poly_c0_ptr = z_poly.c0.storage.as_ref().as_ptr();
    unsafe {
        assert_eq!(
            z_poly_c0_ptr.add(domain_size),
            z_poly.c1.storage.as_ref().as_ptr()
        );
    }
    let z_poly_ptr = z_poly_c0_ptr as *const VEF;
    let z_poly_slice: &[VEF] = unsafe { slice::from_raw_parts(z_poly_ptr, domain_size) };
    let z_poly_vector = unsafe { DeviceSlice::from_slice(&z_poly_slice) };

    assert_eq!(variable_cols.len(), num_polys);
    let variable_cols_ptr = variable_cols[0].storage.as_ref().as_ptr() as *const F;
    let variable_cols_slice =
        unsafe { slice::from_raw_parts(variable_cols_ptr, num_polys * domain_size) };
    let variable_cols_device_slice = unsafe { DeviceSlice::from_slice(variable_cols_slice) };
    let variable_cols_matrix = DeviceMatrix::new(variable_cols_device_slice, domain_size);

    assert_eq!(sigma_cols.len(), num_polys);
    let sigma_cols_ptr = sigma_cols[0].storage.as_ref().as_ptr() as *const F;
    let sigma_cols_slice =
        unsafe { slice::from_raw_parts(sigma_cols_ptr, num_polys * domain_size) };
    let sigma_cols_device_slice = unsafe { DeviceSlice::from_slice(sigma_cols_slice) };
    let sigma_cols_matrix = DeviceMatrix::new(sigma_cols_device_slice, domain_size);

    assert_eq!(omega_values.storage.len(), domain_size);
    let omega_values_ptr = omega_values.storage.as_ref().as_ptr() as *const F;
    let omega_values_slice = unsafe { slice::from_raw_parts(omega_values_ptr, domain_size) };
    let omega_values_vector = unsafe { DeviceSlice::from_slice(&omega_values_slice) };

    assert_eq!(powers_of_alpha.len(), num_partial_products + 1);
    let powers_of_alpha_vector = unsafe { DeviceSlice::from_slice(&powers_of_alpha) };

    assert_eq!(non_residues_by_beta.len(), num_polys);
    let non_residues_by_beta_vector = unsafe { DeviceSlice::from_slice(&non_residues_by_beta) };

    let beta_c0 = unsafe { DeviceVariable::from_ref(&beta.c0.inner[0]) };
    let beta_c1 = unsafe { DeviceVariable::from_ref(&beta.c1.inner[0]) };
    let gamma_c0 = unsafe { DeviceVariable::from_ref(&gamma.c0.inner[0]) };
    let gamma_c1 = unsafe { DeviceVariable::from_ref(&gamma.c1.inner[0]) };

    assert_eq!(quotient.c0.storage.len(), domain_size);
    let quotient_c0_ptr = quotient.c0.storage.as_ref().as_ptr();
    unsafe {
        assert_eq!(
            quotient_c0_ptr.add(domain_size),
            quotient.c1.storage.as_ref().as_ptr()
        );
    }
    let quotient_ptr = quotient_c0_ptr as *mut VEF;
    let mut quotient_slice: &mut [VEF] =
        unsafe { slice::from_raw_parts_mut(quotient_ptr, domain_size) };
    let quotient_vector = unsafe { DeviceSlice::from_mut_slice(&mut quotient_slice) };

    boojum_cuda::ops_complex::partial_products_quotient_terms(
        &partial_products_matrix,
        z_poly_vector,
        &variable_cols_matrix,
        &sigma_cols_matrix,
        omega_values_vector,
        powers_of_alpha_vector,
        non_residues_by_beta_vector,
        beta_c0,
        beta_c1,
        gamma_c0,
        gamma_c1,
        quotient_vector,
        num_cols_per_product,
        get_stream(),
    )
}

// TODO: Rework to accept slices
pub fn lookup_aggregated_table_values<'a>(
    table_cols: &[Poly<'a, LagrangeBasis>],
    beta: &DExt,
    powers_of_gamma: &[EF],
    aggregated_table_values: &mut ComplexPoly<'a, LagrangeBasis>,
    num_cols_per_subarg: usize,
    domain_size: usize,
) -> CudaResult<()> {
    use std::slice;
    type VEF = VectorizedExtensionField;

    let num_polys = num_cols_per_subarg + 1;

    assert_eq!(table_cols.len(), num_polys);
    let table_cols_ptr = table_cols[0].storage.as_ref().as_ptr() as *const F;
    let table_cols_slice =
        unsafe { slice::from_raw_parts(table_cols_ptr, num_polys * domain_size) };
    let table_cols_device_slice = unsafe { DeviceSlice::from_slice(table_cols_slice) };
    let table_cols_matrix = DeviceMatrix::new(table_cols_device_slice, domain_size);

    let beta_c0 = unsafe { DeviceVariable::from_ref(&beta.c0.inner[0]) };
    let beta_c1 = unsafe { DeviceVariable::from_ref(&beta.c1.inner[0]) };

    assert_eq!(powers_of_gamma.len(), num_polys);
    let powers_of_gamma_vector = unsafe { DeviceSlice::from_slice(&powers_of_gamma) };

    assert_eq!(aggregated_table_values.c0.storage.len(), domain_size);
    let aggregated_table_values_c0_ptr = aggregated_table_values.c0.storage.as_ref().as_ptr();
    unsafe {
        assert_eq!(
            aggregated_table_values_c0_ptr.add(domain_size),
            aggregated_table_values.c1.storage.as_ref().as_ptr()
        );
    }
    let aggregated_table_values_ptr = aggregated_table_values_c0_ptr as *mut VEF;
    let mut aggregated_table_values_slice: &mut [VEF] =
        unsafe { slice::from_raw_parts_mut(aggregated_table_values_ptr, domain_size) };
    let aggregated_table_values_vector =
        unsafe { DeviceSlice::from_mut_slice(&mut aggregated_table_values_slice) };

    boojum_cuda::ops_complex::lookup_aggregated_table_values(
        &table_cols_matrix,
        beta_c0,
        beta_c1,
        powers_of_gamma_vector,
        aggregated_table_values_vector,
        num_polys,
        get_stream(),
    )
}

// TODO: Rework to accept slices
pub fn lookup_subargs<'a>(
    variable_cols: &[Poly<'a, LagrangeBasis>],
    subargs_a: &mut [ComplexPoly<'a, LagrangeBasis>],
    subargs_b: &mut [ComplexPoly<'a, LagrangeBasis>],
    beta: &DExt,
    powers_of_gamma: &[EF],
    table_id_col: &Poly<'a, LagrangeBasis>,
    aggregated_table_values_inv: &ComplexPoly<'a, LagrangeBasis>,
    multiplicity_cols: &[Poly<'a, LagrangeBasis>],
    num_cols_per_subarg: usize,
    num_polys: usize,
    domain_size: usize,
) -> CudaResult<()> {
    use std::slice;
    type VEF = VectorizedExtensionField;

    let num_subargs_a = num_polys / num_cols_per_subarg;
    // num_polys should be an even multiple of num_cols_per_subarg
    assert_eq!(num_polys, num_subargs_a * num_cols_per_subarg);
    let num_subargs_b = subargs_b.len();
    assert_eq!(num_subargs_b, 1);

    assert_eq!(variable_cols.len(), num_polys);
    let variable_cols_ptr = variable_cols[0].storage.as_ref().as_ptr() as *const F;
    let variable_cols_slice =
        unsafe { slice::from_raw_parts(variable_cols_ptr, num_polys * domain_size) };
    let variable_cols_device_slice = unsafe { DeviceSlice::from_slice(variable_cols_slice) };
    let variable_cols_matrix = DeviceMatrix::new(variable_cols_device_slice, domain_size);

    assert_eq!(subargs_a.len(), num_subargs_a);
    let subargs_a_ptr = subargs_a[0].c0.storage.as_ref().as_ptr() as *mut VEF;
    let subargs_a_slice =
        unsafe { slice::from_raw_parts_mut(subargs_a_ptr, num_subargs_a * domain_size) };
    let subargs_a_device_slice = unsafe { DeviceSlice::from_mut_slice(subargs_a_slice) };
    let mut subargs_a_matrix = DeviceMatrixMut::new(subargs_a_device_slice, domain_size);

    assert_eq!(subargs_b.len(), num_subargs_b);
    let subargs_b_ptr = subargs_b[0].c0.storage.as_ref().as_ptr() as *mut VEF;
    let subargs_b_slice =
        unsafe { slice::from_raw_parts_mut(subargs_b_ptr, num_subargs_b * domain_size) };
    let subargs_b_device_slice = unsafe { DeviceSlice::from_mut_slice(subargs_b_slice) };
    let mut subargs_b_matrix = DeviceMatrixMut::new(subargs_b_device_slice, domain_size);

    let beta_c0 = unsafe { DeviceVariable::from_ref(&beta.c0.inner[0]) };
    let beta_c1 = unsafe { DeviceVariable::from_ref(&beta.c1.inner[0]) };

    assert_eq!(powers_of_gamma.len(), num_cols_per_subarg + 1);
    let powers_of_gamma_vector = unsafe { DeviceSlice::from_slice(&powers_of_gamma) };

    assert_eq!(table_id_col.storage.len(), domain_size);
    let table_id_col_ptr = table_id_col.storage.as_ref().as_ptr() as *const F;
    let table_id_col_slice = unsafe { slice::from_raw_parts(table_id_col_ptr, domain_size) };
    let table_id_col_vector = unsafe { DeviceSlice::from_slice(&table_id_col_slice) };

    assert_eq!(aggregated_table_values_inv.c0.storage.len(), domain_size);
    let aggregated_table_values_inv_c0_ptr =
        aggregated_table_values_inv.c0.storage.as_ref().as_ptr();
    unsafe {
        assert_eq!(
            aggregated_table_values_inv_c0_ptr.add(domain_size),
            aggregated_table_values_inv.c1.storage.as_ref().as_ptr()
        );
    }
    let aggregated_table_values_inv_ptr = aggregated_table_values_inv_c0_ptr as *const VEF;
    let aggregated_table_values_inv_slice: &[VEF] =
        unsafe { slice::from_raw_parts(aggregated_table_values_inv_ptr, domain_size) };
    let aggregated_table_values_inv_vector =
        unsafe { DeviceSlice::from_slice(&aggregated_table_values_inv_slice) };

    assert_eq!(multiplicity_cols.len(), num_subargs_b);
    let multiplicity_cols_ptr = multiplicity_cols[0].storage.as_ref().as_ptr() as *const F;
    let multiplicity_cols_slice =
        unsafe { slice::from_raw_parts(multiplicity_cols_ptr, num_subargs_b * domain_size) };
    let multiplicity_cols_device_slice =
        unsafe { DeviceSlice::from_slice(multiplicity_cols_slice) };
    let multiplicity_cols_matrix = DeviceMatrix::new(multiplicity_cols_device_slice, domain_size);

    boojum_cuda::ops_complex::lookup_subargs_a_and_b(
        &variable_cols_matrix,
        &mut subargs_a_matrix,
        &mut subargs_b_matrix,
        beta_c0,
        beta_c1,
        powers_of_gamma_vector,
        table_id_col_vector,
        aggregated_table_values_inv_vector,
        &multiplicity_cols_matrix,
        num_cols_per_subarg,
        get_stream(),
    )
}

// TODO: Rework to accept slices
pub fn lookup_quotient_ensure_a_and_b_are_well_formed<'a, 'b>(
    variable_cols: &[Poly<'a, CosetEvaluations>],
    table_cols: &[Poly<'a, CosetEvaluations>],
    subargs_a: &[ComplexPoly<'a, CosetEvaluations>],
    subargs_b: &[ComplexPoly<'a, CosetEvaluations>],
    beta: &DExt,
    powers_of_gamma: &[EF],
    powers_of_alpha: &[EF],
    table_id_col: &Poly<'a, CosetEvaluations>,
    multiplicity_cols: &[Poly<'a, CosetEvaluations>],
    quotient: &mut ComplexPoly<'b, CosetEvaluations>,
    num_cols_per_subarg: usize,
    num_polys: usize,
    domain_size: usize,
) -> CudaResult<()> {
    use std::slice;
    type VEF = VectorizedExtensionField;

    let num_subargs_a = num_polys / num_cols_per_subarg;
    // num_polys should be an even multiple of num_cols_per_subarg
    assert_eq!(num_polys, num_subargs_a * num_cols_per_subarg);
    let num_subargs_b = subargs_b.len();
    assert_eq!(num_subargs_b, 1);
    let num_cols_per_subarg_b = num_cols_per_subarg + 1;

    assert_eq!(variable_cols.len(), num_polys);
    let variable_cols_ptr = variable_cols[0].storage.as_ref().as_ptr() as *const F;
    let variable_cols_slice =
        unsafe { slice::from_raw_parts(variable_cols_ptr, num_polys * domain_size) };
    let variable_cols_device_slice = unsafe { DeviceSlice::from_slice(variable_cols_slice) };
    let variable_cols_matrix = DeviceMatrix::new(variable_cols_device_slice, domain_size);

    assert_eq!(table_cols.len(), num_cols_per_subarg_b);
    let table_cols_ptr = table_cols[0].storage.as_ref().as_ptr() as *const F;
    let table_cols_slice =
        unsafe { slice::from_raw_parts(table_cols_ptr, num_cols_per_subarg_b * domain_size) };
    let table_cols_device_slice = unsafe { DeviceSlice::from_slice(table_cols_slice) };
    let table_cols_matrix = DeviceMatrix::new(table_cols_device_slice, domain_size);

    assert_eq!(subargs_a.len(), num_subargs_a);
    let subargs_a_ptr = subargs_a[0].c0.storage.as_ref().as_ptr() as *const VEF;
    let subargs_a_slice =
        unsafe { slice::from_raw_parts(subargs_a_ptr, num_subargs_a * domain_size) };
    let subargs_a_device_slice = unsafe { DeviceSlice::from_slice(subargs_a_slice) };
    let subargs_a_matrix = DeviceMatrix::new(subargs_a_device_slice, domain_size);

    assert_eq!(subargs_b.len(), num_subargs_b);
    let subargs_b_ptr = subargs_b[0].c0.storage.as_ref().as_ptr() as *const VEF;
    let subargs_b_slice =
        unsafe { slice::from_raw_parts(subargs_b_ptr, num_subargs_b * domain_size) };
    let subargs_b_device_slice = unsafe { DeviceSlice::from_slice(subargs_b_slice) };
    let subargs_b_matrix = DeviceMatrix::new(subargs_b_device_slice, domain_size);

    let beta_c0 = unsafe { DeviceVariable::from_ref(&beta.c0.inner[0]) };
    let beta_c1 = unsafe { DeviceVariable::from_ref(&beta.c1.inner[0]) };

    assert_eq!(powers_of_gamma.len(), num_cols_per_subarg + 1);
    let powers_of_gamma_vector = unsafe { DeviceSlice::from_slice(&powers_of_gamma) };

    assert_eq!(powers_of_alpha.len(), num_subargs_a + num_subargs_b);
    let powers_of_alpha_vector = unsafe { DeviceSlice::from_slice(&powers_of_alpha) };

    assert_eq!(table_id_col.storage.len(), domain_size);
    let table_id_col_ptr = table_id_col.storage.as_ref().as_ptr() as *const F;
    let table_id_col_slice = unsafe { slice::from_raw_parts(table_id_col_ptr, domain_size) };
    let table_id_col_vector = unsafe { DeviceSlice::from_slice(&table_id_col_slice) };

    assert_eq!(multiplicity_cols.len(), num_subargs_b);
    let multiplicity_cols_ptr = multiplicity_cols[0].storage.as_ref().as_ptr() as *const F;
    let multiplicity_cols_slice =
        unsafe { slice::from_raw_parts(multiplicity_cols_ptr, num_subargs_b * domain_size) };
    let multiplicity_cols_device_slice =
        unsafe { DeviceSlice::from_slice(multiplicity_cols_slice) };
    let multiplicity_cols_matrix = DeviceMatrix::new(multiplicity_cols_device_slice, domain_size);

    assert_eq!(quotient.c0.storage.len(), domain_size);
    let quotient_c0_ptr = quotient.c0.storage.as_ref().as_ptr();
    unsafe {
        assert_eq!(
            quotient_c0_ptr.add(domain_size),
            quotient.c1.storage.as_ref().as_ptr()
        );
    }
    let quotient_ptr = quotient_c0_ptr as *mut VEF;
    let mut quotient_slice: &mut [VEF] =
        unsafe { slice::from_raw_parts_mut(quotient_ptr, domain_size) };
    let quotient_vector = unsafe { DeviceSlice::from_mut_slice(&mut quotient_slice) };

    boojum_cuda::ops_complex::lookup_quotient_a_and_b(
        &variable_cols_matrix,
        &table_cols_matrix,
        &subargs_a_matrix,
        &subargs_b_matrix,
        beta_c0,
        beta_c1,
        powers_of_gamma_vector,
        powers_of_alpha_vector,
        table_id_col_vector,
        &multiplicity_cols_matrix,
        quotient_vector,
        num_cols_per_subarg,
        get_stream(),
    )
}

pub fn deep_quotient_except_public_inputs<'a, 'b>(
    variable_cols: &[Poly<'a, CosetEvaluations>],
    maybe_witness_cols: &Option<&Vec<Poly<'a, CosetEvaluations>>>,
    constant_cols: &[Poly<'a, CosetEvaluations>],
    permutation_cols: &[Poly<'a, CosetEvaluations>],
    z_poly: &ComplexPoly<'a, CosetEvaluations>,
    partial_products: &[ComplexPoly<'a, CosetEvaluations>],
    maybe_multiplicity_cols: &Option<&Vec<Poly<'a, CosetEvaluations>>>,
    maybe_lookup_a_polys: &Option<&[ComplexPoly<'a, CosetEvaluations>]>,
    maybe_lookup_b_polys: &Option<&[ComplexPoly<'a, CosetEvaluations>]>,
    maybe_table_cols: &Option<&Vec<Poly<'a, CosetEvaluations>>>,
    quotient_constraint_polys: &[ComplexPoly<'a, CosetEvaluations>],
    evaluations_at_z: &[EF],
    evaluations_at_z_omega: &[EF],
    evaluations_at_zero: &Option<DVec<EF>>,
    challenges: &[EF],
    denom_at_z: &ComplexPoly<'a, CosetEvaluations>,
    denom_at_z_omega: &ComplexPoly<'a, CosetEvaluations>,
    maybe_denom_at_zero: &Option<Poly<'a, CosetEvaluations>>,
    quotient: &mut ComplexPoly<'b, CosetEvaluations>,
) -> CudaResult<()> {
    use std::slice;
    type VEF = VectorizedExtensionField;

    // Handling empty partial products would require special-case logic.
    // For now we don't need it. Assert as a reminder.
    assert!(partial_products.len() > 0);

    let domain_size = z_poly.c0.storage.len();

    assert_eq!(
        maybe_multiplicity_cols.is_none(),
        maybe_lookup_a_polys.is_none()
    );
    assert_eq!(
        maybe_multiplicity_cols.is_none(),
        maybe_lookup_b_polys.is_none()
    );
    assert_eq!(
        maybe_multiplicity_cols.is_none(),
        maybe_table_cols.is_none()
    );
    assert_eq!(
        maybe_multiplicity_cols.is_none(),
        evaluations_at_zero.is_none()
    );
    assert_eq!(
        maybe_multiplicity_cols.is_none(),
        maybe_denom_at_zero.is_none()
    );

    let (multiplicity_cols_len, multiplicity_cols_ptr) =
        if let Some(multiplicity_cols) = maybe_multiplicity_cols {
            (
                multiplicity_cols.len(),
                multiplicity_cols[0].storage.as_ref().as_ptr() as *const F,
            )
        } else {
            (0 as usize, std::ptr::null::<F>())
        };
    let (lookup_a_polys_len, lookup_a_polys_ptr) =
        if let Some(lookup_a_polys) = maybe_lookup_a_polys {
            (
                lookup_a_polys.len(),
                lookup_a_polys[0].c0.storage.as_ref().as_ptr() as *const VEF,
            )
        } else {
            (0 as usize, std::ptr::null::<VEF>())
        };
    let (lookup_b_polys_len, lookup_b_polys_ptr) =
        if let Some(lookup_b_polys) = maybe_lookup_b_polys {
            (
                lookup_b_polys.len(),
                lookup_b_polys[0].c0.storage.as_ref().as_ptr() as *const VEF,
            )
        } else {
            (0 as usize, std::ptr::null::<VEF>())
        };
    let (table_cols_len, table_cols_ptr) = if let Some(table_cols) = maybe_table_cols {
        (
            table_cols.len(),
            table_cols[0].storage.as_ref().as_ptr() as *const F,
        )
    } else {
        (0 as usize, std::ptr::null::<F>())
    };
    let (witness_cols_len, witness_cols_ptr) = if let Some(witness_cols) = maybe_witness_cols {
        (
            witness_cols.len(),
            witness_cols[0].storage.as_ref().as_ptr() as *const F,
        )
    } else {
        (0 as usize, std::ptr::null::<F>())
    };
    let (evaluations_at_zero_len, evaluations_at_zero_ptr) = if evaluations_at_zero.is_some() {
        let evaluations_at_zero_ref = evaluations_at_zero.as_ref().expect("must exist");
        (
            evaluations_at_zero_ref.len(),
            evaluations_at_zero_ref.as_ptr(),
        )
    } else {
        (0 as usize, std::ptr::null::<EF>())
    };
    let (denom_at_zero_len, denom_at_zero_ptr) = if maybe_denom_at_zero.is_some() {
        let denom_at_zero_ref = maybe_denom_at_zero.as_ref().expect("must exist");
        let len = denom_at_zero_ref.storage.len();
        assert_eq!(len, domain_size);
        (len, denom_at_zero_ref.storage.as_ref().as_ptr())
    } else {
        (0 as usize, std::ptr::null::<F>())
    };

    let mut num_terms_at_z = 0;
    num_terms_at_z += variable_cols.len();
    num_terms_at_z += witness_cols_len;
    num_terms_at_z += constant_cols.len();
    num_terms_at_z += permutation_cols.len();
    num_terms_at_z += 1; // z_poly
    num_terms_at_z += partial_products.len();
    num_terms_at_z += multiplicity_cols_len;
    num_terms_at_z += lookup_a_polys_len;
    num_terms_at_z += lookup_b_polys_len;
    num_terms_at_z += table_cols_len;
    num_terms_at_z += quotient_constraint_polys.len();
    assert_eq!(evaluations_at_z.len(), num_terms_at_z);
    assert_eq!(evaluations_at_z_omega.len(), 1);
    assert_eq!(
        evaluations_at_zero_len,
        lookup_a_polys_len + lookup_b_polys_len
    );

    let mut num_terms_from_evals = 0;
    num_terms_from_evals += evaluations_at_z.len();
    num_terms_from_evals += evaluations_at_z_omega.len();
    num_terms_from_evals += evaluations_at_zero_len;
    assert_eq!(challenges.len(), num_terms_from_evals);

    let variable_cols_ptr = variable_cols[0].storage.as_ref().as_ptr() as *const F;
    let variable_cols_slice =
        unsafe { slice::from_raw_parts(variable_cols_ptr, variable_cols.len() * domain_size) };
    let variable_cols_device_slice = unsafe { DeviceSlice::from_slice(variable_cols_slice) };
    let variable_cols_matrix = DeviceMatrix::new(variable_cols_device_slice, domain_size);

    let witness_cols_slice =
        unsafe { slice::from_raw_parts(witness_cols_ptr, witness_cols_len * domain_size) };
    let witness_cols_device_slice = unsafe { DeviceSlice::from_slice(witness_cols_slice) };
    let witness_cols_matrix = DeviceMatrix::new(witness_cols_device_slice, domain_size);

    let constant_cols_ptr = constant_cols[0].storage.as_ref().as_ptr() as *const F;
    let constant_cols_slice =
        unsafe { slice::from_raw_parts(constant_cols_ptr, constant_cols.len() * domain_size) };
    let constant_cols_device_slice = unsafe { DeviceSlice::from_slice(constant_cols_slice) };
    let constant_cols_matrix = DeviceMatrix::new(constant_cols_device_slice, domain_size);

    let permutation_cols_ptr = permutation_cols[0].storage.as_ref().as_ptr() as *const F;
    let permutation_cols_slice = unsafe {
        slice::from_raw_parts(permutation_cols_ptr, permutation_cols.len() * domain_size)
    };
    let permutation_cols_device_slice = unsafe { DeviceSlice::from_slice(permutation_cols_slice) };
    let permutation_cols_matrix = DeviceMatrix::new(permutation_cols_device_slice, domain_size);

    let multiplicity_cols_slice = unsafe {
        slice::from_raw_parts(multiplicity_cols_ptr, multiplicity_cols_len * domain_size)
    };
    let multiplicity_cols_device_slice =
        unsafe { DeviceSlice::from_slice(multiplicity_cols_slice) };
    let multiplicity_cols_matrix = DeviceMatrix::new(multiplicity_cols_device_slice, domain_size);

    let lookup_a_polys_slice =
        unsafe { slice::from_raw_parts(lookup_a_polys_ptr, lookup_a_polys_len * domain_size) };
    let lookup_a_polys_device_slice = unsafe { DeviceSlice::from_slice(lookup_a_polys_slice) };
    let lookup_a_polys_matrix = DeviceMatrix::new(lookup_a_polys_device_slice, domain_size);

    let lookup_b_polys_slice =
        unsafe { slice::from_raw_parts(lookup_b_polys_ptr, lookup_b_polys_len * domain_size) };
    let lookup_b_polys_device_slice = unsafe { DeviceSlice::from_slice(lookup_b_polys_slice) };
    let lookup_b_polys_matrix = DeviceMatrix::new(lookup_b_polys_device_slice, domain_size);

    let table_cols_slice =
        unsafe { slice::from_raw_parts(table_cols_ptr, table_cols_len * domain_size) };
    let table_cols_device_slice = unsafe { DeviceSlice::from_slice(table_cols_slice) };
    let table_cols_matrix = DeviceMatrix::new(table_cols_device_slice, domain_size);

    let z_poly_c0_ptr = z_poly.c0.storage.as_ref().as_ptr();
    unsafe {
        assert_eq!(
            z_poly_c0_ptr.add(domain_size),
            z_poly.c1.storage.as_ref().as_ptr()
        );
    }
    let z_poly_ptr = z_poly_c0_ptr as *const VEF;
    let z_poly_slice: &[VEF] = unsafe { slice::from_raw_parts(z_poly_ptr, domain_size) };
    let z_poly_vector = unsafe { DeviceSlice::from_slice(&z_poly_slice) };

    let partial_products_ptr = partial_products[0].c0.storage.as_ref().as_ptr() as *const VEF;
    let partial_products_slice = unsafe {
        slice::from_raw_parts(partial_products_ptr, partial_products.len() * domain_size)
    };
    let partial_products_device_slice = unsafe { DeviceSlice::from_slice(partial_products_slice) };
    let partial_products_matrix = DeviceMatrix::new(partial_products_device_slice, domain_size);

    let quotient_constraint_polys_ptr =
        quotient_constraint_polys[0].c0.storage.as_ref().as_ptr() as *const VEF;
    let quotient_constraint_polys_slice = unsafe {
        slice::from_raw_parts(
            quotient_constraint_polys_ptr,
            quotient_constraint_polys.len() * domain_size,
        )
    };
    let quotient_constraint_polys_device_slice =
        unsafe { DeviceSlice::from_slice(quotient_constraint_polys_slice) };
    let quotient_constraint_polys_matrix =
        DeviceMatrix::new(quotient_constraint_polys_device_slice, domain_size);

    let evaluations_at_z_vector = unsafe { DeviceSlice::from_slice(&evaluations_at_z) };
    let evaluations_at_z_omega_vector = unsafe { DeviceSlice::from_slice(&evaluations_at_z_omega) };
    let evaluations_at_zero_slice =
        unsafe { slice::from_raw_parts(evaluations_at_zero_ptr, evaluations_at_zero_len) };
    let evaluations_at_zero_vector = unsafe { DeviceSlice::from_slice(evaluations_at_zero_slice) };
    let challenges_vector = unsafe { DeviceSlice::from_slice(&challenges) };

    let denom_at_z_c0_ptr = denom_at_z.c0.storage.as_ref().as_ptr();
    unsafe {
        assert_eq!(
            denom_at_z_c0_ptr.add(domain_size),
            denom_at_z.c1.storage.as_ref().as_ptr()
        );
    }
    let denom_at_z_ptr = denom_at_z_c0_ptr as *const VEF;
    let denom_at_z_slice: &[VEF] = unsafe { slice::from_raw_parts(denom_at_z_ptr, domain_size) };
    let denom_at_z_vector = unsafe { DeviceSlice::from_slice(&denom_at_z_slice) };

    let denom_at_z_omega_c0_ptr = denom_at_z_omega.c0.storage.as_ref().as_ptr();
    unsafe {
        assert_eq!(
            denom_at_z_omega_c0_ptr.add(domain_size),
            denom_at_z_omega.c1.storage.as_ref().as_ptr()
        );
    }
    let denom_at_z_omega_ptr = denom_at_z_omega_c0_ptr as *const VEF;
    let denom_at_z_omega_slice: &[VEF] =
        unsafe { slice::from_raw_parts(denom_at_z_omega_ptr, domain_size) };
    let denom_at_z_omega_vector = unsafe { DeviceSlice::from_slice(&denom_at_z_omega_slice) };

    let denom_at_zero_slice =
        unsafe { slice::from_raw_parts(denom_at_zero_ptr, denom_at_zero_len) };
    let denom_at_zero_vector = unsafe { DeviceSlice::from_slice(&denom_at_zero_slice) };

    let quotient_c0_ptr = quotient.c0.storage.as_ref().as_ptr();
    unsafe {
        assert_eq!(
            quotient_c0_ptr.add(domain_size),
            quotient.c1.storage.as_ref().as_ptr()
        );
    }
    let quotient_ptr = quotient_c0_ptr as *mut VEF;
    let mut quotient_slice: &mut [VEF] =
        unsafe { slice::from_raw_parts_mut(quotient_ptr, domain_size) };
    let quotient_vector = unsafe { DeviceSlice::from_mut_slice(&mut quotient_slice) };

    boojum_cuda::ops_complex::deep_quotient_except_public_inputs(
        &variable_cols_matrix,
        &witness_cols_matrix,
        &constant_cols_matrix,
        &permutation_cols_matrix,
        z_poly_vector,
        &partial_products_matrix,
        &multiplicity_cols_matrix,
        &lookup_a_polys_matrix,
        &lookup_b_polys_matrix,
        &table_cols_matrix,
        &quotient_constraint_polys_matrix,
        evaluations_at_z_vector,
        evaluations_at_z_omega_vector,
        evaluations_at_zero_vector,
        challenges_vector,
        denom_at_z_vector,
        denom_at_z_omega_vector,
        denom_at_zero_vector,
        quotient_vector,
        get_stream(),
    )
}

pub fn deep_quotient_public_input<'a, 'b>(
    values: &Poly<'a, CosetEvaluations>,
    expected_value: F,
    challenge: &[EF],
    quotient: &mut ComplexPoly<'b, CosetEvaluations>,
) -> CudaResult<()> {
    use std::slice;
    type VEF = VectorizedExtensionField;

    assert_eq!(challenge.len(), 1);

    let domain_size = values.storage.len();

    let values_ptr = values.storage.as_ref().as_ptr() as *const F;
    let values_slice = unsafe { slice::from_raw_parts(values_ptr, domain_size) };
    let values_vector = unsafe { DeviceSlice::from_slice(&values_slice) };

    let challenge_vector = unsafe { DeviceSlice::from_slice(&challenge) };

    assert_eq!(quotient.c0.storage.len(), domain_size);
    let quotient_c0_ptr = quotient.c0.storage.as_ref().as_ptr();
    unsafe {
        assert_eq!(
            quotient_c0_ptr.add(domain_size),
            quotient.c1.storage.as_ref().as_ptr()
        );
    }
    let quotient_ptr = quotient_c0_ptr as *mut VEF;
    let mut quotient_slice: &mut [VEF] =
        unsafe { slice::from_raw_parts_mut(quotient_ptr, domain_size) };
    let quotient_vector = unsafe { DeviceSlice::from_mut_slice(&mut quotient_slice) };

    boojum_cuda::ops_complex::deep_quotient_public_input(
        values_vector,
        expected_value,
        challenge_vector,
        quotient_vector,
        get_stream(),
    )
}
