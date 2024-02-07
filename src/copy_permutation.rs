use crate::primitives::arith::{partial_products_num_denom_chunk, partial_products_quotient_terms};

use super::*;

pub fn compute_partial_products(
    raw_trace: &TracePolynomials<LagrangeBasis>,
    raw_setup: &SetupPolynomials<LagrangeBasis>,
    non_residues_by_beta: &SVec<EF>,
    beta: &DExt,
    gamma: &DExt,
    num_cols_per_product: usize,
    storage: &mut GenericArgumentsStorage<LagrangeBasis>,
) -> CudaResult<()> {
    let TracePolynomials { variable_cols, .. } = raw_trace;

    let ArgumentsPolynomials {
        mut z_polys,
        mut partial_products,
        ..
    } = storage.as_polynomials_mut();

    let z_poly = &mut z_polys[0];

    let SetupPolynomials {
        permutation_cols, ..
    } = raw_setup;

    assert_eq!(variable_cols.len(), permutation_cols.len());
    assert_eq!(non_residues_by_beta.len(), variable_cols.len());

    let domain_size = variable_cols[0].domain_size();
    let num_partial_products = variable_cols.chunks(num_cols_per_product).len() - 1;
    assert_eq!(partial_products.len(), num_partial_products);

    let mut omega_values = dvec!(domain_size);

    helpers::compute_domain_elems(&mut omega_values, domain_size)?;

    helpers::set_value(z_poly.c0.storage.as_mut(), &DF::one()?)?;
    helpers::set_zero(z_poly.c1.storage.as_mut())?;

    let mut num: ComplexPoly<LagrangeBasis> = ComplexPoly::empty(domain_size)?;
    let mut denum: ComplexPoly<LagrangeBasis> = ComplexPoly::empty(domain_size)?;

    // The "max opt" pattern here would look like some fully fused kernel followed by
    // a single batch inverse followed by another fused kernel.
    // The pattern written here is simpler and recovers most of the performance
    // without additional temporaries.
    let mut partial_products_iter = partial_products.iter_mut();
    for ((non_residues_chunk, variable_cols_chunk), sigma_cols_chunk) in non_residues_by_beta
        .chunks(num_cols_per_product)
        .zip(variable_cols.chunks(num_cols_per_product))
        .zip(permutation_cols.chunks(num_cols_per_product))
    {
        let num = if let Some(partial_product) = partial_products_iter.next() {
            partial_product
        } else {
            &mut num
        };

        partial_products_num_denom_chunk(
            num,
            &mut denum,
            variable_cols_chunk,
            sigma_cols_chunk,
            &omega_values,
            &non_residues_chunk,
            &beta,
            &gamma,
            num_cols_per_product,
            domain_size,
        )?;
        denum.inverse()?;
        num.mul_assign(&denum)?;

        z_poly.mul_assign(&num)?;
    }
    assert!(partial_products_iter.next().is_none());

    z_poly.shifted_grand_product()?;

    let mut prev_product = z_poly.clone();
    // we already ignored last product
    for partial_product in partial_products.iter_mut() {
        partial_product.mul_assign(&prev_product)?;
        prev_product = partial_product.clone();
    }

    Ok(())
}

// For debugging purposes
#[allow(dead_code)]
pub fn compute_quotient_for_partial_products_naive<'a, 'b>(
    trace: &TracePolynomials<'a, CosetEvaluations>,
    setup: &SetupPolynomials<'a, CosetEvaluations>,
    argument: &ArgumentsPolynomials<'a, CosetEvaluations>,
    omega_values: &Poly<'a, CosetEvaluations>,
    num_cols_per_product: usize,
    beta: DExt,
    gamma: DExt,
    non_residues_by_beta: &DVec<EF>,
    powers_of_alpha: Vec<DExt>,
    quotient: &mut ComplexPoly<'b, CosetEvaluations>,
) -> CudaResult<()>
where
    'a: 'b,
{
    let TracePolynomials { variable_cols, .. } = trace;

    let SetupPolynomials {
        permutation_cols, ..
    } = setup;

    let ArgumentsPolynomials {
        z_polys,
        partial_products,
        ..
    } = argument;

    let z_poly = &z_polys[0];

    let mut non_residues_by_beta_transformed = vec![];
    for el in non_residues_by_beta.to_vec()?.into_iter() {
        let el: DExt = el.into();
        non_residues_by_beta_transformed.push(el);
    }

    // z(w*x) * f(x) - z(x) * g(x) = q(x) * v(x)
    // f(x) = w + non_res * beta * x + omega
    // g(x) = w + beta * sigma + omega

    // partial_product0*g0 - z*f0
    // partial_product1*g1 - partial_product0*f1
    // partial_product2*g2 - partial_product1*f2
    // ..
    // z_shifted*g_n - partial_product_(n-1)*f_n
    assert_eq!(powers_of_alpha.len(), partial_products.len() + 1);
    assert_eq!(non_residues_by_beta.len(), variable_cols.len(),);
    let domain_size = variable_cols[0].domain_size();
    assert_eq!(omega_values.domain_size(), domain_size);

    let mut shifted_z = ComplexPoly::clone(z_poly);
    shifted_z.rotate()?;

    let mut lhs = vec![];
    for p in partial_products.iter() {
        lhs.push(p);
    }
    lhs.push(&shifted_z);

    let mut rhs = vec![];
    rhs.push(z_poly);
    for p in partial_products.iter() {
        rhs.push(p);
    }
    assert_eq!(lhs.len(), rhs.len());
    let omega_values = ComplexPoly::<CosetEvaluations>::from_real(omega_values)?;

    for (
        ((((existing_lhs, existing_rhs), variable_cols_chunk), sigma_cols_chunk), alpha),
        non_residues_chunk,
    ) in lhs
        .iter()
        .zip(rhs.iter())
        .zip(variable_cols.chunks(num_cols_per_product))
        .zip(permutation_cols.chunks(num_cols_per_product))
        .zip(powers_of_alpha.iter())
        .zip(non_residues_by_beta_transformed.chunks(num_cols_per_product))
    {
        let mut num = ComplexPoly::<CosetEvaluations>::one(domain_size)?;
        let mut denum = num.clone();

        for ((non_residue_by_beta, variable_col), sigma_col) in non_residues_chunk
            .iter()
            .zip(variable_cols_chunk.iter())
            .zip(sigma_cols_chunk.iter())
        {
            // numerator w + beta * non_res * x + gamma
            let mut current_num = omega_values.clone();
            current_num.scale_real(&non_residue_by_beta)?;
            let variable_col = variable_col.clone();
            current_num.add_assign_real(&variable_col)?;
            current_num.add_constant(&gamma)?;

            // dnumerator w + beta * sigma(x) + gamma
            let mut current_denum = ComplexPoly::<CosetEvaluations>::from_real(sigma_col)?;
            current_denum.scale_real(&beta)?;
            current_denum.add_assign_real(&variable_col)?;
            current_denum.add_constant(&gamma)?;

            num.mul_assign(&current_num)?;
            denum.mul_assign(&current_denum)?;
        }
        // lhs * denum - rhs * num
        // division will happen on the final quotient
        denum.mul_assign(&existing_lhs)?;
        num.mul_assign(&existing_rhs)?;
        denum.sub_assign(&num)?;
        denum.scale(&alpha)?;
        quotient.add_assign(&denum)?;
    }

    Ok(())
}

pub fn compute_quotient_for_partial_products(
    variable_cols: &[Poly<CosetEvaluations>],
    permutation_cols: &[Poly<CosetEvaluations>],
    z_poly: &ComplexPoly<CosetEvaluations>,
    partial_products: &[ComplexPoly<CosetEvaluations>],
    omega_values: &Poly<CosetEvaluations>,
    num_cols_per_product: usize,
    beta: &DExt,
    gamma: &DExt,
    non_residues_by_beta: &SVec<EF>,
    powers_of_alpha: &SVec<EF>,
    quotient: &mut ComplexPoly<CosetEvaluations>,
) -> CudaResult<()> {
    // z(w*x) * f(x) - z(x) * g(x) = q(x) * v(x)
    // f(x) = w + non_res * beta * x + omega
    // g(x) = w + beta * sigma + omega

    // partial_product0*g0 - z*f0
    // partial_product1*g1 - partial_product0*f1
    // partial_product2*g2 - partial_product1*f2
    // ..
    // z_shifted*g_n - partial_product_(n-1)*f_n
    let num_polys = variable_cols.len();
    let domain_size = variable_cols[0].domain_size();
    assert_eq!(powers_of_alpha.len(), partial_products.len() + 1);
    assert_eq!(non_residues_by_beta.len(), num_polys);
    assert_eq!(omega_values.domain_size(), domain_size);

    partial_products_quotient_terms(
        partial_products,
        z_poly,
        variable_cols,
        permutation_cols,
        omega_values,
        powers_of_alpha,
        non_residues_by_beta,
        beta,
        gamma,
        quotient,
        num_cols_per_product,
        num_polys,
        domain_size,
    )?;

    Ok(())
}
