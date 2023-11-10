use boojum::cs::LookupParameters;

use crate::primitives::arith::{
    lookup_aggregated_table_values, lookup_quotient_ensure_a_and_b_are_well_formed, lookup_subargs,
};

use super::*;

pub fn compute_lookup_argument_over_specialized_cols(
    trace: &TracePolynomials<LagrangeBasis>,
    setup: &SetupPolynomials<LagrangeBasis>,
    table_id_column_idxes: Vec<usize>,
    beta: &DExt,
    powers_of_gamma: &DVec<EF>,
    variables_offset: usize,
    lookup_params: LookupParameters,
    storage: &mut GenericArgumentStorage<LagrangeBasis>,
) -> CudaResult<()> {
    assert!(lookup_params.is_specialized_lookup());
    let TracePolynomials {
        variable_cols,
        witness_cols: _,
        multiplicity_cols,
    } = trace;

    let SetupPolynomials {
        constant_cols,
        table_cols,
        ..
    } = setup;

    let ArgumentPolynomialsMut {
        lookup_a_polys: subargs_a,
        lookup_b_polys: subargs_b,
        ..
    } = storage.as_polynomials_mut();

    assert!(variable_cols.len() > 0);

    // added up multiplicities
    assert_eq!(multiplicity_cols.len(), 1);
    let domain_size = variable_cols[0].domain_size();

    let (
        use_constant_for_table_id,
        _share_table_id,
        _width,
        num_variable_columns_per_subargument,
        num_lookup_columns,
        num_subargs,
    ) = match lookup_params {
        LookupParameters::UseSpecializedColumnsWithTableIdAsVariable {
            width,
            num_repetitions,
            share_table_id,
        } => {
            assert!(share_table_id == false);
            (
                false,
                false,
                width as usize,
                width as usize + 1,
                width as usize + 1,
                num_repetitions,
            )
        }
        LookupParameters::UseSpecializedColumnsWithTableIdAsConstant {
            width,
            num_repetitions,
            share_table_id,
        } => {
            assert!(share_table_id);

            (
                true,
                true,
                width as usize,
                width as usize,
                width as usize + 1,
                num_repetitions,
            )
        }
        _ => unreachable!(),
    };
    assert_eq!(subargs_a.len(), num_subargs);
    assert_eq!(subargs_b.len(), 1);
    assert_eq!(powers_of_gamma.len(), num_lookup_columns);

    let variable_cols_for_lookup = &variable_cols
        [variables_offset..(variables_offset + num_variable_columns_per_subargument * num_subargs)];

    // aggregate table values
    assert_eq!(table_cols.len(), powers_of_gamma.len());
    let mut aggregated_table_values = ComplexPoly::<LagrangeBasis>::zero(domain_size)?;
    lookup_aggregated_table_values(
        &table_cols,
        &beta,
        &powers_of_gamma,
        &mut aggregated_table_values,
        num_variable_columns_per_subargument,
        domain_size,
    )?;

    aggregated_table_values.inverse()?;
    let aggregated_table_values_inv = aggregated_table_values;

    assert!(table_id_column_idxes.len() == 1);
    assert!(use_constant_for_table_id);
    let table_id_col_idx = table_id_column_idxes.get(0).copied().expect("should exist");
    let table_id_col = &constant_cols[table_id_col_idx];

    lookup_subargs(
        &variable_cols_for_lookup,
        subargs_a,
        subargs_b,
        &beta,
        &powers_of_gamma,
        &table_id_col,
        &aggregated_table_values_inv,
        &multiplicity_cols,
        num_variable_columns_per_subargument,
        variable_cols_for_lookup.len(),
        domain_size,
    )?;

    // TODO: is it possible to get a single handle for the whole chunk?
    for a_poly in subargs_a.iter_mut() {
        a_poly.inverse()?;
    }

    debug_assert!(
        {
            let mut a_c0 = F::ZERO;
            let mut a_c1 = F::ZERO;

            for a in subargs_a.iter() {
                let d_sum = a.grand_sum()?;
                let h_sum: EF = d_sum.into();
                let [c0, c1] = h_sum.into_coeffs_in_base();
                a_c0.add_assign(&c0);
                a_c1.add_assign(&c1);
            }

            let mut b_c0 = F::ZERO;
            let mut b_c1 = F::ZERO;

            for b in subargs_b.iter() {
                let d_tmp = b.grand_sum()?;
                let tmp: EF = d_tmp.into();
                let [c0, c1] = tmp.into_coeffs_in_base();
                b_c0.add_assign(&c0);
                b_c1.add_assign(&c1);
            }
            a_c0 == b_c0 && a_c1 == b_c1
        },
        "lookup argument fails: a(0) != b(0)"
    );

    Ok(())
}

#[allow(dead_code)]
pub fn compute_lookup_argument_over_general_purpose_cols(
    _trace: &TracePolynomials<LagrangeBasis>,
    _setup: &SetupPolynomials<LagrangeBasis>,
    _table_id_column_idxes: Vec<usize>,
    _beta: &DExt,
    _powers_of_gamma: Vec<DExt>,
    _variables_offset: usize,
    _lookup_params: LookupParameters,
    _lde_degree: usize,
    _storage: &mut GenericArgumentStorage<LagrangeBasis>,
) -> CudaResult<()> {
    // let BaseTrace {
    //     variable_cols,
    //     witness_cols,
    //     multiplicity_cols: multiplicities,
    // } = trace;

    // let BaseSetup {
    //     permutation_cols,
    //     constant_cols,
    //     table_cols,
    //     h_table_ids,
    // } = setup;
    // assert!(variable_cols.len() > 0);
    // let num_polys = variable_cols.len();
    // let domain_size = variable_cols[0].domain_size();
    // let num_subargs = num_polys / column_elements_per_subargument;
    // assert_eq!(num_polys % column_elements_per_subargument, 0);
    // assert_eq!(multiplicities.len(), 1);

    // assert!(table_id_column_idxes.len() == 0 || table_id_column_idxes.len() == 1);
    // let num_lookup_columns =
    //     column_elements_per_subargument + ((table_id_column_idxes.len() == 1) as usize);
    // assert_eq!(table_cols.len(), num_lookup_columns);

    // let one = DF::one()?;
    // let mut ones = dvec!(domain_size);
    // helpers::set_value(&mut ones, &one)?;
    // let ones = ComplexPoly::<LagrangeBasis>::from_real(Poly::from(ones))?;
    // let mut unified_selector = ComplexPoly::<LagrangeBasis>::one(domain_size)?;

    // // compute unified selector from layer columns
    // for (path, selector_col) in lookup_selector_path.iter().zip(constant_cols.iter()) {
    //     let mut current_col = ComplexPoly::<LagrangeBasis>::from_real(selector_col.clone())?;
    //     if *path {
    //         current_col.negate()?;
    //         current_col.add_assign(&ones)?;
    //     }
    //     unified_selector.mul_assign(&current_col)?;
    // }

    // let mut subargs_a = Vec::with_capacity(num_subargs);
    // let mut subargs_b = Vec::with_capacity(num_subargs);

    // // aggregate table values
    // let mut aggregated_table_values = ComplexPoly::<LagrangeBasis>::zero(domain_size)?;
    // for (table_col, current_gamma) in table_cols.iter().zip(powers_of_gamma.iter()) {
    //     let mut tmp = ComplexPoly::<LagrangeBasis>::from_real(table_col.clone())?;
    //     tmp.scale(current_gamma)?;
    //     aggregated_table_values.add_assign(&tmp)?;
    // }
    // aggregated_table_values.add_constant(&beta)?;
    // aggregated_table_values.inverse()?;
    // let aggregated_table_values_inv = aggregated_table_values;

    // for (variable_cols_per_subarg, multiplicity_col) in variable_cols
    //     .chunks(column_elements_per_subargument)
    //     .zip(multiplicities.iter())
    // {
    //     // aggregate variables
    //     let mut a_poly = ComplexPoly::<LagrangeBasis>::zero(domain_size)?;

    //     let mut gamma_iter = powers_of_gamma.iter();
    //     for variable_col in variable_cols_per_subarg.iter() {
    //         let current_gamma = gamma_iter.next().unwrap();
    //         let mut tmp = ComplexPoly::<LagrangeBasis>::from_real(variable_col.clone())?;
    //         tmp.scale(current_gamma)?;
    //         a_poly.add_assign(&tmp)?;
    //     }

    //     if let Some(table_id_col) = table_id_column_idxes.get(0).copied() {
    //         let current_gamma = gamma_iter.next().unwrap();
    //         let mut tmp =
    //             ComplexPoly::<LagrangeBasis>::from_real(constant_cols[table_id_col].clone())?;
    //         tmp.scale(current_gamma)?;
    //         a_poly.add_assign(&tmp)?;
    //     }
    //     a_poly.add_constant(&beta)?;
    //     a_poly.inverse()?;
    //     a_poly.mul_assign(&unified_selector)?;

    //     let mut b_poly = ComplexPoly::<LagrangeBasis>::from_real(multiplicity_col.clone())?;
    //     b_poly.mul_assign(&aggregated_table_values_inv)?;

    //     subargs_a.push(a_poly);
    //     subargs_b.push(b_poly);
    // }
    // Ok((subargs_a, subargs_b))
    unimplemented!()
}

pub fn compute_quotient_for_lookup_over_specialized_cols<'a, 'b>(
    trace: &TracePolynomials<'a, CosetEvaluations>,
    setup: &SetupPolynomials<'a, CosetEvaluations>,
    argument: &ArgumentPolynomials<'a, CosetEvaluations>,
    lookup_params: LookupParameters,
    beta: &DExt,
    powers_of_gamma: &Option<DVec<EF>>,
    powers_of_alpha: &Option<DVec<EF>>,
    variables_offset: usize,
    num_column_elements_per_subargument: usize,
    table_ids_column_idxes: &[usize],
    quotient: &mut ComplexPoly<'b, CosetEvaluations>,
) -> CudaResult<()>
where
    'a: 'b,
{
    let TracePolynomials {
        variable_cols,
        witness_cols: _,
        multiplicity_cols,
    } = trace;

    let SetupPolynomials {
        constant_cols,
        table_cols,
        ..
    } = setup;

    let ArgumentPolynomials {
        lookup_a_polys,
        lookup_b_polys,
        ..
    } = argument;

    let num_subarguments = match lookup_params {
        LookupParameters::UseSpecializedColumnsWithTableIdAsVariable {
            width: _,
            num_repetitions,
            share_table_id,
        } => {
            assert!(share_table_id == false);
            num_repetitions
        }
        LookupParameters::UseSpecializedColumnsWithTableIdAsConstant {
            width: _,
            num_repetitions,
            share_table_id,
        } => {
            assert!(share_table_id);

            num_repetitions
        }
        _ => unreachable!(),
    };

    let powers_of_alpha_ref = powers_of_alpha.as_ref().expect("must exist");

    assert_eq!(multiplicity_cols.len(), 1);
    assert_eq!(lookup_b_polys.len(), multiplicity_cols.len());
    assert_eq!(lookup_a_polys.len(), num_subarguments);
    assert_eq!(
        powers_of_alpha_ref.len(),
        num_subarguments + multiplicity_cols.len()
    );

    // a(x) * (f(x) + beta) = q0(x)*z(x) // no need selector in specialized mode
    // b(x) * (t(x) + beta) - m(x) = q1(x)*z(x)
    // q(x) = alpha * q0(x) + alpha^2*q1(x)

    assert_eq!(table_cols.len(), num_column_elements_per_subargument + 1);

    let domain_size = variable_cols[0].domain_size();

    let variable_cols_for_specialized = &variable_cols[variables_offset
        ..(variables_offset + num_column_elements_per_subargument * num_subarguments)];

    assert!(table_ids_column_idxes.len() == 1);
    let table_id_col_idx = table_ids_column_idxes
        .get(0)
        .copied()
        .expect("should exist");
    let table_id_col = &constant_cols[table_id_col_idx];

    let powers_of_gamma_ref = powers_of_gamma.as_ref().expect("must exist");
    lookup_quotient_ensure_a_and_b_are_well_formed(
        &variable_cols_for_specialized,
        &table_cols,
        &lookup_a_polys,
        &lookup_b_polys,
        &beta,
        powers_of_gamma_ref,
        powers_of_alpha_ref,
        &table_id_col,
        &multiplicity_cols,
        quotient,
        num_column_elements_per_subargument,
        variable_cols_for_specialized.len(),
        domain_size,
    )?;

    Ok(())
}
