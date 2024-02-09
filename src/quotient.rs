use boojum::cs::{implementations::setup::TreeNode, LookupParameters};

use super::*;

// The incoming quotient is assumed to be empty (not zeroed).
pub fn compute_quotient_by_coset(
    trace_cache: &mut TraceCache,
    setup_cache: &mut SetupCache,
    arguments_cache: &mut ArgumentsCache,
    lookup_params: LookupParameters,
    table_ids_column_idxes: &[usize],
    selector_placement: &TreeNode,
    specialized_gates: &[GateEvaluationParams],
    general_purpose_gates: &[GateEvaluationParams],
    coset_idx: usize,
    domain_size: usize,
    used_lde_degree: usize,
    num_cols_per_product: usize,
    copy_permutation_challenge_z_at_one_equals_one: &EF,
    copy_permutation_challenges_partial_product_terms: &SVec<EF>,
    alpha: &EF,
    lookup_challenges: &Option<SVec<EF>>,
    specialized_cols_challenge_power_offset: usize,
    general_purpose_cols_challenge_power_offset: usize,
    beta: &DExt,
    gamma: &DExt,
    powers_of_gamma_for_lookup: &Option<SVec<EF>>,
    lookup_beta: Option<&DExt>,
    non_residues_by_beta: &SVec<EF>,
    variables_offset: usize,
    quotient: &mut ComplexPoly<CosetEvaluations>,
) -> CudaResult<()> {
    let trace_storage = trace_cache.get_coset_evaluations(coset_idx)?;
    let TracePolynomials {
        variable_cols,
        witness_cols,
        multiplicity_cols,
    } = trace_storage.as_polynomials();
    let setup_storage = setup_cache.get_coset_evaluations(coset_idx)?;
    let SetupPolynomials {
        permutation_cols,
        constant_cols,
        table_cols,
    } = setup_storage.as_polynomials();
    let arguments_storage = arguments_cache.get_coset_evaluations(coset_idx)?;
    let ArgumentsPolynomials {
        z_polys,
        partial_products,
        lookup_a_polys,
        lookup_b_polys,
    } = arguments_storage.as_polynomials();
    let z_poly = &z_polys[0];

    let l0 = compute_l0_over_coset(coset_idx, domain_size, used_lde_degree)?;
    assert_eq!(l0.storage.len(), domain_size);
    mem::d2d(z_poly.as_single_slice(), quotient.as_single_slice_mut())?;
    quotient.sub_constant(&DExt::one()?)?;
    quotient.mul_assign_real(&l0)?;
    quotient.scale(&copy_permutation_challenge_z_at_one_equals_one.into())?;

    if specialized_gates.len() > 0 {
        generic_evaluate_constraints_by_coset(
            &variable_cols,
            &witness_cols,
            &constant_cols,
            specialized_gates,
            selector_placement.clone(),
            domain_size,
            alpha.clone(),
            specialized_cols_challenge_power_offset,
            quotient,
            true,
        )?;
    }

    assert!(general_purpose_gates.len() > 0);
    if general_purpose_gates.len() > 1 {
        generic_evaluate_constraints_by_coset(
            &variable_cols,
            &witness_cols,
            &constant_cols,
            general_purpose_gates,
            selector_placement.clone(),
            domain_size,
            alpha.clone(),
            general_purpose_cols_challenge_power_offset,
            quotient,
            false,
        )?;
    }

    assert_eq!(
        copy_permutation_challenges_partial_product_terms.len(),
        arguments_cache.layout.num_partial_products / 2 + 1
    );

    let coset_omegas = compute_omega_values_for_coset(coset_idx, domain_size, used_lde_degree)?;

    compute_quotient_for_partial_products(
        &variable_cols,
        &permutation_cols,
        &z_poly,
        &partial_products,
        &coset_omegas,
        num_cols_per_product,
        &beta,
        &gamma,
        &non_residues_by_beta,
        &copy_permutation_challenges_partial_product_terms,
        quotient,
    )?;

    if lookup_params.lookup_is_allowed() {
        // lookup
        assert!(lookup_params.is_specialized_lookup());
        let columns_per_subargument = lookup_params.specialized_columns_per_subargument() as usize;

        compute_quotient_for_lookup_over_specialized_cols(
            &variable_cols,
            &multiplicity_cols,
            &constant_cols,
            &table_cols,
            &lookup_a_polys,
            &lookup_b_polys,
            lookup_params,
            lookup_beta.unwrap(),
            &powers_of_gamma_for_lookup,
            &lookup_challenges,
            variables_offset,
            columns_per_subargument,
            table_ids_column_idxes,
            quotient,
        )?;
    } else {
        assert!(powers_of_gamma_for_lookup.is_none());
    }

    divide_by_vanishing_poly_over_coset(&mut quotient.c0, coset_idx, domain_size, used_lde_degree)?;
    divide_by_vanishing_poly_over_coset(&mut quotient.c1, coset_idx, domain_size, used_lde_degree)?;

    Ok(())
}
