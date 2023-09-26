use boojum::cs::{implementations::setup::TreeNode, LookupParameters};

use super::*;

pub fn compute_quotient_by_coset<'a, 'b>(
    trace_storage: &'a GenericTraceStorage<CosetEvaluations>,
    setup_storage: &'a GenericSetupStorage<CosetEvaluations>,
    argument_storage: &'a GenericArgumentStorage<'a, CosetEvaluations>,
    lookup_params: LookupParameters,
    table_ids_column_idxes: &[usize],
    selector_placement: &TreeNode,
    specialized_gates: &[cs_helpers::GateEvaluationParams],
    general_purpose_gates: &[cs_helpers::GateEvaluationParams],
    coset_idx: usize,
    domain_size: usize,
    used_lde_degree: usize,
    num_cols_per_product: usize,
    copy_permutation_challenge_z_at_one_equals_one: &EF,
    copy_permutation_challenges_partial_product_terms: &DVec<EF>,
    alpha: &EF,
    lookup_challenges: &Option<DVec<EF>>,
    specialized_cols_challenge_power_offset: usize,
    general_purpose_cols_challenge_power_offset: usize,
    beta: &DExt,
    gamma: &DExt,
    powers_of_gamma_for_lookup: &Option<DVec<EF>>,
    lookup_beta: Option<&DExt>,
    non_residues_by_beta: &DVec<EF>,
    variables_offset: usize,
    quotient: &mut ComplexPoly<'b, CosetEvaluations>,
) -> CudaResult<()>
where
    'a: 'b,
{
    let trace_polys = trace_storage.as_polynomials();
    let setup_polys = setup_storage.as_polynomials();

    if specialized_gates.len() > 0 {
        generic_evaluate_constraints_by_coset(
            &trace_polys,
            &setup_polys,
            specialized_gates,
            selector_placement.clone(),
            domain_size,
            alpha.clone(),
            specialized_cols_challenge_power_offset,
            quotient,
        )?;
    }

    assert!(general_purpose_gates.len() > 0);
    if general_purpose_gates.len() > 1 {
        generic_evaluate_constraints_by_coset(
            &trace_polys,
            &setup_polys,
            general_purpose_gates,
            selector_placement.clone(),
            domain_size,
            alpha.clone(),
            general_purpose_cols_challenge_power_offset,
            quotient,
        )?;
    }

    let l0 = compute_l0_over_coset(coset_idx, domain_size, used_lde_degree)?;
    assert_eq!(l0.storage.len(), domain_size);
    let l0 = ComplexPoly::<CosetEvaluations>::from_real(l0)?;

    let argument_polys = argument_storage.as_polynomials();

    let mut tmp = ComplexPoly::<CosetEvaluations>::zero(domain_size)?;
    tmp.add_assign(&argument_polys.z_poly)?;
    tmp.sub_constant(&DExt::one()?)?;
    tmp.mul_assign(&l0)?;
    tmp.scale(&copy_permutation_challenge_z_at_one_equals_one.into())?;
    quotient.add_assign(&tmp)?;

    assert_eq!(
        copy_permutation_challenges_partial_product_terms.len(),
        argument_polys.partial_products.len() + 1
    );

    let coset_omegas = compute_omega_values_for_coset(coset_idx, domain_size, used_lde_degree)?;

    // compute_quotient_for_partial_products_naive(
    //     &trace_polys,
    //     &setup_polys,
    //     &argument_polys,
    //     &coset_omegas,
    //     num_cols_per_product,
    //     beta.clone(),
    //     gamma.clone(),
    //     &non_residues_by_beta,
    //     powers_of_alpha_for_copy_permutation.clone(),
    //     quotient,
    // )?;
    compute_quotient_for_partial_products(
        &trace_polys,
        &setup_polys,
        &argument_polys,
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
            &trace_polys,
            &setup_polys,
            &argument_polys,
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
