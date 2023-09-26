use std::alloc::Global;

#[allow(unused_imports)]
use boojum::{
    config::{CSConfig, CSSetupConfig, CSWitnessEvaluationConfig, ProvingCSConfig},
    cs::{
        gates::lookup_marker::LookupFormalGate,
        implementations::{
            copy_permutation::non_residues_for_copy_permutation,
            hints::{DenseVariablesCopyHint, DenseWitnessCopyHint},
            lookup_table,
            polynomial::{GenericPolynomial, LagrangeForm},
            polynomial_storage::{SetupBaseStorage, SetupStorage, TraceHolder, WitnessStorage},
            pow::{NoPow, PoWRunner},
            proof::{self, OracleQuery, Proof, SingleRoundQueries},
            prover::ProofConfig,
            reference_cs::{CSReferenceAssembly, CSReferenceImplementation},
            setup,
            transcript::Transcript,
            utils::{domain_generator_for_size, materialize_powers_serial},
            verifier::VerificationKey,
            witness::{WitnessSet, WitnessVec},
        },
        oracle::{merkle_tree::MerkleTreeWithCap, TreeHasher},
        toolboxes::{gate_config::GateConfigurationHolder, static_toolbox::StaticToolboxHolder},
        traits::{gate::GatePlacementStrategy, GoodAllocator},
        CSGeometry, LookupParameters,
    },
    field::{traits::field_like::TrivialContext, FieldExtension, U64Representable},
    worker::Worker,
};

use crate::arith::{deep_quotient_except_public_inputs, deep_quotient_public_input};
use crate::cs::GpuSetup;

use super::*;

pub fn gpu_prove_from_external_witness_data<
    P: boojum::field::traits::field_like::PrimeFieldLikeVectorized<Base = F>,
    TR: Transcript<F, CompatibleCap = [F; 4]>,
    H: TreeHasher<F, Output = TR::CompatibleCap>,
    POW: PoWRunner,
    A: GoodAllocator,
>(
    cs: CSReferenceAssembly<F, P, ProvingCSConfig>,
    external_witness_data: &WitnessVec<F>,
    proof_config: ProofConfig,
    setup: &GpuSetup<A>,
    vk: &VerificationKey<F, H>,
    transcript_params: TR::TransciptParameters,
    worker: &Worker,
) -> CudaResult<Proof<F, H, EXT>> {
    // TODO: this is a convenient function that is made for externally synthesized circuits
    // but local synthesis also use this fn. re-enable this check once deployments are done
    // assert_eq!(
    //     cs.next_available_place_idx(),
    //     0,
    //     "CS should be empty and hold no data"
    // );

    let num_variable_cols = setup.variables_hint.len();
    let num_multiplicity_cols = cs.num_multipicities_polys();
    dbg!(num_variable_cols);
    let trace_layout = TraceLayout {
        num_variable_cols,
        num_witness_cols: 0,
        num_multiplicity_cols,
    };
    dbg!(&trace_layout);
    let domain_size = cs.max_trace_len;
    let (raw_trace, monomial_trace) = construct_trace_storage_from_external_witness_data(
        trace_layout,
        domain_size,
        setup,
        &external_witness_data,
        &cs.lookup_parameters,
        worker,
    )?;
    let num_public_inputs = external_witness_data.public_inputs_locations.len();
    assert!(num_public_inputs > 0);
    // external witness already knows both locations of public input and values of them
    let mut public_inputs_with_locations = Vec::with_capacity(num_public_inputs);
    for (col, row) in external_witness_data
        .public_inputs_locations
        .iter()
        .cloned()
    {
        let variable_idx = setup.variables_hint[col][row].clone() as usize;
        let value = external_witness_data.all_values[variable_idx];
        public_inputs_with_locations.push((col, row, value));
    }
    gpu_prove_from_trace::<_, _, TR, _, NoPow, _>(
        cs,
        raw_trace,
        monomial_trace,
        public_inputs_with_locations,
        setup,
        proof_config,
        vk,
        transcript_params,
        worker,
    )
}

pub fn gpu_prove<
    P: boojum::field::traits::field_like::PrimeFieldLikeVectorized<Base = F>,
    TR: Transcript<F, CompatibleCap = [F; 4]>,
    H: TreeHasher<F, Output = TR::CompatibleCap>,
    POW: PoWRunner,
    A: GoodAllocator,
>(
    mut cs: CSReferenceAssembly<F, P, ProvingCSConfig>,
    proof_config: ProofConfig,
    setup: &GpuSetup<A>,
    vk: &VerificationKey<F, H>,
    transcript_params: TR::TransciptParameters,
    worker: &Worker,
) -> CudaResult<Proof<F, H, EXT>> {
    assert!(cs.next_available_place_idx() > 0, "CS shouldn't be empty");
    // this is a convenient function that made for locally synthesized circuits
    // neither witness storage is "reusable" nor we have a direct interface to underling storage
    // so it is okey to materialize them on the host then other convenient function for now
    let witness_data = cs.materialize_witness_vec();
    gpu_prove_from_external_witness_data::<_, TR, _, POW, _>(
        cs,
        &witness_data,
        proof_config,
        setup,
        vk,
        transcript_params,
        worker,
    )
}

fn gpu_prove_from_trace<
    P: boojum::field::traits::field_like::PrimeFieldLikeVectorized<Base = F>,
    CFG: CSConfig,
    TR: Transcript<F, CompatibleCap = [F; 4]>,
    H: TreeHasher<F, Output = TR::CompatibleCap>,
    POW: PoWRunner,
    A: GoodAllocator,
>(
    cs: CSReferenceAssembly<F, P, CFG>,
    raw_trace: GenericTraceStorage<LagrangeBasis>,
    monomial_trace: GenericTraceStorage<MonomialBasis>,
    public_inputs_with_locations: Vec<(usize, usize, F)>,
    setup_base: &GpuSetup<A>,
    proof_config: ProofConfig,
    vk: &VerificationKey<F, H>,
    transcript_params: TR::TransciptParameters,
    worker: &Worker,
) -> CudaResult<Proof<F, H, EXT>> {
    assert!(proof_config.fri_lde_factor.is_power_of_two());
    assert!(proof_config.fri_lde_factor > 1);

    let base_system_degree = cs.max_trace_len;
    assert!(base_system_degree.is_power_of_two());

    unsafe {
        assert!(
            _CUDA_CONTEXT.is_some(),
            "prover context should be initialized"
        )
    };

    let cap_size = proof_config.merkle_tree_cap_size;
    assert!(cap_size > 0);

    let table_ids_column_idxes = setup_base.table_ids_column_idxes.clone();

    let mut transcript = TR::new(transcript_params);

    // Commit to verification key, that should be small
    transcript.witness_merkle_tree_cap(&vk.setup_merkle_tree_cap);

    let mut owned_ctx = P::Context::placeholder();
    let _ctx = &mut owned_ctx;

    let selectors_placement = setup_base.selectors_placement.clone();

    let domain_size = setup_base.constant_columns[0].len(); // counted in elements of P
    assert_eq!(base_system_degree, domain_size);
    assert!(domain_size.is_power_of_two());

    let (max_constraint_contribution_degree, _number_of_constant_polys) =
        selectors_placement.compute_stats();

    let quotient_degree_from_general_purpose_gate_terms = if max_constraint_contribution_degree > 0
    {
        max_constraint_contribution_degree - 1
    } else {
        0
    };

    let max_degree_from_specialized_gates = cs
        .evaluation_data_over_specialized_columns
        .evaluators_over_specialized_columns
        .iter()
        .map(|el| el.max_constraint_degree - 1)
        .max()
        .unwrap_or(0);

    let quotient_degree_from_gate_terms = std::cmp::max(
        quotient_degree_from_general_purpose_gate_terms,
        max_degree_from_specialized_gates,
    );

    let min_lde_degree_for_gates = if quotient_degree_from_gate_terms.is_power_of_two() {
        quotient_degree_from_gate_terms
    } else {
        quotient_degree_from_gate_terms.next_power_of_two()
    };

    let quotient_degree = min_lde_degree_for_gates;

    // now we can commit to public inputs also before potentially moving computations to vectorized form
    for (_, _, value) in public_inputs_with_locations.iter().copied() {
        transcript.witness_field_elements(&[value]);
    }

    let used_lde_degree = std::cmp::max(proof_config.fri_lde_factor, quotient_degree);
    let lde_degree = used_lde_degree;
    let fri_lde_degree = proof_config.fri_lde_factor;

    // cap size shouldn't be part of subtree
    // Immediately transfer raw trace to the device and simultaneously compute monomials of the whole trace
    // Then compute coset evaluations + cap of the sub tree for each coset
    assert!(cap_size.is_power_of_two());
    assert!(fri_lde_degree.is_power_of_two());

    let time = std::time::Instant::now();
    // TODO: read data from Assembly pinned storage
    let TraceLayout {
        num_variable_cols, ..
    } = raw_trace.layout.clone();
    let num_trace_polys = raw_trace.num_polys();
    assert_eq!(monomial_trace.domain_size(), domain_size);
    assert_eq!(monomial_trace.num_polys(), num_trace_polys);
    assert_eq!(raw_trace.domain_size(), domain_size);
    let mut trace_holder =
        TraceCache::from_monomial(monomial_trace, fri_lde_degree, used_lde_degree)?;
    let (trace_subtrees, trace_tree_cap) = trace_holder.commit::<H>(cap_size)?;
    // TODO: use cuda callback for transcript
    transcript.witness_merkle_tree_cap(&trace_tree_cap);

    let mut tree_holder = TreeCache::empty(fri_lde_degree);
    tree_holder.set_trace_subtrees(trace_subtrees);

    let h_beta = transcript.get_multiple_challenges_fixed::<2>();
    let h_beta = ExtensionField::<F, 2, EXT>::from_coeff_in_base(h_beta);
    let h_gamma = transcript.get_multiple_challenges_fixed::<2>();
    let h_gamma = ExtensionField::<F, 2, EXT>::from_coeff_in_base(h_gamma);

    let beta = h_beta.into();
    let gamma = h_gamma.into();

    let non_residues =
        non_residues_for_copy_permutation::<F, Global>(domain_size, num_variable_cols);

    let mut h_non_residues_by_beta = vec![];
    for non_residue in non_residues.iter() {
        let mut non_residue_by_beta = h_beta.clone();
        non_residue_by_beta.mul_assign_by_base(non_residue);
        h_non_residues_by_beta.push(non_residue_by_beta);
    }
    let mut non_residues_by_beta = dvec!(h_non_residues_by_beta.len());
    non_residues_by_beta.copy_from_slice(&h_non_residues_by_beta)?;

    let raw_setup = GenericSetupStorage::from_gpu_setup(&setup_base)?;
    let arguments_layout = ArgumentsLayout::from_trace_layout_and_lookup_params(
        raw_trace.layout.clone(),
        quotient_degree,
        cs.lookup_parameters.clone(),
    );
    let mut argument_raw_storage = GenericArgumentStorage::allocate(arguments_layout, domain_size)?;
    let raw_trace_polynomials = raw_trace.as_polynomials();
    let raw_setup_polynomials = raw_setup.as_polynomials();

    compute_partial_products(
        &raw_trace_polynomials,
        &raw_setup_polynomials,
        &non_residues_by_beta,
        &beta,
        &gamma,
        quotient_degree,
        &mut argument_raw_storage,
    )?;

    let num_intermediate_partial_product_relations =
        argument_raw_storage.as_polynomials().partial_products.len();

    let (lookup_beta, powers_of_gamma_for_lookup) = if cs.lookup_parameters
        != LookupParameters::NoLookup
    {
        let h_lookup_beta = transcript.get_multiple_challenges_fixed::<2>();
        let h_lookup_beta = ExtensionField::<F, 2, EXT>::from_coeff_in_base(h_lookup_beta);
        let h_lookup_gamma = transcript.get_multiple_challenges_fixed::<2>();
        let h_lookup_gamma = ExtensionField::<F, 2, EXT>::from_coeff_in_base(h_lookup_gamma);
        let lookup_beta: DExt = h_lookup_beta.into();

        let lookup_evaluator_id = 0;
        let _selector_subpath = setup_base
            .selectors_placement
            .output_placement(lookup_evaluator_id)
            .expect("lookup gate must be placed");

        let variables_offset = cs.parameters.num_columns_under_copy_permutation;

        let powers_of_gamma = match cs.lookup_parameters {
            LookupParameters::NoLookup => {
                unreachable!()
            }
            LookupParameters::TableIdAsConstant { .. }
            | LookupParameters::TableIdAsVariable { .. } => {
                // let columns_per_subargument = cs.lookup_parameters.columns_per_subargument();

                // let mut h_powers_of_gamma: Vec<EF> = vec![];
                // let mut current = EF::ONE;
                // for _ in 0..columns_per_subargument {
                //     h_powers_of_gamma.push(current.into());
                //     // Should this be h_gamma or h_lookup_gamma?
                //     current.mul_assign(&h_lookup_gamma);
                // }
                // let mut powers_of_gamma = dvec!(h_powers_of_gamma.len());
                // powers_of_gamma.copy_from_slice(&h_powers_of_gamma)?;

                // compute_lookup_argument_over_general_purpose_cols(
                //     &base_trace,
                //     &base_setup,
                //     table_ids_column_idxes.clone(),
                //     &lookup_beta,
                //     powers_of_gamma.clone(),
                //     variables_offset,
                //     cs.lookup_parameters,
                //     columns_per_subargument as usize,
                //     selector_subpath.clone(),
                //     lde_degree,
                // )?;

                todo!();
            }
            a @ LookupParameters::UseSpecializedColumnsWithTableIdAsVariable { width, .. }
            | a @ LookupParameters::UseSpecializedColumnsWithTableIdAsConstant { width, .. } => {
                // ensure proper setup
                assert_eq!(
                    cs.evaluation_data_over_specialized_columns
                        .gate_type_ids_for_specialized_columns[0],
                    std::any::TypeId::of::<LookupFormalGate>(),
                    "we expect first specialized gate to be the lookup -"
                );
                let (initial_offset, offset_per_repetition, _) = cs
                    .evaluation_data_over_specialized_columns
                    .offsets_for_specialized_evaluators[0];
                assert_eq!(initial_offset.constants_offset, 0);

                if let LookupParameters::UseSpecializedColumnsWithTableIdAsConstant {
                    share_table_id,
                    ..
                } = a
                {
                    if share_table_id {
                        assert_eq!(offset_per_repetition.constants_offset, 0);
                    }
                }
                let columns_per_subargument = width + 1;

                let mut h_powers_of_gamma = vec![];
                let mut current = EF::ONE;
                for _ in 0..columns_per_subargument {
                    h_powers_of_gamma.push(current.into());
                    current.mul_assign(&h_lookup_gamma);
                }
                let mut powers_of_gamma = dvec!(h_powers_of_gamma.len());
                powers_of_gamma.copy_from_slice(&h_powers_of_gamma)?;

                compute_lookup_argument_over_specialized_cols(
                    &raw_trace_polynomials,
                    &raw_setup_polynomials,
                    table_ids_column_idxes.clone(),
                    &lookup_beta,
                    &powers_of_gamma,
                    variables_offset,
                    cs.lookup_parameters,
                    lde_degree,
                    &mut argument_raw_storage,
                )?;

                powers_of_gamma
            }
        };

        (Some(lookup_beta), Some(powers_of_gamma))
    } else {
        (None, None)
    };

    let argument_monomial_storage = argument_raw_storage.into_monomial()?;
    let mut argument_holder =
        ArgumentCache::from_monomial(argument_monomial_storage, fri_lde_degree, used_lde_degree)?;
    let (argument_subtrees, argument_tree_cap) = argument_holder.commit::<H>(cap_size)?;

    transcript.witness_merkle_tree_cap(&argument_tree_cap);
    tree_holder.set_argument_subtrees(argument_subtrees);

    let h_alpha = transcript.get_multiple_challenges_fixed::<2>();
    let h_alpha = ExtensionField::<F, 2, EXT>::from_coeff_in_base(h_alpha);

    let num_lookup_subarguments = cs.num_sublookup_arguments();
    let num_multiplicities_polys = cs.num_multipicities_polys();
    let total_num_lookup_argument_terms = num_lookup_subarguments + num_multiplicities_polys;

    let total_num_gate_terms_for_specialized_columns = cs
        .evaluation_data_over_specialized_columns
        .evaluators_over_specialized_columns
        .iter()
        .zip(
            cs.evaluation_data_over_specialized_columns
                .gate_type_ids_for_specialized_columns
                .iter(),
        )
        .map(|(evaluator, gate_type_id)| {
            let placement_strategy = cs
                .placement_strategies
                .get(gate_type_id)
                .copied()
                .expect("gate must be allowed");
            let num_repetitions = match placement_strategy {
                GatePlacementStrategy::UseSpecializedColumns {
                    num_repetitions, ..
                } => num_repetitions,
                _ => unreachable!(),
            };
            assert_eq!(evaluator.num_repetitions_on_row, num_repetitions);
            let terms_per_repetition = evaluator.num_quotient_terms;

            terms_per_repetition * num_repetitions
        })
        .sum();

    let total_num_gate_terms_for_general_purpose_columns: usize = cs
        .evaluation_data_over_general_purpose_columns
        .evaluators_over_general_purpose_columns
        .iter()
        .map(|evaluator| evaluator.total_quotient_terms_over_all_repetitions)
        .sum();

    let total_num_terms = total_num_lookup_argument_terms // and lookup is first
        + total_num_gate_terms_for_specialized_columns // then gates over specialized columns
        + total_num_gate_terms_for_general_purpose_columns // all gates terms over general purpose columns
        + 1 // z(1) == 1 copy permutation
        + 1 // z(x * omega) = ...
        + num_intermediate_partial_product_relations // chunking copy permutation part
    ;

    let h_powers_of_alpha: Vec<_, Global> = materialize_powers_serial(h_alpha, total_num_terms);
    let rest = &h_powers_of_alpha[..];
    let (take, rest) = rest.split_at(total_num_lookup_argument_terms);
    let pregenerated_challenges_for_lookup = if total_num_lookup_argument_terms > 0 {
        let h_vec = take.to_vec();
        let mut d_vec = dvec!(h_vec.len());
        d_vec.copy_from_slice(&h_vec)?;
        Some(d_vec)
    } else {
        None
    };
    let (take, rest) = rest.split_at(total_num_gate_terms_for_specialized_columns);
    let _pregenerated_challenges_for_gates_over_specialized_columns = take.to_vec();
    let (take, rest) = rest.split_at(total_num_gate_terms_for_general_purpose_columns);
    let _pregenerated_challenges_for_gates_over_general_purpose_columns = take.to_vec();
    let (take, rest) = rest.split_at(1);
    let copy_permutation_challenge_z_at_one_equals_one = take.to_vec();
    assert!(rest.len() > 0);
    let h_vec = rest.to_vec();
    let mut copy_permutation_challenges_partial_product_terms = dvec!(h_vec.len());
    // TODO: When we use host pinned memory for challenges, we need to make sure
    // asynchronous H2D copies complete before h_vec gets dropped.
    copy_permutation_challenges_partial_product_terms.copy_from_slice(&h_vec)?;

    let mut quotient = ComplexPoly::<LDE>::zero(quotient_degree * domain_size)?;
    let base_monomial_setup = raw_setup.into_monomials()?;
    let mut setup_holder =
        SetupCache::from_monomial(base_monomial_setup, fri_lde_degree, used_lde_degree)?;

    let variables_offset = cs.parameters.num_columns_under_copy_permutation;

    let general_purpose_gates = constraint_evaluation::get_evaluators_of_general_purpose_cols(
        &cs,
        &setup_base.selectors_placement,
    );

    let specialized_gates = constraint_evaluation::get_specialized_evaluators_from_assembly(
        &cs,
        &setup_base.selectors_placement,
    );
    let num_cols_per_product = quotient_degree;
    let specialized_cols_challenge_power_offset = total_num_lookup_argument_terms;
    let general_purpose_cols_challenge_power_offset =
        total_num_lookup_argument_terms + total_num_gate_terms_for_specialized_columns;
    for coset_idx in 0..quotient_degree {
        let mut coset_values = ComplexPoly::<CosetEvaluations>::zero(domain_size)?;
        let trace_coset_values = trace_holder.get_or_compute_coset_evals(coset_idx)?;
        let setup_coset_values = setup_holder.get_or_compute_coset_evals(coset_idx)?;
        let argument_coset_values = argument_holder.get_or_compute_coset_evals(coset_idx)?;
        compute_quotient_by_coset(
            &trace_coset_values,
            &setup_coset_values,
            &argument_coset_values,
            cs.lookup_parameters,
            &setup_base.table_ids_column_idxes,
            &setup_base.selectors_placement,
            &specialized_gates,
            &general_purpose_gates,
            coset_idx,
            domain_size,
            used_lde_degree,
            num_cols_per_product,
            &copy_permutation_challenge_z_at_one_equals_one[0],
            &copy_permutation_challenges_partial_product_terms,
            &h_alpha,
            &pregenerated_challenges_for_lookup,
            specialized_cols_challenge_power_offset,
            general_purpose_cols_challenge_power_offset,
            &beta,
            &gamma,
            &powers_of_gamma_for_lookup,
            lookup_beta.as_ref().clone(),
            &non_residues_by_beta,
            variables_offset,
            &mut coset_values,
        )?;

        let start = coset_idx * domain_size;
        let end = start + domain_size;
        mem::d2d(
            coset_values.c0.storage.as_ref(),
            &mut quotient.c0.storage.as_mut()[start..end],
        )?;
        mem::d2d(
            coset_values.c1.storage.as_ref(),
            &mut quotient.c1.storage.as_mut()[start..end],
        )?;
    }

    let coset: DF = F::multiplicative_generator().into();
    quotient.bitreverse()?;
    let quotient_monomial = quotient.ifft(&coset)?;
    // quotient memory is guaranteed to allow batch ntts for cosets of the quotinet parts
    let quotient_chunks = quotient_monomial.clone().into_degree_n_polys(domain_size)?;

    let quotient_monomial_storage = GenericComplexPolynomialStorage {
        polynomials: quotient_chunks,
    };
    let num_quotient_polys = quotient_monomial_storage.num_polys();
    let mut quotient_holder =
        QuotientCache::from_monomial(quotient_monomial_storage, fri_lde_degree, used_lde_degree)?;
    let (quotient_subtrees, quotient_tree_cap) =
        quotient_holder.commit::<DefaultTreeHasher>(cap_size)?;
    transcript.witness_merkle_tree_cap(&quotient_tree_cap);
    tree_holder.set_quotient_subtrees(quotient_subtrees);

    // deep part
    let h_z = transcript.get_multiple_challenges_fixed::<2>();
    let h_z = ExtensionField::<F, 2, EXT>::from_coeff_in_base(h_z);

    let mut h_z_omega = h_z;
    h_z_omega.mul_assign_by_base(&domain_generator_for_size::<F>(domain_size as u64));

    let num_setup_polys = setup_holder.num_polys();
    let num_argument_polys = argument_holder.num_polys();

    let (h_evaluations_at_z, h_evaluations_at_z_omega, h_evaluations_at_zero) = {
        compute_evaluations_over_lagrange_basis(
            &mut trace_holder,
            &mut setup_holder,
            &mut argument_holder,
            &mut quotient_holder,
            h_z.clone(),
            h_z_omega.clone(),
        )?
    };

    assert_eq!(
        h_evaluations_at_z.len(),
        num_trace_polys + num_setup_polys + num_argument_polys + num_quotient_polys
    );
    let mut evaluations_at_z = dvec!(h_evaluations_at_z.len());
    evaluations_at_z.copy_from_slice(&h_evaluations_at_z)?;
    for h_point in h_evaluations_at_z.clone().into_iter() {
        transcript.witness_field_elements(h_point.as_coeffs_in_base());
    }

    assert_eq!(h_evaluations_at_z_omega.len(), 1);
    let mut evaluations_at_z_omega = dvec!(h_evaluations_at_z_omega.len());
    evaluations_at_z_omega.copy_from_slice(&h_evaluations_at_z_omega)?;
    for h_point in h_evaluations_at_z_omega.clone().into_iter() {
        transcript.witness_field_elements(h_point.as_coeffs_in_base());
    }

    let evaluations_at_zero = if h_evaluations_at_zero.len() > 0 {
        let mut d_vec = dvec!(h_evaluations_at_zero.len());
        d_vec.copy_from_slice(&h_evaluations_at_zero)?;
        Some(d_vec)
    } else {
        None
    };
    for h_point in h_evaluations_at_zero.clone().into_iter() {
        transcript.witness_field_elements(h_point.as_coeffs_in_base());
    }

    // and public inputs should also go into quotient
    let mut public_input_opening_tuples: Vec<(F, Vec<(usize, F)>)> = vec![];
    {
        let omega = domain_generator_for_size::<F>(domain_size as u64);

        for (column, row, value) in public_inputs_with_locations.iter().cloned() {
            let open_at = omega.pow_u64(row as u64);
            let pos = public_input_opening_tuples
                .iter()
                .position(|el| el.0 == open_at);
            if let Some(pos) = pos {
                public_input_opening_tuples[pos].1.push((column, value));
            } else {
                public_input_opening_tuples.push((open_at, vec![(column, value)]));
            }
        }
    }

    let mut total_num_challenges = 0;
    total_num_challenges += h_evaluations_at_z.len();
    total_num_challenges += h_evaluations_at_z_omega.len();
    total_num_challenges += h_evaluations_at_zero.len();
    for (_, subset) in public_input_opening_tuples.iter() {
        total_num_challenges += subset.len();
    }

    let c0 = transcript.get_challenge();
    let c1 = transcript.get_challenge();
    let h_challenge = ExtensionField::<F, 2, EXT>::from_coeff_in_base([c0, c1]);
    let h_challenges: Vec<_, Global> = materialize_powers_serial(h_challenge, total_num_challenges);
    let mut challenges = dvec!(h_challenges.len());
    challenges.copy_from_slice(&h_challenges)?;

    let mut deep_quotient = ComplexPoly::<LDE>::zero(fri_lde_degree * domain_size)?;
    let z = h_z.clone().into();
    let z_omega = h_z_omega.clone().into();
    for coset_idx in 0..fri_lde_degree {
        let trace_polys = trace_holder
            .get_or_compute_coset_evals(coset_idx)?
            .as_polynomials();
        let setup_polys = setup_holder
            .get_or_compute_coset_evals(coset_idx)?
            .as_polynomials();
        let argument_polys = argument_holder
            .get_or_compute_coset_evals(coset_idx)?
            .as_polynomials();
        let quotient_polys = quotient_holder.get_or_compute_coset_evals(coset_idx)?;

        let coset_omegas = compute_omega_values_for_coset(coset_idx, domain_size, used_lde_degree)?;
        let deep_quotient_over_coset = compute_deep_quotiening_over_coset(
            &trace_polys,
            &setup_polys,
            &argument_polys,
            &quotient_polys,
            coset_omegas,
            &evaluations_at_z,
            &evaluations_at_z_omega,
            &evaluations_at_zero,
            &z,
            &z_omega,
            &public_input_opening_tuples,
            &challenges,
        )?;
        let start = coset_idx * domain_size;
        let end = start + domain_size;

        mem::d2d(
            deep_quotient_over_coset.c0.storage.as_ref(),
            &mut deep_quotient.c0.storage.as_mut()[start..end],
        )?;

        mem::d2d(
            deep_quotient_over_coset.c1.storage.as_ref(),
            &mut deep_quotient.c1.storage.as_mut()[start..end],
        )?;
    }
    let basic_pow_bits = proof_config.pow_bits;

    let (
        new_pow_bits,     // updated POW bits if needed
        num_queries,      // num queries
        folding_schedule, // folding schedule
        ..,
    ) = boojum::cs::implementations::prover::compute_fri_schedule(
        proof_config.security_level as u32,
        cap_size,
        basic_pow_bits,
        fri_lde_degree.trailing_zeros(),
        domain_size.trailing_zeros(),
    );

    let first_code_word = CodeWord::new(
        deep_quotient.c0.storage.into_inner(),
        deep_quotient.c1.storage.into_inner(),
        fri_lde_degree,
    );
    let (first_oracle, intermediate_code_words, intermediate_oracles, fri_last_monomials) =
        compute_fri(
            first_code_word.clone(),
            &mut transcript,
            folding_schedule.clone(),
            fri_lde_degree,
            cap_size,
        )?;

    let pow_challenge = if new_pow_bits != 0 {
        println!("Doing PoW");

        let now = std::time::Instant::now();

        // pull enough challenges from the transcript
        let mut num_challenges = 256 / F::CHAR_BITS;
        if num_challenges % F::CHAR_BITS != 0 {
            num_challenges += 1;
        }
        let challenges = transcript.get_multiple_challenges(num_challenges);
        let pow_challenge = POW::run_from_field_elements(challenges, new_pow_bits, worker);

        assert!(F::CAPACITY_BITS >= 32);
        let (low, high) = (pow_challenge as u32, (pow_challenge >> 32) as u32);
        let low = F::from_u64_unchecked(low as u64);
        let high = F::from_u64_unchecked(high as u64);
        transcript.witness_field_elements(&[low, high]);

        println!("PoW for {} bits taken {:?}", new_pow_bits, now.elapsed());

        pow_challenge
    } else {
        0
    };

    let _fri_intermediate_query_storage: Vec<_> = intermediate_code_words
        .iter()
        .zip(intermediate_oracles.iter())
        .map(|(c, o)| (c.clone(), o.clone()))
        .collect();
    // TODO: setup oracle is precomputed, we don't need to recompute it
    // transfer host based oracle to the device then Robert's function will use it
    let (setup_subtrees, setup_tree_cap) = setup_holder.commit::<H>(cap_size)?;
    debug_assert_eq!(setup_tree_cap, vk.setup_merkle_tree_cap);
    tree_holder.set_setup_subtrees(setup_subtrees);

    use boojum::cs::implementations::transcript::BoolsBuffer;
    let max_needed_bits = (domain_size * fri_lde_degree).trailing_zeros() as usize;
    // let mut bools_buffer = BoolsBuffer::new(max_needed_bits);
    let mut bools_buffer = BoolsBuffer {
        available: vec![],
        max_needed: max_needed_bits,
    };

    let num_bits_for_in_coset_index = max_needed_bits - fri_lde_degree.trailing_zeros() as usize;

    // get query schedule first, sort then reduce number of coset evals
    let mut query_details_for_cosets = vec![vec![]; fri_lde_degree];

    for query_idx in 0..num_queries {
        let query_index_lsb_first_bits = bools_buffer.get_bits(&mut transcript, max_needed_bits);
        // we consider it to be some convenient for us encoding of coset + inner index.

        let inner_idx =
            u64_from_lsb_first_bits(&query_index_lsb_first_bits[0..num_bits_for_in_coset_index])
                as usize;
        let coset_idx =
            u64_from_lsb_first_bits(&query_index_lsb_first_bits[num_bits_for_in_coset_index..])
                as usize;
        query_details_for_cosets[coset_idx].push((query_idx, coset_idx, inner_idx));
    }

    let mut fri_layers = vec![];
    fri_layers.push((&first_code_word, &first_oracle));
    for l in intermediate_code_words
        .iter()
        .zip(intermediate_oracles.iter())
    {
        fri_layers.push(l)
    }

    let mut round_queries = vec![];
    for (coset_idx, rounds_for_coset) in query_details_for_cosets.iter().cloned().enumerate() {
        for (query_idx, actual_coset_idx, row_idx) in rounds_for_coset.iter().cloned() {
            assert_eq!(coset_idx, actual_coset_idx);
            // In theory queries should happen through tree holder
            // but it is okey to construct queries through polynomial storages
            // due to too many lifetime definitions
            let witness_query = trace_holder.query(
                coset_idx,
                fri_lde_degree,
                row_idx,
                domain_size,
                &tree_holder,
            )?;
            let setup_query = setup_holder.query(
                coset_idx,
                fri_lde_degree,
                row_idx,
                domain_size,
                &tree_holder,
            )?;
            let argument_query = argument_holder.query(
                coset_idx,
                fri_lde_degree,
                row_idx,
                domain_size,
                &tree_holder,
            )?;
            let quotient_query = quotient_holder.query(
                coset_idx,
                fri_lde_degree,
                row_idx,
                domain_size,
                &tree_holder,
            )?;

            let mut fri_queries = vec![];
            let mut row_idx = row_idx;
            let mut domain_size = domain_size;
            for ((code_word, fri_oracle), schedule) in
                fri_layers.iter().cloned().zip(folding_schedule.iter())
            {
                let fri_round_query = fri_oracle.query::<H, _>(
                    code_word,
                    coset_idx,
                    fri_lde_degree,
                    row_idx,
                    domain_size,
                )?;

                fri_queries.push(fri_round_query);

                row_idx >>= schedule;
                domain_size >>= schedule;
            }

            let round_query = SingleRoundQueries {
                witness_query,
                setup_query,
                stage_2_query: argument_query,
                quotient_query,
                fri_queries,
            };
            round_queries.push((query_idx, round_query))
        }
    }

    round_queries.sort_by(|a, b| a.0.cmp(&b.0));

    let queries_per_fri_repetition: Vec<_> = round_queries.into_iter().map(|r| r.1).collect();
    synchronize_streams()?;
    println!("FRI Queries are done {:?}", time.elapsed());
    let public_inputs_only_values = public_inputs_with_locations
        .into_iter()
        .map(|(_, _, value)| value)
        .collect();
    let proof = Proof::<F, H, EXT> {
        public_inputs: public_inputs_only_values,
        witness_oracle_cap: trace_tree_cap.clone(),
        stage_2_oracle_cap: argument_tree_cap.clone(),
        quotient_oracle_cap: quotient_tree_cap.clone(),
        final_fri_monomials: fri_last_monomials.clone(),
        values_at_z: h_evaluations_at_z.clone(),
        values_at_z_omega: h_evaluations_at_z_omega.clone(),
        values_at_0: h_evaluations_at_zero.clone(),
        pow_challenge,
        fri_base_oracle_cap: first_oracle.get_tree_cap()?,
        fri_intermediate_oracles_caps: intermediate_oracles
            .iter()
            .map(|el| el.get_tree_cap().expect(""))
            .collect(),
        queries_per_fri_repetition,

        _marker: std::marker::PhantomData,
        proof_config,
    };
    Ok(proof)
}

pub(crate) fn u64_from_lsb_first_bits(bits: &[bool]) -> u64 {
    let mut result = 0u64;
    for (shift, bit) in bits.iter().enumerate() {
        result |= (*bit as u64) << shift;
    }

    result
}

pub fn compute_evaluations_over_lagrange_basis<'a>(
    trace_holder: &mut TraceCache,
    setup_holder: &mut SetupCache,
    argument_holder: &mut ArgumentCache<'a>,
    quotient_holder: &mut QuotientCache<'a>,
    z: EF,
    z_omega: EF,
) -> CudaResult<(Vec<EF>, Vec<EF>, Vec<EF>)> {
    // all polynomials should be opened at "z"
    // additionally, copy permutation polynomial should be opened at "z*w"
    // lookup polynomials should be opened at "0"
    // we should follow order of the evaluations as in the reference impl
    let trace_storage = trace_holder.get_or_compute_coset_evals(0)?;
    let setup_storage = setup_holder.get_or_compute_coset_evals(0)?;
    let domain_size = trace_storage.domain_size();
    let precomputed_bases_for_z = PrecomputedBasisForBarycentric::precompute(domain_size, z)?;

    let trace_evals_at_z = trace_storage.barycentric_evaluate(&precomputed_bases_for_z)?;
    let setup_evals_at_z = setup_storage.barycentric_evaluate(&precomputed_bases_for_z)?;

    let quotient_storage = quotient_holder.get_or_compute_coset_evals(0)?;
    let quotient_evals_at_z = quotient_storage.barycentric_evaluate(&precomputed_bases_for_z)?;

    // We can evaluate those polys cheaper but use barycentric for now
    let precomputed_bases_for_zero =
        PrecomputedBasisForBarycentric::precompute(domain_size, EF::ZERO)?;

    // evaluate z(x) at z_omega direcly in monomial
    let z_at_z_omega = argument_holder
        .get_z_monomial()
        .evaluate_at_ext(&z_omega.into())?;
    let argument_storage = argument_holder.get_or_compute_coset_evals(0)?;
    let ArgumentPolynomials {
        partial_products,
        lookup_a_polys,
        lookup_b_polys,
        ..
    } = argument_storage.as_polynomials();
    let argument_evals_at_z = argument_storage.barycentric_evaluate(&precomputed_bases_for_z)?;

    let lookup_evals_at_zero = if lookup_a_polys.len() > 0 {
        assert_eq!(lookup_b_polys.len(), 1);
        assert_multiset_adjacent_ext(&[lookup_a_polys, lookup_b_polys]);
        let num_lookup_polys = lookup_a_polys.len() + lookup_b_polys.len();
        let lookup_polys = unsafe {
            let len = 2 * domain_size * num_lookup_polys;
            std::slice::from_raw_parts(lookup_a_polys[0].c0.storage.as_ref().as_ptr(), len)
        };
        batch_barycentric_evaluate_ext(
            &(lookup_polys, domain_size, num_lookup_polys), // (slice, domain size, num polys)
            &precomputed_bases_for_zero,
            domain_size,
            num_lookup_polys,
        )?
    } else {
        vec![]
    };

    let TraceLayout {
        num_variable_cols,
        num_witness_cols,
        num_multiplicity_cols,
    } = trace_storage.layout.clone();
    assert_eq!(
        trace_evals_at_z.len(),
        num_variable_cols + num_witness_cols + num_multiplicity_cols
    );
    let SetupLayout {
        num_permutation_cols,
        num_constant_cols,
        num_table_cols,
    } = setup_storage.layout.clone();
    assert_eq!(
        setup_evals_at_z.len(),
        num_permutation_cols + num_constant_cols + num_table_cols
    );
    let num_partial_product_poyls = partial_products.len();
    assert_eq!(argument_evals_at_z.len(), argument_holder.num_polys());
    // decompose storage into sub polynomials
    let mut trace_evals_at_z_iter = trace_evals_at_z;
    let variable_evals_at_z: Vec<EF> = trace_evals_at_z_iter.drain(0..num_variable_cols).collect();

    let witness_evals_at_z: Vec<EF> = trace_evals_at_z_iter.drain(0..num_witness_cols).collect();
    let multiplicity_evals_at_z: Vec<EF> = trace_evals_at_z_iter
        .drain(0..num_multiplicity_cols)
        .collect();
    let mut setup_evals_at_z_iter = setup_evals_at_z;
    let permutations_at_z: Vec<EF> = setup_evals_at_z_iter
        .drain(0..num_permutation_cols)
        .collect();
    let constants_cols_at_z: Vec<EF> = setup_evals_at_z_iter.drain(0..num_constant_cols).collect();
    let table_cols_at_z: Vec<EF> = setup_evals_at_z_iter.drain(0..num_table_cols).collect();
    assert_eq!(setup_evals_at_z_iter.len(), 0);

    let (copy_permutation_evals_at_z, lookup_evals_at_z) =
        argument_evals_at_z.split_at(1 + num_partial_product_poyls);

    let mut polynomials_at_z = vec![];
    let mut polynomials_at_z_omega = vec![];
    let mut polynomials_at_zero = vec![];

    polynomials_at_z.extend_from_slice(&variable_evals_at_z);
    polynomials_at_z.extend_from_slice(&witness_evals_at_z);
    polynomials_at_z.extend_from_slice(&constants_cols_at_z);
    polynomials_at_z.extend_from_slice(&permutations_at_z);
    polynomials_at_z.extend_from_slice(copy_permutation_evals_at_z);
    polynomials_at_z.extend_from_slice(&multiplicity_evals_at_z);
    polynomials_at_z.extend_from_slice(&lookup_evals_at_z);
    polynomials_at_z.extend_from_slice(&table_cols_at_z);
    polynomials_at_z.extend_from_slice(&quotient_evals_at_z);

    polynomials_at_z_omega.push(z_at_z_omega.into());

    polynomials_at_zero.extend(lookup_evals_at_zero);

    Ok((
        polynomials_at_z,
        polynomials_at_z_omega,
        polynomials_at_zero,
    ))
}

#[allow(dead_code)]
pub fn barycentric_evaluate_at_zero(poly: &ComplexPoly<LagrangeBasis>) -> CudaResult<DExt> {
    let coset: DF = F::multiplicative_generator().into();
    let mut values = poly.clone();
    values.bitreverse()?;
    let monomial = values.ifft(&coset)?;
    let result = monomial.grand_sum()?;

    Ok(result)
}

pub fn compute_denom_at_base_point<'a>(
    roots: &Poly<'a, CosetEvaluations>,
    point: &DF,
) -> CudaResult<Poly<'a, CosetEvaluations>> {
    // TODO: This pattern is a temporary workaround, not optimal
    let mut denom = Poly::<CosetEvaluations>::zero(roots.domain_size())?;
    denom
        .storage
        .copy_from_device_slice(roots.storage.as_ref())?;
    denom.sub_constant(point)?;
    denom.inverse()?;
    Ok(denom)
}

pub fn compute_denom_at_ext_point<'a>(
    roots: &Poly<'a, CosetEvaluations>,
    point: &DExt,
) -> CudaResult<ComplexPoly<'a, CosetEvaluations>> {
    // TODO: This pattern is a temporary workaround, not optimal
    let mut denom = ComplexPoly::<CosetEvaluations>::zero(roots.domain_size())?;
    denom
        .c0
        .storage
        .copy_from_device_slice(roots.storage.as_ref())?;
    denom.sub_constant(point)?;
    denom.inverse()?;
    Ok(denom)
}

fn compute_deep_quotiening_over_coset(
    trace_polys: &TracePolynomials<CosetEvaluations>,
    setup_polys: &SetupPolynomials<CosetEvaluations>,
    argument_polys: &ArgumentPolynomials<CosetEvaluations>,
    quotient_poly_constraints: &GenericComplexPolynomialStorage<CosetEvaluations>,
    roots: Poly<CosetEvaluations>,
    evaluations_at_z: &DVec<EF>,
    evaluations_at_z_omega: &DVec<EF>,
    evaluations_at_zero: &Option<DVec<EF>>,
    z: &DExt,
    z_omega: &DExt,
    public_input_opening_tuples: &[(F, Vec<(usize, F)>)],
    challenges: &DVec<EF>,
) -> CudaResult<ComplexPoly<'static, CosetEvaluations>> {
    let domain_size = trace_polys.variable_cols[0].domain_size();
    // TODO:
    // When we add an API to allocate empty vectors without pre-zeroing them,
    // construct quotient that way. It will be fine because
    // deep_quotient_except_public_inputs sets quotient disregarding initial values.
    let mut quotient = ComplexPoly::<CosetEvaluations>::zero(domain_size)?;

    let denom_at_z = compute_denom_at_ext_point(&roots, &z)?;
    let denom_at_z_omega = compute_denom_at_ext_point(&roots, &z_omega)?;
    let denom_at_zero = compute_denom_at_base_point(&roots, &DF::zero()?)?;

    let (maybe_multiplicity_cols, maybe_lookup_a_polys, maybe_lookup_b_polys, maybe_table_cols) =
        if argument_polys.lookup_a_polys.len() > 0 {
            (
                Some(&trace_polys.multiplicity_cols),
                Some(argument_polys.lookup_a_polys),
                Some(argument_polys.lookup_b_polys),
                Some(&setup_polys.table_cols),
            )
        } else {
            (None, None, None, None)
        };

    let maybe_witness_cols = if trace_polys.witness_cols.len() > 0 {
        Some(&trace_polys.witness_cols)
    } else {
        None
    };

    let num_public_inputs: usize = public_input_opening_tuples
        .iter()
        .map(|(_open_at, set)| set.len())
        .sum();

    deep_quotient_except_public_inputs(
        &trace_polys.variable_cols,
        &maybe_witness_cols,
        &setup_polys.constant_cols,
        &setup_polys.permutation_cols,
        &argument_polys.z_poly,
        &argument_polys.partial_products,
        &maybe_multiplicity_cols,
        &maybe_lookup_a_polys,
        &maybe_lookup_b_polys,
        &maybe_table_cols,
        &quotient_poly_constraints.polynomials,
        &evaluations_at_z,
        &evaluations_at_z_omega,
        &evaluations_at_zero,
        &challenges[..challenges.len() - num_public_inputs],
        &denom_at_z,
        &denom_at_z_omega,
        &denom_at_zero,
        &mut quotient,
    )?;

    // public inputs
    // One kernel per term isn't optimal, but it's easy, and should be negligible anyway
    let mut challenge_id = challenges.len() - num_public_inputs;
    for (open_at, set) in public_input_opening_tuples.into_iter() {
        let open_at_df: DF = open_at.clone().into();
        let denom_at_point = compute_denom_at_base_point(&roots, &open_at_df)?;
        // deep_quotient_add_opening accumulates into quotient_for_row, so it does need to be pre-zeroed
        let mut quotient_for_row = ComplexPoly::<CosetEvaluations>::zero(domain_size)?;
        for (column, expected_value) in set.into_iter() {
            deep_quotient_public_input(
                &trace_polys.variable_cols[*column],
                *expected_value,
                &challenges[challenge_id..challenge_id + 1],
                &mut quotient_for_row,
            )?;
            challenge_id += 1;
        }
        // TODO: If mul_assign (and similar calls) can handle ComplexPolys interacting with Polys,
        // we wouldn't need the denom_at_point_ext intermediate
        let mut denom_at_point_ext = ComplexPoly::<CosetEvaluations>::zero(domain_size)?;
        denom_at_point_ext
            .c0
            .storage
            .copy_from_device_slice(&denom_at_point.storage.as_ref())?;
        quotient_for_row.mul_assign(&denom_at_point_ext)?;
        quotient.add_assign(&quotient_for_row)?;
    }

    Ok(quotient)
}
