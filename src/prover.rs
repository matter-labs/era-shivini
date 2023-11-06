use std::{
    alloc::Global, collections::HashMap, mem::ManuallyDrop, ops::Deref, sync::atomic::AtomicU32,
};

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
            setup::{self, TreeNode},
            transcript::Transcript,
            utils::{domain_generator_for_size, materialize_powers_serial},
            verifier::VerificationKey,
            witness::{WitnessSet, WitnessVec},
        },
        oracle::{merkle_tree::MerkleTreeWithCap, TreeHasher},
        toolboxes::{gate_config::GateConfigurationHolder, static_toolbox::StaticToolboxHolder},
        traits::{gate::GatePlacementStrategy, GoodAllocator},
        CSGeometry, LookupParameters, Place, Variable, Witness,
    },
    field::{traits::field_like::TrivialContext, FieldExtension, U64Representable},
    worker::Worker,
};
use smallvec::SmallVec;

use crate::cs::GpuSetup;
use crate::{
    arith::{deep_quotient_except_public_inputs, deep_quotient_public_input},
    cs::PACKED_PLACEHOLDER_BITMASK,
};

use super::*;

pub fn gpu_prove_from_external_witness_data<
    P: boojum::field::traits::field_like::PrimeFieldLikeVectorized<Base = F>,
    TR: Transcript<F, CompatibleCap = [F; 4]>,
    H: TreeHasher<F, Output = TR::CompatibleCap>,
    POW: PoWRunner,
    A: GoodAllocator,
>(
    // service layer reuses assembly for repeated proving, so that just borrow it
    cs: &CSReferenceAssembly<F, P, ProvingCSConfig>,
    external_witness_data: &WitnessVec<F>, // TODO: read data from Assembly pinned storage
    proof_config: ProofConfig,
    setup: &GpuSetup<A>,
    vk: &VerificationKey<F, H>,
    transcript_params: TR::TransciptParameters,
    worker: &Worker,
) -> CudaResult<GpuProof<A>> {
    // TODO: this is a convenient function that is made for externally synthesized circuits
    // but local synthesis also use this fn. re-enable this check once deployments are done
    // assert_eq!(
    //     cs.next_available_place_idx(),
    //     0,
    //     "CS should be empty and hold no data"
    // );
    unsafe {
        assert!(
            _cuda_context.is_some(),
            "prover context should be initialized"
        )
    };

    let num_variable_cols = setup.variables_hint.len();
    let num_witness_cols = setup.witnesses_hint.len();
    let num_multiplicity_cols = cs.num_multipicities_polys();
    let trace_layout = TraceLayout {
        num_variable_cols,
        num_witness_cols,
        num_multiplicity_cols,
    };
    dbg!(&trace_layout);
    let domain_size = cs.max_trace_len;
    let start = std::time::Instant::now();
    let quotient_degree = compute_quotient_degree(&cs, &setup.selectors_placement);
    let used_lde_degree = usize::max(quotient_degree, proof_config.fri_lde_factor);
    let (raw_trace, monomial_trace, subtrees, trace_tree_cap) =
        construct_trace_storage_from_remote_witness_data(
            trace_layout.clone(),
            used_lde_degree,
            proof_config.fri_lde_factor,
            domain_size,
            setup,
            &external_witness_data,
            &cs.lookup_parameters,
            worker,
        )?;
    synchronize_streams()?;
    println!("on device variable assignment takes {:?}", start.elapsed());
    let num_public_inputs = external_witness_data.public_inputs_locations.len();
    // assert!(num_public_inputs > 0); // TODO
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
    gpu_prove_from_trace::<_, TR, _, NoPow, _>(
        cs,
        raw_trace,
        monomial_trace,
        subtrees,
        trace_tree_cap,
        public_inputs_with_locations,
        setup,
        proof_config,
        vk,
        transcript_params,
        worker,
    )
}
// allocate both hints and result through same allocator
pub fn materialize_variable_cols_from_hints<
    P: boojum::field::traits::field_like::PrimeFieldLikeVectorized<Base = F>,
    A: GoodAllocator,
>(
    cs: &CSReferenceAssembly<F, P, ProvingCSConfig>,
    hints: &[Vec<u32, A>],
    worker: &Worker,
) -> Vec<Vec<F, A>> {
    let domain_size = cs.max_trace_len;
    assert!(domain_size.is_power_of_two());
    let num_cols = hints.len();
    // assume that dag is resolved;
    use boojum::dag::WitnessSource;
    let vars_storage = &cs.variables_storage.read().unwrap();
    assert!(vars_storage.try_get_value(Place::placeholder()).is_some());

    let mut result: Vec<_> = (0..num_cols)
        .map(|_| {
            let mut col = Vec::with_capacity_in(domain_size, A::default());
            col.resize(domain_size, F::ZERO);
            col
        })
        .collect();
    worker.scope(result.len(), |scope, chunk_size| {
        for (vars_chunk, polys_chunk) in hints.chunks(chunk_size).zip(result.chunks_mut(chunk_size))
        {
            scope.spawn(move |_| {
                debug_assert_eq!(vars_chunk.len(), polys_chunk.len());
                for (vars_column, poly) in vars_chunk.iter().zip(polys_chunk.iter_mut()) {
                    unsafe {
                        poly.set_len(domain_size);
                    }
                    for (var, dst) in vars_column.iter().zip(poly.iter_mut()) {
                        if var & PACKED_PLACEHOLDER_BITMASK == 0 {
                            let place =
                                Place::from_variable(Variable::from_variable_index(*var as u64));
                            *dst = vars_storage.get_value_unchecked(place);
                        } else {
                            // we can use 0 as a substitue for all undefined variables,
                            // or add ZK into them
                        }
                    }
                }
            });
        }
    });
    result
}

pub fn materialzie_witness_cols_from_hints<
    P: boojum::field::traits::field_like::PrimeFieldLikeVectorized<Base = F>,
    A: GoodAllocator,
>(
    cs: &CSReferenceAssembly<F, P, ProvingCSConfig>,
    hints: &[Vec<u32, A>],
    worker: &Worker,
) -> Vec<Vec<F, A>> {
    let domain_size = cs.max_trace_len;
    assert!(domain_size.is_power_of_two());
    // assume that dag is resolved;
    use boojum::dag::WitnessSource;
    let vars_storage = &cs.variables_storage.read().unwrap();
    assert!(vars_storage.try_get_value(Place::placeholder()).is_some());
    let num_cols = hints.len();
    let mut result: Vec<_> = (0..num_cols)
        .map(|_| {
            let mut col = Vec::with_capacity_in(domain_size, A::default());
            col.resize(domain_size, F::ZERO); // TODO: unsafe set_len?
            col
        })
        .collect();
    worker.scope(result.len(), |scope, chunk_size| {
        for (vars_chunk, polys_chunk) in hints.chunks(chunk_size).zip(result.chunks_mut(chunk_size))
        {
            scope.spawn(move |_| {
                debug_assert_eq!(vars_chunk.len(), polys_chunk.len());
                for (vars_column, poly) in vars_chunk.iter().zip(polys_chunk.iter_mut()) {
                    unsafe {
                        poly.set_len(domain_size);
                    }
                    for (var, dst) in vars_column.iter().zip(poly.iter_mut()) {
                        if var & PACKED_PLACEHOLDER_BITMASK == 0 {
                            let place =
                                Place::from_witness(Witness::from_witness_index(*var as u64));
                            *dst = vars_storage.get_value_unchecked(place);
                        } else {
                            // we can use 0 as a substitue for all undefined variables,
                            // or add ZK into them
                        }
                    }
                }
            });
        }
    });

    result
}

pub fn materialize_multiplicities_polynomials<
    P: boojum::field::traits::field_like::PrimeFieldLikeVectorized<Base = F>,
    A: GoodAllocator,
>(
    cs: &CSReferenceAssembly<F, P, ProvingCSConfig>,
    worker: &Worker,
) -> Vec<Vec<F, A>> {
    if cs.lookup_parameters == LookupParameters::NoLookup {
        return vec![];
    }
    let domain_size = cs.max_trace_len;
    assert!(domain_size.is_power_of_two());
    let flattening_iter = cs.lookup_multiplicities.iter().flat_map(|el| el.iter());
    let mut result: Vec<_> = (0..cs.num_multipicities_polys())
        .map(|_| {
            let mut col = Vec::with_capacity_in(domain_size, A::default());
            col.resize(domain_size, F::ZERO);
            col
        })
        .collect();
    for (idx, dst) in result.iter_mut().enumerate() {
        let num_to_skip = idx * cs.max_trace_len;
        let src_it = flattening_iter.clone().skip(num_to_skip);

        worker.scope(dst.len(), |scope, chunk_size| {
            for (idx, dst) in dst.chunks_mut(chunk_size).enumerate() {
                let src = src_it.clone().skip(idx * chunk_size);
                scope.spawn(move |_| {
                    for (dst, src) in dst.iter_mut().zip(src) {
                        *dst = F::from_u64_unchecked(AtomicU32::load(
                            src,
                            std::sync::atomic::Ordering::SeqCst,
                        ) as u64);
                    }
                });
            }
        });
    }

    result
}

pub fn gpu_prove<
    P: boojum::field::traits::field_like::PrimeFieldLikeVectorized<Base = F>,
    TR: Transcript<F, CompatibleCap = [F; 4]>,
    H: TreeHasher<F, Output = TR::CompatibleCap>,
    POW: PoWRunner,
    A: GoodAllocator,
>(
    cs: &mut CSReferenceAssembly<F, P, ProvingCSConfig>,
    proof_config: ProofConfig,
    setup: &GpuSetup<A>,
    vk: &VerificationKey<F, H>,
    transcript_params: TR::TransciptParameters,
    worker: &Worker,
) -> CudaResult<GpuProof<A>> {
    unsafe {
        assert!(
            _cuda_context.is_some(),
            "prover context should be initialized"
        )
    };

    cs.wait_for_witness();
    use boojum::dag::WitnessSource;
    let time = std::time::Instant::now();
    assert!(cs.next_available_place_idx() > 0, "CS shouldn't be empty");
    let domain_size = cs.max_trace_len;
    assert!(domain_size.is_power_of_two());
    let h_variables_cols = materialize_variable_cols_from_hints(&cs, &setup.variables_hint, worker);
    let h_witnesses_cols = materialzie_witness_cols_from_hints(&cs, &setup.witnesses_hint, worker);
    let h_multiplicities = materialize_multiplicities_polynomials(&cs, worker);
    println!(
        "materialization of {} trace cols are done on host in {:?} with {} cores",
        h_variables_cols.len() + h_witnesses_cols.len() + h_multiplicities.len(),
        time.elapsed(),
        worker.num_cores
    );
    let quotient_degree = compute_quotient_degree(&cs, &setup.selectors_placement);
    let used_lde_degree = std::cmp::max(proof_config.fri_lde_factor, quotient_degree);
    let (raw_trace, monomial_trace, subtrees, trace_tree_cap) =
        construct_trace_storage_from_local_witness_data(
            h_variables_cols,
            h_witnesses_cols,
            h_multiplicities,
            used_lde_degree,
            domain_size,
            &proof_config,
        )?;
    let num_public_inputs = cs.public_inputs.len();
    let mut public_inputs_with_locations = Vec::with_capacity(num_public_inputs);
    let vars_storage = &cs.variables_storage.read().unwrap();
    for (col, row) in cs.public_inputs.iter().cloned() {
        let variable_idx = setup.variables_hint[col][row].clone() as u64;
        let place = Place::from_variable(Variable::from_variable_index(variable_idx));
        let value = vars_storage.get_value_unchecked(place);
        public_inputs_with_locations.push((col, row, value));
    }
    gpu_prove_from_trace::<_, TR, _, POW, _>(
        &cs,
        raw_trace,
        monomial_trace,
        subtrees,
        trace_tree_cap,
        public_inputs_with_locations,
        setup,
        proof_config,
        vk,
        transcript_params,
        worker,
    )
}

pub fn compute_quotient_degree<
    P: boojum::field::traits::field_like::PrimeFieldLikeVectorized<Base = F>,
>(
    cs: &CSReferenceAssembly<F, P, ProvingCSConfig>,
    selectors_placement: &TreeNode,
) -> usize {
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

    min_lde_degree_for_gates
}

fn gpu_prove_from_trace<
    P: boojum::field::traits::field_like::PrimeFieldLikeVectorized<Base = F>,
    TR: Transcript<F, CompatibleCap = [F; 4]>,
    H: TreeHasher<F, Output = TR::CompatibleCap>,
    POW: PoWRunner,
    A: GoodAllocator,
>(
    cs: &CSReferenceAssembly<F, P, ProvingCSConfig>,
    raw_trace: GenericTraceStorage<LagrangeBasis>,
    monomial_trace: GenericTraceStorage<MonomialBasis>,
    trace_subtrees: Vec<SubTree>,
    trace_tree_cap: Vec<[F; 4]>,
    public_inputs_with_locations: Vec<(usize, usize, F)>,
    setup_base: &GpuSetup<A>,
    proof_config: ProofConfig,
    vk: &VerificationKey<F, H>,
    transcript_params: TR::TransciptParameters,
    worker: &Worker,
) -> CudaResult<GpuProof<A>> {
    assert!(proof_config.fri_lde_factor.is_power_of_two());
    assert!(proof_config.fri_lde_factor > 1);
    assert!(cs.max_trace_len.is_power_of_two());

    let cap_size = proof_config.merkle_tree_cap_size;
    assert!(cap_size > 0);

    let table_ids_column_idxes = setup_base.table_ids_column_idxes.clone();

    let mut transcript = TR::new(transcript_params);
    transcript.witness_merkle_tree_cap(&vk.setup_merkle_tree_cap);
    for (_, _, value) in public_inputs_with_locations.iter().copied() {
        transcript.witness_field_elements(&[value]);
    }

    let selectors_placement = setup_base.selectors_placement.clone();
    let domain_size = cs.max_trace_len; // counted in elements of P
    assert_eq!(setup_base.constant_columns[0].len(), domain_size);

    let quotient_degree = compute_quotient_degree(&cs, &selectors_placement);
    let fri_lde_degree = proof_config.fri_lde_factor;
    let used_lde_degree = std::cmp::max(fri_lde_degree, quotient_degree);

    // cap size shouldn't be part of subtree
    // Immediately transfer raw trace to the device and simultaneously compute monomials of the whole trace
    // Then compute coset evaluations + cap of the sub tree for each coset
    assert!(cap_size.is_power_of_two());
    assert!(fri_lde_degree.is_power_of_two());
    let coset_cap_size = if cap_size < fri_lde_degree {
        1
    } else {
        assert!(cap_size.is_power_of_two());
        1 << (cap_size.trailing_zeros() - fri_lde_degree.trailing_zeros())
    };

    let time = std::time::Instant::now();
    let trace_layout = raw_trace.layout.clone();
    let TraceLayout {
        num_variable_cols,
        num_witness_cols,
        num_multiplicity_cols,
    } = trace_layout.clone();
    let num_trace_polys = raw_trace.num_polys();
    assert_eq!(monomial_trace.domain_size(), domain_size);
    assert_eq!(monomial_trace.num_polys(), num_trace_polys);
    assert_eq!(raw_trace.domain_size(), domain_size);

    assert_eq!(trace_subtrees.len(), proof_config.fri_lde_factor);
    assert_eq!(trace_tree_cap.len(), proof_config.merkle_tree_cap_size);
    let mut trace_holder =
        TraceCache::from_monomial(monomial_trace, fri_lde_degree, used_lde_degree)?;
    let mut tree_holder = TreeCache::empty(fri_lde_degree);
    tree_holder.set_trace_subtrees(trace_subtrees);
    // TODO: use cuda callback for transcript
    transcript.witness_merkle_tree_cap(&trace_tree_cap);

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
        let lookup_gamma: DExt = h_lookup_gamma.into();

        let lookup_evaluator_id = 0;
        let selector_subpath = setup_base
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
                let columns_per_subargument = cs.lookup_parameters.columns_per_subargument();

                let mut h_powers_of_gamma = vec![];
                let mut current = EF::ONE;
                for _ in 0..columns_per_subargument {
                    h_powers_of_gamma.push(current.into());
                    // Should this be h_gamma or h_lookup_gamma?
                    current.mul_assign(&h_lookup_gamma);
                }
                let mut powers_of_gamma = dvec!(h_powers_of_gamma.len());
                powers_of_gamma.copy_from_slice(&h_powers_of_gamma)?;

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
                powers_of_gamma
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
                    &mut argument_raw_storage,
                )?;

                powers_of_gamma
            }
        };

        (Some(lookup_beta), Some(powers_of_gamma))
    } else {
        (None, None)
    };
    drop(raw_trace);
    let argument_monomial_storage = argument_raw_storage.into_monomial()?;
    let mut argument_holder =
        ArgumentCache::from_monomial(argument_monomial_storage, fri_lde_degree, used_lde_degree)?;
    let (argument_subtrees, argument_tree_cap) = argument_holder.commit::<H>(cap_size)?;

    transcript.witness_merkle_tree_cap(&argument_tree_cap);
    tree_holder.set_argument_subtrees(argument_subtrees);

    let h_alpha = transcript.get_multiple_challenges_fixed::<2>();
    let h_alpha = ExtensionField::<F, 2, EXT>::from_coeff_in_base(h_alpha);
    let alpha: DExt = h_alpha.into();

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
            fri_lde_degree,
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
            coset_idx,
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
        final_expected_degree,
    ) = boojum::cs::implementations::prover::compute_fri_schedule(
        proof_config.security_level as u32,
        cap_size,
        basic_pow_bits,
        fri_lde_degree.trailing_zeros(),
        domain_size.trailing_zeros(),
    );

    let first_codeword = unsafe {
        // use deep quotient storage as adjacent full storage as usual
        let (ptr, len, cap, allocator) = deep_quotient
            .c0
            .storage
            .into_inner()
            .into_raw_parts_with_alloc();
        debug_assert_eq!(len, fri_lde_degree * domain_size);
        let storage = DVec::from_raw_parts_in(ptr, 2 * len, 2 * cap, allocator);
        CodeWord::new_base_assuming_adjacent(storage, fri_lde_degree)
    };
    dbg!(first_codeword.length());

    let (mut fri_holder, final_fri_monomials) = compute_fri::<_, A>(
        first_codeword,
        &mut transcript,
        folding_schedule.clone(),
        fri_lde_degree,
        cap_size,
        worker,
    )?;
    assert_eq!(final_fri_monomials[0].len(), final_expected_degree);
    assert_eq!(final_fri_monomials[1].len(), final_expected_degree);
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
    let mut query_idx_and_coset_idx_map = vec![0usize; num_queries];
    for query_idx in 0..num_queries {
        let query_index_lsb_first_bits = bools_buffer.get_bits(&mut transcript, max_needed_bits);
        // we consider it to be some convenient for us encoding of coset + inner index.

        let inner_idx =
            u64_from_lsb_first_bits(&query_index_lsb_first_bits[0..num_bits_for_in_coset_index])
                as u32;
        let coset_idx =
            u64_from_lsb_first_bits(&query_index_lsb_first_bits[num_bits_for_in_coset_index..])
                as usize;
        assert!(inner_idx < u32::MAX);
        query_details_for_cosets[coset_idx].push(inner_idx);
        query_idx_and_coset_idx_map[query_idx] = coset_idx;
    }

    let mut d_query_details = vec![];
    for queries_for_coset in query_details_for_cosets.iter().cloned() {
        let mut d_queries_for_coset = svec!(queries_for_coset.len());
        mem::h2d(&queries_for_coset, &mut d_queries_for_coset)?;
        d_query_details.push(d_queries_for_coset);
    }

    //our typical FRI schedule is  [3,3,3,3,3,2]
    let effective_query_indexes =
        compute_effective_indexes_for_fri_layers(&fri_holder, &query_details_for_cosets)?;

    let public_inputs: Vec<_> = public_inputs_with_locations
        .iter()
        .cloned()
        .map(|i| i.2)
        .collect();

    let mut gpu_proof = GpuProof::<A>::allocate(
        trace_layout.num_polys(),
        setup_holder.num_polys(),
        // num polys is double here because argument and quotient polys have their elements in the extension field
        argument_holder.num_polys_in_base(),
        quotient_holder.num_polys_in_base(),
        domain_size,
        proof_config,
        cs.lookup_parameters.clone(),
        num_queries,
        query_details_for_cosets.clone(),
        query_idx_and_coset_idx_map,
        folding_schedule.clone(),
        public_inputs.clone(),
        pow_challenge,
        h_evaluations_at_z.clone(),
        h_evaluations_at_z_omega.clone(),
        h_evaluations_at_zero.clone(),
    );

    // TODO: consider to do setup query on the host
    let (setup_subtrees, setup_tree_cap) = setup_holder.commit::<H>(cap_size)?;
    assert_eq!(setup_base.setup_tree.get_cap(), setup_tree_cap);
    tree_holder.setup_setup_subtrees(setup_subtrees);
    // tree_holder.set_setup_tree_from_host_data(&setup_base.setup_tree);

    for (coset_idx, indexes_for_coset) in d_query_details.iter().enumerate() {
        let num_queries_for_coset = indexes_for_coset.len();
        trace_holder.batch_query_for_coset::<H, A>(
            coset_idx,
            indexes_for_coset,
            num_queries_for_coset,
            domain_size,
            &mut gpu_proof.witness_all_leaf_elems[coset_idx],
            &mut gpu_proof.witness_all_proofs[coset_idx],
            &tree_holder,
        )?;

        argument_holder.batch_query_for_coset::<H, A>(
            coset_idx,
            &indexes_for_coset,
            num_queries_for_coset,
            domain_size,
            &mut gpu_proof.stage_2_all_leaf_elems[coset_idx],
            &mut gpu_proof.stage_2_all_proofs[coset_idx],
            &tree_holder,
        )?;

        quotient_holder.batch_query_for_coset::<H, A>(
            coset_idx,
            &indexes_for_coset,
            num_queries_for_coset,
            domain_size,
            &mut gpu_proof.quotient_all_leaf_elems[coset_idx],
            &mut gpu_proof.quotient_all_proofs[coset_idx],
            &tree_holder,
        )?;

        setup_holder.batch_query::<H, A>(
            coset_idx,
            &indexes_for_coset,
            num_queries_for_coset,
            domain_size,
            &mut gpu_proof.setup_all_leaf_elems[coset_idx],
            &mut gpu_proof.setup_all_proofs[coset_idx],
            &tree_holder,
        )?;
    }
    // FIXME: query FRI oracles
    let mut domain_size_for_fri_layers = fri_lde_degree * domain_size;
    for (layer_idx, schedule) in folding_schedule.iter().enumerate() {
        for coset_idx in 0..fri_lde_degree {
            if layer_idx == 0 {
                fri_holder.base_oracle_batch_query::<H, A>(
                    layer_idx,
                    &effective_query_indexes[0][coset_idx],
                    effective_query_indexes[0][coset_idx].len(),
                    domain_size_for_fri_layers,
                    &mut gpu_proof.fri_base_oracle_leaf_elems[coset_idx],
                    &mut gpu_proof.fri_base_oracle_proofs[coset_idx],
                )?;
            } else {
                fri_holder.intermediate_oracle_batch_query::<H, A>(
                    layer_idx,
                    &effective_query_indexes[layer_idx][coset_idx],
                    effective_query_indexes[layer_idx][coset_idx].len(),
                    domain_size_for_fri_layers,
                    &mut gpu_proof.fri_intermediate_oracles_leaf_elems[layer_idx - 1][coset_idx],
                    &mut gpu_proof.fri_intermediate_oracles_proofs[layer_idx - 1][coset_idx],
                )?;
            }
        }
        domain_size_for_fri_layers >>= schedule;
    }

    synchronize_streams()?;
    println!("FRI Queries are done {:?}", time.elapsed());

    gpu_proof.public_inputs = public_inputs;
    gpu_proof.witness_oracle_cap = trace_tree_cap;
    gpu_proof.stage_2_oracle_cap = argument_tree_cap;
    gpu_proof.quotient_oracle_cap = quotient_tree_cap;
    gpu_proof.setup_oracle_cap = setup_base.setup_tree.get_cap();
    gpu_proof.fri_base_oracle_cap = fri_holder.base_oracle.get_tree_cap()?;
    gpu_proof.fri_intermediate_oracles_caps = fri_holder
        .intermediate_oracles
        .into_iter()
        .map(|o| o.get_tree_cap().expect("fri oracle cap"))
        .collect();
    gpu_proof.final_fri_monomials = final_fri_monomials;

    Ok((gpu_proof))
}

pub struct GpuProof<A: GoodAllocator> {
    proof_config: ProofConfig,
    public_inputs: Vec<F>,
    pow_challenge: u64,
    domain_size: usize,
    num_trace_polys: usize,
    num_setup_polys: usize,
    num_arguments_polys: usize,
    num_quotient_polys: usize,

    query_details: Vec<Vec<u32>>,
    query_map: Vec<usize>,
    witness_oracle_cap: Vec<[F; 4]>,
    // only inner vectors needs to be located in the pinned memory
    witness_all_leaf_elems: Vec<Vec<F, A>>,
    witness_all_proofs: Vec<Vec<F, A>>,

    stage_2_oracle_cap: Vec<[F; 4]>,
    stage_2_all_leaf_elems: Vec<Vec<F, A>>,
    stage_2_all_proofs: Vec<Vec<F, A>>,

    quotient_oracle_cap: Vec<[F; 4]>,
    quotient_all_leaf_elems: Vec<Vec<F, A>>,
    quotient_all_proofs: Vec<Vec<F, A>>,

    setup_oracle_cap: Vec<[F; 4]>,
    setup_all_leaf_elems: Vec<Vec<F, A>>,
    setup_all_proofs: Vec<Vec<F, A>>,

    fri_base_oracle_cap: Vec<[F; 4]>,
    fri_base_oracle_leaf_elems: Vec<Vec<F, A>>,
    fri_base_oracle_proofs: Vec<Vec<F, A>>,

    fri_intermediate_oracles_caps: Vec<Vec<[F; 4]>>,
    fri_intermediate_oracles_leaf_elems: Vec<Vec<Vec<F, A>>>,
    fri_intermediate_oracles_proofs: Vec<Vec<Vec<F, A>>>,
    fri_folding_schedule: Vec<usize>,
    // last monomials doesn't need to be allocated on the pinned memory
    // since intermediate transfer uses pinned then global allocator.
    final_fri_monomials: [Vec<F>; 2],

    values_at_z: Vec<EF, A>,
    values_at_z_omega: Vec<EF, A>,
    values_at_z_zero: Vec<EF, A>,
}

impl<A: GoodAllocator> GpuProof<A> {
    pub fn allocate(
        num_trace_polys: usize,
        num_setup_polys: usize,
        num_arguments_polys: usize,
        num_quotient_polys: usize,
        domain_size: usize,
        proof_config: ProofConfig,
        lookup_params: LookupParameters,
        num_queries: usize,
        query_details: Vec<Vec<u32>>,
        query_map: Vec<usize>,
        folding_schedule: Vec<usize>,
        public_inputs: Vec<F>,
        pow_challenge: u64,
        values_at_z: Vec<EF, A>,
        values_at_z_omega: Vec<EF, A>,
        values_at_z_zero: Vec<EF, A>,
    ) -> Self {
        let ProofConfig {
            fri_lde_factor,
            merkle_tree_cap_size,
            fri_folding_schedule,
            security_level,
            pow_bits,
        } = proof_config.clone();

        assert!(
            fri_folding_schedule.is_none(),
            "we do not yet support externally provided FRI schedule"
        );

        assert!(domain_size.is_power_of_two());
        assert!(values_at_z.len() > 0);
        assert!(values_at_z_omega.len() > 0);
        assert_eq!(query_details.len(), fri_lde_factor);
        if lookup_params.lookup_is_allowed() {
            assert!(values_at_z_zero.len() > 0);
        }

        assert_eq!(
            num_queries,
            query_details.iter().map(|coset| coset.len()).sum()
        );
        let typical_capacity_for_merkle_caps = merkle_tree_cap_size;
        // it is guaranteed that number of leaf elems is the largest in the witness tree than other trees
        let typical_capacity_for_leaf = num_queries * num_trace_polys;
        let typical_capacity_for_proof =
            num_queries * NUM_EL_PER_HASH * domain_size.trailing_zeros() as usize;
        let num_fri_layers = folding_schedule.len();
        let mut proof = GpuProof {
            proof_config,
            public_inputs,
            pow_challenge,
            domain_size,
            num_trace_polys,
            num_setup_polys,
            num_arguments_polys,
            num_quotient_polys,
            query_details,
            query_map,
            witness_oracle_cap: Vec::with_capacity(typical_capacity_for_merkle_caps),
            stage_2_oracle_cap: Vec::with_capacity(typical_capacity_for_merkle_caps),
            quotient_oracle_cap: Vec::with_capacity(typical_capacity_for_merkle_caps),
            setup_oracle_cap: Vec::with_capacity(typical_capacity_for_merkle_caps),
            fri_base_oracle_cap: Vec::with_capacity(typical_capacity_for_merkle_caps),
            fri_intermediate_oracles_caps: vec![
                Vec::with_capacity(typical_capacity_for_merkle_caps);
                num_fri_layers
            ],
            witness_all_leaf_elems: vec![],
            witness_all_proofs: vec![],
            stage_2_all_leaf_elems: vec![],
            stage_2_all_proofs: vec![],
            quotient_all_leaf_elems: vec![],
            quotient_all_proofs: vec![],
            setup_all_leaf_elems: vec![],
            setup_all_proofs: vec![],
            fri_base_oracle_leaf_elems: vec![],
            fri_base_oracle_proofs: vec![],
            fri_intermediate_oracles_leaf_elems: vec![],
            fri_intermediate_oracles_proofs: vec![],
            fri_folding_schedule: folding_schedule.clone(),
            values_at_z,
            values_at_z_omega,
            values_at_z_zero,
            final_fri_monomials: [vec![], vec![]],
        };
        proof
            .fri_intermediate_oracles_leaf_elems
            .resize(folding_schedule.len(), vec![]);
        proof
            .fri_intermediate_oracles_proofs
            .resize(folding_schedule.len(), vec![]);

        for _ in 0..fri_lde_factor {
            proof.witness_all_leaf_elems.push(Vec::with_capacity_in(
                typical_capacity_for_leaf,
                A::default(),
            ));
            proof.witness_all_proofs.push(Vec::with_capacity_in(
                typical_capacity_for_proof,
                A::default(),
            ));
            proof.stage_2_all_leaf_elems.push(Vec::with_capacity_in(
                typical_capacity_for_leaf,
                A::default(),
            ));
            proof.stage_2_all_proofs.push(Vec::with_capacity_in(
                typical_capacity_for_proof,
                A::default(),
            ));
            proof.quotient_all_leaf_elems.push(Vec::with_capacity_in(
                typical_capacity_for_leaf,
                A::default(),
            ));
            proof.quotient_all_proofs.push(Vec::with_capacity_in(
                typical_capacity_for_leaf,
                A::default(),
            ));
            proof.setup_all_leaf_elems.push(Vec::with_capacity_in(
                typical_capacity_for_leaf,
                A::default(),
            ));
            proof.setup_all_proofs.push(Vec::with_capacity_in(
                typical_capacity_for_leaf,
                A::default(),
            ));
            proof.fri_base_oracle_leaf_elems.push(Vec::with_capacity_in(
                typical_capacity_for_leaf,
                A::default(),
            ));
            proof.fri_base_oracle_proofs.push(Vec::with_capacity_in(
                typical_capacity_for_leaf,
                A::default(),
            ));
            for (layer_idx, _) in folding_schedule.iter().enumerate() {
                proof.fri_intermediate_oracles_leaf_elems[layer_idx].push(Vec::with_capacity_in(
                    typical_capacity_for_leaf,
                    A::default(),
                ));
                proof.fri_intermediate_oracles_proofs[layer_idx].push(Vec::with_capacity_in(
                    typical_capacity_for_leaf,
                    A::default(),
                ));
            }
        }

        proof
    }
}

impl<A: GoodAllocator> Into<Proof<F, DefaultTreeHasher, EXT>> for GpuProof<A> {
    fn into(self) -> Proof<F, DefaultTreeHasher, EXT> {
        let GpuProof {
            proof_config,
            public_inputs,
            pow_challenge,
            query_details,
            query_map,
            num_trace_polys,
            num_setup_polys,
            num_arguments_polys,
            num_quotient_polys,
            witness_oracle_cap,
            witness_all_leaf_elems,
            witness_all_proofs,
            stage_2_oracle_cap,
            stage_2_all_leaf_elems,
            stage_2_all_proofs,
            quotient_oracle_cap,
            quotient_all_leaf_elems,
            quotient_all_proofs,
            setup_oracle_cap,
            setup_all_leaf_elems,
            setup_all_proofs,
            fri_base_oracle_cap,
            fri_base_oracle_leaf_elems,
            fri_base_oracle_proofs,
            fri_intermediate_oracles_caps,
            fri_intermediate_oracles_leaf_elems,
            fri_intermediate_oracles_proofs,
            fri_folding_schedule,
            values_at_z,
            values_at_z_omega,
            values_at_z_zero,
            final_fri_monomials,
            domain_size,
        } = self;
        let mut num_queries = vec![];
        for indexes in query_details.iter() {
            num_queries.push(indexes.len());
        }
        let mut offsets_for_query_in_coset = vec![0; proof_config.fri_lde_factor];
        let mut queries_per_fri_repetition = vec![];
        let coset_cap_size = coset_cap_size(
            proof_config.merkle_tree_cap_size,
            proof_config.fri_lde_factor,
        );
        assert!(coset_cap_size.is_power_of_two());

        for coset_idx in query_map.into_iter() {
            assert!(coset_idx < proof_config.fri_lde_factor);
            let num_queries_in_coset = num_queries[coset_idx];
            let query_idx_in_coset = offsets_for_query_in_coset[coset_idx];
            // witnesses
            let witness_query = construct_single_query_from_flattened_batch_sources(
                &witness_all_leaf_elems[coset_idx],
                &witness_all_proofs[coset_idx],
                coset_cap_size,
                num_queries_in_coset,
                query_idx_in_coset,
                num_trace_polys,
                1,
                domain_size,
            );

            // arguments
            let stage_2_query = construct_single_query_from_flattened_batch_sources(
                &stage_2_all_leaf_elems[coset_idx],
                &stage_2_all_proofs[coset_idx],
                coset_cap_size,
                num_queries_in_coset,
                query_idx_in_coset,
                num_arguments_polys,
                1,
                domain_size,
            );

            // quotient
            let quotient_query = construct_single_query_from_flattened_batch_sources(
                &quotient_all_leaf_elems[coset_idx],
                &quotient_all_proofs[coset_idx],
                coset_cap_size,
                num_queries_in_coset,
                query_idx_in_coset,
                num_quotient_polys,
                1,
                domain_size,
            );

            // setup
            let setup_query = construct_single_query_from_flattened_batch_sources(
                &setup_all_leaf_elems[coset_idx],
                &setup_all_proofs[coset_idx],
                coset_cap_size,
                num_queries_in_coset,
                query_idx_in_coset,
                num_setup_polys,
                1,
                domain_size,
            );

            // fri
            let mut domain_size_for_fri = proof_config.fri_lde_factor * domain_size;
            let mut fri_queries = vec![];
            for (layer_idx, schedule) in fri_folding_schedule.iter().enumerate() {
                let num_els_per_leaf = 1 << schedule;
                let fri_query = if layer_idx == 0 {
                    construct_single_query_from_flattened_batch_sources(
                        &fri_base_oracle_leaf_elems[coset_idx],
                        &fri_base_oracle_proofs[coset_idx],
                        proof_config.merkle_tree_cap_size,
                        num_queries_in_coset,
                        query_idx_in_coset,
                        2,
                        num_els_per_leaf,
                        domain_size_for_fri,
                    )
                } else {
                    let layer_idx = layer_idx - 1;
                    construct_single_query_from_flattened_batch_sources(
                        &fri_intermediate_oracles_leaf_elems[layer_idx][coset_idx],
                        &fri_intermediate_oracles_proofs[layer_idx][coset_idx],
                        proof_config.merkle_tree_cap_size,
                        num_queries_in_coset,
                        query_idx_in_coset,
                        2,
                        num_els_per_leaf,
                        domain_size_for_fri,
                    )
                };
                fri_queries.push(fri_query);
                domain_size_for_fri >>= schedule
            }
            assert_eq!(fri_queries.len(), fri_folding_schedule.len());

            let round_queries = SingleRoundQueries {
                witness_query,
                stage_2_query,
                quotient_query,
                setup_query,
                fri_queries,
            };

            queries_per_fri_repetition.push(round_queries);
            offsets_for_query_in_coset[coset_idx] += 1;
        }

        let proof = Proof::<F, DefaultTreeHasher, EXT> {
            public_inputs: public_inputs.clone(),
            witness_oracle_cap: witness_oracle_cap,
            stage_2_oracle_cap: stage_2_oracle_cap,
            quotient_oracle_cap: quotient_oracle_cap,
            final_fri_monomials,
            values_at_z: values_at_z.to_vec(),
            values_at_z_omega: values_at_z_omega.to_vec(),
            values_at_0: values_at_z_zero.to_vec(),
            pow_challenge,
            fri_base_oracle_cap: fri_base_oracle_cap,
            fri_intermediate_oracles_caps,
            queries_per_fri_repetition,

            _marker: std::marker::PhantomData,
            proof_config,
        };

        proof
    }
}

fn construct_single_query_from_flattened_batch_sources(
    leaf_elems: &[F],
    proof_elems: &[F],
    cap_size: usize,
    num_queries: usize,
    query_idx_in_coset: usize,
    num_cols: usize,
    num_elems_per_leaf: usize,
    domain_size: usize,
) -> OracleQuery<F, DefaultTreeHasher> {
    let leaf_elements = construct_single_query_for_leaf_source_from_batch_sources(
        leaf_elems,
        num_queries,
        query_idx_in_coset,
        num_cols,
        num_elems_per_leaf,
    );

    let proof = construct_single_query_for_merkle_path_from_batch_sources(
        proof_elems,
        cap_size,
        num_queries,
        query_idx_in_coset,
        num_elems_per_leaf,
        domain_size,
    );

    let mut query: OracleQuery<F, DefaultTreeHasher> = OracleQuery {
        leaf_elements,
        proof,
    };

    query
}

pub(crate) fn construct_single_query_for_leaf_source_from_batch_sources(
    leaf_elems: &[F],
    num_queries: usize,
    query_idx: usize,
    num_cols: usize,
    num_elems_per_leaf: usize,
) -> Vec<F> {
    assert!(!leaf_elems.is_empty());
    assert_eq!(leaf_elems.len() % (num_queries * num_elems_per_leaf), 0);
    let chunk_size = num_elems_per_leaf * num_queries;
    let leaf_chunks = leaf_elems.chunks(chunk_size);
    assert_eq!(leaf_chunks.len(), num_cols);

    let mut result = Vec::with_capacity(num_elems_per_leaf * num_cols);
    for chunk in leaf_chunks {
        assert_eq!(chunk.len(), chunk_size);
        let start = query_idx * num_elems_per_leaf;
        let end = start + num_elems_per_leaf;
        assert!(end <= chunk_size);
        result.extend_from_slice(&chunk[start..end]);
    }
    assert_eq!(result.len(), result.capacity());
    result
}

pub(crate) fn construct_single_query_for_merkle_path_from_batch_sources(
    proof_elems: &[F],
    cap_size: usize,
    num_queries: usize,
    query_idx: usize,
    num_elems_per_leaf: usize,
    domain_size: usize,
) -> Vec<[F; 4]> {
    assert_eq!(proof_elems.len() % num_queries * NUM_EL_PER_HASH, 0);
    assert!(domain_size.is_power_of_two());
    assert!(num_elems_per_leaf.is_power_of_two());
    let num_leafs = domain_size / num_elems_per_leaf;
    let num_layers = (num_leafs.trailing_zeros() - cap_size.trailing_zeros()) as usize;
    assert_eq!(
        proof_elems.len(),
        num_queries * num_layers * NUM_EL_PER_HASH
    );
    let mut result = Vec::with_capacity(num_layers);
    for layer in proof_elems.chunks(num_queries * NUM_EL_PER_HASH) {
        let mut tmp = [F::ZERO; NUM_EL_PER_HASH];
        for col_idx in 0..NUM_EL_PER_HASH {
            let idx = col_idx * num_queries + query_idx;
            tmp[col_idx] = layer[idx];
        }
        result.push(tmp);
    }
    assert_eq!(result.len(), result.capacity());

    result
}

pub(crate) fn u64_from_lsb_first_bits(bits: &[bool]) -> u64 {
    let mut result = 0u64;
    for (shift, bit) in bits.iter().enumerate() {
        result |= (*bit as u64) << shift;
    }

    result
}

pub fn compute_evaluations_over_lagrange_basis<'a, A: GoodAllocator>(
    trace_holder: &mut TraceCache,
    setup_holder: &mut SetupCache,
    argument_holder: &mut ArgumentCache<'a>,
    quotient_holder: &mut QuotientCache<'a>,
    z: EF,
    z_omega: EF,
    lde_degree: usize,
) -> CudaResult<(Vec<EF, A>, Vec<EF, A>, Vec<EF, A>)> {
    // all polynomials should be opened at "z"
    // additionally, copy permutation polynomial should be opened at "z*w"
    // lookup polynomials should be opened at "0"
    // we should follow order of the evaluations as in the reference impl
    let trace_storage = trace_holder.get_or_compute_coset_evals(0)?;
    let setup_storage = setup_holder.get_or_compute_coset_evals(0)?;
    let domain_size = trace_storage.domain_size();
    let precomputed_bases_for_z = PrecomputedBasisForBarycentric::precompute(domain_size, z)?;

    let trace_evals_at_z = trace_storage.barycentric_evaluate::<A>(&precomputed_bases_for_z)?;
    let setup_evals_at_z = setup_storage.barycentric_evaluate::<A>(&precomputed_bases_for_z)?;

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
    let num_lookup_polys = lookup_a_polys.len() + lookup_b_polys.len();
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
    let num_trace_polys = trace_holder.num_polys();
    let num_setup_polys = setup_holder.num_polys();
    let num_argument_polys = argument_holder.num_polys();
    let num_quotient_polys = quotient_holder.num_polys();
    let num_all_polys_at_z = trace_holder.num_polys()
        + setup_holder.num_polys()
        + argument_holder.num_polys()
        + quotient_holder.num_polys();
    let mut polynomials_at_z = Vec::with_capacity_in(num_all_polys_at_z, A::default());
    let mut polynomials_at_z_omega = Vec::with_capacity_in(1, A::default());
    let mut polynomials_at_zero = Vec::with_capacity_in(argument_holder.num_polys(), A::default());

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
    coset_idx: usize,
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

    let (
        maybe_multiplicity_cols,
        maybe_lookup_a_polys,
        maybe_lookup_b_polys,
        maybe_table_cols,
        maybe_denom_at_zero,
    ) = if argument_polys.lookup_a_polys.len() > 0 {
        (
            Some(&trace_polys.multiplicity_cols),
            Some(argument_polys.lookup_a_polys),
            Some(argument_polys.lookup_b_polys),
            Some(&setup_polys.table_cols),
            Some(compute_denom_at_base_point(&roots, &DF::zero()?)?),
        )
    } else {
        (None, None, None, None, None)
    };

    let maybe_witness_cols = if trace_polys.witness_cols.len() > 0 {
        Some(&trace_polys.witness_cols)
    } else {
        None
    };

    let num_public_inputs: usize = public_input_opening_tuples
        .iter()
        .map(|(open_at, set)| set.len())
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
        &maybe_denom_at_zero,
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
            .copy_from_device_slice(&denom_at_point.storage.as_ref());
        quotient_for_row.mul_assign(&denom_at_point_ext)?;
        quotient.add_assign(&quotient_for_row)?;
    }

    Ok(quotient)
}
