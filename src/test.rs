use crate::cs::{materialize_permutation_cols_from_indexes_into, GpuSetup};

use super::*;
use std::{path::Path, sync::Arc};

use boojum::cs::{implementations::polynomial_storage::SetupBaseStorage, Variable};
use boojum::{
    config::{CSConfig, CSSetupConfig, DevCSConfig, ProvingCSConfig, SetupCSConfig},
    cs::{
        gates::{
            ConstantAllocatableCS, ConstantsAllocatorGate, FmaGateInBaseFieldWithoutConstant,
            NopGate, PublicInputGate, ReductionGate,
        },
        implementations::{
            pow::NoPow, proof::Proof, prover::ProofConfig, reference_cs::CSReferenceAssembly,
            setup::FinalizationHintsForProver,
        },
        oracle::merkle_tree::MerkleTreeWithCap,
        traits::{cs::ConstraintSystem, gate::GatePlacementStrategy},
        CSGeometry,
    },
    field::{goldilocks::GoldilocksExt2, U64Representable},
    gadgets::{
        sha256::sha256,
        tables::{
            ch4::{create_ch4_table, Ch4Table},
            chunk4bits::{create_4bit_chunk_split_table, Split4BitChunkTable},
            maj4::{create_maj4_table, Maj4Table},
            trixor4::{create_tri_xor_table, TriXor4Table},
        },
        traits::{
            round_function::{BuildableCircuitRoundFunction, CircuitRoundFunction},
            witnessable::WitnessHookable,
        },
        u8::UInt8,
    },
    implementations::poseidon2::Poseidon2Goldilocks,
    worker::Worker,
};

use boojum::field::traits::field_like::PrimeFieldLikeVectorized;

#[allow(dead_code)]
pub type DefaultDevCS = CSReferenceAssembly<F, F, DevCSConfig>;
type P = F;
use crate::gpu_proof_config::GpuProofConfig;
use serial_test::serial;

#[serial]
#[test]
#[ignore]
fn test_proof_comparison_for_poseidon_gate_with_private_witnesses() {
    let (setup_cs, finalization_hint) =
        init_or_synth_cs_with_poseidon2_and_private_witnesses::<SetupCSConfig, true>(None);
    let worker = Worker::new();
    let prover_config = init_proof_cfg();
    let (setup_base, setup, vk, setup_tree, vars_hint, wits_hint) = setup_cs.get_full_setup(
        &worker,
        prover_config.fri_lde_factor,
        prover_config.merkle_tree_cap_size,
    );
    let domain_size = setup_cs.max_trace_len;
    let _ctx = ProverContext::dev(domain_size).expect("init gpu prover context");
    let gpu_setup = GpuSetup::<Global>::from_setup_and_hints(
        setup_base.clone(),
        clone_reference_tree(&setup_tree),
        vars_hint.clone(),
        wits_hint.clone(),
        &worker,
    )
    .unwrap();

    assert!(domain_size.is_power_of_two());
    let actual_proof = {
        let (proving_cs, _) = init_or_synth_cs_with_poseidon2_and_private_witnesses::<
            ProvingCSConfig,
            true,
        >(finalization_hint.as_ref());
        let witness = proving_cs.witness.unwrap();
        let (reusable_cs, _) = init_or_synth_cs_with_poseidon2_and_private_witnesses::<
            ProvingCSConfig,
            false,
        >(finalization_hint.as_ref());
        let config = GpuProofConfig::from_assembly(&reusable_cs);
        let proof = gpu_prove_from_external_witness_data::<
            DefaultTranscript,
            DefaultTreeHasher,
            NoPow,
            Global,
        >(
            &config,
            &witness,
            prover_config.clone(),
            &gpu_setup,
            &vk,
            (),
            &worker,
        )
        .expect("gpu proof");

        proof
    };

    let expected_proof = {
        let (proving_cs, _) = init_or_synth_cs_with_poseidon2_and_private_witnesses::<
            ProvingCSConfig,
            true,
        >(finalization_hint.as_ref());
        let worker = Worker::new();
        let prover_config = init_proof_cfg();

        proving_cs.prove_from_precomputations::<GoldilocksExt2, DefaultTranscript, DefaultTreeHasher, NoPow>(
            prover_config,
            &setup_base,
            &setup,
            &setup_tree,
            &vk,
            &vars_hint,
            &wits_hint,
            (),
            &worker,
        )
    };
    let actual_proof = actual_proof.into();
    compare_proofs(&expected_proof, &actual_proof);
}

fn init_or_synth_cs_with_poseidon2_and_private_witnesses<CFG: CSConfig, const DO_SYNTH: bool>(
    finalization_hint: Option<&FinalizationHintsForProver>,
) -> (
    CSReferenceAssembly<F, F, CFG>,
    Option<FinalizationHintsForProver>,
) {
    let geometry = CSGeometry {
        num_columns_under_copy_permutation: 100,
        num_witness_columns: 30,
        num_constant_columns: 4,
        max_allowed_constraint_degree: 4,
    };

    use boojum::cs::cs_builder_reference::*;
    let builder_impl = CsReferenceImplementationBuilder::<F, F, CFG>::new(geometry, 1 << 20);
    use boojum::cs::cs_builder::new_builder;
    let builder = new_builder::<_, F>(builder_impl);

    let builder = Poseidon2Goldilocks::configure_builder(
        builder,
        GatePlacementStrategy::UseGeneralPurposeColumns,
    );
    let builder = ConstantsAllocatorGate::configure_builder(
        builder,
        GatePlacementStrategy::UseGeneralPurposeColumns,
    );
    let builder =
        NopGate::configure_builder(builder, GatePlacementStrategy::UseGeneralPurposeColumns);

    let mut owned_cs = builder.build(1 << 25);
    // quick and dirty way of testing with private witnesses
    fn synthesize<CS: ConstraintSystem<F>>(cs: &mut CS) -> [Variable; 8] {
        use rand::{Rng, SeedableRng};
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        type R = Poseidon2Goldilocks;
        let num_gates = 1 << 16;
        let mut prev_state = [cs.allocate_constant(F::ZERO); 12];
        let _to_keep = [cs.allocate_constant(F::ZERO); 4];
        for _ in 0..num_gates {
            let to_absorb =
                cs.alloc_multiple_variables_from_witnesses([F::from_u64_unchecked(rng.gen()); 8]);
            let to_keep = R::split_capacity_elements(&prev_state);
            prev_state = R::absorb_with_replacement(cs, to_absorb, to_keep);
            prev_state = R::compute_round_function(cs, prev_state);
        }

        Poseidon2Goldilocks::state_into_commitment::<8>(&prev_state)
    }

    if DO_SYNTH {
        let output = synthesize(&mut owned_cs);
        let next_available_row = owned_cs.next_available_row();
        for (column, var) in output.into_iter().enumerate() {
            // TODO: Ask Sait
            // I'm not sure it's ok to add a gate only if we synthesized the witness.
            // This may yield inconsistencies between a fully synthesized cs created
            // with DO_SYNTH=true and a reusable cs created with DO_SYNTH=false.
            // On the other hand, in the "ordinary" zksync circuits I'm fairly sure
            // place_gate, place_variable, and set_public are called during synthesis.
            let gate = PublicInputGate::new(var);
            owned_cs.place_gate(&gate, next_available_row);
            owned_cs.place_variable(var, next_available_row, column);
            owned_cs.set_public(column, next_available_row);
        }
    }

    // imitates control flow of synthesis_utils::init_or_synthesize_assembly
    if <CFG::SetupConfig as CSSetupConfig>::KEEP_SETUP {
        let (_, finalization_hint) = owned_cs.pad_and_shrink();
        (owned_cs.into_assembly(), Some(finalization_hint))
    } else {
        let hint = finalization_hint.unwrap();
        if DO_SYNTH {
            owned_cs.pad_and_shrink_using_hint(hint);
            (owned_cs.into_assembly(), None)
        } else {
            (owned_cs.into_assembly_for_repeated_proving(hint), None)
        }
    }
}

#[serial]
#[test]
#[ignore]
fn test_permutation_polys() {
    let (setup_cs, _finalization_hint) =
        init_or_synth_cs_for_sha256::<DevCSConfig, Global, true>(None);

    let worker = Worker::new();
    let prover_config = init_proof_cfg();

    let (setup_base, _setup, _vk, setup_tree, _vars_hint, _wits_hint) = setup_cs.get_full_setup(
        &worker,
        prover_config.fri_lde_factor,
        prover_config.merkle_tree_cap_size,
    );
    let (variables_hint, wits_hint) = setup_cs.create_copy_hints();
    let expected_permutation_polys = setup_base.copy_permutation_polys.clone();

    let domain_size = setup_cs.max_trace_len;
    let _ctx = ProverContext::dev(domain_size).expect("init gpu prover context");

    let num_copy_permutation_polys = variables_hint.maps.len();
    let gpu_setup = GpuSetup::<Global>::from_setup_and_hints(
        setup_base,
        setup_tree,
        variables_hint,
        wits_hint,
        &worker,
    )
    .expect("gpu setup");
    println!("Gpu setup is made");

    let mut actual_copy_permutation_polys =
        GenericStorage::allocate(num_copy_permutation_polys, domain_size);
    let copy_permutation_polys_as_slice_view = actual_copy_permutation_polys.as_single_slice_mut();
    println!("GenericSetupStorage is allocated");
    let variable_indexes =
        construct_indexes_from_hint(&gpu_setup.variables_hint, domain_size, &worker).unwrap();
    materialize_permutation_cols_from_indexes_into(
        copy_permutation_polys_as_slice_view,
        &variable_indexes,
        num_copy_permutation_polys,
        domain_size,
    )
    .unwrap();
    println!("Permutation polynomials are constructed");

    for (expected, actual) in expected_permutation_polys
        .into_iter()
        .map(|p| Arc::try_unwrap(p).unwrap())
        .map(|p| p.storage)
        .map(|p| P::vec_into_base_vec(p))
        .zip(
            actual_copy_permutation_polys
                .into_poly_storage()
                .polynomials
                .into_iter()
                .map(|p| p.storage.into_inner())
                .map(|p| p.to_vec().unwrap()),
        )
    {
        assert_eq!(expected, actual);
    }
}

#[serial]
#[test]
#[ignore]
fn test_setup_comparison() {
    let (setup_cs, _) = init_or_synth_cs_for_sha256::<DevCSConfig, Global, true>(None);

    let worker = Worker::new();
    let prover_config = init_proof_cfg();

    let (setup_base, _setup, _vk, setup_tree, vars_hint, wits_hint) = setup_cs.get_full_setup(
        &worker,
        prover_config.fri_lde_factor,
        prover_config.merkle_tree_cap_size,
    );

    let _expected_permutation_polys = setup_base.copy_permutation_polys.clone();

    let domain_size = setup_cs.max_trace_len;
    let _ctx = ProverContext::dev(domain_size).expect("init gpu prover context");

    let expected_setup = GenericSetupStorage::from_host_values(&setup_base).unwrap();

    let gpu_setup = GpuSetup::<Global>::from_setup_and_hints(
        setup_base, setup_tree, vars_hint, wits_hint, &worker,
    )
    .expect("gpu setup");

    let actual_setup = GenericSetupStorage::from_gpu_setup(&gpu_setup, &worker).unwrap();

    assert_eq!(
        expected_setup.inner.to_vec().unwrap(),
        actual_setup.inner.to_vec().unwrap(),
    );

    let expected_monomial = expected_setup.into_monomials().unwrap();
    let actual_monomial = actual_setup.into_monomials().unwrap();

    assert_eq!(
        expected_monomial.inner.to_vec().unwrap(),
        actual_monomial.inner.to_vec().unwrap(),
    );
}

fn clone_reference_tree(
    input: &MerkleTreeWithCap<F, DefaultTreeHasher>,
) -> MerkleTreeWithCap<F, DefaultTreeHasher> {
    MerkleTreeWithCap {
        cap_size: input.cap_size,
        leaf_hashes: input.leaf_hashes.clone(),
        node_hashes_enumerated_from_leafs: input.node_hashes_enumerated_from_leafs.clone(),
    }
}

#[cfg(feature = "allocator_stats")]
#[serial]
#[test]
#[ignore]
fn test_dry_runs() {
    let (setup_cs, finalization_hint) =
        init_or_synth_cs_for_sha256::<DevCSConfig, Global, true>(None);
    let (proving_cs, _) =
        init_or_synth_cs_for_sha256::<ProvingCSConfig, Global, true>(finalization_hint.as_ref());
    let witness = proving_cs.witness.unwrap();
    let (reusable_cs, _) =
        init_or_synth_cs_for_sha256::<ProvingCSConfig, Global, false>(finalization_hint.as_ref());
    let config = GpuProofConfig::from_assembly(&reusable_cs);
    let worker = Worker::new();
    let prover_config = init_proof_cfg();
    let (setup_base, _setup, vk, setup_tree, vars_hint, wits_hint) = setup_cs.get_full_setup(
        &worker,
        prover_config.fri_lde_factor,
        prover_config.merkle_tree_cap_size,
    );
    let domain_size = setup_cs.max_trace_len;
    let _ctx = ProverContext::dev(domain_size).expect("init gpu prover context");
    let gpu_setup = GpuSetup::<Global>::from_setup_and_hints(
        setup_base.clone(),
        clone_reference_tree(&setup_tree),
        vars_hint.clone(),
        wits_hint.clone(),
        &worker,
    )
    .unwrap();

    assert!(domain_size.is_power_of_two());
    let candidates = CacheStrategy::get_strategy_candidates(
        &config,
        &prover_config,
        &gpu_setup,
        &vk.fixed_parameters,
    );
    for (_, strategy) in candidates.iter().copied() {
        let proof = || {
            let _ = crate::prover::gpu_prove_from_external_witness_data_with_cache_strategy::<
                DefaultTranscript,
                DefaultTreeHasher,
                NoPow,
                Global,
            >(
                &config,
                &witness,
                prover_config.clone(),
                &gpu_setup,
                &vk,
                (),
                &worker,
                strategy,
            )
            .expect("gpu proof");
        };
        dry_run_start();
        proof();
        dry_run_stop().unwrap();
        let dry = _alloc()
            .stats
            .lock()
            .unwrap()
            .allocations_at_maximum_block_count_at_maximum_tail_index
            .clone();
        let dry_tail_index = dry.tail_index();
        _setup_cache_reset();
        _alloc().stats.lock().unwrap().reset();
        assert_eq!(_alloc().stats.lock().unwrap().allocations.tail_index(), 0);
        proof();
        let wet = _alloc()
            .stats
            .lock()
            .unwrap()
            .allocations_at_maximum_block_count_at_maximum_tail_index
            .clone();
        let wet_tail_index = wet.tail_index();
        _setup_cache_reset();
        _alloc().stats.lock().unwrap().reset();
        assert_eq!(_alloc().stats.lock().unwrap().allocations.tail_index(), 0);
        assert_eq!(dry_tail_index, wet_tail_index);
    }
}

#[serial]
#[test]
#[ignore]
fn test_proof_comparison_for_sha256() {
    let (setup_cs, finalization_hint) =
        init_or_synth_cs_for_sha256::<DevCSConfig, Global, true>(None);

    let worker = Worker::new();
    let prover_config = init_proof_cfg();
    let (setup_base, setup, vk, setup_tree, vars_hint, wits_hint) = setup_cs.get_full_setup(
        &worker,
        prover_config.fri_lde_factor,
        prover_config.merkle_tree_cap_size,
    );
    let domain_size = setup_cs.max_trace_len;
    let _ctx = ProverContext::dev(domain_size).expect("init gpu prover context");
    let gpu_setup = GpuSetup::<Global>::from_setup_and_hints(
        setup_base.clone(),
        clone_reference_tree(&setup_tree),
        vars_hint.clone(),
        wits_hint.clone(),
        &worker,
    )
    .unwrap();

    assert!(domain_size.is_power_of_two());
    let actual_proof = {
        let (proving_cs, _) = init_or_synth_cs_for_sha256::<ProvingCSConfig, Global, true>(
            finalization_hint.as_ref(),
        );
        let witness = proving_cs.witness.unwrap();
        let (reusable_cs, _) = init_or_synth_cs_for_sha256::<ProvingCSConfig, Global, false>(
            finalization_hint.as_ref(),
        );
        let config = GpuProofConfig::from_assembly(&reusable_cs);

        let proof = gpu_prove_from_external_witness_data::<
            DefaultTranscript,
            DefaultTreeHasher,
            NoPow,
            Global,
        >(
            &config,
            &witness,
            prover_config.clone(),
            &gpu_setup,
            &vk,
            (),
            &worker,
        )
        .expect("gpu proof");

        proof
    };

    let expected_proof = {
        let (proving_cs, _) = init_or_synth_cs_for_sha256::<ProvingCSConfig, Global, true>(
            finalization_hint.as_ref(),
        );
        let worker = Worker::new();
        let prover_config = init_proof_cfg();

        proving_cs.prove_from_precomputations::<GoldilocksExt2, DefaultTranscript, DefaultTreeHasher, NoPow>(
            prover_config,
            &setup_base,
            &setup,
            &setup_tree,
            &vk,
            &vars_hint,
            &wits_hint,
            (),
            &worker,
        )
    };
    let actual_proof = actual_proof.into();
    compare_proofs(&expected_proof, &actual_proof);
}

fn init_or_synth_cs_for_sha256<CFG: CSConfig, A: GoodAllocator, const DO_SYNTH: bool>(
    finalization_hint: Option<&FinalizationHintsForProver>,
) -> (
    CSReferenceAssembly<F, F, CFG, A>,
    Option<FinalizationHintsForProver>,
) {
    use blake2::Digest;
    // let len = 10 * 64 + 64 - 9;
    // let len = 2 * (1 << 10);
    let len = 2 * (1 << 2);
    use rand::{Rng, SeedableRng};
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);

    let mut input = vec![];
    for _ in 0..len {
        let byte: u8 = rng.gen();
        input.push(byte);
    }

    let mut hasher = sha2::Sha256::new();
    hasher.update(&input);
    let reference_output = hasher.finalize();

    let geometry = CSGeometry {
        num_columns_under_copy_permutation: 32,
        num_witness_columns: 0,
        num_constant_columns: 4,
        max_allowed_constraint_degree: 4,
    };

    use boojum::cs::cs_builder_reference::*;
    let builder_impl = CsReferenceImplementationBuilder::<F, F, CFG>::new(geometry, 1 << 19);
    use boojum::cs::cs_builder::new_builder;
    let builder = new_builder::<_, F>(builder_impl);

    let builder = builder.allow_lookup(
        boojum::cs::LookupParameters::UseSpecializedColumnsWithTableIdAsConstant {
            width: 4,
            num_repetitions: 8,
            share_table_id: true,
        },
    );

    let builder = ConstantsAllocatorGate::configure_builder(
        builder,
        GatePlacementStrategy::UseGeneralPurposeColumns,
    );
    let builder = FmaGateInBaseFieldWithoutConstant::configure_builder(
        builder,
        GatePlacementStrategy::UseGeneralPurposeColumns,
    );
    let builder = ReductionGate::<F, 4>::configure_builder(
        builder,
        GatePlacementStrategy::UseGeneralPurposeColumns,
    );
    // not present in boojum/src/gadgets/sha256
    // let builder = PublicInputGate::configure_builder(
    //     builder,
    //     GatePlacementStrategy::UseGeneralPurposeColumns,
    // );
    let builder =
        NopGate::configure_builder(builder, GatePlacementStrategy::UseGeneralPurposeColumns);

    let mut owned_cs = builder.build(1 << 25);

    // add tables
    let table = create_tri_xor_table();
    owned_cs.add_lookup_table::<TriXor4Table, 4>(table);

    let table = create_ch4_table();
    owned_cs.add_lookup_table::<Ch4Table, 4>(table);

    let table = create_maj4_table();
    owned_cs.add_lookup_table::<Maj4Table, 4>(table);

    let table = create_4bit_chunk_split_table::<F, 1>();
    owned_cs.add_lookup_table::<Split4BitChunkTable<1>, 4>(table);

    let table = create_4bit_chunk_split_table::<F, 2>();
    owned_cs.add_lookup_table::<Split4BitChunkTable<2>, 4>(table);

    if DO_SYNTH {
        let mut circuit_input = vec![];

        let cs = &mut owned_cs;

        for el in input.iter() {
            let el = UInt8::allocate_checked(cs, *el);
            circuit_input.push(el);
        }

        let output = sha256(cs, &circuit_input);
        dbg!(output.len());

        // not present in boojum/src/gadgets/sha256
        // let mut next_available_row = cs.next_available_row();
        // for (column, var) in output.iter().enumerate() {
        //     let gate = PublicInputGate::new(var.get_variable());
        //     cs.place_gate(&gate, next_available_row);
        //     cs.place_variable(var.get_variable(), next_available_row, column);
        //     cs.set_public(column, next_available_row);
        // }
        let output = hex::encode(&output.witness_hook(&*cs)().unwrap());
        let reference_output = hex::encode(reference_output.as_slice());
        assert_eq!(output, reference_output);
    }

    // imitates control flow of synthesis_utils::init_or_synthesize_assembly
    if <CFG::SetupConfig as CSSetupConfig>::KEEP_SETUP {
        let (_, finalization_hint) = owned_cs.pad_and_shrink();
        let owned_cs = owned_cs.into_assembly();
        (owned_cs, Some(finalization_hint))
    } else {
        let hint = finalization_hint.unwrap();
        if DO_SYNTH {
            owned_cs.pad_and_shrink_using_hint(hint);
            let owned_cs = owned_cs.into_assembly();
            (owned_cs, None)
        } else {
            (owned_cs.into_assembly_for_repeated_proving(hint), None)
        }
    }
}

fn compare_proofs(
    expected_proof: &Proof<F, DefaultTreeHasher, EXT>,
    actual_proof: &Proof<F, DefaultTreeHasher, EXT>,
) {
    assert_eq!(expected_proof.public_inputs, actual_proof.public_inputs);
    assert_eq!(
        expected_proof.witness_oracle_cap,
        actual_proof.witness_oracle_cap
    );
    assert_eq!(
        expected_proof.stage_2_oracle_cap,
        actual_proof.stage_2_oracle_cap
    );
    assert_eq!(
        expected_proof.quotient_oracle_cap,
        actual_proof.quotient_oracle_cap
    );

    assert_eq!(expected_proof.values_at_z, actual_proof.values_at_z);
    assert_eq!(
        expected_proof.values_at_z_omega,
        actual_proof.values_at_z_omega
    );
    assert_eq!(expected_proof.values_at_0, actual_proof.values_at_0);
    assert_eq!(
        expected_proof.fri_base_oracle_cap,
        actual_proof.fri_base_oracle_cap
    );
    assert_eq!(
        expected_proof.fri_intermediate_oracles_caps,
        actual_proof.fri_intermediate_oracles_caps
    );
    assert_eq!(
        expected_proof.final_fri_monomials,
        actual_proof.final_fri_monomials
    );
    assert_eq!(expected_proof.pow_challenge, actual_proof.pow_challenge);
    assert_eq!(
        expected_proof.queries_per_fri_repetition.len(),
        actual_proof.queries_per_fri_repetition.len(),
    );

    for (_query_idx, (expected_fri_query, actual_fri_query)) in expected_proof
        .queries_per_fri_repetition
        .iter()
        .zip(actual_proof.queries_per_fri_repetition.iter())
        .enumerate()
    {
        // leaf elems
        assert_eq!(
            expected_fri_query.witness_query.leaf_elements.len(),
            actual_fri_query.witness_query.leaf_elements.len()
        );
        assert_eq!(
            expected_fri_query.witness_query.leaf_elements,
            actual_fri_query.witness_query.leaf_elements
        );

        assert_eq!(
            expected_fri_query.stage_2_query.leaf_elements.len(),
            actual_fri_query.stage_2_query.leaf_elements.len(),
        );
        assert_eq!(
            expected_fri_query.stage_2_query.leaf_elements,
            actual_fri_query.stage_2_query.leaf_elements
        );

        assert_eq!(
            expected_fri_query.quotient_query.leaf_elements.len(),
            actual_fri_query.quotient_query.leaf_elements.len()
        );
        assert_eq!(
            expected_fri_query.quotient_query.leaf_elements,
            actual_fri_query.quotient_query.leaf_elements
        );
        assert_eq!(
            expected_fri_query.setup_query.leaf_elements.len(),
            actual_fri_query.setup_query.leaf_elements.len()
        );
        assert_eq!(
            expected_fri_query.setup_query.leaf_elements,
            actual_fri_query.setup_query.leaf_elements
        );

        assert_eq!(
            expected_fri_query.fri_queries.len(),
            actual_fri_query.fri_queries.len(),
        );

        for (_layer_idx, (expected, actual)) in expected_fri_query
            .fri_queries
            .iter()
            .zip(actual_fri_query.fri_queries.iter())
            .enumerate()
        {
            assert_eq!(expected.leaf_elements.len(), actual.leaf_elements.len());
            assert_eq!(expected.leaf_elements, actual.leaf_elements);
        }

        // merkle paths
        assert_eq!(
            expected_fri_query.witness_query.proof.len(),
            actual_fri_query.witness_query.proof.len(),
        );
        assert_eq!(
            expected_fri_query.witness_query.proof,
            actual_fri_query.witness_query.proof
        );
        assert_eq!(
            expected_fri_query.stage_2_query.proof.len(),
            actual_fri_query.stage_2_query.proof.len()
        );
        assert_eq!(
            expected_fri_query.quotient_query.proof,
            actual_fri_query.quotient_query.proof
        );

        assert_eq!(
            expected_fri_query.setup_query.proof.len(),
            actual_fri_query.setup_query.proof.len(),
        );

        assert_eq!(
            expected_fri_query.setup_query.proof,
            actual_fri_query.setup_query.proof,
        );

        for (_layer_idx, (expected, actual)) in expected_fri_query
            .fri_queries
            .iter()
            .zip(actual_fri_query.fri_queries.iter())
            .enumerate()
        {
            assert_eq!(expected.proof.len(), actual.proof.len());
            assert_eq!(expected.proof, actual.proof);
        }
    }
}

#[serial]
#[test]
#[ignore]
fn test_reference_proof_for_sha256() {
    let (mut cs, _) = init_or_synth_cs_for_sha256::<DevCSConfig, Global, true>(None);

    let worker = Worker::new();
    let prover_config = init_proof_cfg();

    let (base_setup, setup, vk, setup_tree, vars_hint, wits_hint) = cs.get_full_setup(
        &worker,
        prover_config.fri_lde_factor,
        prover_config.merkle_tree_cap_size,
    );
    let witness_set = cs.take_witness_using_hints(&worker, &vars_hint, &wits_hint);
    let _proof = cs.prove_cpu_basic::<GoldilocksExt2, DefaultTranscript, DefaultTreeHasher, NoPow>(
        &worker,
        witness_set,
        &base_setup,
        &setup,
        &setup_tree,
        &vk,
        prover_config,
        (),
    );
}

pub fn init_proof_cfg() -> ProofConfig {
    let mut prover_config = ProofConfig::default();
    prover_config.fri_lde_factor = 2;
    prover_config.pow_bits = 0;
    prover_config.merkle_tree_cap_size = 32;

    prover_config
}

#[cfg(test)]
#[cfg(feature = "zksync")]
mod zksync {
    use std::path::PathBuf;

    use super::*;

    use crate::cs::PACKED_PLACEHOLDER_BITMASK;
    use boojum::cs::implementations::fast_serialization::MemcopySerializable;
    use circuit_definitions::circuit_definitions::base_layer::ZkSyncBaseLayerCircuit;
    use era_cudart_sys::CudaError;

    pub type ZksyncProof = Proof<F, DefaultTreeHasher, GoldilocksExt2>;

    const TEST_DATA_ROOT_DIR: &str = "./test_data";
    const DEFAULT_CIRCUIT_INPUT: &str = "default.circuit";

    use crate::synthesis_utils::{
        init_cs_for_external_proving, init_or_synthesize_assembly, synth_circuit_for_proving,
        synth_circuit_for_setup, CircuitWrapper,
    };

    #[allow(dead_code)]
    pub type BaseLayerCircuit = ZkSyncBaseLayerCircuit;

    fn scan_directory<P: AsRef<Path>>(dir: P) -> Vec<PathBuf> {
        let mut file_paths = vec![];
        for entry in std::fs::read_dir(dir).unwrap() {
            let entry = entry.unwrap();
            let path = entry.path();
            if path.is_file() {
                file_paths.push(path);
            }
        }
        file_paths.sort_by(|a, b| a.cmp(b));

        file_paths
    }

    fn scan_directory_for_circuits<P: AsRef<Path>>(dir: P) -> Vec<CircuitWrapper> {
        let mut circuits = vec![];
        let file_paths = scan_directory(dir);
        for path in file_paths {
            let file_extension = path.extension().unwrap().to_string_lossy().to_string();
            if file_extension.contains("circuit") {
                let file = std::fs::File::open(path).unwrap();
                let circuit: CircuitWrapper = bincode::deserialize_from(file).expect("deserialize");
                circuits.push(circuit);
            }
        }

        circuits
    }

    #[allow(dead_code)]
    fn scan_directory_for_setups<P: AsRef<Path>>(dir: P) -> Vec<SetupBaseStorage<F, F>> {
        let mut circuits = vec![];
        let file_paths = scan_directory(dir);
        for path in file_paths {
            let file_extension = path.extension().unwrap().to_string_lossy().to_string();
            if file_extension.contains("setup") {
                let file = std::fs::File::open(path).unwrap();
                let circuit: SetupBaseStorage<F, F> =
                    bincode::deserialize_from(file).expect("deserialize");
                circuits.push(circuit);
            }
        }

        circuits
    }

    fn scan_directory_for_proofs<P: AsRef<Path>>(dir: P) -> Vec<ZksyncProof> {
        let mut proofs = vec![];
        let file_paths = scan_directory(dir);
        for path in file_paths {
            let file_extension = path.extension().unwrap().to_string_lossy().to_string();
            if file_extension.contains("proof") {
                let file = std::fs::File::open(path).unwrap();
                let proof: ZksyncProof = bincode::deserialize_from(file).expect("deserialize");
                proofs.push(proof);
            }
        }

        proofs
    }

    #[serial]
    #[test]
    #[ignore]
    fn test_single_shot_zksync_setup_comparison() {
        let circuit = get_circuit_from_env();
        let _ctx = ProverContext::create().expect("gpu prover context");

        println!(
            "{} {}",
            circuit.numeric_circuit_type(),
            circuit.short_description()
        );
        let worker = Worker::new();
        let (setup_cs, _) = synth_circuit_for_setup(circuit.clone());
        let proof_cfg = circuit.proof_config();

        let (setup_base, _setup, _vk, setup_tree, vars_hint, wits_hint) = setup_cs.get_full_setup(
            &worker,
            proof_cfg.fri_lde_factor,
            proof_cfg.merkle_tree_cap_size,
        );

        let _expected_permutation_polys = setup_base.copy_permutation_polys.clone();

        let _domain_size = setup_cs.max_trace_len;

        let expected_setup = GenericSetupStorage::from_host_values(&setup_base).unwrap();

        let gpu_setup = GpuSetup::<Global>::from_setup_and_hints(
            setup_base, setup_tree, vars_hint, wits_hint, &worker,
        )
        .expect("gpu setup");

        let actual_setup = GenericSetupStorage::from_gpu_setup(&gpu_setup, &worker).unwrap();

        assert_eq!(
            expected_setup.inner.to_vec().unwrap(),
            actual_setup.inner.to_vec().unwrap(),
        );

        let expected_monomial = expected_setup.into_monomials().unwrap();
        let actual_monomial = actual_setup.into_monomials().unwrap();

        assert_eq!(
            expected_monomial.inner.to_vec().unwrap(),
            actual_monomial.inner.to_vec().unwrap(),
        );
    }

    #[serial]
    #[test]
    fn compare_proofs_for_all_zksync_circuits() -> CudaResult<()> {
        let worker = &Worker::new();
        let _ctx = ProverContext::create().expect("gpu prover context");

        for main_dir in ["base", "leaf", "node", "tip"] {
            let data_dir = format!("./test_data/{}", main_dir);
            dbg!(&data_dir);
            let circuits = scan_directory_for_circuits(&data_dir);
            let reference_proofs = scan_directory_for_proofs(&data_dir);

            for (circuit, _reference_proof) in
                circuits.into_iter().zip(reference_proofs.into_iter())
            {
                let reference_proof_path =
                    format!("{}/{}.cpu.proof", data_dir, circuit.numeric_circuit_type());

                let reference_proof_path = Path::new(&reference_proof_path);

                let gpu_proof_path =
                    format!("{}/{}.gpu.proof", data_dir, circuit.numeric_circuit_type());

                let gpu_proof_path = Path::new(&gpu_proof_path);

                if reference_proof_path.exists() && gpu_proof_path.exists() {
                    continue;
                }

                println!(
                    "{} {}",
                    circuit.numeric_circuit_type(),
                    circuit.short_description()
                );

                let proof_config = circuit.proof_config();

                let (setup_cs, finalization_hint) = synth_circuit_for_setup(circuit.clone());
                let (setup_base, _setup, vk, setup_tree, vars_hint, wits_hint) = setup_cs
                    .get_full_setup(
                        worker,
                        proof_config.fri_lde_factor,
                        proof_config.merkle_tree_cap_size,
                    );

                let gpu_setup = GpuSetup::<Global>::from_setup_and_hints(
                    setup_base.clone(),
                    setup_tree,
                    vars_hint.clone(),
                    wits_hint.clone(),
                    &worker,
                )?;

                println!("gpu proving");

                let gpu_proof = {
                    let proving_cs = synth_circuit_for_proving(circuit.clone(), &finalization_hint);
                    let witness = proving_cs.witness.unwrap();
                    let config = GpuProofConfig::from_circuit_wrapper(&circuit);
                    let proof = gpu_prove_from_external_witness_data::<
                        DefaultTranscript,
                        DefaultTreeHasher,
                        NoPow,
                        Global,
                    >(
                        &config,
                        &witness,
                        proof_config.clone(),
                        &gpu_setup,
                        &vk,
                        (),
                        worker,
                    )
                    .expect("gpu proof");
                    proof
                };

                let reference_proof_file = std::fs::File::open(reference_proof_path).unwrap();
                let reference_proof = bincode::deserialize_from(&reference_proof_file).unwrap();
                let actual_proof = gpu_proof.into();
                compare_proofs(&reference_proof, &actual_proof);
                assert!(circuit.verify_proof(&vk, &actual_proof));
                let proof_file = std::fs::File::create(gpu_proof_path).unwrap();

                bincode::serialize_into(proof_file, &actual_proof).expect("write proof into file");
            }
        }

        Ok(())
    }

    #[serial]
    #[test]
    #[ignore]
    fn generate_reference_proofs_for_all_zksync_circuits() {
        let worker = &Worker::new();

        for main_dir in ["base", "leaf", "node", "tip"] {
            let data_dir = format!("./test_data/{}", main_dir);
            dbg!(&data_dir);
            let circuits = scan_directory_for_circuits(&data_dir);

            for circuit in circuits.into_iter() {
                if Path::new(&format!(
                    "{}/{}.cpu.proof",
                    data_dir,
                    circuit.numeric_circuit_type(),
                ))
                .exists()
                {
                    continue;
                }
                println!(
                    "{} {}",
                    circuit.numeric_circuit_type(),
                    circuit.short_description()
                );

                let proof_config = circuit.proof_config();

                let (setup_cs, finalization_hint) =
                    init_or_synthesize_assembly::<SetupCSConfig, true>(circuit.clone(), None);
                let finalization_hint = finalization_hint.unwrap();
                let (setup_base, setup, vk, setup_tree, vars_hint, witness_hints) = setup_cs
                    .get_full_setup(
                        worker,
                        proof_config.fri_lde_factor,
                        proof_config.merkle_tree_cap_size,
                    );

                println!("reference proving");
                let reference_proof = {
                    let (proving_cs, _finalization_hint) =
                        init_or_synthesize_assembly::<ProvingCSConfig, true>(
                            circuit.clone(),
                            Some(&finalization_hint),
                        );
                    proving_cs.prove_from_precomputations::<EXT, DefaultTranscript, DefaultTreeHasher, NoPow>(
                        proof_config,
                        &setup_base,
                        &setup,
                        &setup_tree,
                        &vk,
                        &vars_hint,
                        &witness_hints,
                        (),
                        worker,
                    )
                };
                assert!(circuit.verify_proof(&vk, &reference_proof));
                let proof_file = std::fs::File::create(format!(
                    "{}/{}.cpu.proof",
                    data_dir,
                    circuit.numeric_circuit_type()
                ))
                .unwrap();

                bincode::serialize_into(proof_file, &reference_proof)
                    .expect("write proof into file");
            }
        }
    }

    #[serial]
    #[test]
    #[ignore]
    fn compare_proofs_for_single_zksync_circuit() {
        let circuit = get_circuit_from_env();
        let _ctx = ProverContext::create().expect("gpu prover context");

        println!(
            "{} {}",
            circuit.numeric_circuit_type(),
            circuit.short_description()
        );
        let worker = &Worker::new();

        let (setup_cs, finalization_hint) = synth_circuit_for_setup(circuit.clone());
        let proof_cfg = circuit.proof_config();
        let (setup_base, setup, vk, setup_tree, vars_hint, wits_hint) = setup_cs.get_full_setup(
            worker,
            proof_cfg.fri_lde_factor,
            proof_cfg.merkle_tree_cap_size,
        );

        println!(
            "trace length size 2^{}",
            setup_base.copy_permutation_polys[0]
                .domain_size()
                .trailing_zeros()
        );

        let proving_cs = synth_circuit_for_proving(circuit.clone(), &finalization_hint);

        let gpu_setup = GpuSetup::<Global>::from_setup_and_hints(
            setup_base.clone(),
            clone_reference_tree(&setup_tree),
            vars_hint.clone(),
            wits_hint.clone(),
            &worker,
        )
        .expect("gpu setup");

        println!("gpu proving");
        let gpu_proof = {
            let witness = proving_cs.witness.as_ref().unwrap();
            let config = GpuProofConfig::from_circuit_wrapper(&circuit);
            gpu_prove_from_external_witness_data::<
                DefaultTranscript,
                DefaultTreeHasher,
                NoPow,
                Global,
            >(
                &config,
                &witness,
                proof_cfg.clone(),
                &gpu_setup,
                &vk,
                (),
                worker,
            )
            .expect("gpu proof")
        };

        println!("cpu proving");
        let reference_proof = {
            // we can't clone assembly lets synth it again
            let proving_cs = synth_circuit_for_proving(circuit.clone(), &finalization_hint);
            proving_cs
                .prove_from_precomputations::<EXT, DefaultTranscript, DefaultTreeHasher, NoPow>(
                    proof_cfg.clone(),
                    &setup_base,
                    &setup,
                    &setup_tree,
                    &vk,
                    &vars_hint,
                    &wits_hint,
                    (),
                    worker,
                )
        };
        let start = std::time::Instant::now();
        let actual_proof = gpu_proof.into();
        println!("proof transformation takes {:?}", start.elapsed());
        circuit.verify_proof(&vk, &actual_proof);
        compare_proofs(&reference_proof, &actual_proof);
    }

    #[serial]
    #[test]
    #[ignore]
    fn benchmark_single_circuit() {
        let circuit = get_circuit_from_env();
        println!(
            "{} {}",
            circuit.numeric_circuit_type(),
            circuit.short_description()
        );
        let worker = &Worker::new();
        let (setup_cs, finalization_hint) = synth_circuit_for_setup(circuit.clone());
        let proof_cfg = circuit.proof_config();
        let (setup_base, _setup, vk, setup_tree, vars_hint, wits_hint) = setup_cs.get_full_setup(
            worker,
            proof_cfg.fri_lde_factor,
            proof_cfg.merkle_tree_cap_size,
        );
        let proving_cs = synth_circuit_for_proving(circuit.clone(), &finalization_hint);
        let witness = proving_cs.witness.unwrap();
        let config = GpuProofConfig::from_circuit_wrapper(&circuit);
        let gpu_setup = {
            let _ctx = ProverContext::create().expect("gpu prover context");
            GpuSetup::<Global>::from_setup_and_hints(
                setup_base.clone(),
                clone_reference_tree(&setup_tree),
                vars_hint.clone(),
                wits_hint.clone(),
                &worker,
            )
            .expect("gpu setup")
        };
        let proof_fn = || {
            let _ = gpu_prove_from_external_witness_data::<
                DefaultTranscript,
                DefaultTreeHasher,
                NoPow,
                Global,
            >(
                &config,
                &witness,
                proof_cfg.clone(),
                &gpu_setup,
                &vk,
                (),
                worker,
            )
            .expect("gpu proof");
        };
        loop {
            for i in 0..40 {
                let num_blocks = 2560 - i * 64;
                println!("num_blocks = {num_blocks}");
                let ctx = ProverContext::create_limited(num_blocks).expect("gpu prover context");
                // technically not needed because CacheStrategy::get calls it internally,
                // but nice for peace of mind
                _setup_cache_reset();
                let strategy =
                    CacheStrategy::get::<DefaultTranscript, DefaultTreeHasher, NoPow, Global>(
                        &config,
                        &witness,
                        proof_cfg.clone(),
                        &gpu_setup,
                        &vk,
                        (),
                        worker,
                    );
                // technically not needed because CacheStrategy::get calls it internally,
                // but nice for peace of mind
                _setup_cache_reset();
                let strategy = match strategy {
                    Ok(s) => s,
                    Err(CudaError::ErrorMemoryAllocation) => {
                        println!("no cache strategy for {num_blocks}  found");
                        return;
                    }
                    Err(e) => panic!("unexpected error: {e}"),
                };
                println!("strategy: {:?}", strategy);
                println!("warmup with determined strategy");
                proof_fn();
                _setup_cache_reset();
                println!("first run");
                let start = std::time::Instant::now();
                proof_fn();
                println!("◆ total: {:?}", start.elapsed());
                println!("second run");
                let start = std::time::Instant::now();
                proof_fn();
                println!("◆ total: {:?}", start.elapsed());
                drop(ctx);
            }
        }
    }

    #[serial]
    #[test]
    #[ignore]
    fn profile_single_circuit() {
        let circuit = get_circuit_from_env();
        println!(
            "{} {}",
            circuit.numeric_circuit_type(),
            circuit.short_description()
        );
        let worker = &Worker::new();
        let (setup_cs, finalization_hint) = synth_circuit_for_setup(circuit.clone());
        let proof_cfg = circuit.proof_config();
        let (setup_base, _setup, vk, setup_tree, vars_hint, wits_hint) = setup_cs.get_full_setup(
            worker,
            proof_cfg.fri_lde_factor,
            proof_cfg.merkle_tree_cap_size,
        );
        let proving_cs = synth_circuit_for_proving(circuit.clone(), &finalization_hint);
        let witness = proving_cs.witness.unwrap();
        let reusable_cs = init_cs_for_external_proving(circuit.clone(), &finalization_hint);
        let config = GpuProofConfig::from_assembly(&reusable_cs);
        let gpu_setup = {
            let _ctx = ProverContext::create().expect("gpu prover context");
            GpuSetup::<Global>::from_setup_and_hints(
                setup_base.clone(),
                clone_reference_tree(&setup_tree),
                vars_hint.clone(),
                wits_hint.clone(),
                &worker,
            )
            .expect("gpu setup")
        };
        let proof_fn = || {
            let _ = gpu_prove_from_external_witness_data::<
                DefaultTranscript,
                DefaultTreeHasher,
                NoPow,
                Global,
            >(
                &config,
                &witness,
                proof_cfg.clone(),
                &gpu_setup,
                &vk,
                (),
                worker,
            )
            .expect("gpu proof");
        };
        let ctx = ProverContext::create().expect("gpu prover context");
        println!("warmup");
        proof_fn();
        _setup_cache_reset();
        #[cfg(feature = "nvtx")]
        nvtx::range_push!("test");
        #[cfg(feature = "nvtx")]
        nvtx::range_push!("first run");
        println!("first run");
        let start = std::time::Instant::now();
        proof_fn();
        println!("◆ total: {:?}", start.elapsed());
        #[cfg(feature = "nvtx")]
        nvtx::range_pop!();
        #[cfg(feature = "nvtx")]
        nvtx::range_push!("second run");
        println!("second run");
        let start = std::time::Instant::now();
        proof_fn();
        println!("◆ total: {:?}", start.elapsed());
        #[cfg(feature = "nvtx")]
        nvtx::range_pop!();
        #[cfg(feature = "nvtx")]
        nvtx::range_push!("third run");
        println!("third run");
        let start = std::time::Instant::now();
        proof_fn();
        println!("◆ total: {:?}", start.elapsed());
        #[cfg(feature = "nvtx")]
        nvtx::range_pop!();
        #[cfg(feature = "nvtx")]
        nvtx::range_pop!();
        drop(ctx);
        return;
    }

    #[serial]
    #[test]
    #[ignore]
    #[should_panic(expected = "placeholder found in a public input location")]
    fn test_public_input_placeholder_fail() {
        let (setup_cs, finalization_hint) =
            init_or_synth_cs_for_sha256::<DevCSConfig, Global, true>(None);
        let worker = Worker::new();
        let proof_config = init_proof_cfg();
        let (setup_base, _, vk, setup_tree, vars_hint, wits_hint) = setup_cs.get_full_setup(
            &worker,
            proof_config.fri_lde_factor,
            proof_config.merkle_tree_cap_size,
        );
        let domain_size = setup_cs.max_trace_len;
        let _ctx = ProverContext::dev(domain_size).expect("init gpu prover context");
        let (proving_cs, _) = init_or_synth_cs_for_sha256::<ProvingCSConfig, Global, true>(
            finalization_hint.as_ref(),
        );
        let mut witness = proving_cs.witness.as_ref().unwrap().clone();
        let (reusable_cs, _) = init_or_synth_cs_for_sha256::<ProvingCSConfig, Global, false>(
            finalization_hint.as_ref(),
        );
        let config = GpuProofConfig::from_assembly(&reusable_cs);
        let mut gpu_setup = GpuSetup::<Global>::from_setup_and_hints(
            setup_base.clone(),
            clone_reference_tree(&setup_tree),
            vars_hint.clone(),
            wits_hint.clone(),
            &worker,
        )
        .expect("gpu setup");
        witness.public_inputs_locations = vec![(0, 0)];
        gpu_setup.variables_hint[0][0] = PACKED_PLACEHOLDER_BITMASK;
        let _ = gpu_prove_from_external_witness_data::<
            DefaultTranscript,
            DefaultTreeHasher,
            NoPow,
            Global,
        >(
            &config,
            &witness,
            proof_config.clone(),
            &gpu_setup,
            &vk,
            (),
            &worker,
        )
        .expect("gpu proof");
    }

    #[serial]
    #[test]
    #[ignore]
    fn test_reference_proof_for_circuit() {
        let circuit = get_circuit_from_env();
        println!(
            "{} {}",
            circuit.numeric_circuit_type(),
            circuit.short_description()
        );
        let worker = &Worker::new();

        let (setup_cs, finalization_hint) = synth_circuit_for_setup(circuit.clone());
        let proof_cfg = circuit.proof_config();
        let (setup_base, setup, vk, setup_tree, vars_hint, witness_hints) = setup_cs
            .get_full_setup(
                worker,
                proof_cfg.fri_lde_factor,
                proof_cfg.merkle_tree_cap_size,
            );

        println!(
            "trace length size 2^{}",
            setup_base.copy_permutation_polys[0]
                .domain_size()
                .trailing_zeros()
        );

        let _proving_cs = synth_circuit_for_proving(circuit.clone(), &finalization_hint);
        let reference_proof = {
            // we can't clone assembly lets synth it again
            let mut proving_cs = synth_circuit_for_proving(circuit.clone(), &finalization_hint);
            let _witness_set =
                proving_cs.take_witness_using_hints(&worker, &vars_hint, &witness_hints);
            proving_cs
                .prove_from_precomputations::<EXT, DefaultTranscript, DefaultTreeHasher, NoPow>(
                    proof_cfg.clone(),
                    &setup_base,
                    &setup,
                    &setup_tree,
                    &vk,
                    &vars_hint,
                    &witness_hints,
                    (),
                    worker,
                )
        };
        circuit.verify_proof(&vk, &reference_proof);
    }

    #[serial]
    #[test]
    #[ignore]
    fn test_generate_reference_setups_for_all_zksync_circuits() {
        let _worker = Worker::new();

        for main_dir in ["base", "leaf", "node", "tip"] {
            let data_dir = format!("./test_data/{}", main_dir);
            let circuits = scan_directory_for_circuits(&data_dir);

            let worker = &Worker::new();
            for circuit in circuits {
                println!(
                    "{} {}",
                    circuit.numeric_circuit_type(),
                    circuit.short_description()
                );

                let setup_file_path = format!(
                    "{}/{}.reference.setup",
                    data_dir,
                    circuit.numeric_circuit_type()
                );

                let (setup_cs, _finalization_hint) = synth_circuit_for_setup(circuit);
                let reference_base_setup = setup_cs.create_base_setup(&worker, &mut ());

                let setup_file = std::fs::File::create(&setup_file_path).unwrap();
                reference_base_setup
                    .write_into_buffer(&setup_file)
                    .expect("write gpu setup into file");
                println!("Setup written into file {}", setup_file_path);
            }
        }
    }

    #[serial]
    #[test]
    #[ignore]
    fn test_generate_gpu_setups_for_all_zksync_circuits() {
        let _worker = Worker::new();
        let _ctx = ProverContext::create().expect("gpu context");
        let worker = &Worker::new();

        for main_dir in ["base", "leaf", "node", "tip"] {
            let data_dir = format!("{}/{}", TEST_DATA_ROOT_DIR, main_dir);
            let circuits = scan_directory_for_circuits(&data_dir);

            for circuit in circuits {
                println!(
                    "{} {}",
                    circuit.numeric_circuit_type(),
                    circuit.short_description()
                );

                let setup_file_path =
                    format!("{}/{}.gpu.setup", data_dir, circuit.numeric_circuit_type());

                let proof_cfg = circuit.proof_config();
                let (setup_cs, _finalization_hint) = synth_circuit_for_setup(circuit);
                let (
                    reference_base_setup,
                    _setup,
                    _vk,
                    reference_setup_tree,
                    _vars_hint,
                    _witness_hints,
                ) = setup_cs.get_full_setup(
                    worker,
                    proof_cfg.fri_lde_factor,
                    proof_cfg.merkle_tree_cap_size,
                );
                let (variables_hint, wits_hint) = setup_cs.create_copy_hints();

                let gpu_setup = GpuSetup::<Global>::from_setup_and_hints(
                    reference_base_setup,
                    reference_setup_tree,
                    variables_hint,
                    wits_hint,
                    &worker,
                )
                .expect("gpu setup");

                let setup_file = std::fs::File::create(&setup_file_path).unwrap();
                bincode::serialize_into(&setup_file, &gpu_setup).unwrap();
                println!("Setup written into file {}", setup_file_path);
            }
        }
    }

    fn get_circuit_from_env() -> CircuitWrapper {
        let circuit_file_path = if let Ok(circuit_file) = std::env::var("CIRCUIT_FILE") {
            circuit_file
        } else {
            std::env::args()
                // --circuit=/path/to/circuit prevents rebuilds
                .filter(|arg| arg.contains("--circuit"))
                .map(|arg| {
                    let parts: Vec<&str> = arg.splitn(2, '=').collect();
                    assert_eq!(parts.len(), 2);
                    let circuit_file_path = parts[1];
                    dbg!(circuit_file_path);
                    circuit_file_path.to_string()
                })
                .collect::<Vec<String>>()
                .pop()
                .unwrap_or(format!("./test_data/{}", DEFAULT_CIRCUIT_INPUT))
        };

        let data = std::fs::read(circuit_file_path).expect("circuit file");
        bincode::deserialize(&data).expect("circuit")
    }
}
