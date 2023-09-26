use crate::cs::{materialize_permutation_cols_from_transformed_hints_into, GpuSetup};

use super::*;
use std::{path::Path, sync::Arc};

use boojum::{
    config::{CSConfig, DevCSConfig, ProvingCSConfig, SetupCSConfig},
    cs::{
        gates::{
            ConstantsAllocatorGate, FmaGateInBaseFieldWithoutConstant, NopGate, ReductionGate,
        },
        implementations::{
            pow::NoPow, proof::Proof, prover::ProofConfig, reference_cs::CSReferenceAssembly,
        },
        oracle::merkle_tree::MerkleTreeWithCap,
        traits::{cs::ConstraintSystem, gate::GatePlacementStrategy},
        CSGeometry,
    },
    field::goldilocks::GoldilocksExt2,
    gadgets::{
        sha256::sha256,
        tables::{
            ch4::{create_ch4_table, Ch4Table},
            chunk4bits::{create_4bit_chunk_split_table, Split4BitChunkTable},
            maj4::{create_maj4_table, Maj4Table},
            trixor4::{create_tri_xor_table, TriXor4Table},
        },
        traits::witnessable::WitnessHookable,
        u8::UInt8,
    },
    worker::Worker,
};
use boojum::cs::implementations::setup::FinalizationHintsForProver;

use boojum::field::traits::field_like::PrimeFieldLikeVectorized;

#[allow(dead_code)]
pub type DefaultDevCS = CSReferenceAssembly<F, F, DevCSConfig>;
type P = F;
use serial_test::serial;

#[serial]
#[test]
#[ignore]
fn test_permutation_polys() {
    let setup_cs = init_cs_for_sha256::<SetupCSConfig>();

    let worker = Worker::new();
    let prover_config = init_proof_cfg();

    let (setup_base, _setup, _vk, setup_tree, _vars_hint, _wits_hint) = setup_cs.get_full_setup(
        &worker,
        prover_config.fri_lde_factor,
        prover_config.merkle_tree_cap_size,
    );
    let (variables_hint, _) = setup_cs.create_copy_hints();
    let expected_permutation_polys = setup_base.copy_permutation_polys.clone();

    let domain_size = setup_cs.max_trace_len;
    let _ctx = ProverContext::dev(domain_size).expect("init gpu prover context");

    let num_copy_permutation_polys = variables_hint.maps.len();
    let gpu_setup =
        GpuSetup::<Global>::from_setup_and_hints(setup_base, setup_tree, variables_hint)
            .expect("gpu setup");
    println!("Gpu setup is made");

    let mut actual_copy_permutation_polys =
        GenericStorage::allocate(num_copy_permutation_polys, domain_size).unwrap();
    let copy_permutation_polys_as_slice_view = actual_copy_permutation_polys.as_single_slice_mut();
    println!("GenericSetupStorage is allocated");

    materialize_permutation_cols_from_transformed_hints_into(
        copy_permutation_polys_as_slice_view,
        &gpu_setup.variables_hint,
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
                .into_poly_storage::<LagrangeBasis>()
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
fn test_variable_assignment() {
    let mut cs = init_cs_for_sha256::<DevCSConfig>();
    let worker = Worker::new();
    let prover_config = init_proof_cfg();
    let domain_size = cs.max_trace_len;
    let _ctx = ProverContext::dev(domain_size).expect("init gpu prover context");

    let (setup_base, _setup, _vk, setup_tree, vars_hint, _wits_hint) = cs.get_full_setup(
        &worker,
        prover_config.fri_lde_factor,
        prover_config.merkle_tree_cap_size,
    );
    let (variables_hint, witness_hints) = cs.create_copy_hints();
    let setup =
        GpuSetup::<Global>::from_setup_and_hints(setup_base.clone(), setup_tree, vars_hint.clone())
            .unwrap();

    let witness_set = cs.take_witness_using_hints(&worker, &variables_hint, &witness_hints);

    let (expected_raw_trace_storage, expected_monomial_trace_storage) =
        construct_trace_storage(&witness_set).unwrap();

    let external_witness_data = cs.materialize_witness_vec();

    let (actual_raw_trace_storage, actual_monomial_trace_storage) =
        construct_trace_storage_from_external_witness_data(
            expected_raw_trace_storage.layout.clone(),
            domain_size,
            &setup,
            &external_witness_data,
            &cs.lookup_parameters,
            &worker,
        )
        .unwrap();

    synchronize_streams().unwrap();

    let expected = expected_raw_trace_storage.storage.inner.to_vec().unwrap();
    let actual = actual_raw_trace_storage.storage.inner.to_vec().unwrap();
    assert_eq!(expected, actual);
    let expected = expected_monomial_trace_storage
        .storage
        .inner
        .to_vec()
        .unwrap();
    let actual = actual_monomial_trace_storage
        .storage
        .inner
        .to_vec()
        .unwrap();
    assert_eq!(expected, actual);
}

#[serial]
#[test]
#[ignore]
fn test_setup_comparison() {
    let setup_cs = init_cs_for_sha256::<SetupCSConfig>();

    let worker = Worker::new();
    let prover_config = init_proof_cfg();

    let (setup_base, _setup, _vk, setup_tree, vars_hint, _wits_hint) = setup_cs.get_full_setup(
        &worker,
        prover_config.fri_lde_factor,
        prover_config.merkle_tree_cap_size,
    );

    let _expected_permutation_polys = setup_base.copy_permutation_polys.clone();

    let domain_size = setup_cs.max_trace_len;
    let _ctx = ProverContext::dev(domain_size).expect("init gpu prover context");

    let expected_setup = GenericSetupStorage::from_host_values(&setup_base).unwrap();

    let gpu_setup = GpuSetup::<Global>::from_setup_and_hints(setup_base, setup_tree, vars_hint)
        .expect("gpu setup");

    let actual_setup = GenericSetupStorage::from_gpu_setup(&gpu_setup).unwrap();

    assert_eq!(
        expected_setup.storage.inner.to_vec().unwrap(),
        actual_setup.storage.inner.to_vec().unwrap(),
    );

    let expected_monomial = expected_setup.into_monomials().unwrap();
    let actual_monomial = actual_setup.into_monomials().unwrap();

    assert_eq!(
        expected_monomial.storage.inner.to_vec().unwrap(),
        actual_monomial.storage.inner.to_vec().unwrap(),
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

#[serial]
#[test]
#[ignore]
fn test_proof_comparison_for_sha256() {
    let setup_cs = init_cs_for_sha256::<SetupCSConfig>();

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
    )
    .unwrap();

    assert!(domain_size.is_power_of_two());
    let actual_proof = {
        let proving_cs = init_cs_for_sha256::<ProvingCSConfig>();
        let proof = prover::gpu_prove::<_, DefaultTranscript, DefaultTreeHasher, NoPow, Global>(
            proving_cs,
            prover_config,
            &gpu_setup,
            &vk,
            (),
            &worker,
        )
        .expect("gpu proof");

        proof
    };

    let expected_proof = {
        let proving_cs = init_cs_for_sha256::<ProvingCSConfig>();
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

    compare_proofs(&expected_proof, &actual_proof);
}

#[allow(dead_code)]
pub fn init_reusable_cs_for_sha256(
    finalization_hint: &FinalizationHintsForProver,
) -> CSReferenceAssembly<F, F, ProvingCSConfig> {
    use blake2::Digest;
    // let len = 10 * 64 + 64 - 9;
    // let len = 2 * (1 << 10);
    let len = 2 * (1 << 2);
    use rand::{Rng, SeedableRng};
    let mut rng = rand::rngs::StdRng::seed_from_u64(42 as u64);

    let mut input = vec![];
    for _ in 0..len {
        let byte: u8 = rng.gen();
        input.push(byte);
    }

    let mut hasher = sha2::Sha256::new();
    hasher.update(&input);
    let _reference_output = hasher.finalize();

    let geometry = CSGeometry {
        num_columns_under_copy_permutation: 20,
        num_witness_columns: 0,
        num_constant_columns: 4,
        max_allowed_constraint_degree: 4,
    };

    use boojum::cs::cs_builder_reference::*;
    let builder_impl =
        CsReferenceImplementationBuilder::<F, F, ProvingCSConfig>::new(geometry, 1 << 25, 1 << 19);
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
    let builder =
        NopGate::configure_builder(builder, GatePlacementStrategy::UseGeneralPurposeColumns);

    let mut owned_cs = builder.build(());

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
    let owned_cs = owned_cs.into_assembly_for_repeated_proving(&finalization_hint);

    owned_cs
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
    assert_eq!(
        expected_proof.final_fri_monomials,
        actual_proof.final_fri_monomials
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
        expected_proof.queries_per_fri_repetition.len(),
        actual_proof.queries_per_fri_repetition.len(),
    );
    for (expected_fri_query, actual_fri_query) in expected_proof
        .queries_per_fri_repetition
        .iter()
        .zip(actual_proof.queries_per_fri_repetition.iter())
    {
        assert_eq!(
            expected_fri_query.witness_query.leaf_elements,
            actual_fri_query.witness_query.leaf_elements
        );
        assert_eq!(
            expected_fri_query.witness_query.proof,
            actual_fri_query.witness_query.proof
        );
        assert_eq!(
            expected_fri_query.stage_2_query.leaf_elements,
            actual_fri_query.stage_2_query.leaf_elements
        );
        assert_eq!(
            expected_fri_query.stage_2_query.proof,
            actual_fri_query.stage_2_query.proof
        );
        assert_eq!(
            expected_fri_query.quotient_query.leaf_elements,
            actual_fri_query.quotient_query.leaf_elements
        );
        assert_eq!(
            expected_fri_query.quotient_query.proof,
            actual_fri_query.quotient_query.proof
        );

        assert_eq!(
            expected_fri_query.setup_query.leaf_elements,
            actual_fri_query.setup_query.leaf_elements
        );
        assert_eq!(
            expected_fri_query.setup_query.proof,
            actual_fri_query.setup_query.proof,
        );
        assert_eq!(expected_fri_query.fri_queries, actual_fri_query.fri_queries,);
    }
    assert_eq!(expected_proof.pow_challenge, actual_proof.pow_challenge);
}

#[test]
#[ignore]
fn test_reference_proof_for_sha256() {
    let mut cs = init_cs_for_sha256::<DevCSConfig>();

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

pub fn init_cs_for_sha256<CFG: CSConfig>() -> CSReferenceAssembly<F, F, CFG> {
    use blake2::Digest;
    // let len = 10 * 64 + 64 - 9;
    // let len = 2 * (1 << 10);
    let len = 2 * (1 << 2);
    use rand::{Rng, SeedableRng};
    let mut rng = rand::rngs::StdRng::seed_from_u64(42 as u64);

    let mut input = vec![];
    for _ in 0..len {
        let byte: u8 = rng.gen();
        input.push(byte);
    }

    let mut hasher = sha2::Sha256::new();
    hasher.update(&input);
    let reference_output = hasher.finalize();

    let geometry = CSGeometry {
        num_columns_under_copy_permutation: 20,
        num_witness_columns: 0,
        num_constant_columns: 4,
        max_allowed_constraint_degree: 4,
    };

    use boojum::cs::cs_builder_reference::*;
    let builder_impl =
        CsReferenceImplementationBuilder::<F, F, CFG>::new(geometry, 1 << 25, 1 << 19);
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
    let builder =
        NopGate::configure_builder(builder, GatePlacementStrategy::UseGeneralPurposeColumns);

    let mut owned_cs = builder.build(());

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

    let mut circuit_input = vec![];

    let cs = &mut owned_cs;

    for el in input.iter() {
        let el = UInt8::allocate_checked(cs, *el);
        circuit_input.push(el);
    }

    let output = sha256(cs, &circuit_input);
    let output = hex::encode(&(output.witness_hook(&*cs))().unwrap());
    let reference_output = hex::encode(reference_output.as_slice());
    assert_eq!(output, reference_output);

    owned_cs.pad_and_shrink();
    let mut owned_cs = owned_cs.into_assembly();
    owned_cs.wait_for_witness();
    let _worker = Worker::new_with_num_threads(8);
    // assert!(owned_cs.check_if_satisfied(&worker));

    owned_cs
}

#[cfg(test)]
#[cfg(feature = "zksync")]
mod zksync {
    use std::path::PathBuf;

    use super::*;

    use boojum::cs::implementations::fast_serialization::MemcopySerializable;
    use boojum::cs::implementations::polynomial_storage::SetupBaseStorage;
    use circuit_definitions::aux_definitions::witness_oracle::VmWitnessOracle;
    use circuit_definitions::circuit_definitions::base_layer::ZkSyncBaseLayerCircuit;
    use circuit_definitions::ZkSyncDefaultRoundFunction;

    pub type ZksyncProof = Proof<F, DefaultTreeHasher, GoldilocksExt2>;

    const TEST_DATA_ROOT_DIR: &str = "./test_data";
    const DEFAULT_CIRCUIT_INPUT: &str = "default.circuit";

    use crate::synthesis_utils::{
        init_cs_for_external_proving, init_or_synthesize_assembly, synth_circuit_for_proving,
        synth_circuit_for_setup, CircuitWrapper,
    };

    #[allow(dead_code)]
    pub type BaseLayerCircuit =
        ZkSyncBaseLayerCircuit<F, VmWitnessOracle<F>, ZkSyncDefaultRoundFunction>;

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
    fn compare_proofs_for_all_zksync_circuits() -> CudaResult<()> {
        let worker = &Worker::new();
        let _ctx = ProverContext::create()?;

        for main_dir in ["base", "leaf", "node"] {
            let data_dir = format!("./test_data/{}", main_dir);
            dbg!(&data_dir);
            let circuits = scan_directory_for_circuits(&data_dir);
            let reference_proofs = scan_directory_for_proofs(&data_dir);

            for (circuit, _reference_proof) in
                circuits.into_iter().zip(reference_proofs.into_iter())
            {
                let reference_proof_path =
                    format!("{}/{}.cpu.proof", data_dir, circuit.numeric_circuit_type());

                let reference_proof_path = std::path::Path::new(&reference_proof_path);

                let gpu_proof_path =
                    format!("{}/{}.gpu.proof", data_dir, circuit.numeric_circuit_type());

                let gpu_proof_path = std::path::Path::new(&gpu_proof_path);

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
                let (setup_base, _setup, vk, setup_tree, vars_hint, _witness_hints) = setup_cs
                    .get_full_setup(
                        worker,
                        proof_config.fri_lde_factor,
                        proof_config.merkle_tree_cap_size,
                    );

                let gpu_setup = GpuSetup::<Global>::from_setup_and_hints(
                    setup_base.clone(),
                    setup_tree,
                    vars_hint.clone(),
                )?;

                println!("gpu proving");

                let gpu_proof = {
                    let proving_cs = synth_circuit_for_proving(circuit.clone(), &finalization_hint);
                    gpu_prove::<_, DefaultTranscript, DefaultTreeHasher, NoPow, Global>(
                        proving_cs,
                        proof_config,
                        &gpu_setup,
                        &vk,
                        (),
                        worker,
                    )
                    .expect("gpu proof")
                };

                let reference_proof_file = std::fs::File::open(reference_proof_path).unwrap();
                assert!(circuit.verify_proof(&vk, &gpu_proof));
                let reference_proof = bincode::deserialize_from(&reference_proof_file).unwrap();
                compare_proofs(&reference_proof, &gpu_proof);
                let proof_file = std::fs::File::create(gpu_proof_path).unwrap();

                bincode::serialize_into(proof_file, &gpu_proof).expect("write proof into file");
            }
        }

        Ok(())
    }

    #[serial]
    #[test]
    #[ignore]
    fn generate_reference_proofs_for_all_zksync_circuits() -> CudaResult<()> {
        let worker = &Worker::new();
        let _ctx = ProverContext::create()?;

        for main_dir in ["base", "leaf", "node"] {
            let data_dir = format!("./test_data/{}", main_dir);
            dbg!(&data_dir);
            let circuits = scan_directory_for_circuits(&data_dir);

            for circuit in circuits.into_iter() {
                if std::path::Path::new(&format!(
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

                let (cs, finalization_hint) =
                    init_or_synthesize_assembly::<DevCSConfig, true>(circuit.clone(), None);
                let _finalization_hint = finalization_hint.unwrap();
                let (setup_base, setup, vk, setup_tree, vars_hint, witness_hints) = cs
                    .get_full_setup(
                        worker,
                        proof_config.fri_lde_factor,
                        proof_config.merkle_tree_cap_size,
                    );

                println!("cpu proving");
                let reference_proof = {
                    cs.prove_from_precomputations::<EXT, DefaultTranscript, DefaultTreeHasher, NoPow>(
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

        Ok(())
    }

    #[serial]
    #[test]
    #[ignore]
    fn compare_proofs_for_single_zksync_circuit_in_single_shot() {
        let circuit_file_path = if let Ok(circuit_file) = std::env::var("CIRCUIT_FILE") {
            circuit_file
        } else {
            format!("./test_data/{}", DEFAULT_CIRCUIT_INPUT)
        };
        let data = std::fs::read(circuit_file_path).expect("circuit file");
        let circuit: CircuitWrapper = bincode::deserialize(&data).expect("circuit");
        let _ctx = ProverContext::create().expect("gpu prover context");

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

        let proving_cs = synth_circuit_for_proving(circuit.clone(), &finalization_hint);

        println!("gpu proving");
        let gpu_setup = GpuSetup::<Global>::from_setup_and_hints(
            setup_base.clone(),
            clone_reference_tree(&setup_tree),
            vars_hint.clone(),
        )
        .expect("gpu setup");
        let gpu_proof = {
            gpu_prove::<_, DefaultTranscript, DefaultTreeHasher, NoPow, Global>(
                proving_cs,
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
                    &witness_hints,
                    (),
                    worker,
                )
        };
        circuit.verify_proof(&vk, &gpu_proof);
        compare_proofs(&reference_proof, &gpu_proof);
    }

    #[serial]
    #[test]
    #[ignore]
    fn compare_proofs_with_external_synthesis_for_single_zksync_circuit_in_single_shot() {
        let circuit_file_path = if let Ok(circuit_file) = std::env::var("CIRCUIT_FILE") {
            circuit_file
        } else {
            format!("./test_data/{}", DEFAULT_CIRCUIT_INPUT)
        };
        let data = std::fs::read(circuit_file_path).expect("circuit file");
        let circuit: CircuitWrapper = bincode::deserialize(&data).expect("circuit");
        let _ctx = ProverContext::create().expect("gpu prover context");

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

        let mut proving_cs = synth_circuit_for_proving(circuit.clone(), &finalization_hint);

        println!("gpu proving");
        let gpu_setup = GpuSetup::<Global>::from_setup_and_hints(
            setup_base.clone(),
            clone_reference_tree(&setup_tree),
            vars_hint.clone(),
        )
        .expect("gpu setup");
        let gpu_proof = {
            let witness = proving_cs.materialize_witness_vec();
            let reusable_cs = init_cs_for_external_proving(circuit.clone(), &finalization_hint);
            gpu_prove_from_external_witness_data::<
                _,
                DefaultTranscript,
                DefaultTreeHasher,
                NoPow,
                Global,
            >(
                reusable_cs,
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
                    &witness_hints,
                    (),
                    worker,
                )
        };
        circuit.verify_proof(&vk, &gpu_proof);
        compare_proofs(&reference_proof, &gpu_proof);
    }

    #[test]
    #[ignore]
    fn test_reference_proof_for_circuit() {
        let circuit_file_path = if let Ok(circuit_file) = std::env::var("CIRCUIT_FILE") {
            circuit_file
        } else {
            format!("./test_data/{}", DEFAULT_CIRCUIT_INPUT)
        };
        let data = std::fs::read(circuit_file_path).expect("circuit file");
        let circuit: CircuitWrapper = bincode::deserialize(&data).expect("circuit");
        let _ctx = ProverContext::create().expect("gpu prover context");

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

        for main_dir in ["base", "leaf", "node"] {
            let data_dir = format!("./test_data/{}", main_dir);
            let circuits = scan_directory_for_circuits(main_dir);

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

        for main_dir in ["base", "leaf", "node"] {
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
                let (variables_hint, _) = setup_cs.create_copy_hints();

                let gpu_setup = GpuSetup::<Global>::from_setup_and_hints(
                    reference_base_setup,
                    reference_setup_tree,
                    variables_hint,
                )
                .expect("gpu setup");

                let setup_file = std::fs::File::create(&setup_file_path).unwrap();
                bincode::serialize_into(&setup_file, &gpu_setup).unwrap();
                println!("Setup written into file {}", setup_file_path);
            }
        }
    }
}
