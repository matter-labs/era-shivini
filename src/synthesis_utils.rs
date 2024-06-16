use boojum::config::{
    CSConfig, CSSetupConfig, CSWitnessEvaluationConfig, ProvingCSConfig, SetupCSConfig,
};
use boojum::cs::cs_builder::new_builder;
use boojum::cs::cs_builder_reference::CsReferenceImplementationBuilder;
use boojum::cs::implementations::pow::NoPow;
use boojum::cs::implementations::proof::Proof;
use boojum::cs::implementations::prover::ProofConfig;
use boojum::cs::implementations::reference_cs::{CSReferenceAssembly, CSReferenceImplementation};
use boojum::cs::implementations::setup::FinalizationHintsForProver;
use boojum::cs::implementations::verifier::{VerificationKey, Verifier};
use boojum::cs::traits::GoodAllocator;
use boojum::cs::{CSGeometry, GateConfigurationHolder, StaticToolboxHolder};
use boojum::field::goldilocks::{GoldilocksExt2, GoldilocksField};
use circuit_definitions::circuit_definitions::base_layer::ZkSyncBaseLayerCircuit;
use circuit_definitions::circuit_definitions::recursion_layer::ZkSyncRecursiveLayerCircuit;
#[allow(unused_imports)]
use circuit_definitions::circuit_definitions::{
    ZkSyncUniformCircuitInstance, ZkSyncUniformSynthesisFunction,
};
use circuit_definitions::{
    base_layer_proof_config, recursion_layer_proof_config, ZkSyncDefaultRoundFunction,
};

use crate::{DefaultTranscript, DefaultTreeHasher};

type F = GoldilocksField;
type P = F;
#[allow(dead_code)]
type ZksyncProof = Proof<F, DefaultTreeHasher, GoldilocksExt2>;
#[allow(dead_code)]
type EXT = GoldilocksExt2;

#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub(crate) enum CircuitWrapper {
    Base(ZkSyncBaseLayerCircuit),
    Recursive(ZkSyncRecursiveLayerCircuit),
}

impl CircuitWrapper {
    pub fn geometry(&self) -> CSGeometry {
        match self {
            CircuitWrapper::Base(inner) => inner.geometry(),
            CircuitWrapper::Recursive(inner) => inner.geometry(),
        }
    }
    pub fn size_hint(&self) -> (Option<usize>, Option<usize>) {
        match self {
            CircuitWrapper::Base(inner) => inner.size_hint(),
            CircuitWrapper::Recursive(inner) => inner.size_hint(),
        }
    }

    #[allow(dead_code)]
    pub fn numeric_circuit_type(&self) -> u8 {
        match self {
            CircuitWrapper::Base(inner) => inner.numeric_circuit_type(),
            CircuitWrapper::Recursive(inner) => inner.numeric_circuit_type(),
        }
    }

    #[allow(dead_code)]
    pub fn short_description(&self) -> &str {
        match self {
            CircuitWrapper::Base(inner) => inner.short_description(),
            CircuitWrapper::Recursive(inner) => inner.short_description(),
        }
    }

    #[allow(dead_code)]
    pub fn into_base_layer(self) -> ZkSyncBaseLayerCircuit {
        match self {
            CircuitWrapper::Base(inner) => inner,
            _ => unimplemented!(),
        }
    }

    #[allow(dead_code)]
    pub fn into_recursive_layer(self) -> ZkSyncRecursiveLayerCircuit {
        match self {
            CircuitWrapper::Recursive(inner) => inner,
            _ => unimplemented!(),
        }
    }

    #[allow(dead_code)]
    pub fn as_base_layer(&self) -> &ZkSyncBaseLayerCircuit {
        match self {
            CircuitWrapper::Base(inner) => inner,
            _ => unimplemented!(),
        }
    }

    #[allow(dead_code)]
    pub fn as_recursive_layer(&self) -> &ZkSyncRecursiveLayerCircuit {
        match self {
            CircuitWrapper::Recursive(inner) => inner,
            _ => unimplemented!(),
        }
    }

    #[allow(dead_code)]
    pub fn is_base_layer(&self) -> bool {
        matches!(self, CircuitWrapper::Base(_))
    }

    #[allow(dead_code)]
    pub fn proof_config(&self) -> ProofConfig {
        match self {
            CircuitWrapper::Base(_) => base_layer_proof_config(),
            CircuitWrapper::Recursive(_) => recursion_layer_proof_config(),
        }
    }

    #[allow(dead_code)]
    pub fn verify_proof(
        &self,
        vk: &VerificationKey<F, DefaultTreeHasher>,
        proof: &ZksyncProof,
    ) -> bool {
        let verifier = self.get_verifier();
        verifier.verify::<DefaultTreeHasher, DefaultTranscript, NoPow>((), vk, proof)
    }

    pub(crate) fn get_verifier(&self) -> Verifier<F, EXT> {
        match self {
            CircuitWrapper::Base(inner) => get_verifier_for_base_layer_circuit(inner),
            CircuitWrapper::Recursive(inner) => get_verifier_for_recursive_layer_circuit(inner),
        }
    }
}

pub(crate) fn get_verifier_for_base_layer_circuit(
    circuit: &ZkSyncBaseLayerCircuit,
) -> Verifier<F, EXT> {
    use circuit_definitions::circuit_definitions::verifier_builder::dyn_verifier_builder_for_circuit_type;
    let verifier_builder = dyn_verifier_builder_for_circuit_type(circuit.numeric_circuit_type());
    verifier_builder.create_verifier()
}

pub(crate) fn get_verifier_for_recursive_layer_circuit(
    circuit: &ZkSyncRecursiveLayerCircuit,
) -> Verifier<F, EXT> {
    let verifier_builder = circuit.into_dyn_verifier_builder();
    verifier_builder.create_verifier()
}

#[allow(dead_code)]
pub(crate) fn synth_circuit_for_setup(
    circuit: CircuitWrapper,
) -> (
    CSReferenceAssembly<F, P, SetupCSConfig>,
    FinalizationHintsForProver,
) {
    let (cs, some_finalization_hint) = init_or_synthesize_assembly::<_, true>(circuit, None);
    assert!(cs.next_available_place_idx() > 0);
    (cs, some_finalization_hint.expect("finalization hint"))
}

#[allow(dead_code)]
pub(crate) fn synth_circuit_for_proving(
    circuit: CircuitWrapper,
    hint: &FinalizationHintsForProver,
) -> CSReferenceAssembly<F, F, ProvingCSConfig> {
    let (cs, some_finalization_hint) = init_or_synthesize_assembly::<_, true>(circuit, Some(hint));
    assert!(some_finalization_hint.is_none());
    assert!(cs.next_available_place_idx() > 0);
    cs
}

// called by zksync-era
pub fn init_base_layer_cs_for_repeated_proving(
    circuit: ZkSyncBaseLayerCircuit,
    hint: &FinalizationHintsForProver,
) -> CSReferenceAssembly<F, F, ProvingCSConfig> {
    init_cs_for_external_proving(CircuitWrapper::Base(circuit), hint)
}

// called by zksync-era
pub fn init_recursive_layer_cs_for_repeated_proving(
    circuit: ZkSyncRecursiveLayerCircuit,
    hint: &FinalizationHintsForProver,
) -> CSReferenceAssembly<F, F, ProvingCSConfig> {
    init_cs_for_external_proving(CircuitWrapper::Recursive(circuit), hint)
}

pub(crate) fn init_cs_for_external_proving(
    circuit: CircuitWrapper,
    hint: &FinalizationHintsForProver,
) -> CSReferenceAssembly<F, F, ProvingCSConfig> {
    let (cs, some_finalization_hint) = init_or_synthesize_assembly::<_, false>(circuit, Some(hint));
    assert!(some_finalization_hint.is_none());
    assert_eq!(cs.next_available_place_idx(), 0);
    cs
}

// in init_or_synthesize_assembly, we expect CFG to be either
// ProvingCSConfig or SetupCSConfig
pub trait AllowInitOrSynthesize: CSConfig {}

impl AllowInitOrSynthesize for ProvingCSConfig {}

impl AllowInitOrSynthesize for SetupCSConfig {}

pub(crate) fn init_or_synthesize_assembly<CFG: AllowInitOrSynthesize, const DO_SYNTH: bool>(
    circuit: CircuitWrapper,
    finalization_hint: Option<&FinalizationHintsForProver>,
) -> (
    CSReferenceAssembly<F, F, CFG>,
    Option<FinalizationHintsForProver>,
) {
    let geometry = circuit.geometry();
    let (max_trace_len, num_vars) = circuit.size_hint();

    let builder_impl =
        CsReferenceImplementationBuilder::<F, P, CFG>::new(geometry, max_trace_len.unwrap());
    let builder = new_builder::<_, F>(builder_impl);
    let round_function = ZkSyncDefaultRoundFunction::default();

    // if we are proving then we need finalization hint
    assert_eq!(
        finalization_hint.is_some(),
        <CFG::WitnessConfig as CSWitnessEvaluationConfig>::EVALUATE_WITNESS
    );
    assert_eq!(
        finalization_hint.is_none(),
        <CFG::SetupConfig as CSSetupConfig>::KEEP_SETUP
    );
    // if we are just creating reusable assembly then cs shouldn't be configured for setup
    if !DO_SYNTH {
        assert_eq!(<CFG::SetupConfig as CSSetupConfig>::KEEP_SETUP, false);
        assert_eq!(
            <CFG::WitnessConfig as CSWitnessEvaluationConfig>::EVALUATE_WITNESS,
            true
        );
    }

    fn into_assembly<
        CFG: CSConfig,
        GC: GateConfigurationHolder<F>,
        T: StaticToolboxHolder,
        A: GoodAllocator,
    >(
        mut cs: CSReferenceImplementation<F, P, CFG, GC, T>,
        do_synth: bool,
        finalization_hint: Option<&FinalizationHintsForProver>,
    ) -> (
        CSReferenceAssembly<F, F, CFG, A>,
        Option<FinalizationHintsForProver>,
    ) {
        if <CFG::SetupConfig as CSSetupConfig>::KEEP_SETUP {
            let (_, finalization_hint) = cs.pad_and_shrink();
            (cs.into_assembly(), Some(finalization_hint))
        } else {
            let hint = finalization_hint.unwrap();
            if do_synth {
                cs.pad_and_shrink_using_hint(hint);
                (cs.into_assembly(), None)
            } else {
                (cs.into_assembly_for_repeated_proving(hint), None)
            }
        }
    }

    let builder_arg = num_vars.unwrap();

    match circuit {
        CircuitWrapper::Base(base_circuit) => match base_circuit {
            ZkSyncBaseLayerCircuit::MainVM(inner) => {
                let builder = inner.configure_builder_proxy(builder);
                let mut cs = builder.build(builder_arg);
                inner.add_tables_proxy(&mut cs);
                if DO_SYNTH {
                    inner.synthesize_proxy(&mut cs);
                }
                into_assembly(cs, DO_SYNTH, finalization_hint)
            }
            ZkSyncBaseLayerCircuit::CodeDecommittmentsSorter(inner) => {
                let builder = inner.configure_builder_proxy(builder);
                let mut cs = builder.build(builder_arg);
                inner.add_tables_proxy(&mut cs);
                if DO_SYNTH {
                    inner.synthesize_proxy(&mut cs);
                }
                into_assembly(cs, DO_SYNTH, finalization_hint)
            }
            ZkSyncBaseLayerCircuit::CodeDecommitter(inner) => {
                let builder = inner.configure_builder_proxy(builder);
                let mut cs = builder.build(builder_arg);
                inner.add_tables_proxy(&mut cs);
                if DO_SYNTH {
                    inner.synthesize_proxy(&mut cs);
                }
                into_assembly(cs, DO_SYNTH, finalization_hint)
            }
            ZkSyncBaseLayerCircuit::LogDemuxer(inner) => {
                let builder = inner.configure_builder_proxy(builder);
                let mut cs = builder.build(builder_arg);
                inner.add_tables_proxy(&mut cs);
                if DO_SYNTH {
                    inner.synthesize_proxy(&mut cs);
                }
                into_assembly(cs, DO_SYNTH, finalization_hint)
            }
            ZkSyncBaseLayerCircuit::KeccakRoundFunction(inner) => {
                let builder = inner.configure_builder_proxy(builder);
                let mut cs = builder.build(builder_arg);
                inner.add_tables_proxy(&mut cs);
                if DO_SYNTH {
                    inner.synthesize_proxy(&mut cs);
                }
                into_assembly(cs, DO_SYNTH, finalization_hint)
            }
            ZkSyncBaseLayerCircuit::Sha256RoundFunction(inner) => {
                let builder = inner.configure_builder_proxy(builder);
                let mut cs = builder.build(builder_arg);
                inner.add_tables_proxy(&mut cs);
                if DO_SYNTH {
                    inner.synthesize_proxy(&mut cs);
                }
                into_assembly(cs, DO_SYNTH, finalization_hint)
            }
            ZkSyncBaseLayerCircuit::ECRecover(inner) => {
                let builder = inner.configure_builder_proxy(builder);
                let mut cs = builder.build(builder_arg);
                inner.add_tables_proxy(&mut cs);
                if DO_SYNTH {
                    inner.synthesize_proxy(&mut cs);
                }
                into_assembly(cs, DO_SYNTH, finalization_hint)
            }
            ZkSyncBaseLayerCircuit::RAMPermutation(inner) => {
                let builder = inner.configure_builder_proxy(builder);
                let mut cs = builder.build(builder_arg);
                inner.add_tables_proxy(&mut cs);
                if DO_SYNTH {
                    inner.synthesize_proxy(&mut cs);
                }
                into_assembly(cs, DO_SYNTH, finalization_hint)
            }
            ZkSyncBaseLayerCircuit::StorageSorter(inner) => {
                let builder = inner.configure_builder_proxy(builder);
                let mut cs = builder.build(builder_arg);
                inner.add_tables_proxy(&mut cs);
                if DO_SYNTH {
                    inner.synthesize_proxy(&mut cs);
                }
                into_assembly(cs, DO_SYNTH, finalization_hint)
            }
            ZkSyncBaseLayerCircuit::StorageApplication(inner) => {
                let builder = inner.configure_builder_proxy(builder);
                let mut cs = builder.build(builder_arg);
                inner.add_tables_proxy(&mut cs);
                if DO_SYNTH {
                    inner.synthesize_proxy(&mut cs);
                }
                into_assembly(cs, DO_SYNTH, finalization_hint)
            }
            ZkSyncBaseLayerCircuit::EventsSorter(inner) => {
                let builder = inner.configure_builder_proxy(builder);
                let mut cs = builder.build(builder_arg);
                inner.add_tables_proxy(&mut cs);
                if DO_SYNTH {
                    inner.synthesize_proxy(&mut cs);
                }
                into_assembly(cs, DO_SYNTH, finalization_hint)
            }
            ZkSyncBaseLayerCircuit::L1MessagesSorter(inner) => {
                let builder = inner.configure_builder_proxy(builder);
                let mut cs = builder.build(builder_arg);
                inner.add_tables_proxy(&mut cs);
                if DO_SYNTH {
                    inner.synthesize_proxy(&mut cs);
                }
                into_assembly(cs, DO_SYNTH, finalization_hint)
            }
            ZkSyncBaseLayerCircuit::L1MessagesHasher(inner) => {
                let builder = inner.configure_builder_proxy(builder);
                let mut cs = builder.build(builder_arg);
                inner.add_tables_proxy(&mut cs);
                if DO_SYNTH {
                    inner.synthesize_proxy(&mut cs);
                }
                into_assembly(cs, DO_SYNTH, finalization_hint)
            }
            ZkSyncBaseLayerCircuit::TransientStorageSorter(inner) => {
                let builder = inner.configure_builder_proxy(builder);
                let mut cs = builder.build(builder_arg);
                inner.add_tables_proxy(&mut cs);
                if DO_SYNTH {
                    inner.synthesize_proxy(&mut cs);
                }
                into_assembly(cs, DO_SYNTH, finalization_hint)
            }
            ZkSyncBaseLayerCircuit::Secp256r1Verify(inner) => {
                let builder = inner.configure_builder_proxy(builder);
                let mut cs = builder.build(builder_arg);
                inner.add_tables_proxy(&mut cs);
                if DO_SYNTH {
                    inner.synthesize_proxy(&mut cs);
                }
                into_assembly(cs, DO_SYNTH, finalization_hint)
            }
            ZkSyncBaseLayerCircuit::EIP4844Repack(inner) => {
                let builder = inner.configure_builder_proxy(builder);
                let mut cs = builder.build(builder_arg);
                inner.add_tables_proxy(&mut cs);
                if DO_SYNTH {
                    inner.synthesize_proxy(&mut cs);
                }
                into_assembly(cs, DO_SYNTH, finalization_hint)
            }
        },
        CircuitWrapper::Recursive(recursive_circuit) => match recursive_circuit {
            ZkSyncRecursiveLayerCircuit::SchedulerCircuit(inner) => {
                let builder = inner.configure_builder_proxy(builder);
                let mut cs = builder.build(builder_arg);
                inner.add_tables(&mut cs);
                if DO_SYNTH {
                    inner.synthesize_into_cs(&mut cs, &round_function);
                }
                into_assembly(cs, DO_SYNTH, finalization_hint)
            }
            ZkSyncRecursiveLayerCircuit::NodeLayerCircuit(inner) => {
                let builder = inner.configure_builder_proxy(builder);
                let mut cs = builder.build(builder_arg);
                inner.add_tables(&mut cs);
                if DO_SYNTH {
                    inner.synthesize_into_cs(&mut cs, &round_function);
                }
                into_assembly(cs, DO_SYNTH, finalization_hint)
            }
            ZkSyncRecursiveLayerCircuit::LeafLayerCircuitForMainVM(inner)
            | ZkSyncRecursiveLayerCircuit::LeafLayerCircuitForCodeDecommittmentsSorter(inner)
            | ZkSyncRecursiveLayerCircuit::LeafLayerCircuitForCodeDecommitter(inner)
            | ZkSyncRecursiveLayerCircuit::LeafLayerCircuitForLogDemuxer(inner)
            | ZkSyncRecursiveLayerCircuit::LeafLayerCircuitForKeccakRoundFunction(inner)
            | ZkSyncRecursiveLayerCircuit::LeafLayerCircuitForSha256RoundFunction(inner)
            | ZkSyncRecursiveLayerCircuit::LeafLayerCircuitForECRecover(inner)
            | ZkSyncRecursiveLayerCircuit::LeafLayerCircuitForRAMPermutation(inner)
            | ZkSyncRecursiveLayerCircuit::LeafLayerCircuitForStorageSorter(inner)
            | ZkSyncRecursiveLayerCircuit::LeafLayerCircuitForStorageApplication(inner)
            | ZkSyncRecursiveLayerCircuit::LeafLayerCircuitForEventsSorter(inner)
            | ZkSyncRecursiveLayerCircuit::LeafLayerCircuitForL1MessagesSorter(inner)
            | ZkSyncRecursiveLayerCircuit::LeafLayerCircuitForL1MessagesHasher(inner)
            | ZkSyncRecursiveLayerCircuit::LeafLayerCircuitForTransientStorageSorter(inner)
            | ZkSyncRecursiveLayerCircuit::LeafLayerCircuitForSecp256r1Verify(inner)
            | ZkSyncRecursiveLayerCircuit::LeafLayerCircuitForEIP4844Repack(inner) => {
                let builder = inner.configure_builder_proxy(builder);
                let mut cs = builder.build(builder_arg);
                inner.add_tables(&mut cs);
                if DO_SYNTH {
                    inner.synthesize_into_cs(&mut cs, &round_function);
                }
                into_assembly(cs, DO_SYNTH, finalization_hint)
            }
            ZkSyncRecursiveLayerCircuit::RecursionTipCircuit(inner) => {
                let builder = inner.configure_builder_proxy(builder);
                let mut cs = builder.build(builder_arg);
                inner.add_tables(&mut cs);
                if DO_SYNTH {
                    inner.synthesize_into_cs(&mut cs, &round_function);
                }
                into_assembly(cs, DO_SYNTH, finalization_hint)
            }
        },
    }
}
