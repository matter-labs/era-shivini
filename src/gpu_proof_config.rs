use crate::synthesis_utils::{
    get_verifier_for_base_layer_circuit, get_verifier_for_recursive_layer_circuit,
};
use boojum::config::ProvingCSConfig;
use boojum::cs::implementations::reference_cs::CSReferenceAssembly;
use boojum::cs::implementations::verifier::{
    TypeErasedGateEvaluationVerificationFunction, Verifier,
};
use boojum::cs::traits::evaluator::{
    GatePlacementType, PerChunkOffset, TypeErasedGateEvaluationFunction,
};
use boojum::cs::traits::gate::GatePlacementStrategy;
use boojum::field::goldilocks::{GoldilocksExt2, GoldilocksField};
use boojum::field::traits::field_like::PrimeFieldLikeVectorized;
use boojum::field::FieldExtension;
use circuit_definitions::circuit_definitions::base_layer::ZkSyncBaseLayerCircuit;
use circuit_definitions::circuit_definitions::recursion_layer::ZkSyncRecursiveLayerCircuit;
use std::any::TypeId;
use std::collections::HashMap;

type F = GoldilocksField;
type EXT = GoldilocksExt2;

pub(crate) struct EvaluatorData {
    pub debug_name: String,
    pub unique_name: String,
    pub max_constraint_degree: usize,
    pub num_quotient_terms: usize,
    pub total_quotient_terms_over_all_repetitions: usize,
    pub num_repetitions_on_row: usize,
    pub placement_type: GatePlacementType,
}

impl<P: PrimeFieldLikeVectorized<Base = F>> From<&TypeErasedGateEvaluationFunction<F, P>>
    for EvaluatorData
{
    fn from(value: &TypeErasedGateEvaluationFunction<F, P>) -> Self {
        let debug_name = value.debug_name.clone();
        let unique_name = value.unique_name.clone();
        let max_constraint_degree = value.max_constraint_degree;
        let num_quotient_terms = value.num_quotient_terms;
        let total_quotient_terms_over_all_repetitions =
            value.total_quotient_terms_over_all_repetitions;
        let num_repetitions_on_row = value.num_repetitions_on_row;
        let placement_type = value.placement_type;
        Self {
            debug_name,
            unique_name,
            max_constraint_degree,
            num_quotient_terms,
            total_quotient_terms_over_all_repetitions,
            num_repetitions_on_row,
            placement_type,
        }
    }
}

impl<EXT: FieldExtension<2, BaseField = F>>
    From<&TypeErasedGateEvaluationVerificationFunction<F, EXT>> for EvaluatorData
{
    fn from(value: &TypeErasedGateEvaluationVerificationFunction<F, EXT>) -> Self {
        let debug_name = value.debug_name.clone();
        let unique_name = value.unique_name.clone();
        let max_constraint_degree = value.max_constraint_degree;
        let num_quotient_terms = value.num_quotient_terms;
        let total_quotient_terms_over_all_repetitions =
            value.total_quotient_terms_over_all_repetitions;
        let num_repetitions_on_row = value.num_repetitions_on_row;
        let placement_type = value.placement_type;
        Self {
            debug_name,
            unique_name,
            max_constraint_degree,
            num_quotient_terms,
            total_quotient_terms_over_all_repetitions,
            num_repetitions_on_row,
            placement_type,
        }
    }
}

pub struct GpuProofConfig {
    pub(crate) gate_type_ids_for_specialized_columns: Vec<TypeId>,
    pub(crate) evaluators_over_specialized_columns: Vec<EvaluatorData>,
    pub(crate) offsets_for_specialized_evaluators: Vec<(PerChunkOffset, PerChunkOffset, usize)>,
    pub(crate) evaluators_over_general_purpose_columns: Vec<EvaluatorData>,
    pub(crate) placement_strategies: HashMap<TypeId, GatePlacementStrategy>,
}

impl GpuProofConfig {
    pub fn from_assembly<P: PrimeFieldLikeVectorized<Base = F>>(
        cs: &CSReferenceAssembly<F, P, ProvingCSConfig>,
    ) -> Self {
        let evaluation_data_over_specialized_columns = &cs.evaluation_data_over_specialized_columns;
        let gate_type_ids_for_specialized_columns = evaluation_data_over_specialized_columns
            .gate_type_ids_for_specialized_columns
            .clone();
        let evaluators_over_specialized_columns = evaluation_data_over_specialized_columns
            .evaluators_over_specialized_columns
            .iter()
            .map(|x| x.into())
            .collect();
        let evaluators_over_general_purpose_columns = cs
            .evaluation_data_over_general_purpose_columns
            .evaluators_over_general_purpose_columns
            .iter()
            .map(|x| x.into())
            .collect();
        let offsets_for_specialized_evaluators = evaluation_data_over_specialized_columns
            .offsets_for_specialized_evaluators
            .clone();
        let placement_strategies = cs.placement_strategies.clone();
        Self {
            gate_type_ids_for_specialized_columns,
            evaluators_over_specialized_columns,
            offsets_for_specialized_evaluators,
            evaluators_over_general_purpose_columns,
            placement_strategies,
        }
    }

    pub fn from_verifier(verifier: &Verifier<F, EXT>) -> Self {
        let gate_type_ids_for_specialized_columns =
            verifier.gate_type_ids_for_specialized_columns.clone();
        let evaluators_over_specialized_columns = verifier
            .evaluators_over_specialized_columns
            .iter()
            .map(|x| x.into())
            .collect();
        let offsets_for_specialized_evaluators =
            verifier.offsets_for_specialized_evaluators.clone();
        let evaluators_over_general_purpose_columns = verifier
            .evaluators_over_general_purpose_columns
            .iter()
            .map(|x| x.into())
            .collect();
        let placement_strategies = verifier.placement_strategies.clone();
        Self {
            gate_type_ids_for_specialized_columns,
            evaluators_over_specialized_columns,
            offsets_for_specialized_evaluators,
            evaluators_over_general_purpose_columns,
            placement_strategies,
        }
    }

    pub fn from_base_layer_circuit(circuit: &ZkSyncBaseLayerCircuit) -> Self {
        Self::from_verifier(&get_verifier_for_base_layer_circuit(circuit))
    }

    pub fn from_recursive_layer_circuit(circuit: &ZkSyncRecursiveLayerCircuit) -> Self {
        Self::from_verifier(&get_verifier_for_recursive_layer_circuit(circuit))
    }

    #[cfg(test)]
    pub(crate) fn from_circuit_wrapper(wrapper: &crate::synthesis_utils::CircuitWrapper) -> Self {
        use crate::synthesis_utils::CircuitWrapper::*;
        match wrapper {
            Base(circuit) => Self::from_base_layer_circuit(circuit),
            Recursive(circuit) => Self::from_recursive_layer_circuit(circuit),
        }
    }
}
