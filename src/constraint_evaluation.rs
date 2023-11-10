use boojum::{
    config::CSConfig,
    cs::{
        gates::lookup_marker::LookupFormalGate,
        implementations::{reference_cs::CSReferenceAssembly, setup::TreeNode},
        traits::{evaluator::PerChunkOffset, gate::GatePlacementStrategy},
    },
};

use super::*;

pub fn get_evaluators_of_general_purpose_cols<
    P: boojum::field::traits::field_like::PrimeFieldLikeVectorized<Base = F>,
    CFG: CSConfig,
>(
    cs: &CSReferenceAssembly<F, P, CFG>,
    selectors_placement: &TreeNode,
) -> Vec<GateEvaluationParams> {
    let mut gates = vec![];
    for (evaluator_idx, (evaluator, _gate_type_id)) in cs
        .evaluation_data_over_general_purpose_columns
        .evaluators_over_general_purpose_columns
        .iter()
        .zip(
            cs.evaluation_data_over_general_purpose_columns
                .gate_type_ids_for_general_purpose_columns
                .iter(),
        )
        .enumerate()
    {
        if evaluator.debug_name
            == "boojum::cs::gates::lookup_marker::LookupGateMarkerFormalEvaluator"
        {
            continue;
        }
        // FIXME ignore NOPGate
        if evaluator.unique_name == "nop_gate_constraint_evaluator" {
            continue;
        }

        let per_chunk_offset = match evaluator.placement_type {
            boojum::cs::traits::evaluator::GatePlacementType::UniqueOnRow => PerChunkOffset::zero(),
            boojum::cs::traits::evaluator::GatePlacementType::MultipleOnRow {
                per_chunk_offset,
            } => per_chunk_offset,
        };

        if let Some(path) = selectors_placement.output_placement(evaluator_idx) {
            let mask = pack_path(&path);
            let count = path.len() as u32;
            let num_repetitions = evaluator.num_repetitions_on_row as u32;
            let cuda_id = boojum_cuda::gates::find_gate_id_by_name(&evaluator.unique_name)
                .expect(&format!("gate id found for {}", evaluator.unique_name));
            let gate = GateEvaluationParams {
                id: cuda_id,
                selector_mask: mask,
                selector_count: count,
                repetitions_count: num_repetitions,
                initial_variables_offset: 0,
                initial_witnesses_offset: 0,
                initial_constants_offset: 0,
                repetition_variables_offset: per_chunk_offset.variables_offset as u32,
                repetition_witnesses_offset: per_chunk_offset.witnesses_offset as u32,
                repetition_constants_offset: per_chunk_offset.constants_offset as u32,
            };
            gates.push(gate);
        } else {
            debug_assert!(evaluator.num_quotient_terms == 0);
        }
    }
    assert!(gates.len() > 0);
    gates
}

pub fn get_specialized_evaluators_from_assembly<
    P: boojum::field::traits::field_like::PrimeFieldLikeVectorized<Base = F>,
    CFG: CSConfig,
>(
    cs: &CSReferenceAssembly<F, P, CFG>,
    selectors_placement: &TreeNode,
) -> Vec<GateEvaluationParams> {
    if cs
        .evaluation_data_over_specialized_columns
        .evaluators_over_specialized_columns
        .len()
        < 1
    {
        return vec![];
    }

    let (_deg, _constants_for_gates_over_general_purpose_columns) =
        selectors_placement.compute_stats();
    let mut gates = vec![];
    for (idx, (evaluator, gate_type_id)) in cs
        .evaluation_data_over_specialized_columns
        .evaluators_over_specialized_columns
        .iter()
        .zip(
            cs.evaluation_data_over_specialized_columns
                .gate_type_ids_for_specialized_columns
                .iter(),
        )
        .enumerate()
    {
        if evaluator.debug_name
            == "boojum::cs::gates::lookup_marker::LookupGateMarkerFormalEvaluator"
        {
            continue;
        }
        if evaluator.unique_name == "nop_gate_constraint_evaluator" {
            continue;
        }
        if gate_type_id == &std::any::TypeId::of::<LookupFormalGate>() {
            continue;
        }
        assert_ne!(
            evaluator.total_quotient_terms_over_all_repetitions, 0,
            "evaluator {} has not contribution to quotient",
            &evaluator.debug_name
        );

        let num_terms = evaluator.num_quotient_terms;
        let placement_strategy = cs
            .placement_strategies
            .get(&gate_type_id)
            .copied()
            .expect("gate must be allowed");
        let GatePlacementStrategy::UseSpecializedColumns {
            num_repetitions,
            share_constants,
        } = placement_strategy
        else {
            unreachable!();
        };
        assert_eq!(evaluator.num_repetitions_on_row, num_repetitions);

        let total_terms = num_terms * num_repetitions;

        let (initial_offset, per_repetition_offset, total_constants_available) = cs
            .evaluation_data_over_specialized_columns
            .offsets_for_specialized_evaluators[idx];

        let _placement_data = (
            num_repetitions,
            share_constants,
            initial_offset,
            per_repetition_offset,
            total_constants_available,
            total_terms,
        );

        let cuda_id = boojum_cuda::gates::find_gate_id_by_name(&evaluator.unique_name)
            .expect(&format!("gate id found for {}", evaluator.unique_name));

        let gate = GateEvaluationParams {
            id: cuda_id,
            selector_mask: 0,
            selector_count: 0,
            repetitions_count: num_repetitions as u32,
            initial_variables_offset: initial_offset.variables_offset as u32,
            initial_witnesses_offset: initial_offset.witnesses_offset as u32,
            initial_constants_offset: initial_offset.constants_offset as u32,
            repetition_variables_offset: per_repetition_offset.variables_offset as u32,
            repetition_witnesses_offset: per_repetition_offset.witnesses_offset as u32,
            repetition_constants_offset: per_repetition_offset.constants_offset as u32,
        };

        gates.push(gate);
    }
    gates
}

pub fn multi_polys_as_single_slice<'a, P: PolyForm>(polys: &[Poly<'a, P>]) -> &'a [F] {
    if polys.is_empty() {
        return &[];
    };
    assert_adjacent_base(polys);
    unsafe {
        let len = polys.len() * polys[0].domain_size();
        std::slice::from_raw_parts(polys[0].storage.as_ref().as_ptr(), len)
    }
}

#[allow(dead_code)]
pub fn multi_polys_as_single_slice_mut<'a, P: PolyForm>(polys: &mut [Poly<'a, P>]) -> &'a mut [F] {
    if polys.is_empty() {
        return &mut [];
    };
    let polys = multi_polys_as_single_slice(polys);
    unsafe { std::slice::from_raw_parts_mut(polys.as_ptr() as _, polys.len()) }
}

pub fn generic_evaluate_constraints_by_coset<'a, 'b>(
    trace_polys: &TracePolynomials<'a, CosetEvaluations>,
    setup_polys: &SetupPolynomials<'a, CosetEvaluations>,
    gates: &[cs_helpers::GateEvaluationParams],
    _selectors_placement: TreeNode,
    domain_size: usize,
    challenge: EF,
    challenge_power_offset: usize,
    quotient: &mut ComplexPoly<'b, CosetEvaluations>,
) -> CudaResult<()>
where
    'a: 'b,
{
    assert_eq!(
        trace_polys.variable_cols[0].domain_size(),
        quotient.domain_size()
    );

    let mut tmp = ComplexPoly::<CosetEvaluations>::zero(domain_size)?;
    let quotient_as_single_slice = unsafe {
        let len = 2 * tmp.domain_size();
        std::slice::from_raw_parts_mut(tmp.c0.storage.as_mut().as_mut_ptr(), len)
    };
    let TracePolynomials {
        variable_cols,
        witness_cols,
        multiplicity_cols: _,
    } = trace_polys;
    let SetupPolynomials { constant_cols, .. } = setup_polys;

    let all_variable_cols = multi_polys_as_single_slice(&variable_cols);
    let all_witness_cols = multi_polys_as_single_slice(&witness_cols);
    let all_constant_cols = multi_polys_as_single_slice(&constant_cols);
    let d_challenge = challenge.into();
    cs_helpers::constraint_evaluation(
        gates,
        all_variable_cols,
        all_witness_cols,
        all_constant_cols,
        &d_challenge,
        challenge_power_offset,
        quotient_as_single_slice,
        domain_size,
    )?;

    quotient.add_assign(&tmp)?;

    Ok(())
}

fn pack_path(path: &[bool]) -> u32 {
    let mut pack = 0;
    for (idx, p) in path.iter().cloned().enumerate() {
        if p {
            pack |= 1 << idx
        }
    }

    pack
}
