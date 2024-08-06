use boojum::{
    cs::{implementations::witness::WitnessVec, traits::GoodAllocator, LookupParameters},
    field::U64Representable,
    worker::Worker,
};
use era_cudart::slice::CudaSlice;
use std::ops::Range;

use super::*;

use crate::cs::variable_assignment;
use crate::data_structures::cache::StorageCache;

#[allow(dead_code)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum TracePolyType {
    Variable,
    Witness,
    Multiplicity,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct TraceLayout {
    pub num_variable_cols: usize,
    pub num_witness_cols: usize,
    pub num_multiplicity_cols: usize,
}

impl GenericStorageLayout for TraceLayout {
    type PolyType = TracePolyType;

    fn num_polys(&self) -> usize {
        self.num_variable_cols + self.num_witness_cols + self.num_multiplicity_cols
    }

    fn poly_range(&self, poly_type: Self::PolyType) -> (Range<usize>, Self) {
        let start = match poly_type {
            TracePolyType::Variable => 0,
            TracePolyType::Witness => self.num_variable_cols,
            TracePolyType::Multiplicity => self.num_variable_cols + self.num_witness_cols,
        };
        let len = match poly_type {
            TracePolyType::Variable => self.num_variable_cols,
            TracePolyType::Witness => self.num_witness_cols,
            TracePolyType::Multiplicity => self.num_multiplicity_cols,
        };
        let range = start..start + len;
        let layout = TraceLayout {
            num_variable_cols: match poly_type {
                TracePolyType::Variable => len,
                _ => 0,
            },
            num_witness_cols: match poly_type {
                TracePolyType::Witness => len,
                _ => 0,
            },
            num_multiplicity_cols: match poly_type {
                TracePolyType::Multiplicity => len,
                _ => 0,
            },
        };
        (range, layout)
    }
}

pub type GenericTraceStorage<P> = GenericStorage<P, TraceLayout>;

impl<'a> LeafSourceQuery for TracePolynomials<'a, CosetEvaluations> {
    fn get_leaf_sources(
        &self,
        _coset_idx: usize,
        _lde_degree: usize,
        _domain_size: usize,
        row_idx: usize,
        _: usize,
    ) -> CudaResult<Vec<F>> {
        let TracePolynomials {
            variable_cols,
            witness_cols,
            multiplicity_cols,
        } = self;

        let num_polys = variable_cols.len() + witness_cols.len() + multiplicity_cols.len();

        let mut leaf_sources = Vec::with_capacity(num_polys);
        for col in variable_cols.iter() {
            let el = col.storage.clone_el_to_host(row_idx)?;
            leaf_sources.push(el);
        }

        for col in witness_cols.iter() {
            let el = col.storage.clone_el_to_host(row_idx)?;
            leaf_sources.push(el);
        }

        for col in multiplicity_cols.iter() {
            let el = col.storage.clone_el_to_host(row_idx)?;
            leaf_sources.push(el);
        }
        assert_eq!(leaf_sources.len(), num_polys);

        Ok(leaf_sources)
    }
}

pub struct TracePolynomials<'a, P: PolyForm> {
    pub variable_cols: Vec<Poly<'a, P>>,
    pub witness_cols: Vec<Poly<'a, P>>,
    pub multiplicity_cols: Vec<Poly<'a, P>>,
}

impl<P: PolyForm> GenericTraceStorage<P> {
    pub fn as_polynomials(&self) -> TracePolynomials<P> {
        let TraceLayout {
            num_variable_cols,
            num_witness_cols,
            num_multiplicity_cols,
        } = self.layout;

        let all_polys = self.as_polys();
        let mut all_polys_iter = all_polys.into_iter();

        let mut variable_cols = vec![];
        for _ in 0..num_variable_cols {
            variable_cols.push(all_polys_iter.next().unwrap());
        }
        let mut witness_cols = vec![];
        for _ in 0..num_witness_cols {
            witness_cols.push(all_polys_iter.next().unwrap());
        }

        let mut multiplicity_cols = vec![];
        for _ in 0..num_multiplicity_cols {
            multiplicity_cols.push(all_polys_iter.next().unwrap());
        }
        assert!(all_polys_iter.next().is_none());
        assert_multiset_adjacent_base(&[&variable_cols, &witness_cols, &multiplicity_cols]);

        TracePolynomials {
            variable_cols,
            witness_cols,
            multiplicity_cols,
        }
    }
}

impl GenericTraceStorage<LagrangeBasis> {
    pub fn fill_from_remote_witness_data(
        variable_indexes: &DVec<u32>,
        witness_indexes: &DVec<u32>,
        witness_data: &WitnessVec<F>,
        lookup_parameters: &LookupParameters,
        worker: &Worker,
        mut storage: GenericTraceStorage<Undefined>,
    ) -> CudaResult<Self> {
        let trace_layout = storage.layout.clone();
        let num_polys = storage.num_polys();
        let domain_size = storage.domain_size();
        let TraceLayout {
            num_variable_cols,
            num_witness_cols,
            num_multiplicity_cols,
        } = trace_layout;
        assert_eq!(num_variable_cols * domain_size, variable_indexes.len());
        assert_eq!(num_witness_cols * domain_size, witness_indexes.len());
        assert!(num_multiplicity_cols <= 1);
        let WitnessVec {
            all_values,
            multiplicities,
            ..
        } = witness_data;
        let mut d_variable_values = dvec!(all_values.len());
        let pending_callbacks =
            mem::h2d_buffered(&all_values, &mut d_variable_values, domain_size / 2, worker)?;
        get_h2d_stream().synchronize()?;
        drop(pending_callbacks);
        let remaining_raw_storage = storage.as_single_slice_mut();
        assert_eq!(remaining_raw_storage.len(), num_polys * domain_size);
        let (variables_raw_storage, remaining_raw_storage) =
            remaining_raw_storage.split_at_mut(num_variable_cols * domain_size);
        variable_assignment(variable_indexes, &d_variable_values, variables_raw_storage)?;
        let size_of_all_witness_cols = num_witness_cols * domain_size;
        let (witnesses_raw_storage, multiplicities_raw_storage) =
            remaining_raw_storage.split_at_mut(size_of_all_witness_cols);
        if !witness_indexes.is_empty() {
            variable_assignment(witness_indexes, &d_variable_values, witnesses_raw_storage)?;
        } else {
            assert!(witnesses_raw_storage.is_empty());
        }
        drop(d_variable_values);
        // we can transform and pad multiplicities on the host then transfer to the device
        if lookup_parameters.lookup_is_allowed() {
            let size_of_all_multiplicity_cols = num_multiplicity_cols * domain_size;
            assert_eq!(
                multiplicities_raw_storage.len(),
                size_of_all_multiplicity_cols
            );
            let num_actual_multiplicities = multiplicities.len();
            // we receive witness data from network so that they are minimal in size
            // and may needs padding
            assert!(num_actual_multiplicities <= multiplicities_raw_storage.len());
            let mut transformed_multiplicities = vec![F::ZERO; num_actual_multiplicities];
            if !is_dry_run()? {
                worker.scope(num_actual_multiplicities, |scope, chunk_size| {
                    for (src_chunk, dst_chunk) in multiplicities
                        .chunks(chunk_size)
                        .zip(transformed_multiplicities.chunks_mut(chunk_size))
                    {
                        scope.spawn(|_| {
                            for (src, dst) in src_chunk.iter().zip(dst_chunk.iter_mut()) {
                                *dst = F::from_u64_unchecked(*src as u64);
                            }
                        })
                    }
                });
            }
            let (actual_multiplicities_raw_storage, padding) =
                multiplicities_raw_storage.split_at_mut(num_actual_multiplicities);
            let pending_callbacks = mem::h2d_buffered(
                &transformed_multiplicities,
                actual_multiplicities_raw_storage,
                domain_size / 2,
                worker,
            )?;
            if !padding.is_empty() {
                helpers::set_zero(padding)?;
            }
            get_h2d_stream().synchronize()?;
            drop(pending_callbacks);
        } else {
            assert!(multiplicities_raw_storage.is_empty())
        };
        let result = unsafe { storage.transmute() };
        Ok(result)
    }
}

impl GenericTraceStorage<CosetEvaluations> {
    pub(crate) fn barycentric_evaluate<A: GoodAllocator>(
        &self,
        bases: &PrecomputedBasisForBarycentric,
    ) -> CudaResult<Vec<EF, A>> {
        batch_barycentric_evaluate_base(self, bases, self.domain_size(), self.num_polys())
    }
}

pub type TraceCache = StorageCache<TraceLayout, ()>;
