use boojum::{
    cs::{
        implementations::{
            proof::OracleQuery,
            witness::{WitnessSet, WitnessVec},
        },
        oracle::TreeHasher,
        traits::GoodAllocator,
        LookupParameters,
    },
    field::U64Representable,
    worker::Worker,
};

use cudart::event::CudaEventCreateFlags;

use crate::cs::{variable_assignment, GpuSetup};

use super::*;

#[derive(Clone, Debug)]
pub struct TraceLayout {
    pub num_variable_cols: usize,
    pub num_witness_cols: usize,
    pub num_multiplicity_cols: usize,
}

impl TraceLayout {
    pub fn from_witness_set(witness_set: &WitnessSet<F>) -> Self {
        assert!(witness_set.variables.len() > 0);
        assert!(witness_set.multiplicities.len() < 2);
        Self {
            num_variable_cols: witness_set.variables.len(),
            num_witness_cols: witness_set.witness.len(),
            num_multiplicity_cols: witness_set.multiplicities.len(),
        }
    }

    #[allow(dead_code)]
    pub fn new(
        num_variable_cols: usize,
        num_witness_cols: usize,
        num_multiplicity_cols: usize,
    ) -> Self {
        Self {
            num_variable_cols,
            num_witness_cols,
            num_multiplicity_cols,
        }
    }

    pub fn num_polys(&self) -> usize {
        self.num_variable_cols + self.num_witness_cols + self.num_multiplicity_cols
    }
}

#[derive(Clone)]
pub struct GenericTraceStorage<P: PolyForm> {
    pub storage: GenericStorage,
    pub coset_idx: Option<usize>,
    pub layout: TraceLayout,
    form: std::marker::PhantomData<P>,
}

impl<P: PolyForm> GenericTraceStorage<P> {
    pub fn allocate(domain_size: usize, layout: TraceLayout) -> CudaResult<Self> {
        assert!(domain_size.is_power_of_two());
        assert!(layout.num_variable_cols > 0);
        let storage = GenericStorage::allocate(layout.num_polys(), domain_size)?;

        let new = GenericTraceStorage {
            storage,
            coset_idx: None,
            layout,
            form: std::marker::PhantomData,
        };

        Ok(new)
    }
    pub fn num_polys(&self) -> usize {
        let num_polys = self.layout.num_polys();
        assert_eq!(num_polys, self.storage.num_polys);
        self.storage.num_polys
    }

    pub fn domain_size(&self) -> usize {
        self.storage.domain_size
    }
}

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
    pub fn as_polynomials<'a>(&'a self) -> TracePolynomials<'a, P> {
        let GenericTraceStorage {
            storage, layout, ..
        } = self;
        let TraceLayout {
            num_variable_cols,
            num_witness_cols,
            num_multiplicity_cols,
        } = layout.clone();

        let all_polys = storage.as_poly_storage();
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
        assert_multiset_adjacent(&[&variable_cols, &witness_cols, &multiplicity_cols]);

        TracePolynomials {
            variable_cols,
            witness_cols,
            multiplicity_cols,
        }
    }
}

pub fn construct_trace_storage_from_external_witness_data<A: GoodAllocator>(
    trace_layout: TraceLayout,
    domain_size: usize,
    setup: &GpuSetup<A>,
    witness_data: &WitnessVec<F>,
    lookup_parameters: &LookupParameters,
    worker: &Worker,
) -> CudaResult<(
    GenericTraceStorage<LagrangeBasis>,
    GenericTraceStorage<MonomialBasis>,
)> {
    let num_polys = trace_layout.num_polys();
    dbg!(num_polys);
    dbg!(domain_size);

    let TraceLayout {
        num_variable_cols,
        num_witness_cols,
        num_multiplicity_cols,
    } = trace_layout;

    dbg!(num_variable_cols);
    dbg!(num_witness_cols);
    dbg!(num_multiplicity_cols);

    let GpuSetup { variables_hint, .. } = setup;

    let WitnessVec {
        all_values,
        multiplicities,
        ..
    } = witness_data;

    let mut raw_trace_storage = GenericStorage::allocate(num_polys, domain_size)?;
    let remaining_storage = raw_trace_storage.as_single_slice_mut();
    assert_eq!(remaining_storage.len(), num_polys * domain_size);

    let size_of_all_variable_cols = num_variable_cols * domain_size;
    let (variables_storage, remaining_storage) =
        remaining_storage.split_at_mut(size_of_all_variable_cols);
    let _d_variable_values = variable_assignment(variables_hint, all_values, variables_storage)?;

    // we can transform and pad multiplicities on the host then transfer to the device
    // TODO: consider to make a select function which allows values that are generic in type
    if lookup_parameters.lookup_is_allowed() {
        let size_of_all_witness_cols = num_witness_cols * domain_size;
        assert_eq!(size_of_all_witness_cols, 0);
        let (_, multiplicities_storage) = remaining_storage.split_at_mut(size_of_all_witness_cols);
        let size_of_all_multiplicity_cols = num_multiplicity_cols * domain_size;
        assert_eq!(multiplicities_storage.len(), size_of_all_multiplicity_cols);
        let num_actual_multiplicities = multiplicities.len();
        // we receive witness data from network so that they are minimal in size
        // and may needs padding
        assert!(num_actual_multiplicities <= multiplicities_storage.len());
        let mut transformed_multiplicities = vec![F::ZERO; num_actual_multiplicities];
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

        mem::h2d(
            &transformed_multiplicities,
            &mut multiplicities_storage[..num_actual_multiplicities],
        )?;
    } else {
        assert!(remaining_storage.is_empty())
    }

    let raw_trace_storage = GenericTraceStorage {
        storage: raw_trace_storage,
        coset_idx: None,
        layout: trace_layout.clone(),
        form: std::marker::PhantomData,
    };

    let monomial_trace_storage = raw_trace_storage.into_monomials()?;

    Ok((raw_trace_storage, monomial_trace_storage))
}

#[allow(dead_code)]
pub fn construct_trace_storage(
    witness_set: &WitnessSet<F>,
) -> CudaResult<(
    GenericTraceStorage<LagrangeBasis>,
    GenericTraceStorage<MonomialBasis>,
)> {
    let WitnessSet {
        variables,
        witness,
        multiplicities,
        ..
    } = witness_set;

    let trace_layout = TraceLayout::from_witness_set(witness_set);
    let num_polys = trace_layout.num_polys();

    let domain_size = variables[0].domain_size();
    let mut raw_storage = GenericStorage::allocate(num_polys, domain_size)?;
    let mut monomial_storage = GenericStorage::allocate(num_polys, domain_size)?;
    for ((src, raw_poly), monomial) in variables
        .iter()
        .chain(witness.iter())
        .chain(multiplicities.iter())
        .zip(raw_storage.as_mut().chunks_mut(domain_size))
        .zip(monomial_storage.as_mut().chunks_mut(domain_size))
    {
        let transferred = CudaEvent::create_with_flags(CudaEventCreateFlags::DISABLE_TIMING)?;
        mem::h2d(&src.storage, raw_poly)?;
        transferred.record(get_h2d_stream())?;
        // we overlap data transfer and ntt computation here
        // so we are fine with many kernel calls
        // TODO: chunk columns by 8
        get_stream().wait_event(&transferred, CudaStreamWaitEventFlags::DEFAULT)?;
        ntt::ifft_into(raw_poly, monomial)?;
    }
    let raw_storage = GenericTraceStorage {
        storage: raw_storage,
        coset_idx: None,
        layout: trace_layout.clone(),
        form: std::marker::PhantomData,
    };
    let monomial_storage = GenericTraceStorage {
        storage: monomial_storage,
        coset_idx: None,
        layout: trace_layout,
        form: std::marker::PhantomData,
    };
    Ok((raw_storage, monomial_storage))
}

impl GenericTraceStorage<LagrangeBasis> {
    pub fn into_monomials(&self) -> CudaResult<GenericTraceStorage<MonomialBasis>> {
        let trace_layout = self.layout.clone();
        let num_polys = trace_layout.num_polys();
        let domain_size = self.domain_size();

        let mut monomial_storage = GenericStorage::allocate(num_polys, domain_size)?;

        ntt::batch_ntt_into(
            self.storage.as_single_slice(),
            monomial_storage.as_single_slice_mut(),
            false,
            true,
            domain_size,
            num_polys,
        )?;

        ntt::batch_bitreverse(monomial_storage.as_single_slice_mut(), domain_size)?;

        let monomials = GenericTraceStorage {
            storage: monomial_storage,
            coset_idx: None,
            layout: trace_layout,
            form: std::marker::PhantomData,
        };

        Ok(monomials)
    }
}

impl GenericTraceStorage<MonomialBasis> {
    #[allow(dead_code)]
    pub fn from_host_values(witness_set: &WitnessSet<F>) -> CudaResult<Self> {
        let WitnessSet {
            variables,
            witness,
            multiplicities,
            ..
        } = witness_set;
        let trace_layout = TraceLayout::from_witness_set(witness_set);
        let num_polys = trace_layout.num_polys();

        let domain_size = variables[0].domain_size();
        let coset = DF::one()?;
        let mut storage = GenericStorage::allocate(num_polys, domain_size)?;
        for (src, poly) in variables
            .iter()
            .chain(witness.iter())
            .chain(multiplicities.iter())
            .zip(storage.as_mut().chunks_mut(domain_size))
        {
            mem::h2d(&src.storage, poly)?;
            // we overlap data transfer and ntt computation here
            // so we are fine with many kernel calls
            ntt::ifft(poly, &coset)?;
        }

        Ok(Self {
            storage,
            coset_idx: None,
            form: std::marker::PhantomData,
            layout: trace_layout,
        })
    }

    pub fn into_coset_eval(
        &self,
        coset_idx: usize,
        lde_degree: usize,
        coset_storage: &mut GenericTraceStorage<CosetEvaluations>,
    ) -> CudaResult<()> {
        let num_polys = self.num_polys();
        let Self { storage, .. } = self;
        let domain_size = storage.domain_size;

        // let mut coset_storage = GenericStorage::allocate(num_polys, domain_size)?;
        ntt::batch_coset_fft_into(
            storage.as_ref(),
            coset_storage.storage.as_mut(),
            coset_idx,
            domain_size,
            lde_degree,
            num_polys,
        )?;

        Ok(())
    }

    #[allow(dead_code)]
    pub fn into_raw_trace(self) -> CudaResult<GenericTraceStorage<LagrangeBasis>> {
        let num_polys = self.num_polys();
        let Self {
            mut storage,
            layout,
            ..
        } = self;
        let domain_size = storage.domain_size;
        let inner_storage = storage.as_mut();

        ntt::batch_bitreverse(inner_storage, domain_size)?;
        let is_input_in_bitreversed = true;

        ntt::batch_ntt(
            inner_storage,
            is_input_in_bitreversed,
            false,
            domain_size,
            num_polys,
        )?;

        let new: GenericTraceStorage<LagrangeBasis> = GenericTraceStorage {
            storage,
            layout,
            coset_idx: None,
            form: std::marker::PhantomData,
        };

        Ok(new)
    }
}

impl GenericTraceStorage<CosetEvaluations> {
    pub fn build_subtree_for_coset(
        &self,
        coset_cap_size: usize,
        coset_idx: usize,
    ) -> CudaResult<(SubTree, Vec<[F; 4]>)> {
        let domain_size = self.domain_size();
        let Self { storage, .. } = self;
        let leaf_sources = <GenericStorage as AsRef<DVec<F>>>::as_ref(&storage);
        let mut subtree = dvec!(2 * NUM_EL_PER_HASH * domain_size);
        let subtree_root =
            compute_tree_cap(leaf_sources, &mut subtree, domain_size, coset_cap_size, 1)?;
        let subtree = SubTree::new(subtree, domain_size, coset_cap_size, coset_idx);
        Ok((subtree, subtree_root))
    }

    pub(crate) fn barycentric_evaluate(
        &self,
        bases: &PrecomputedBasisForBarycentric,
    ) -> CudaResult<Vec<EF>> {
        batch_barycentric_evaluate_base(&self.storage, bases, self.domain_size(), self.num_polys())
    }
}

// we want a holder that either
// - recomputes coset evals each time
// - or keeps coset evals
pub struct TraceCache {
    monomials: GenericTraceStorage<MonomialBasis>,
    cosets: Vec<Option<GenericTraceStorage<CosetEvaluations>>>,
    tmp_coset: GenericTraceStorage<CosetEvaluations>,
    fri_lde_degree: usize,
    used_lde_degree: usize,
}

impl TraceCache {
    pub fn from_monomial(
        monomial_trace: GenericTraceStorage<MonomialBasis>,
        fri_lde_degree: usize,
        used_lde_degree: usize,
    ) -> CudaResult<Self> {
        assert!(fri_lde_degree.is_power_of_two());
        assert!(used_lde_degree.is_power_of_two());
        let cosets = vec![None; fri_lde_degree];
        let tmp_coset = GenericTraceStorage::allocate(
            monomial_trace.domain_size(),
            monomial_trace.layout.clone(),
        )?;
        Ok(Self {
            monomials: monomial_trace,
            cosets: cosets,
            tmp_coset,
            fri_lde_degree,
            used_lde_degree,
        })
    }

    pub fn commit<H: TreeHasher<F, Output = [F; 4]>>(
        &mut self,
        cap_size: usize,
    ) -> CudaResult<(Vec<SubTree>, Vec<[F; 4]>)> {
        let fri_lde_degree = self.fri_lde_degree;
        let _used_lde_degree = self.used_lde_degree;
        let coset_cap_size = coset_cap_size(cap_size, self.fri_lde_degree);
        let mut trace_subtrees = vec![];
        let mut trace_subtree_caps = vec![];

        assert_eq!(self.cosets.len(), fri_lde_degree);

        for coset_idx in 0..fri_lde_degree {
            let coset_values = self.get_or_compute_coset_evals(coset_idx)?;
            let (subtree, subtree_cap) =
                coset_values.build_subtree_for_coset(coset_cap_size, coset_idx)?;
            trace_subtree_caps.push(subtree_cap);
            trace_subtrees.push(subtree);
        }

        let trace_tree_cap = trace_subtree_caps.compute_cap::<H>(&mut trace_subtrees, cap_size)?;

        Ok((trace_subtrees, trace_tree_cap))
    }

    pub fn get_or_compute_coset_evals(
        &mut self,
        coset_idx: usize,
    ) -> CudaResult<&GenericTraceStorage<CosetEvaluations>> {
        assert!(coset_idx < self.used_lde_degree);

        if REMEMBER_COSETS == false || coset_idx >= self.fri_lde_degree {
            self.monomials
                .into_coset_eval(coset_idx, self.used_lde_degree, &mut self.tmp_coset)?;
            return Ok(&self.tmp_coset);
        }

        if self.cosets[coset_idx].is_none() {
            let mut current_storage = GenericTraceStorage::allocate(
                self.monomials.domain_size(),
                self.monomials.layout.clone(),
            )?;
            self.monomials.into_coset_eval(
                coset_idx,
                self.used_lde_degree,
                &mut current_storage,
            )?;
            self.cosets[coset_idx] = Some(current_storage);
        }

        return Ok(self.cosets[coset_idx].as_ref().unwrap());
    }

    pub fn query<H: TreeHasher<F, Output = [F; 4]>>(
        &mut self,
        coset_idx: usize,
        fri_lde_degree: usize,
        row_idx: usize,
        domain_size: usize,
        tree_holder: &TreeCache,
    ) -> CudaResult<OracleQuery<F, H>> {
        let leaf_sources = self.get_or_compute_coset_evals(coset_idx)?;
        tree_holder.get_trace_subtrees().query(
            &leaf_sources.as_polynomials(),
            coset_idx,
            fri_lde_degree,
            row_idx,
            domain_size,
        )
    }

    #[allow(dead_code)]
    pub fn num_polys(&self) -> usize {
        self.monomials.num_polys()
    }
}
