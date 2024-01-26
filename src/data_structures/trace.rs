use boojum::{
    cs::{
        implementations::{proof::OracleQuery, witness::WitnessVec},
        oracle::TreeHasher,
        traits::GoodAllocator,
        LookupParameters,
    },
    field::U64Representable,
    worker::Worker,
};
use std::ops::Deref;
use std::rc::Rc;

use crate::cs::{variable_assignment, GpuSetup};

use super::*;

#[derive(Clone, Debug)]
pub struct TraceLayout {
    pub num_variable_cols: usize,
    pub num_witness_cols: usize,
    pub num_multiplicity_cols: usize,
}

impl TraceLayout {
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
        assert_multiset_adjacent_base(&[&variable_cols, &witness_cols, &multiplicity_cols]);

        TracePolynomials {
            variable_cols,
            witness_cols,
            multiplicity_cols,
        }
    }
}

pub fn construct_trace_storage_from_remote_witness_data<A: GoodAllocator>(
    trace_layout: TraceLayout,
    used_lde_degree: usize,
    fri_lde_degree: usize,
    domain_size: usize,
    setup: &GpuSetup<A>,
    witness_data: &WitnessVec<F>,
    lookup_parameters: &LookupParameters,
    worker: &Worker,
) -> CudaResult<(
    GenericTraceStorage<LagrangeBasis>,
    GenericTraceStorage<MonomialBasis>,
    Vec<SubTree>,
    Vec<[F; 4]>,
)> {
    let num_polys = trace_layout.num_polys();
    dbg!(num_polys);
    dbg!(domain_size);

    let TraceLayout {
        num_variable_cols,
        num_witness_cols,
        num_multiplicity_cols,
    } = trace_layout;

    let GpuSetup {
        variables_hint,
        witnesses_hint,
        setup_tree,
        ..
    } = setup;
    let setup_root = setup_tree.get_cap();
    let cap_size = setup_root.len();
    dbg!(&cap_size); // TODO
    assert_eq!(num_variable_cols, variables_hint.len());
    assert_eq!(num_witness_cols, witnesses_hint.len());
    assert!(num_multiplicity_cols <= 1);

    let WitnessVec {
        all_values,
        multiplicities,
        ..
    } = witness_data;
    // let inner_h2d_stream = CudaStream::create()?;
    let inner_h2d_stream = get_stream();
    let mut d_variable_values = dvec!(all_values.len());
    mem::h2d_on_stream(&all_values, &mut d_variable_values, &inner_h2d_stream)?;

    let mut raw_storage = GenericStorage::allocate(num_polys, domain_size)?;
    let mut monomial_storage = GenericStorage::allocate(num_polys, domain_size)?;
    let remaining_raw_storage = raw_storage.as_single_slice_mut();
    let remaining_monomial_storage = monomial_storage.as_single_slice_mut();
    assert_eq!(remaining_raw_storage.len(), num_polys * domain_size);
    assert_eq!(remaining_monomial_storage.len(), num_polys * domain_size);
    // assign variable values
    let (variables_raw_storage, remaining_raw_storage) =
        remaining_raw_storage.split_at_mut(num_variable_cols * domain_size);
    let (variables_monomial_storage, remaining_monomial_storage) =
        remaining_monomial_storage.split_at_mut(num_variable_cols * domain_size);

    for ((variables, d_variables_raw), d_variables_monomial) in variables_hint
        .iter()
        .zip(variables_raw_storage.chunks_mut(domain_size))
        .zip(variables_monomial_storage.chunks_mut(domain_size))
    {
        let transferred = CudaEvent::create_with_flags(CudaEventCreateFlags::DISABLE_TIMING)?;
        let mut d_variable_indexes = dvec!(variables.len());
        mem::h2d_on_stream(variables, &mut d_variable_indexes, &inner_h2d_stream)?;
        transferred.record(&inner_h2d_stream)?;
        get_stream().wait_event(&transferred, CudaStreamWaitEventFlags::DEFAULT)?;
        variable_assignment(&d_variable_indexes, &d_variable_values, d_variables_raw)?;
        let (_, padding) = d_variables_raw.split_at_mut(d_variable_indexes.len());
        if !padding.is_empty() {
            helpers::set_zero(padding)?;
        }
        ntt::intt_into(d_variables_raw, d_variables_monomial)?;
        ntt::bitreverse(d_variables_monomial)?;
    }

    // now witness values
    let size_of_all_witness_cols = num_witness_cols * domain_size;
    let (witnesses_raw_storage, multiplicities_raw_storage) =
        remaining_raw_storage.split_at_mut(size_of_all_witness_cols);
    let (witnesses_monomial_storage, multiplicities_monomial_storage) =
        remaining_monomial_storage.split_at_mut(size_of_all_witness_cols);
    // hints may not be proper rectangular, so look for at least one non-empty col
    let has_witnesses = witnesses_hint.iter().any(|v| !v.is_empty());
    if has_witnesses {
        for ((witnesses, d_witnesses_raw), d_witnesses_monomial) in witnesses_hint
            .iter()
            .zip(witnesses_raw_storage.chunks_mut(domain_size))
            .zip(witnesses_monomial_storage.chunks_mut(domain_size))
        {
            let transferred = CudaEvent::create_with_flags(CudaEventCreateFlags::DISABLE_TIMING)?;
            let mut d_witness_indexes = dvec!(witnesses.len());
            mem::h2d_on_stream(witnesses, &mut d_witness_indexes, &inner_h2d_stream)?;
            transferred.record(&inner_h2d_stream)?;
            get_stream().wait_event(&transferred, CudaStreamWaitEventFlags::DEFAULT)?;
            variable_assignment(&d_witness_indexes, &d_variable_values, d_witnesses_raw)?;
            let (_, padding) = d_witnesses_raw.split_at_mut(d_witness_indexes.len());
            if !padding.is_empty() {
                helpers::set_zero(padding)?;
            }
            ntt::intt_into(d_witnesses_raw, d_witnesses_monomial)?;
            ntt::bitreverse(d_witnesses_monomial)?;
        }
    } else {
        assert!(witnesses_raw_storage.is_empty());
    }

    // we can transform and pad multiplicities on the host then transfer to the device
    // TODO: consider to make a select function which allows values that are generic in type
    if lookup_parameters.lookup_is_allowed() {
        let size_of_all_multiplicity_cols = num_multiplicity_cols * domain_size;
        assert_eq!(
            multiplicities_raw_storage.len(),
            size_of_all_multiplicity_cols
        );
        assert_eq!(
            multiplicities_monomial_storage.len(),
            size_of_all_multiplicity_cols
        );
        let num_actual_multiplicities = multiplicities.len();
        // we receive witness data from network so that they are minimal in size
        // and may needs padding
        assert!(num_actual_multiplicities <= multiplicities_raw_storage.len());
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
        let (actual_multiplicities_raw_storage, padding) =
            multiplicities_raw_storage.split_at_mut(num_actual_multiplicities);
        let transferred = CudaEvent::create_with_flags(CudaEventCreateFlags::DISABLE_TIMING)?;
        mem::h2d_on_stream(
            &transformed_multiplicities,
            actual_multiplicities_raw_storage,
            &inner_h2d_stream,
        )?;
        transferred.record(&inner_h2d_stream)?;
        if !padding.is_empty() {
            helpers::set_zero(padding)?;
        }
        get_stream().wait_event(&transferred, CudaStreamWaitEventFlags::DEFAULT)?;
        // Reminder to change ntt into batch ntt if we ever use more than one multiplicity col
        assert_eq!(num_multiplicity_cols, 1);
        ntt::intt_into(multiplicities_raw_storage, multiplicities_monomial_storage)?;
        ntt::bitreverse(multiplicities_monomial_storage)?;
    } else {
        assert!(multiplicities_raw_storage.is_empty())
    }

    let raw_trace_storage = GenericTraceStorage {
        storage: raw_storage,
        coset_idx: None,
        layout: trace_layout.clone(),
        form: std::marker::PhantomData,
    };
    let monomial_trace_storage = GenericTraceStorage {
        storage: monomial_storage,
        coset_idx: None,
        layout: trace_layout.clone(),
        form: std::marker::PhantomData,
    };
    let coset_cap_size = coset_cap_size(cap_size, fri_lde_degree);
    let mut first_coset_storage = GenericTraceStorage::allocate(domain_size, trace_layout.clone())?;
    monomial_trace_storage.into_coset_eval(0, used_lde_degree, &mut first_coset_storage)?;
    let (first_subtree, first_subtree_root) =
        first_coset_storage.build_subtree_for_coset(coset_cap_size, 0)?;
    let mut second_coset_storage =
        GenericTraceStorage::allocate(domain_size, trace_layout.clone())?;
    monomial_trace_storage.into_coset_eval(1, used_lde_degree, &mut second_coset_storage)?;
    let (second_subree, second_subtree_root) =
        second_coset_storage.build_subtree_for_coset(coset_cap_size, 1)?;

    let mut subtrees = vec![first_subtree, second_subree];
    let mut subtree_roots = vec![first_subtree_root, second_subtree_root];
    let trace_tree_cap = subtree_roots.compute_cap::<DefaultTreeHasher>(&mut subtrees, cap_size)?;

    Ok((
        raw_trace_storage,
        monomial_trace_storage,
        subtrees,
        trace_tree_cap,
    ))
}

impl GenericTraceStorage<LagrangeBasis> {
    #[allow(dead_code)]
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
        ntt::batch_coset_ntt_into(
            storage.as_ref(),
            coset_storage.storage.as_mut(),
            coset_idx,
            domain_size,
            lde_degree,
            num_polys,
        )?;

        Ok(())
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

    pub(crate) fn barycentric_evaluate<A: GoodAllocator>(
        &self,
        bases: &PrecomputedBasisForBarycentric,
    ) -> CudaResult<Vec<EF, A>> {
        batch_barycentric_evaluate_base(&self.storage, bases, self.domain_size(), self.num_polys())
    }
}

impl AsSingleSlice for &GenericTraceStorage<CosetEvaluations> {
    fn domain_size(&self) -> usize {
        self.storage.domain_size()
    }

    fn num_polys(&self) -> usize {
        GenericTraceStorage::num_polys(&self)
    }

    fn as_single_slice(&self) -> &[F] {
        self.storage.as_single_slice()
    }
}

// we want a holder that either
// - recomputes coset evals each time
// - or keeps coset evals
pub struct TraceCache {
    monomials: GenericTraceStorage<MonomialBasis>,
    cosets: Vec<Option<Rc<GenericTraceStorage<CosetEvaluations>>>>,
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
        Ok(Self {
            monomials: monomial_trace,
            cosets,
            fri_lde_degree,
            used_lde_degree,
        })
    }

    #[allow(dead_code)]
    pub fn commit<H: TreeHasher<F, Output = [F; 4]>>(
        &mut self,
        cap_size: usize,
    ) -> CudaResult<(Vec<SubTree>, Vec<[F; 4]>)> {
        let fri_lde_degree = self.fri_lde_degree;
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
    ) -> CudaResult<Rc<GenericTraceStorage<CosetEvaluations>>> {
        assert!(coset_idx < self.used_lde_degree);

        if REMEMBER_COSETS == false || coset_idx >= self.fri_lde_degree {
            let mut tmp_coset = GenericTraceStorage::allocate(
                self.monomials.domain_size(),
                self.monomials.layout.clone(),
            )?;
            self.monomials
                .into_coset_eval(coset_idx, self.used_lde_degree, &mut tmp_coset)?;
            return Ok(Rc::new(tmp_coset));
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
            self.cosets[coset_idx] = Some(Rc::new(current_storage));
        }

        return Ok(self.cosets[coset_idx].as_ref().unwrap().clone());
    }

    #[allow(dead_code)]
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

    pub fn batch_query_for_coset<H: TreeHasher<F, Output = [F; 4]>, A: GoodAllocator>(
        &mut self,
        coset_idx: usize,
        indexes: &DVec<u32, SmallStaticDeviceAllocator>,
        num_queries: usize,
        domain_size: usize,
        h_all_leaf_elems: &mut Vec<F, A>,
        h_all_proofs: &mut Vec<F, A>,
        tree_holder: &TreeCache,
    ) -> CudaResult<()> {
        let leaf_sources = self.get_or_compute_coset_evals(coset_idx)?;
        let oracle_data = tree_holder.get_trace_subtree(coset_idx);
        batch_query::<H, A>(
            indexes,
            num_queries,
            leaf_sources.deref(),
            leaf_sources.num_polys(),
            oracle_data,
            oracle_data.cap_size,
            domain_size,
            1,
            h_all_leaf_elems,
            h_all_proofs,
        )
    }

    pub fn num_polys(&self) -> usize {
        self.monomials.num_polys()
    }

    #[allow(dead_code)]
    pub fn layout(&self) -> TraceLayout {
        self.monomials.layout.clone()
    }
}
