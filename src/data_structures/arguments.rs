use boojum::cs::{implementations::proof::OracleQuery, oracle::TreeHasher, LookupParameters};
use std::ops::Deref;
use std::rc::Rc;
use cudart::event::{CudaEvent, CudaEventCreateFlags};
use cudart::stream::{CudaStream, CudaStreamCreateFlags, CudaStreamWaitEventFlags};

use super::*;

pub struct GenericArgumentStorage<'a, P: PolyForm> {
    pub storage: GenericComplexPolynomialStorage<'a, P>,
    pub coset_idx: Option<usize>,
    pub layout: ArgumentsLayout,
    form: std::marker::PhantomData<P>,
}

pub struct ArgumentPolynomials<'a, P: PolyForm> {
    pub z_poly: &'a ComplexPoly<'a, P>,
    pub partial_products: &'a [ComplexPoly<'a, P>],
    pub lookup_a_polys: &'a [ComplexPoly<'a, P>],
    pub lookup_b_polys: &'a [ComplexPoly<'a, P>],
}

pub struct ArgumentPolynomialsMut<'a, P: PolyForm> {
    pub z_poly: &'a mut ComplexPoly<'a, P>,
    pub partial_products: &'a mut [ComplexPoly<'a, P>],
    pub lookup_a_polys: &'a mut [ComplexPoly<'a, P>],
    pub lookup_b_polys: &'a mut [ComplexPoly<'a, P>],
}

#[derive(Clone, Debug)]
pub struct ArgumentsLayout {
    pub num_partial_products: usize,
    pub num_lookup_a_polys: usize,
    pub num_lookup_b_polys: usize,
}

impl ArgumentsLayout {
    pub fn from_trace_layout_and_lookup_params(
        trace_layout: TraceLayout,
        quotient_degree: usize,
        lookup_params: LookupParameters,
    ) -> Self {
        let TraceLayout {
            num_variable_cols,
            num_witness_cols: _,
            num_multiplicity_cols: _,
        } = trace_layout;

        let mut num_partial_products = num_variable_cols / quotient_degree;
        if num_variable_cols % quotient_degree != 0 {
            num_partial_products += 1;
        }
        num_partial_products -= 1; // ignore last partial product

        let (num_lookup_a_polys, num_lookup_b_polys) =
            if lookup_params == LookupParameters::NoLookup {
                (0, 0)
            } else {
                match lookup_params {
                    LookupParameters::UseSpecializedColumnsWithTableIdAsVariable {
                        width: _,
                        num_repetitions,
                        share_table_id,
                    } => {
                        assert!(share_table_id == false);
                        (num_repetitions, 1)
                    }
                    LookupParameters::UseSpecializedColumnsWithTableIdAsConstant {
                        width: _,
                        num_repetitions,
                        share_table_id,
                    } => {
                        assert!(share_table_id);

                        (num_repetitions, 1)
                    }
                    _ => unreachable!(),
                }
            };

        Self {
            num_partial_products,
            num_lookup_a_polys,
            num_lookup_b_polys,
        }
    }

    pub fn num_polys(&self) -> usize {
        1 + self.num_partial_products + self.num_lookup_a_polys + self.num_lookup_b_polys
    }
}

impl<'a, P: PolyForm> GenericArgumentStorage<'a, P> {
    pub fn num_polys(&self) -> usize {
        self.layout.num_polys()
    }

    pub fn domain_size(&self) -> usize {
        self.storage.polynomials[0].c0.storage.len()
    }

    pub fn allocate(layout: ArgumentsLayout, domain_size: usize) -> CudaResult<Self> {
        let storage = GenericComplexPolynomialStorage::allocate(layout.num_polys(), domain_size)?;

        Ok(Self {
            storage,
            layout,
            coset_idx: None,
            form: std::marker::PhantomData,
        })
    }

    pub fn split_into_arguments(&self) -> (&[ComplexPoly<P>], &[ComplexPoly<P>]) {
        let num_copy_permutation_polys = 1 + self.layout.num_partial_products;
        self.storage
            .polynomials
            .split_at(num_copy_permutation_polys)
    }

    pub fn as_polynomials(&'a self) -> ArgumentPolynomials<'a, P> {
        let (copy_permutation_polys, lookup_polys) = self.split_into_arguments();
        let (lookup_a_polys, lookup_b_polys) =
            lookup_polys.split_at(self.layout.num_lookup_a_polys);

        ArgumentPolynomials {
            z_poly: &copy_permutation_polys[0],
            partial_products: &copy_permutation_polys[1..],
            lookup_a_polys,
            lookup_b_polys,
        }
    }

    pub fn as_polynomials_mut(&self) -> ArgumentPolynomialsMut<P> {
        unsafe { std::mem::transmute(self.as_polynomials()) }
    }

    #[allow(dead_code)]
    pub fn clone(&self) -> CudaResult<Self> {
        let _num_polys = self.num_polys();
        let _domain_size = self.domain_size();

        let new = self.storage.clone()?;

        Ok(Self {
            storage: new,
            coset_idx: self.coset_idx.clone(),
            layout: self.layout.clone(),
            form: std::marker::PhantomData,
        })
    }
}

impl<'a> GenericArgumentStorage<'a, LagrangeBasis> {
    pub fn into_monomial(self) -> CudaResult<GenericArgumentStorage<'static, MonomialBasis>> {
        let num_polys = self.num_polys();
        let domain_size = self.domain_size();

        let Self {
            mut storage,
            layout,
            ..
        } = self;
        let num_ntts = 2 * num_polys;

        let storage_slice = storage.as_single_slice_mut();
        let l2_chunk_elems = get_l2_chunk_elems(domain_size)?;
        let mut num_cols_processed = 0;
        for storage_chunk in storage_slice.chunks_mut(l2_chunk_elems) {
            let num_cols_this_chunk = storage_chunk.len() / domain_size;
            ntt::batch_ntt(storage_chunk, false, true, domain_size, num_cols_this_chunk)?;
            ntt::batch_bitreverse(storage_chunk, domain_size)?;
            num_cols_processed += num_cols_this_chunk;
        }
        assert_eq!(num_cols_processed, num_ntts);

        let monomial_storage = unsafe { std::mem::transmute(storage) };

        Ok(GenericArgumentStorage {
            storage: monomial_storage,
            coset_idx: None,
            layout,
            form: std::marker::PhantomData,
        })
    }
}

impl<'a> GenericArgumentStorage<'a, MonomialBasis> {
    pub fn into_coset_eval<'b>(
        &self,
        coset_idx: usize,
        lde_degree: usize,
        coset_storage: &mut GenericArgumentStorage<'b, CosetEvaluations>,
    ) -> CudaResult<()> {
        let num_polys = self.num_polys();
        let domain_size = self.domain_size();
        let num_coset_ffts = 2 * num_polys;

        // let storage_slice = self.storage.as_single_slice();
        // let coset_storage_slice = coset_storage.storage.as_single_slice_mut();
        // let l2_chunk_elems = get_l2_chunk_elems(domain_size)?;
        // let mut num_cols_processed = 0;
        // for (storage_chunk, coset_storage_chunk) in storage_slice
        //     .chunks(l2_chunk_elems)
        //     .zip(coset_storage_slice.chunks_mut(l2_chunk_elems))
        // {
        //     let num_cols_this_chunk = storage_chunk.len() / domain_size;
        //     ntt::batch_coset_ntt_into(
        //         storage_chunk,
        //         coset_storage_chunk,
        //         coset_idx,
        //         domain_size,
        //         lde_degree,
        //         num_cols_this_chunk,
        //     )?;
        //     num_cols_processed += num_cols_this_chunk;
        // }
        // assert_eq!(num_cols_processed, num_coset_ffts);

        let storage_slice = self.storage.as_single_slice();
        let coset_storage_slice = coset_storage.storage.as_single_slice_mut();
        let l2_chunk_elems = get_l2_chunk_elems(domain_size)?;
        let mut num_cols_processed = 0;
        let main_stream = get_stream();
        let stream0 = CudaStream::create_with_flags(CudaStreamCreateFlags::NON_BLOCKING)?;
        let stream1 = CudaStream::create_with_flags(CudaStreamCreateFlags::NON_BLOCKING)?;
        set_l2_persistence_for_twiddles(&stream0)?;
        set_l2_persistence_for_twiddles(&stream1)?;
        let start_event = CudaEvent::create_with_flags(CudaEventCreateFlags::DISABLE_TIMING)?;
        start_event.record(&main_stream)?;
        stream0.wait_event(&start_event, CudaStreamWaitEventFlags::DEFAULT)?;
        stream1.wait_event(&start_event, CudaStreamWaitEventFlags::DEFAULT)?;
        for (storage_chunk, coset_storage_chunk) in storage_slice
            .chunks(l2_chunk_elems)
            .zip(coset_storage_slice.chunks_mut(l2_chunk_elems))
        {
            let num_cols_this_chunk = storage_chunk.len() / domain_size;
            let num_cols_stream0 = num_cols_this_chunk / 2;
            let num_cols_stream1 = num_cols_this_chunk - num_cols_stream0;
            let elems_stream0 = num_cols_stream0 * domain_size;
            // ntt::batch_coset_ntt_into(
            //     storage_chunk,
            //     coset_storage_chunk,
            //     coset_idx,
            //     domain_size,
            //     lde_degree,
            //     num_cols_this_chunk,
            // )?;
            if num_cols_stream0 > 0 {
                ntt::batch_coset_ntt_raw_into_with_stream(
                    &storage_chunk[..elems_stream0],
                    &mut coset_storage_chunk[..elems_stream0],
                    false,
                    false,
                    coset_idx,
                    domain_size,
                    lde_degree,
                    num_cols_stream0,
                    &stream0,
                )?;
            }
            if num_cols_stream1 > 0 {
                ntt::batch_coset_ntt_raw_into_with_stream(
                    &storage_chunk[elems_stream0..],
                    &mut coset_storage_chunk[elems_stream0..],
                    false,
                    false,
                    coset_idx,
                    domain_size,
                    lde_degree,
                    num_cols_stream1,
                    &stream1,
                )?;
            }
            num_cols_processed += num_cols_this_chunk;
        }
        let end_event0 = CudaEvent::create_with_flags(CudaEventCreateFlags::DISABLE_TIMING)?;
        let end_event1 = CudaEvent::create_with_flags(CudaEventCreateFlags::DISABLE_TIMING)?;
        end_event0.record(&stream0)?;
        end_event1.record(&stream1)?;
        main_stream.wait_event(&end_event0, CudaStreamWaitEventFlags::DEFAULT)?;
        main_stream.wait_event(&end_event1, CudaStreamWaitEventFlags::DEFAULT)?;

        stream0.destroy()?;
        stream1.destroy()?;
        end_event0.destroy()?;
        end_event1.destroy()?;

        assert_eq!(num_cols_processed, num_coset_ffts);

        Ok(())
    }
}

impl<'a> GenericArgumentStorage<'a, CosetEvaluations> {
    pub fn build_subtree_for_coset(
        &self,
        cap_size: usize,
        coset_idx: usize,
    ) -> CudaResult<(SubTree, Vec<[F; 4]>)> {
        let domain_size = self.domain_size();
        let Self { storage, .. } = self;
        let leaf_sources = storage.as_single_slice();
        let mut subtree = dvec!(2 * NUM_EL_PER_HASH * domain_size);
        let subtree_root = compute_tree_cap(leaf_sources, &mut subtree, domain_size, cap_size, 1)?;

        let subtree = SubTree::new(subtree, domain_size, cap_size, coset_idx);
        Ok((subtree, subtree_root))
    }

    pub(crate) fn barycentric_evaluate(
        &self,
        bases: &PrecomputedBasisForBarycentric,
    ) -> CudaResult<Vec<EF>> {
        batch_barycentric_evaluate_ext(&self.storage, bases, self.domain_size(), self.num_polys())
    }
}

impl<'a> AsSingleSlice for GenericArgumentStorage<'a, CosetEvaluations> {
    fn domain_size(&self) -> usize {
        self.storage.domain_size()
    }

    fn num_polys(&self) -> usize {
        self.storage.polynomials.len()
    }

    fn as_single_slice(&self) -> &[F] {
        self.storage.as_single_slice()
    }

    fn as_single_slice_mut(&mut self) -> &mut [F] {
        self.storage.as_single_slice_mut()
    }
}

impl<'a> AsSingleSlice for &GenericArgumentStorage<'a, CosetEvaluations> {
    fn domain_size(&self) -> usize {
        self.storage.domain_size()
    }

    fn num_polys(&self) -> usize {
        self.storage.polynomials.len()
    }

    fn as_single_slice(&self) -> &[F] {
        self.storage.as_single_slice()
    }
}
impl<'a> LeafSourceQuery for ArgumentPolynomials<'a, CosetEvaluations> {
    fn get_leaf_sources(
        &self,
        _coset_idx: usize,
        _lde_degree: usize,
        _domain_size: usize,
        row_idx: usize,
        _: usize,
    ) -> CudaResult<Vec<F>> {
        let _leaf_sources: Vec<F> = vec![];

        let mut values = vec![];
        let el = self.z_poly.c0.storage.clone_el_to_host(row_idx)?;
        values.push(el);
        let el = self.z_poly.c1.storage.clone_el_to_host(row_idx)?;
        values.push(el);

        for p in self.partial_products.iter() {
            let el = p.c0.storage.clone_el_to_host(row_idx)?;
            values.push(el);
            let el = p.c1.storage.clone_el_to_host(row_idx)?;
            values.push(el);
        }

        if self.lookup_a_polys.len() > 0 {
            for p in self.lookup_a_polys.iter() {
                let el = p.c0.storage.clone_el_to_host(row_idx)?;
                values.push(el);
                let el = p.c1.storage.clone_el_to_host(row_idx)?;
                values.push(el);
            }

            for p in self.lookup_b_polys.iter() {
                let el = p.c0.storage.clone_el_to_host(row_idx)?;
                values.push(el);
                let el = p.c1.storage.clone_el_to_host(row_idx)?;
                values.push(el);
            }
        }

        Ok(values)
    }
}

pub struct ArgumentCache<'a> {
    monomials: GenericArgumentStorage<'a, MonomialBasis>,
    cosets: Vec<Option<Rc<GenericArgumentStorage<'a, CosetEvaluations>>>>,
    fri_lde_degree: usize,
    used_lde_degree: usize,
}

impl<'a> ArgumentCache<'a> {
    pub fn from_monomial(
        monomials: GenericArgumentStorage<'a, MonomialBasis>,
        fri_lde_degree: usize,
        used_lde_degree: usize,
    ) -> CudaResult<Self> {
        assert!(fri_lde_degree.is_power_of_two());
        assert!(used_lde_degree.is_power_of_two());

        let cosets = vec![None; fri_lde_degree];

        Ok(Self {
            monomials,
            cosets,
            fri_lde_degree,
            used_lde_degree,
        })
    }

    pub fn num_polys(&self) -> usize {
        self.monomials.num_polys()
    }
    pub fn num_polys_in_base(&self) -> usize {
        2 * self.monomials.num_polys()
    }

    pub fn commit<H: TreeHasher<F, Output = [F; 4]>>(
        &mut self,
        cap_size: usize,
    ) -> CudaResult<(Vec<SubTree>, Vec<[F; 4]>)> {
        let fri_lde_degree = self.fri_lde_degree;
        let _used_lde_degree = self.used_lde_degree;
        let coset_cap_size = coset_cap_size(cap_size, self.fri_lde_degree);
        let mut setup_subtrees = vec![];
        let mut setup_subtree_caps = vec![];

        assert_eq!(self.cosets.len(), fri_lde_degree);

        for coset_idx in 0..fri_lde_degree {
            let coset_values = self.get_or_compute_coset_evals(coset_idx)?;
            let (subtree, subtree_cap) =
                coset_values.build_subtree_for_coset(coset_cap_size, coset_idx)?;
            setup_subtree_caps.push(subtree_cap);
            setup_subtrees.push(subtree);
        }

        let setup_tree_cap = setup_subtree_caps.compute_cap::<H>(&mut setup_subtrees, cap_size)?;

        Ok((setup_subtrees, setup_tree_cap))
    }

    pub fn get_or_compute_coset_evals(
        &mut self,
        coset_idx: usize,
    ) -> CudaResult<Rc<GenericArgumentStorage<CosetEvaluations>>> {
        assert!(coset_idx < self.used_lde_degree);

        if REMEMBER_COSETS == false || coset_idx >= self.fri_lde_degree {
            let mut tmp_coset = GenericArgumentStorage::allocate(
                self.monomials.layout.clone(),
                self.monomials.domain_size(),
            )?;
            self.monomials
                .into_coset_eval(coset_idx, self.used_lde_degree, &mut tmp_coset)?;
            return Ok(Rc::new(tmp_coset));
        }

        if self.cosets[coset_idx].is_none() {
            let mut current_storage = GenericArgumentStorage::allocate(
                self.monomials.layout.clone(),
                self.monomials.domain_size(),
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
        tree_holder.get_argument_subtrees().query(
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
        let num_polys = self.num_polys_in_base();
        let leaf_sources = self.get_or_compute_coset_evals(coset_idx)?;
        let oracle_data = tree_holder.get_argument_subtree(coset_idx);
        batch_query::<H, A>(
            indexes,
            num_queries,
            leaf_sources.deref(),
            num_polys,
            oracle_data,
            oracle_data.cap_size,
            domain_size,
            1,
            h_all_leaf_elems,
            h_all_proofs,
        )
    }

    pub fn get_z_monomial(&'a self) -> &ComplexPoly<'a, MonomialBasis> {
        self.monomials.as_polynomials().z_poly
    }

    #[allow(dead_code)]
    pub fn layout(&self) -> ArgumentsLayout {
        self.monomials.layout.clone()
    }
}

pub struct QuotientCache<'a> {
    monomials: GenericComplexPolynomialStorage<'a, MonomialBasis>,
    cosets: Vec<Option<Rc<GenericComplexPolynomialStorage<'a, CosetEvaluations>>>>,
    fri_lde_degree: usize,
    used_lde_degree: usize,
}

impl<'a> QuotientCache<'a> {
    pub fn from_monomial(
        monomials: GenericComplexPolynomialStorage<'a, MonomialBasis>,
        fri_lde_degree: usize,
        used_lde_degree: usize,
    ) -> CudaResult<Self> {
        assert!(fri_lde_degree.is_power_of_two());
        assert!(used_lde_degree.is_power_of_two());

        let cosets = vec![None; fri_lde_degree];

        Ok(Self {
            monomials,
            cosets,
            fri_lde_degree,
            used_lde_degree,
        })
    }

    pub fn num_polys(&self) -> usize {
        self.monomials.num_polys()
    }

    pub fn num_polys_in_base(&self) -> usize {
        2 * self.monomials.num_polys()
    }

    pub fn commit<H: TreeHasher<F, Output = [F; 4]>>(
        &mut self,
        cap_size: usize,
    ) -> CudaResult<(Vec<SubTree>, Vec<[F; 4]>)> {
        let fri_lde_degree = self.fri_lde_degree;
        let _used_lde_degree = self.used_lde_degree;
        let coset_cap_size = coset_cap_size(cap_size, self.fri_lde_degree);
        let mut setup_subtrees = vec![];
        let mut setup_subtree_caps = vec![];

        assert_eq!(self.cosets.len(), fri_lde_degree);

        for coset_idx in 0..fri_lde_degree {
            let coset_values = self.get_or_compute_coset_evals(coset_idx)?;
            let (subtree, subtree_cap) =
                coset_values.build_subtree_for_coset(coset_cap_size, coset_idx)?;
            setup_subtree_caps.push(subtree_cap);
            setup_subtrees.push(subtree);
        }

        let setup_tree_cap = setup_subtree_caps.compute_cap::<H>(&mut setup_subtrees, cap_size)?;

        Ok((setup_subtrees, setup_tree_cap))
    }

    pub fn get_or_compute_coset_evals(
        &mut self,
        coset_idx: usize,
    ) -> CudaResult<Rc<GenericComplexPolynomialStorage<'a, CosetEvaluations>>> {
        assert!(coset_idx < self.used_lde_degree);

        if REMEMBER_COSETS == false || coset_idx >= self.fri_lde_degree {
            let mut tmp_coset = GenericComplexPolynomialStorage::allocate(
                self.monomials.num_polys(),
                self.monomials.domain_size(),
            )?;
            self.monomials
                .into_coset_eval(coset_idx, self.used_lde_degree, &mut tmp_coset)?;
            return Ok(Rc::new(tmp_coset));
        }

        if self.cosets[coset_idx].is_none() {
            let mut current_storage = GenericComplexPolynomialStorage::allocate(
                self.monomials.num_polys(),
                self.monomials.domain_size(),
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
        tree_holder.get_quotient_subtrees().query(
            leaf_sources.as_ref(),
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
        let num_polys = self.num_polys_in_base();
        let leaf_sources = self.get_or_compute_coset_evals(coset_idx)?;
        let oracle_data = tree_holder.get_quotient_subtree(coset_idx);
        batch_query::<H, A>(
            indexes,
            num_queries,
            leaf_sources.deref(),
            num_polys,
            oracle_data,
            oracle_data.cap_size,
            domain_size,
            1,
            h_all_leaf_elems,
            h_all_proofs,
        )
    }
}
