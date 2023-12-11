use boojum::config::ProvingCSConfig;
use boojum::cs::{
    implementations::{
        polynomial_storage::SetupBaseStorage, prover::ProofConfig,
        reference_cs::CSReferenceAssembly,
    },
    oracle::TreeHasher,
};
use std::ops::Deref;
use std::rc::Rc;

use crate::cs::{materialize_permutation_cols_from_transformed_hints_into, GpuSetup};
use crate::prover::compute_quotient_degree;

use super::*;

use nvtx::{range_pop, range_push};

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SetupLayout {
    pub num_permutation_cols: usize,
    pub num_constant_cols: usize,
    pub num_table_cols: usize,
}

impl SetupLayout {
    pub fn from_setup<A: GoodAllocator>(setup: &GpuSetup<A>) -> Self {
        assert!(setup.variables_hint.len() > 0);
        assert!(setup.constant_columns.len() > 0);
        Self {
            num_permutation_cols: setup.variables_hint.len(),
            num_constant_cols: setup.constant_columns.len(),
            num_table_cols: setup.lookup_tables_columns.len(),
        }
    }

    pub fn from_base_setup_and_hints<
        P: boojum::field::traits::field_like::PrimeFieldLikeVectorized<Base = F>,
    >(
        base_setup: &SetupBaseStorage<F, P, Global, Global>,
    ) -> Self {
        assert!(base_setup.copy_permutation_polys.len() > 0);
        assert!(base_setup.constant_columns.len() > 0);
        Self {
            num_permutation_cols: base_setup.copy_permutation_polys.len(),
            num_constant_cols: base_setup.constant_columns.len(),
            num_table_cols: base_setup.lookup_tables_columns.len(),
        }
    }

    pub fn num_polys(&self) -> usize {
        self.num_permutation_cols + self.num_constant_cols + self.num_table_cols
    }
}

pub struct GenericSetupStorage<P: PolyForm> {
    pub storage: GenericStorage,
    pub layout: SetupLayout,
    #[allow(dead_code)]
    coset_idx: Option<usize>,
    form: std::marker::PhantomData<P>,
}

impl<P: PolyForm> GenericSetupStorage<P> {
    pub fn num_polys(&self) -> usize {
        self.layout.num_polys()
    }

    pub fn domain_size(&self) -> usize {
        self.storage.domain_size()
    }

    pub fn as_polynomials<'a>(&'a self) -> SetupPolynomials<'a, P> {
        let GenericSetupStorage {
            storage,
            layout,
            coset_idx: _,
            form: _,
        } = self;

        let SetupLayout {
            num_permutation_cols,
            num_constant_cols,
            num_table_cols,
        } = *layout;

        let all_polys = storage.as_poly_storage();
        let mut all_polys_iter = all_polys.into_iter();

        let mut permutation_cols = vec![];
        for _ in 0..num_permutation_cols {
            permutation_cols.push(all_polys_iter.next().unwrap());
        }
        let mut constant_cols = vec![];
        for _ in 0..num_constant_cols {
            constant_cols.push(all_polys_iter.next().unwrap());
        }

        let mut table_cols = vec![];
        for _ in 0..num_table_cols {
            table_cols.push(all_polys_iter.next().unwrap());
        }
        assert!(all_polys_iter.next().is_none());
        assert_multiset_adjacent_base(&[&permutation_cols, &constant_cols, &table_cols]);

        SetupPolynomials {
            permutation_cols,
            constant_cols,
            table_cols,
        }
    }

    #[allow(dead_code)]
    pub fn clone(&self) -> CudaResult<Self> {
        range_push!("GenericSetupStorage::clone");

        let num_polys = self.num_polys();
        let domain_size = self.domain_size();
        let src = self.as_single_slice();

        let mut storage = GenericStorage::allocate(num_polys, domain_size)?;
        let dst = storage.as_single_slice_mut();
        println!("\nLength of GenericSetupStorage: dst.len() {}\n", dst.len());
        mem::d2d(src, dst)?;

        range_pop!();

        Ok(Self {
            storage,
            layout: self.layout.clone(),
            coset_idx: self.coset_idx.clone(),
            form: std::marker::PhantomData,
        })
    }

    pub fn allocate(layout: SetupLayout, domain_size: usize) -> CudaResult<Self> {
        let storage = GenericStorage::allocate(layout.num_polys(), domain_size)?;
        Ok(Self {
            storage,
            layout,
            coset_idx: None,
            form: std::marker::PhantomData,
        })
    }
}

impl<P: PolyForm> AsSingleSlice for GenericSetupStorage<P> {
    fn as_single_slice(&self) -> &[F] {
        self.storage.as_single_slice()
    }

    fn as_single_slice_mut(&mut self) -> &mut [F] {
        self.storage.as_single_slice_mut()
    }

    fn domain_size(&self) -> usize {
        self.storage.domain_size
    }

    fn num_polys(&self) -> usize {
        assert_eq!(self.storage.num_polys, self.layout.num_polys());
        self.layout.num_polys()
    }
}
impl<P: PolyForm> AsSingleSlice for &GenericSetupStorage<P> {
    fn as_single_slice(&self) -> &[F] {
        self.storage.as_single_slice()
    }

    fn domain_size(&self) -> usize {
        self.storage.domain_size
    }

    fn num_polys(&self) -> usize {
        assert_eq!(self.storage.num_polys, self.layout.num_polys());
        self.layout.num_polys()
    }
}

impl<'a> LeafSourceQuery for SetupPolynomials<'a, CosetEvaluations> {
    fn get_leaf_sources(
        &self,
        _coset_idx: usize,
        _domain_size: usize,
        _lde_degree: usize,
        row_idx: usize,
        _: usize,
    ) -> CudaResult<Vec<F>> {
        let mut leaf_sources = vec![];

        for col in self.permutation_cols.iter() {
            let el = col.storage.clone_el_to_host(row_idx)?;
            leaf_sources.push(el);
        }

        for col in self.constant_cols.iter() {
            let el = col.storage.clone_el_to_host(row_idx)?;
            leaf_sources.push(el);
        }

        for col in self.table_cols.iter() {
            let el = col.storage.clone_el_to_host(row_idx)?;
            leaf_sources.push(el);
        }

        Ok(leaf_sources)
    }
}

pub struct SetupPolynomials<'a, P: PolyForm> {
    pub permutation_cols: Vec<Poly<'a, P>>,
    pub constant_cols: Vec<Poly<'a, P>>,
    pub table_cols: Vec<Poly<'a, P>>,
}

impl GenericSetupStorage<LagrangeBasis> {
    pub fn from_gpu_setup<A: GoodAllocator>(
        setup: &GpuSetup<A>,
    ) -> CudaResult<GenericSetupStorage<LagrangeBasis>> {
        range_push!("GenericSetupStorage::from_gpu_setup");

        let GpuSetup {
            constant_columns,
            lookup_tables_columns,
            variables_hint,
            layout,
            ..
        } = setup;

        let domain_size = constant_columns[0].len();
        assert!(domain_size.is_power_of_two());
        for col in constant_columns.iter().chain(lookup_tables_columns.iter()) {
            assert_eq!(col.len(), domain_size);
        }

        let num_copy_permutation_polys = variables_hint.len();
        let mut storage = GenericSetupStorage::allocate(layout.clone(), domain_size)?;

        let size_of_all_copy_perm_polys = num_copy_permutation_polys * domain_size;
        let (copy_permutation_storage, remaining_polys) = storage
            .as_single_slice_mut()
            .split_at_mut(size_of_all_copy_perm_polys);
        assert_eq!(
            remaining_polys.len(),
            (constant_columns.len() + lookup_tables_columns.len()) * domain_size
        );
        materialize_permutation_cols_from_transformed_hints_into(
            copy_permutation_storage,
            variables_hint,
            domain_size,
        )?;

        for (dst, src) in remaining_polys
            .chunks_mut(domain_size)
            .zip(constant_columns.iter().chain(lookup_tables_columns.iter()))
        {
            mem::h2d(src, dst)?;
        }

        range_pop!();

        Ok(storage)
    }

    #[allow(dead_code)]
    pub fn from_host_values<
        PP: boojum::field::traits::field_like::PrimeFieldLikeVectorized<Base = F>,
    >(
        base_setup: &SetupBaseStorage<F, PP>,
    ) -> CudaResult<GenericSetupStorage<LagrangeBasis>> {
        let SetupBaseStorage {
            copy_permutation_polys,
            constant_columns,
            lookup_tables_columns,
            table_ids_column_idxes: _,
            selectors_placement: _,
        } = base_setup;
        let domain_size = copy_permutation_polys[0].domain_size();
        assert!(domain_size.is_power_of_two());
        assert_eq!(domain_size, constant_columns[0].domain_size());

        let permutation_cols: Vec<&[F]> = copy_permutation_polys
            .iter()
            .map(|p| PP::slice_into_base_slice(&p.storage))
            .collect();
        let constant_cols: Vec<&[F]> = constant_columns
            .iter()
            .map(|p| PP::slice_into_base_slice(&p.storage))
            .collect();
        let table_cols: Vec<&[F]> = lookup_tables_columns
            .iter()
            .map(|p| PP::slice_into_base_slice(&p.storage))
            .collect();
        let domain_size = permutation_cols[0].len();

        let setup_layout = SetupLayout::from_base_setup_and_hints(base_setup);
        let mut storage = GenericSetupStorage::allocate(setup_layout, domain_size)?;

        for (dst, src) in storage
            .storage
            .as_single_slice_mut()
            .chunks_mut(domain_size)
            .zip(
                permutation_cols
                    .iter()
                    .chain(constant_cols.iter())
                    .chain(table_cols.iter()),
            )
        {
            mem::h2d(src, dst)?;
        }

        Ok(storage)
    }

    pub fn into_monomials(mut self) -> CudaResult<GenericSetupStorage<MonomialBasis>> {
        let domain_size = self.domain_size();
        let num_polys = self.num_polys();

        ntt::batch_ntt(
            self.as_single_slice_mut(),
            false,
            true,
            domain_size,
            num_polys,
        )?;

        ntt::batch_bitreverse(self.as_single_slice_mut(), domain_size)?;

        let monomials = unsafe { std::mem::transmute(self) };

        Ok(monomials)
    }
}

impl GenericSetupStorage<MonomialBasis> {
    pub fn into_coset_eval(
        &self,
        coset_idx: usize,
        lde_degree: usize,
        coset_storage: &mut GenericSetupStorage<CosetEvaluations>,
    ) -> CudaResult<()> {
        let domain_size = self.domain_size();
        let num_polys = self.num_polys();

        assert_eq!(coset_storage.domain_size(), domain_size);
        assert_eq!(coset_storage.num_polys(), num_polys);

        ntt::batch_coset_fft_into(
            self.as_single_slice(),
            coset_storage.as_single_slice_mut(),
            coset_idx,
            domain_size,
            lde_degree,
            num_polys,
        )?;

        Ok(())
    }
}

impl GenericSetupStorage<CosetEvaluations> {
    pub fn build_subtree_for_coset(
        &self,
        cap_size: usize,
        coset_idx: usize,
    ) -> CudaResult<(SubTree, Vec<[F; 4]>)> {
        let domain_size = self.domain_size();
        let leaf_sources = self.as_single_slice();
        let mut subtree = dvec!(2 * NUM_EL_PER_HASH * domain_size);
        let subtree_root = compute_tree_cap(leaf_sources, &mut subtree, domain_size, cap_size, 1)?;
        let subtree = SubTree::new(subtree, domain_size, cap_size, coset_idx);
        Ok((subtree, subtree_root))
    }

    pub(crate) fn barycentric_evaluate<A: GoodAllocator>(
        &self,
        bases: &PrecomputedBasisForBarycentric,
    ) -> CudaResult<Vec<EF, A>> {
        batch_barycentric_evaluate_base(self, bases, self.domain_size(), self.num_polys())
    }
}

pub struct SetupCache {
    pub evals_on_trace_domain: GenericSetupStorage<LagrangeBasis>,
    pub monomials: GenericSetupStorage<MonomialBasis>,
    cosets: Vec<Option<Rc<GenericSetupStorage<CosetEvaluations>>>>,
    pub variables_hint: Option<Vec<DVec<u32>>>,
    pub witnesses_hint: Option<Vec<DVec<u32>>>,
    pub fri_oracles_subtrees: Option<Vec<SubTree>>,
    pub fri_oracles_caps: Option<Vec<[F; 4]>>,
    pub fri_lde_degree: usize,
    pub used_lde_degree: usize,
}

impl SetupCache {
    pub fn new<
        A: GoodAllocator,
        P: boojum::field::traits::field_like::PrimeFieldLikeVectorized<Base = F>,
    >(
        setup: &GpuSetup<A>,
        proof_config: &ProofConfig,
        cs: &CSReferenceAssembly<F, P, ProvingCSConfig>,
    ) -> CudaResult<Self> {
        let quotient_degree = compute_quotient_degree(&cs, &setup.selectors_placement);
        let fri_lde_degree = proof_config.fri_lde_factor;
        let used_lde_degree = usize::max(fri_lde_degree, quotient_degree);

        assert!(fri_lde_degree.is_power_of_two());
        assert!(used_lde_degree.is_power_of_two());

        let evals_on_trace_domain = GenericSetupStorage::<_>::from_gpu_setup(setup)?;
        let tmp = evals_on_trace_domain.clone()?;
        let monomials = tmp.into_monomials()?;

        let cosets = vec![None; fri_lde_degree];

        let this = Self {
            evals_on_trace_domain,
            monomials,
            cosets,
            variables_hint: None,
            witnesses_hint: None,
            fri_oracles_subtrees: None,
            fri_oracles_caps: None,
            fri_lde_degree,
            used_lde_degree,
        };

        Ok(this)
    }

    pub fn num_polys(&self) -> usize {
        self.monomials.num_polys()
    }

    pub fn commit<H: TreeHasher<F, Output = [F; 4]>>(
        &mut self,
        cap_size: usize,
    ) -> CudaResult<(&Vec<SubTree>, &Vec<[F; 4]>)> {
        assert_eq!(
            self.fri_oracles_subtrees.is_none(),
            self.fri_oracles_caps.is_none()
        );
        if self.fri_oracles_subtrees.is_none() {
            let fri_lde_degree = self.fri_lde_degree;
            let coset_cap_size = coset_cap_size(cap_size, self.fri_lde_degree);
            let mut setup_subtrees = Vec::with_capacity(fri_lde_degree);
            let mut setup_subtree_caps = Vec::with_capacity(fri_lde_degree);

            assert_eq!(self.cosets.len(), fri_lde_degree);

            for coset_idx in 0..fri_lde_degree {
                let coset_values = self.get_or_compute_coset_evals(coset_idx)?;
                let (subtree, subtree_cap) =
                    coset_values.build_subtree_for_coset(coset_cap_size, coset_idx)?;
                setup_subtree_caps.push(subtree_cap);
                setup_subtrees.push(subtree);
            }

            self.fri_oracles_caps =
                Some(setup_subtree_caps.compute_cap::<H>(&mut setup_subtrees, cap_size)?);
            self.fri_oracles_subtrees = Some(setup_subtrees);
        }

        let subtrees = self.fri_oracles_subtrees.as_ref().unwrap();
        let caps = self.fri_oracles_caps.as_ref().unwrap();
        Ok((subtrees, caps))
    }

    pub fn get_or_compute_coset_evals(
        &mut self,
        coset_idx: usize,
    ) -> CudaResult<Rc<GenericSetupStorage<CosetEvaluations>>> {
        assert!(coset_idx < self.used_lde_degree);

        if REMEMBER_COSETS == false || coset_idx >= self.fri_lde_degree {
            let mut tmp_coset = GenericSetupStorage::allocate(
                self.monomials.layout.clone(),
                self.monomials.domain_size(),
            )?;
            self.monomials
                .into_coset_eval(coset_idx, self.used_lde_degree, &mut tmp_coset)?;
            return Ok(Rc::new(tmp_coset));
        }

        if self.cosets[coset_idx].is_none() {
            let domain_size = self.monomials.domain_size();

            let mut current_storage =
                GenericSetupStorage::allocate(self.monomials.layout.clone(), domain_size)?;

            self.monomials.into_coset_eval(
                coset_idx,
                self.used_lde_degree,
                &mut current_storage,
            )?;
            self.cosets[coset_idx] = Some(Rc::new(current_storage));
        }

        return Ok(self.cosets[coset_idx].as_ref().unwrap().clone());
    }

    // #[allow(dead_code)]
    // pub fn query<H: TreeHasher<F, Output = [F; 4]>>(
    //     &mut self,
    //     coset_idx: usize,
    //     fri_lde_degree: usize,
    //     row_idx: usize,
    //     domain_size: usize,
    //     tree_holder: &TreeCache,
    // ) -> CudaResult<OracleQuery<F, H>> {
    //     let leaf_sources = self.get_or_compute_coset_evals(coset_idx)?;
    //     tree_holder.get_setup_subtrees().query(
    //         &leaf_sources.as_polynomials(),
    //         coset_idx,
    //         fri_lde_degree,
    //         row_idx,
    //         domain_size,
    //     )
    // }

    pub fn batch_query<H: TreeHasher<F, Output = [F; 4]>, A: GoodAllocator>(
        &mut self,
        coset_idx: usize,
        indexes: &DVec<u32, SmallStaticDeviceAllocator>,
        num_queries: usize,
        domain_size: usize,
        h_all_leaf_elems: &mut Vec<F, A>,
        h_all_proofs: &mut Vec<F, A>,
        // tree_holder: &TreeCache,
    ) -> CudaResult<()> {
        let leaf_sources = self.get_or_compute_coset_evals(coset_idx)?;
        let oracle_data = &self.fri_oracles_subtrees.as_ref().unwrap()[coset_idx];
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

    #[allow(dead_code)]
    pub fn layout(&self) -> SetupLayout {
        self.monomials.layout.clone()
    }
}
