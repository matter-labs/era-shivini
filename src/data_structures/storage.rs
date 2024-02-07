use super::*;
use std::fmt::Debug;
use std::marker::PhantomData;
use std::ops::Range;
use std::rc::Rc;

pub trait GenericStorageLayout: Debug + Copy + Clone + Eq + PartialEq {
    type PolyType: Copy + Clone;
    fn num_polys(&self) -> usize;
    fn poly_range(&self, poly_type: Self::PolyType) -> (Range<usize>, Self);
}

impl GenericStorageLayout for usize {
    type PolyType = ();

    fn num_polys(&self) -> usize {
        *self
    }

    fn poly_range(&self, _: Self::PolyType) -> (Range<usize>, Self) {
        (0..*self, *self)
    }
}

#[derive(Clone)]
pub struct GenericStorage<P: PolyForm, L: GenericStorageLayout> {
    pub(crate) inner: DVec<F>,
    pub(crate) layout: L,
    pub(crate) num_polys: usize,
    pub(crate) domain_size: usize,
    poly_form: PhantomData<P>,
}

impl<P: PolyForm, L: GenericStorageLayout> AsSingleSlice for GenericStorage<P, L> {
    fn domain_size(&self) -> usize {
        self.domain_size
    }

    fn num_polys(&self) -> usize {
        self.num_polys
    }

    fn as_single_slice(&self) -> &[F] {
        &self.inner
    }

    fn as_single_slice_mut(&mut self) -> &mut [F] {
        &mut self.inner
    }
}

impl<P: PolyForm, L: GenericStorageLayout> GenericStorage<P, L> {
    pub fn new(inner: DVec<F>, layout: L, domain_size: usize) -> Self {
        let num_polys = layout.num_polys();
        assert_eq!(inner.len(), num_polys * domain_size);
        Self {
            inner,
            layout,
            num_polys,
            domain_size,
            poly_form: PhantomData,
        }
    }

    pub fn into_poly_storage(self) -> GenericPolynomialStorage<'static, P> {
        let Self {
            inner,
            num_polys,
            domain_size,
            ..
        } = self;
        let chunks = inner.into_adjacent_chunks(domain_size);
        assert_eq!(chunks.len(), num_polys);

        let mut polynomials = vec![];
        for chunk in chunks {
            let poly = Poly::from(chunk);
            polynomials.push(poly)
        }
        let new = GenericPolynomialStorage { polynomials };
        new
    }

    pub fn as_polys(&self) -> Vec<Poly<P>> {
        self.inner
            .chunks(self.domain_size)
            .map(Poly::from)
            .collect()
    }

    #[allow(dead_code)]
    pub fn as_polys_mut(&mut self) -> Vec<Poly<P>> {
        self.inner
            .chunks_mut(self.domain_size)
            .map(Poly::from)
            .collect()
    }

    pub fn as_complex_polys(&self) -> Vec<ComplexPoly<P>> {
        assert_eq!(self.num_polys % 2, 0);
        self.inner
            .chunks(self.domain_size)
            .map(Poly::from)
            .array_chunks::<2>()
            .map(|[c0, c1]| ComplexPoly::new(c0, c1))
            .collect()
    }

    pub fn as_complex_polys_mut(&mut self) -> Vec<ComplexPoly<P>> {
        assert_eq!(self.num_polys % 2, 0);
        self.inner
            .chunks_mut(self.domain_size)
            .map(Poly::from)
            .array_chunks::<2>()
            .map(|[c0, c1]| ComplexPoly::new(c0, c1))
            .collect()
    }

    pub fn into_complex_poly_storage(
        self,
    ) -> CudaResult<GenericComplexPolynomialStorage<'static, P>> {
        let Self {
            inner,
            num_polys,
            domain_size,
            ..
        } = self;
        let chunks = inner.into_adjacent_chunks(domain_size);
        assert_eq!(chunks.len(), num_polys);
        assert_eq!(chunks.len() % 2, 0);

        let mut polynomials = vec![];
        for [c0, c1] in chunks.into_iter().array_chunks::<2>() {
            let poly = ComplexPoly::new(Poly::from(c0), Poly::from(c1));
            polynomials.push(poly)
        }
        assert_eq!(polynomials.len(), num_polys >> 1);
        let new = GenericComplexPolynomialStorage { polynomials };

        Ok(new)
    }

    pub unsafe fn transmute<U: PolyForm>(self) -> GenericStorage<U, L> {
        GenericStorage::new(self.inner, self.layout, self.domain_size)
    }
}

impl<L: GenericStorageLayout> GenericStorage<Undefined, L> {
    pub fn allocate(layout: L, domain_size: usize) -> Self {
        let num_polys = layout.num_polys();
        let inner = dvec!(num_polys * domain_size);
        Self::new(inner, layout, domain_size)
    }
}

impl<L: GenericStorageLayout> GenericStorage<LagrangeBasis, L> {
    pub fn into_monomials(mut self) -> CudaResult<GenericStorage<MonomialBasis, L>> {
        let domain_size = self.domain_size;
        let num_polys = self.num_polys;
        let input = self.as_single_slice_mut();
        ntt::batch_ntt_with_epilogue(
            input,
            false,
            true,
            domain_size,
            num_polys,
            |chunk, stream| ntt::batch_bitreverse_on_stream(chunk, domain_size, stream),
        )?;
        let result = unsafe { self.transmute() };
        Ok(result)
    }

    pub fn fill_monomials(
        &self,
        mut storage: GenericStorage<Undefined, L>,
    ) -> CudaResult<GenericStorage<MonomialBasis, L>> {
        assert_eq!(storage.layout, self.layout);
        let domain_size = self.domain_size;
        assert_eq!(storage.domain_size, domain_size);
        let num_polys = self.num_polys;
        let inputs = self.as_single_slice();
        let outputs = storage.as_single_slice_mut();
        ntt::batch_ntt_with_epilogue_into(
            inputs,
            outputs,
            false,
            true,
            domain_size,
            num_polys,
            |chunk, stream| ntt::batch_bitreverse_on_stream(chunk, domain_size, stream),
        )?;
        let result = unsafe { storage.transmute() };
        Ok(result)
    }

    #[allow(dead_code)]
    pub fn create_monomials(&self) -> CudaResult<GenericStorage<MonomialBasis, L>> {
        let storage = GenericStorage::allocate(self.layout, self.domain_size);
        self.fill_monomials(storage)
    }
}

impl<L: GenericStorageLayout> GenericStorage<MonomialBasis, L> {
    pub fn into_evaluations(mut self) -> CudaResult<GenericStorage<LagrangeBasis, L>> {
        let domain_size = self.domain_size;
        let num_polys = self.num_polys;
        let input = self.as_single_slice_mut();
        ntt::batch_ntt_with_epilogue(
            input,
            false,
            false,
            domain_size,
            num_polys,
            |chunk, stream| ntt::batch_bitreverse_on_stream(chunk, domain_size, stream),
        )?;
        let evaluations = unsafe { self.transmute() };
        Ok(evaluations)
    }

    pub fn fill_evaluations(
        &self,
        mut storage: GenericStorage<Undefined, L>,
    ) -> CudaResult<GenericStorage<LagrangeBasis, L>> {
        assert_eq!(storage.layout, self.layout);
        let domain_size = self.domain_size;
        assert_eq!(storage.domain_size, domain_size);
        let num_polys = self.num_polys;
        let inputs = self.as_single_slice();
        let outputs = storage.as_single_slice_mut();
        ntt::batch_ntt_with_epilogue_into(
            inputs,
            outputs,
            false,
            false,
            domain_size,
            num_polys,
            |chunk, stream| ntt::batch_bitreverse_on_stream(chunk, domain_size, stream),
        )?;
        let result = unsafe { storage.transmute() };
        Ok(result)
    }

    pub fn create_evaluations(&self) -> CudaResult<GenericStorage<LagrangeBasis, L>> {
        let storage = GenericStorage::allocate(self.layout, self.domain_size);
        self.fill_evaluations(storage)
    }

    pub fn into_coset_evaluations(
        mut self,
        coset_idx: usize,
        lde_degree: usize,
    ) -> CudaResult<GenericStorage<CosetEvaluations, L>> {
        let domain_size = self.domain_size;
        let num_polys = self.num_polys;
        let inputs = self.as_single_slice_mut();
        ntt::batch_coset_ntt(inputs, coset_idx, domain_size, lde_degree, num_polys)?;
        let result = unsafe { self.transmute() };
        Ok(result)
    }

    pub fn fill_coset_evaluations(
        &self,
        coset_idx: usize,
        lde_degree: usize,
        mut storage: GenericStorage<Undefined, L>,
    ) -> CudaResult<GenericStorage<CosetEvaluations, L>> {
        assert_eq!(storage.layout, self.layout);
        let domain_size = self.domain_size;
        assert_eq!(storage.domain_size, domain_size);
        let num_polys = self.num_polys;
        let inputs = self.as_single_slice();
        let outputs = storage.as_single_slice_mut();
        ntt::batch_coset_ntt_into(
            inputs,
            outputs,
            coset_idx,
            domain_size,
            lde_degree,
            num_polys,
        )?;
        let result = unsafe { storage.transmute() };
        Ok(result)
    }

    pub fn create_coset_evaluations(
        &self,
        coset_idx: usize,
        lde_degree: usize,
    ) -> CudaResult<GenericStorage<CosetEvaluations, L>> {
        let storage = GenericStorage::allocate(self.layout, self.domain_size);
        self.fill_coset_evaluations(coset_idx, lde_degree, storage)
    }

    pub fn fill_coset_evaluations_subset(
        &self,
        coset_idx: usize,
        lde_degree: usize,
        subset: L::PolyType,
        mut storage: GenericStorage<Undefined, L>,
    ) -> CudaResult<GenericStorage<CosetEvaluations, L>> {
        let (range, layout) = self.layout.poly_range(subset);
        assert!(range.end <= self.num_polys);
        assert_eq!(storage.layout, layout);
        let domain_size = self.domain_size;
        assert_eq!(storage.domain_size(), domain_size);
        let num_polys = range.len();
        let inputs = &self.as_single_slice()[range.start * domain_size..range.end * domain_size];
        let outputs = storage.as_single_slice_mut();
        if num_polys != 0 {
            ntt::batch_coset_ntt_into(
                inputs,
                outputs,
                coset_idx,
                domain_size,
                lde_degree,
                num_polys,
            )?;
        }
        let result = unsafe { storage.transmute() };
        Ok(result)
    }

    pub fn create_coset_evaluations_subset(
        &self,
        coset_idx: usize,
        lde_degree: usize,
        subset: L::PolyType,
    ) -> CudaResult<GenericStorage<CosetEvaluations, L>> {
        let (_, layout) = self.layout.poly_range(subset);
        let storage = GenericStorage::allocate(layout, self.domain_size());
        self.fill_coset_evaluations_subset(coset_idx, lde_degree, subset, storage)
    }
}

impl<L: GenericStorageLayout> GenericStorage<CosetEvaluations, L> {
    pub fn into_monomials(
        mut self,
        coset_idx: usize,
        lde_degree: usize,
    ) -> CudaResult<GenericStorage<MonomialBasis, L>> {
        let domain_size = self.domain_size;
        let num_polys = self.num_polys;
        let inputs = self.as_single_slice_mut();
        ntt::batch_inverse_coset_ntt(inputs, coset_idx, domain_size, lde_degree, num_polys)?;
        let result = unsafe { self.transmute() };
        Ok(result)
    }

    pub fn fill_monomials(
        &self,
        coset_idx: usize,
        lde_degree: usize,
        mut storage: GenericStorage<Undefined, L>,
    ) -> CudaResult<GenericStorage<MonomialBasis, L>> {
        assert_eq!(storage.layout, self.layout);
        let domain_size = self.domain_size;
        assert_eq!(storage.domain_size, domain_size);
        let num_polys = self.num_polys;
        let inputs = self.as_single_slice();
        let outputs = storage.as_single_slice_mut();
        ntt::batch_inverse_coset_ntt_into(
            inputs,
            outputs,
            coset_idx,
            domain_size,
            lde_degree,
            num_polys,
        )?;
        let result = unsafe { storage.transmute() };
        Ok(result)
    }

    pub fn create_monomials(
        &self,
        coset_idx: usize,
        lde_degree: usize,
    ) -> CudaResult<GenericStorage<MonomialBasis, L>> {
        let storage = GenericStorage::allocate(self.layout, self.domain_size);
        self.fill_monomials(coset_idx, lde_degree, storage)
    }

    pub fn build_subtree_for_coset(
        &self,
        cap_size: usize,
        coset_idx: usize,
        mut nodes: DVec<F>,
    ) -> CudaResult<(SubTree, Vec<[F; 4]>)> {
        let domain_size = self.domain_size();
        let leaf_sources = self.as_single_slice();
        let subtree_root = compute_tree_cap(leaf_sources, &mut nodes, domain_size, cap_size, 1)?;
        let subtree = SubTree::new(Rc::new(nodes), domain_size, cap_size, coset_idx);
        Ok((subtree, subtree_root))
    }
}

impl<P: PolyForm, L: GenericStorageLayout> AsMut<DVec<F>> for GenericStorage<P, L> {
    fn as_mut(&mut self) -> &mut DVec<F> {
        &mut self.inner
    }
}

impl<P: PolyForm, L: GenericStorageLayout> AsRef<DVec<F>> for GenericStorage<P, L> {
    fn as_ref(&self) -> &DVec<F> {
        &self.inner
    }
}

pub struct GenericPolynomialStorage<'a, P: PolyForm> {
    pub(crate) polynomials: Vec<Poly<'a, P>>,
}

impl<'a, P: PolyForm> GenericPolynomialStorage<'a, P> {
    #[allow(dead_code)]
    pub fn allocate(num_polys: usize, domain_size: usize) -> CudaResult<Self> {
        let storage = GenericStorage::allocate(num_polys, domain_size);
        let storage = unsafe { storage.transmute() };
        Ok(storage.into_poly_storage())
    }

    #[allow(dead_code)]
    pub fn num_polys(&self) -> usize {
        self.polynomials.len()
    }
}

impl<'a, P: PolyForm> AsSingleSlice for GenericPolynomialStorage<'a, P> {
    fn domain_size(&self) -> usize {
        self.polynomials[0].domain_size()
    }

    fn num_polys(&self) -> usize {
        self.polynomials.len()
    }

    fn as_single_slice(&self) -> &[F] {
        assert_adjacent_base(&self.polynomials);
        let num_polys = self.polynomials.len();
        let domain_size = self.polynomials[0].domain_size();
        let len = num_polys * domain_size;
        unsafe { std::slice::from_raw_parts(self.polynomials[0].storage.as_ref().as_ptr(), len) }
    }

    fn as_single_slice_mut(&mut self) -> &mut [F] {
        assert_adjacent_base(&self.polynomials);
        let num_polys = self.polynomials.len();
        let domain_size = self.polynomials[0].domain_size();
        let len = num_polys * domain_size;
        unsafe {
            std::slice::from_raw_parts_mut(self.polynomials[0].storage.as_mut().as_mut_ptr(), len)
        }
    }
}

pub struct GenericComplexPolynomialStorage<'a, P: PolyForm> {
    pub polynomials: Vec<ComplexPoly<'a, P>>,
}

impl<'a> LeafSourceQuery for GenericComplexPolynomialStorage<'a, CosetEvaluations> {
    fn get_leaf_sources(
        &self,
        _coset_idx: usize,
        _lde_degree: usize,
        _domain_size: usize,
        row_idx: usize,
        _: usize,
    ) -> CudaResult<Vec<F>> {
        let mut leaf_sources = vec![];
        for p in self.polynomials.iter() {
            let el = p.c0.storage.clone_el_to_host(row_idx)?;
            leaf_sources.push(el);
            let el = p.c1.storage.clone_el_to_host(row_idx)?;
            leaf_sources.push(el);
        }
        assert_eq!(leaf_sources.len(), 2 * self.polynomials.len());
        Ok(leaf_sources)
    }
}

impl<'a, P: PolyForm> AsSingleSlice for GenericComplexPolynomialStorage<'a, P> {
    fn domain_size(&self) -> usize {
        self.polynomials[0].domain_size()
    }

    fn num_polys(&self) -> usize {
        self.polynomials.len()
    }

    fn as_single_slice(&self) -> &[F] {
        assert_adjacent_ext(&self.polynomials);
        let num_polys = 2 * self.polynomials.len();
        let domain_size = self.polynomials[0].domain_size();
        let len = num_polys * domain_size;
        unsafe { std::slice::from_raw_parts(self.polynomials[0].c0.storage.as_ref().as_ptr(), len) }
    }

    fn as_single_slice_mut(&mut self) -> &mut [F] {
        assert_adjacent_ext(&self.polynomials);
        let num_polys = 2 * self.polynomials.len();
        let domain_size = self.polynomials[0].domain_size();
        let len = num_polys * domain_size;
        unsafe {
            std::slice::from_raw_parts_mut(
                self.polynomials[0].c0.storage.as_mut().as_mut_ptr(),
                len,
            )
        }
    }
}

impl<'a, P: PolyForm> GenericComplexPolynomialStorage<'a, P> {
    pub fn allocate(num_polys: usize, domain_size: usize) -> CudaResult<Self> {
        let storage = GenericStorage::allocate(2 * num_polys, domain_size);
        let storage = unsafe { storage.transmute() };
        storage.into_complex_poly_storage()
    }

    fn num_polys(&self) -> usize {
        self.polynomials.len()
    }
}

impl<'a> GenericComplexPolynomialStorage<'a, MonomialBasis> {
    pub fn into_coset_eval<'b>(
        &self,
        coset_idx: usize,
        lde_degree: usize,
        result: &mut GenericComplexPolynomialStorage<'b, CosetEvaluations>,
    ) -> CudaResult<()> {
        let num_polys = self.polynomials.len();
        let domain_size = self.polynomials[0].domain_size();
        let num_coset_ffts = 2 * num_polys;
        ntt::batch_coset_ntt_into(
            self.as_single_slice(),
            result.as_single_slice_mut(),
            coset_idx,
            domain_size,
            lde_degree,
            num_coset_ffts,
        )?;

        Ok(())
    }
}

impl<'a> GenericComplexPolynomialStorage<'a, CosetEvaluations> {
    pub fn build_subtree_for_coset(
        &self,
        cap_size: usize,
        coset_idx: usize,
    ) -> CudaResult<(SubTree, Vec<[F; 4]>)> {
        let domain_size = self.polynomials[0].domain_size();
        let leaf_sources = self.as_single_slice();
        let mut subtree = dvec!(2 * NUM_EL_PER_HASH * domain_size);
        primitives::tree::build_tree(leaf_sources, &mut subtree, domain_size, cap_size, 1)?;
        let subtree_root = get_tree_cap_from_nodes(&subtree, cap_size)?;
        let subtree = SubTree::new(Rc::new(subtree), domain_size, cap_size, coset_idx);

        Ok((subtree, subtree_root))
    }

    pub(crate) fn barycentric_evaluate(
        &self,
        bases: &PrecomputedBasisForBarycentric,
    ) -> CudaResult<Vec<EF>> {
        batch_barycentric_evaluate_ext(self, bases, self.domain_size(), self.num_polys())
    }
}
