use super::*;

#[derive(Clone)]
pub struct GenericStorage {
    pub(crate) inner: DVec<F>,
    pub(crate) num_polys: usize,
    pub(crate) domain_size: usize,
}

impl AsSingleSlice for GenericStorage {
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

impl GenericStorage {
    pub fn allocate(num_polys: usize, domain_size: usize) -> CudaResult<Self> {
        let storage = dvec!(num_polys * domain_size);
        Ok(Self {
            inner: storage,
            num_polys,
            domain_size,
        })
    }

    #[allow(dead_code)]
    pub fn into_poly_storage<P: PolyForm>(self) -> GenericPolynomialStorage<'static, P> {
        let Self {
            inner,
            num_polys,
            domain_size,
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

    pub fn as_poly_storage<'a, P: PolyForm>(&'a self) -> Vec<Poly<'a, P>> {
        let Self {
            inner,
            num_polys,
            domain_size,
        } = self;
        let num_polys = *num_polys;
        let domain_size = *domain_size;
        let mut storage_ref = &inner[..];
        let mut all_polys_ref = vec![];
        for _ in 0..num_polys {
            let (values, remaining) = storage_ref.split_at(domain_size);
            let poly = Poly::<'a, P>::from(values);
            all_polys_ref.push(poly);
            storage_ref = remaining;
        }
        assert_eq!(all_polys_ref.len() * domain_size, inner.len());
        all_polys_ref
    }

    pub fn into_complex_poly_storage<P: PolyForm>(
        self,
    ) -> CudaResult<GenericComplexPolynomialStorage<'static, P>> {
        let Self {
            inner,
            num_polys,
            domain_size,
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
}

impl AsMut<DVec<F>> for GenericStorage {
    fn as_mut(&mut self) -> &mut DVec<F> {
        &mut self.inner
    }
}
impl AsRef<DVec<F>> for GenericStorage {
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
        let storage = GenericStorage::allocate(num_polys, domain_size)?;
        Ok(storage.into_poly_storage())
    }

    #[allow(dead_code)]
    pub fn num_polys(&self) -> usize {
        self.polynomials.len()
    }
}

impl<'a, P: PolyForm> AsSingleSlice for GenericPolynomialStorage<'a, P> {
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

    fn domain_size(&self) -> usize {
        self.polynomials[0].domain_size()
    }

    fn num_polys(&self) -> usize {
        self.polynomials.len()
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

    fn domain_size(&self) -> usize {
        self.polynomials[0].domain_size()
    }

    fn num_polys(&self) -> usize {
        self.polynomials.len()
    }
}
impl<'a, P: PolyForm> AsSingleSlice for &GenericComplexPolynomialStorage<'a, P> {
    fn as_single_slice(&self) -> &[F] {
        assert_adjacent_ext(&self.polynomials);
        let num_polys = 2 * self.polynomials.len();
        let domain_size = self.polynomials[0].domain_size();
        let len = num_polys * domain_size;
        unsafe { std::slice::from_raw_parts(self.polynomials[0].c0.storage.as_ref().as_ptr(), len) }
    }
    
    fn domain_size(&self) -> usize {
        self.polynomials[0].domain_size()
    }

    fn num_polys(&self) -> usize {
        self.polynomials.len()
    }
}

impl<'a, P: PolyForm> GenericComplexPolynomialStorage<'a, P> {
    pub fn allocate(num_polys: usize, domain_size: usize) -> CudaResult<Self> {
        let storage = GenericStorage::allocate(2 * num_polys, domain_size)?;
        storage.into_complex_poly_storage()
    }

    pub fn clone(&self) -> CudaResult<Self> {
        let num_polys = self.polynomials.len();
        assert!(num_polys > 0);
        let domain_size = self.polynomials[0].domain_size();

        let mut new_storage = Self::allocate(num_polys, domain_size)?;

        let src = self.as_single_slice();
        let dst = new_storage.as_single_slice_mut();

        mem::d2d(src, dst)?;

        Ok(new_storage)
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
        ntt::batch_coset_fft_into(
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
        crate::primitives::tree::build_tree(leaf_sources, &mut subtree, domain_size, cap_size, 1)?;
        let subtree_root = get_tree_cap_from_nodes(&subtree, cap_size)?;
        let subtree = SubTree::new(subtree, domain_size, cap_size, coset_idx);

        Ok((subtree, subtree_root))
    }

    pub(crate) fn barycentric_evaluate(
        &self,
        bases: &PrecomputedBasisForBarycentric,
    ) -> CudaResult<Vec<EF>> {
        batch_barycentric_evaluate_ext(self, bases, self.domain_size(), self.num_polys())
    }
}
