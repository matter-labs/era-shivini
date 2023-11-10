use super::*;

pub trait PolyForm: Clone {}

#[derive(Debug, Clone)]
pub struct LagrangeBasis;
impl PolyForm for LagrangeBasis {}

#[derive(Debug, Clone)]
pub struct LDE;
impl PolyForm for LDE {}

#[derive(Debug, Clone)]
pub struct MonomialBasis;
impl PolyForm for MonomialBasis {}

#[derive(Debug, Clone)]
pub struct CosetEvaluations;
impl PolyForm for CosetEvaluations {}

pub(crate) struct PrecomputedBasisForBarycentric {
    pub(crate) bases: DVec<F>,
}

impl PrecomputedBasisForBarycentric {
    pub fn precompute(domain_size: usize, point: EF) -> CudaResult<Self> {
        let mut bases = dvec!(2 * domain_size);
        let coset = F::multiplicative_generator();
        arith::precompute_barycentric_bases(&mut bases, domain_size, coset, point)?;
        Ok(Self { bases })
    }
}

pub(crate) fn batch_barycentric_evaluate_base<S: AsSingleSlice, A: GoodAllocator>(
    source: &S,
    bases: &PrecomputedBasisForBarycentric,
    domain_size: usize,
    num_polys: usize,
) -> CudaResult<Vec<EF, A>> {
    let source = source.as_single_slice();
    assert_eq!(source.len(), num_polys * domain_size);
    arith::barycentric_evaluate_base_at_ext(source, &bases.bases, domain_size, num_polys)
}
pub(crate) fn batch_barycentric_evaluate_ext<S: AsSingleSlice, A: GoodAllocator>(
    source: &S,
    bases: &PrecomputedBasisForBarycentric,
    domain_size: usize,
    num_polys: usize,
) -> CudaResult<Vec<EF, A>> {
    let source = source.as_single_slice();
    assert_eq!(source.len(), 2 * num_polys * domain_size);
    arith::barycentric_evaluate_ext_at_ext(source, &bases.bases, domain_size, num_polys)
}

#[derive(Debug)]
pub enum PolyStorage<'a> {
    Borrowed(&'a [F]),
    Owned(DVec<F>),
}

impl<'a> AsRef<[F]> for PolyStorage<'a> {
    fn as_ref(&self) -> &[F] {
        match self {
            PolyStorage::Borrowed(inner) => *inner,
            PolyStorage::Owned(ref inner) => inner,
        }
    }
}
impl<'a> AsMut<[F]> for PolyStorage<'a> {
    fn as_mut(&mut self) -> &mut [F] {
        match self {
            PolyStorage::Borrowed(_inner) => unimplemented!(),
            PolyStorage::Owned(ref mut inner) => inner,
        }
    }
}

impl<'a> PolyStorage<'a> {
    pub fn len(&self) -> usize {
        match self {
            PolyStorage::Borrowed(inner) => inner.len(),
            PolyStorage::Owned(ref inner) => inner.len(),
        }
    }

    pub fn into_inner(self) -> DVec<F> {
        match self {
            PolyStorage::Borrowed(_) => unimplemented!(),
            PolyStorage::Owned(inner) => inner,
        }
    }

    pub fn copy_from_device_slice(&mut self, other: &[F]) -> CudaResult<()> {
        match self {
            PolyStorage::Borrowed(_) => unimplemented!(),
            PolyStorage::Owned(inner) => {
                mem::d2d(other, inner)?;
            }
        }

        Ok(())
    }

    pub fn clone_el_to_host(&self, pos: usize) -> CudaResult<F> {
        assert!(pos < self.len());
        let mut h_values = vec![F::ZERO];
        match self {
            PolyStorage::Borrowed(inner) => {
                mem::d2h(&inner[pos..pos + 1], &mut h_values)?;
            }
            PolyStorage::Owned(inner) => {
                mem::d2h(&inner[pos..pos + 1], &mut h_values)?;
            }
        }
        Ok(h_values.pop().unwrap())
    }
}

impl<'a> Clone for PolyStorage<'a> {
    fn clone(&self) -> Self {
        let domain_size = self.len();
        assert!(domain_size.is_power_of_two());
        let mut new = dvec!(domain_size);
        match self {
            PolyStorage::Borrowed(inner) => {
                // TODO: shallow or deep copy?
                mem::d2d(inner, &mut new).expect("clone");
            }
            PolyStorage::Owned(inner) => {
                mem::d2d(inner, &mut new).expect("clone");
            }
        };

        PolyStorage::Owned(new)
    }
}

#[derive(Debug, Clone)]
pub struct Poly<'a, P: PolyForm> {
    pub storage: PolyStorage<'a>,
    marker: std::marker::PhantomData<P>,
}

impl<'a, P: PolyForm> Poly<'a, P> {
    pub fn is_owned(&self) -> bool {
        match self.storage {
            PolyStorage::Borrowed(_) => false,
            PolyStorage::Owned(_) => true,
        }
    }
    pub fn zero(domain_size: usize) -> CudaResult<Self> {
        let mut storage = dvec!(domain_size);
        helpers::set_zero(&mut storage)?;
        Ok(Self {
            storage: PolyStorage::Owned(storage),
            marker: std::marker::PhantomData,
        })
    }

    #[allow(dead_code)]
    pub fn one(domain_size: usize) -> CudaResult<Self> {
        let mut storage = dvec!(domain_size);
        let el = DF::one()?;
        helpers::set_value(&mut storage, &el)?;
        Ok(Self {
            storage: PolyStorage::Owned(storage),
            marker: std::marker::PhantomData,
        })
    }

    pub fn domain_size(&self) -> usize {
        assert!(self.storage.len().is_power_of_two());
        self.storage.len()
    }
}

#[derive(Debug, Clone)]
pub struct ComplexPoly<'a, P: PolyForm> {
    pub c0: Poly<'a, P>,
    pub c1: Poly<'a, P>,
}

impl<'a, P: PolyForm> AsSingleSlice for ComplexPoly<'a, P> {
    fn domain_size(&self) -> usize {
        self.c0.domain_size()
    }

    fn num_polys(&self) -> usize {
        1
    }

    fn as_single_slice(&self) -> &[F] {
        // assert_adjacent(&[self.c0, self.c1]); // TODO
        unsafe {
            let len = 2 * self.c0.domain_size();
            std::slice::from_raw_parts(self.c0.storage.as_ref().as_ptr(), len)
        }
    }

    fn as_single_slice_mut(&mut self) -> &mut [F] {
        // assert_adjacent(&[self.c0, self.c1]); // TODO
        unsafe {
            let len = 2 * self.c0.domain_size();
            std::slice::from_raw_parts_mut(self.c0.storage.as_mut().as_mut_ptr(), len)
        }
    }
}

impl<'a, P: PolyForm> From<DVec<F>> for Poly<'a, P> {
    fn from(values: DVec<F>) -> Self {
        Poly {
            storage: PolyStorage::Owned(values),
            marker: std::marker::PhantomData,
        }
    }
}

impl<'a, P: PolyForm> From<&'a [F]> for Poly<'a, P> {
    fn from(values: &'a [F]) -> Self {
        Poly {
            storage: PolyStorage::Borrowed(values),
            marker: std::marker::PhantomData,
        }
    }
}

impl<'a, P: PolyForm> ComplexPoly<'a, P> {
    pub fn new(c0: Poly<'a, P>, c1: Poly<'a, P>) -> Self {
        let c0_ptr = c0.storage.as_ref().as_ptr();
        let c0_len = c0.storage.len();
        assert_eq!(c0_len, c1.storage.len());
        unsafe {
            assert_eq!(c0_ptr.add(c0_len), c1.storage.as_ref().as_ptr());
        }
        Self { c0, c1 }
    }

    #[allow(dead_code)]
    pub fn is_owned(&self) -> bool {
        let c0 = match self.c0.storage {
            PolyStorage::Borrowed(_) => false,
            PolyStorage::Owned(_) => true,
        };
        let c1 = match self.c1.storage {
            PolyStorage::Borrowed(_) => false,
            PolyStorage::Owned(_) => true,
        };

        c0 && c1
    }

    pub fn zero(domain_size: usize) -> CudaResult<Self> {
        let mut chunks = dvec!(2 * domain_size)
            .into_adjacent_chunks(domain_size)
            .into_iter();
        let mut c0 = chunks.next().unwrap();
        let mut c1 = chunks.next().unwrap();
        assert!(chunks.next().is_none());

        helpers::set_zero(&mut c0)?;
        helpers::set_zero(&mut c1)?;
        Ok(Self {
            c0: Poly::from(c0),
            c1: Poly::from(c1),
        })
    }
    pub fn one(domain_size: usize) -> CudaResult<Self> {
        let mut chunks = dvec!(2 * domain_size)
            .into_adjacent_chunks(domain_size)
            .into_iter();
        let mut c0 = chunks.next().unwrap();
        let mut c1 = chunks.next().unwrap();
        assert!(chunks.next().is_none());

        let el = DF::one()?;
        helpers::set_value(&mut c0, &el)?;
        helpers::set_zero(&mut c1)?;
        Ok(Self {
            c0: Poly::from(c0),
            c1: Poly::from(c1),
        })
    }

    pub fn domain_size(&self) -> usize {
        self.c0.domain_size()
    }
}

impl<'a> Poly<'a, CosetEvaluations> {
    #[allow(dead_code)]
    pub fn ifft(mut self, coset: &DF) -> CudaResult<Poly<'a, MonomialBasis>> {
        ntt::ifft(self.storage.as_mut(), coset)?;
        Ok(Poly {
            storage: self.storage,
            marker: std::marker::PhantomData,
        })
    }

    #[allow(dead_code)]
    pub fn lde_from_trace_values(
        &mut self,
        domain_size: usize,
        lde_degree: usize,
    ) -> CudaResult<()> {
        // first coset has base trace lagranage basis values
        ntt::lde_from_lagrange_basis(self.storage.as_mut(), domain_size, lde_degree)
    }
}

impl<'a> Poly<'a, LDE> {
    pub fn ifft(mut self, coset: &DF) -> CudaResult<Poly<'a, MonomialBasis>> {
        ntt::ifft(self.storage.as_mut(), coset)?;
        Ok(Poly {
            storage: self.storage,
            marker: std::marker::PhantomData,
        })
    }
}

impl<'a> Poly<'a, LagrangeBasis> {
    #[allow(dead_code)]
    pub fn ifft(mut self, coset: &DF) -> CudaResult<Poly<'a, MonomialBasis>> {
        ntt::ifft(self.storage.as_mut(), &coset)?;
        Ok(Poly {
            storage: self.storage,
            marker: std::marker::PhantomData,
        })
    }

    pub fn grand_sum(&self) -> CudaResult<DF> {
        let tmp_size = helpers::calculate_tmp_buffer_size_for_grand_sum(self.domain_size())?;
        let mut tmp = dvec!(tmp_size);
        let sum = arith::grand_sum(self.storage.as_ref(), &mut tmp)?;
        let sum: DF = DF::from(sum);

        Ok(sum)
    }
}

impl<'a> Poly<'a, MonomialBasis> {
    #[allow(dead_code)]
    pub fn coset_fft(
        mut self,
        coset_idx: usize,
        lde_degree: usize,
    ) -> CudaResult<Poly<'a, CosetEvaluations>> {
        ntt::coset_fft(self.storage.as_mut(), coset_idx, lde_degree)?;
        Ok(Poly {
            storage: self.storage,
            marker: std::marker::PhantomData,
        })
    }

    #[allow(dead_code)]
    pub fn fft(mut self, coset: &DF) -> CudaResult<Poly<'a, LagrangeBasis>> {
        ntt::fft(self.storage.as_mut(), coset)?;

        Ok(Poly {
            storage: self.storage,
            marker: std::marker::PhantomData,
        })
    }

    #[allow(dead_code)]
    pub fn lde(self, lde_degree: usize) -> CudaResult<Poly<'a, LDE>> {
        let mut result = Poly::zero(self.domain_size() * lde_degree)?;
        self.lde_into(&mut result, lde_degree)?;

        Ok(result)
    }

    #[allow(dead_code)]
    pub fn lde_into(self, result: &mut Poly<LDE>, lde_degree: usize) -> CudaResult<()> {
        ntt::lde(self.storage.as_ref(), result.storage.as_mut(), lde_degree)
    }

    #[allow(dead_code)]
    pub fn evaluate_at_ext(&self, at: &DExt) -> CudaResult<DExt> {
        arith::evaluate_base_at_ext(self.storage.as_ref(), at)
    }

    // use for monomials until we have a barycentric
    #[allow(dead_code)]
    pub fn grand_sum(&self) -> CudaResult<DF> {
        let tmp_size = helpers::calculate_tmp_buffer_size_for_grand_sum(self.domain_size())?;
        let mut tmp = dvec!(tmp_size);
        let sum = arith::grand_sum(self.storage.as_ref(), &mut tmp)?;
        let sum: DF = DF::from(sum);

        Ok(sum)
    }

    #[allow(dead_code)]
    pub fn bitreverse(&mut self) -> CudaResult<()> {
        ntt::bitreverse(self.storage.as_mut())
    }
}

impl<'a> ComplexPoly<'a, CosetEvaluations> {
    #[allow(dead_code)]
    pub fn rotate(&mut self) -> CudaResult<()> {
        self.c0.bitreverse()?;
        helpers::rotate_left(self.c0.storage.as_mut())?;
        self.c0.bitreverse()?;

        self.c1.bitreverse()?;
        helpers::rotate_left(self.c1.storage.as_mut())?;
        self.c1.bitreverse()?;

        Ok(())
    }

    #[allow(dead_code)]
    pub fn ifft(self, coset: &DF) -> CudaResult<ComplexPoly<'a, MonomialBasis>> {
        let Self { c0, c1 } = self;
        let c0 = c0.ifft(coset)?;
        let c1 = c1.ifft(coset)?;

        Ok(ComplexPoly { c0, c1 })
    }
}
impl<'a> ComplexPoly<'a, LDE> {
    pub fn ifft(self, coset: &DF) -> CudaResult<ComplexPoly<'a, MonomialBasis>> {
        let Self { c0, c1 } = self;
        let c0 = c0.ifft(coset)?;
        let c1 = c1.ifft(coset)?;

        Ok(ComplexPoly { c0, c1 })
    }
}

impl<'a> ComplexPoly<'a, LagrangeBasis> {
    #[allow(dead_code)]
    pub fn ifft(self, coset: &DF) -> CudaResult<ComplexPoly<'a, MonomialBasis>> {
        let Self { c0, c1 } = self;
        let c0 = c0.ifft(&coset)?;
        let c1 = c1.ifft(&coset)?;

        Ok(ComplexPoly { c0, c1 })
    }

    pub fn grand_sum(&self) -> CudaResult<DExt> {
        let sum_c0 = self.c0.grand_sum()?;
        let sum_c1 = self.c1.grand_sum()?;

        Ok(DExt::new(sum_c0, sum_c1))
    }
}

impl<'a> ComplexPoly<'a, MonomialBasis> {
    #[allow(dead_code)]
    pub fn lde(self, lde_degree: usize) -> CudaResult<ComplexPoly<'a, LDE>> {
        let lde_size = self.domain_size() * lde_degree;
        let mut c0 = Poly::zero(lde_size)?;
        let mut c1 = Poly::zero(lde_size)?;
        self.c0.lde_into(&mut c0, lde_degree)?;
        self.c1.lde_into(&mut c1, lde_degree)?;

        Ok(ComplexPoly { c0, c1 })
    }

    pub fn evaluate_at_ext(&self, at: &DExt) -> CudaResult<DExt> {
        arith::evaluate_ext_at_ext(self.c0.storage.as_ref(), self.c1.storage.as_ref(), at)
    }

    #[allow(dead_code)]
    pub fn bitreverse(&mut self) -> CudaResult<()> {
        self.c0.bitreverse()?;
        self.c1.bitreverse()?;

        Ok(())
    }

    #[allow(dead_code)]
    pub fn grand_sum(&self) -> CudaResult<DExt> {
        let sum_c0 = self.c0.grand_sum()?;
        let sum_c1 = self.c1.grand_sum()?;

        Ok(DExt::new(sum_c0, sum_c1))
    }

    #[allow(dead_code)]
    pub fn coset_fft(
        self,
        coset_idx: usize,
        lde_degree: usize,
    ) -> CudaResult<ComplexPoly<'a, CosetEvaluations>> {
        let Self { c0, c1 } = self;
        let c0 = c0.coset_fft(coset_idx, lde_degree)?;
        let c1 = c1.coset_fft(coset_idx, lde_degree)?;

        Ok(ComplexPoly { c0, c1 })
    }

    pub fn into_degree_n_polys(
        self,
        domain_size: usize,
    ) -> CudaResult<Vec<ComplexPoly<'a, MonomialBasis>>> {
        let ComplexPoly { c0, c1 } = self;

        let c0_chunks = c0.storage.into_inner().into_adjacent_chunks(domain_size);
        let c1_chunks = c1.storage.into_inner().into_adjacent_chunks(domain_size);
        let all_polys = dvec!(2 * c0_chunks.len() * domain_size);

        let mut all_polys_iter = all_polys.into_adjacent_chunks(domain_size).into_iter();

        let mut chunks = vec![];
        for (c0_chunk, c1_chunk) in c0_chunks.into_iter().zip(c1_chunks.into_iter()) {
            // we want both c0 and c1 to be adjacent
            let mut new_c0 = all_polys_iter.next().unwrap();
            let mut new_c1 = all_polys_iter.next().unwrap();
            new_c0.copy_from_device_slice(&c0_chunk)?;
            new_c1.copy_from_device_slice(&c1_chunk)?;
            let poly = ComplexPoly::new(Poly::from(new_c0), Poly::from(new_c1));
            chunks.push(poly);
        }

        Ok(chunks)
    }
}

macro_rules! impl_common_poly {
    ($form:tt) => {
        impl<'a> Poly<'a, $form> {
            #[allow(dead_code)]
            pub fn add_assign<'b>(&mut self, other: &Poly<'b, $form>) -> CudaResult<()> {
                arith::add_assign(self.storage.as_mut(), other.storage.as_ref())
            }

            #[allow(dead_code)]
            pub fn sub_assign<'b>(&mut self, other: &Poly<'b, $form>) -> CudaResult<()> {
                let mut other = other.clone();
                other.negate()?;
                // arith::sub_assign(self.storage.as_mut(), other.storage.as_ref())
                arith::add_assign(self.storage.as_mut(), other.storage.as_ref())
            }

            #[allow(dead_code)]
            pub fn mul_assign<'b>(&mut self, other: &Poly<'b, $form>) -> CudaResult<()> {
                assert_eq!(self.storage.len(), other.storage.len());
                arith::mul_assign(self.storage.as_mut(), other.storage.as_ref())
            }

            #[allow(dead_code)]
            pub fn square(&mut self) -> CudaResult<()> {
                let other = self.clone();
                arith::mul_assign(self.storage.as_mut(), other.storage.as_ref())
            }

            #[allow(dead_code)]
            pub fn add_constant(&mut self, value: &DF) -> CudaResult<()> {
                arith::add_constant(self.storage.as_mut(), value)
            }

            #[allow(dead_code)]
            pub fn sub_constant(&mut self, value: &DF) -> CudaResult<()> {
                let mut h_value: F = value.clone().into();
                h_value.negate();
                let value: DF = h_value.into();
                arith::add_constant(self.storage.as_mut(), &value)
            }

            #[allow(dead_code)]
            pub fn scale(&mut self, value: &DF) -> CudaResult<()> {
                arith::scale(self.storage.as_mut(), value)
            }

            #[allow(dead_code)]
            pub fn negate(&mut self) -> CudaResult<()> {
                arith::negate(self.storage.as_mut())
            }

            #[allow(dead_code)]
            pub fn inverse(&mut self) -> CudaResult<()> {
                arith::inverse(self.storage.as_mut())
            }

            #[allow(dead_code)]
            pub fn bitreverse(&mut self) -> CudaResult<()> {
                ntt::bitreverse(self.storage.as_mut())
            }

            #[allow(dead_code)]
            pub fn shifted_grand_product(&mut self) -> CudaResult<()> {
                let domain_size = self.storage.len();
                let tmp_size = helpers::calculate_tmp_buffer_size_for_grand_product(domain_size)?;
                let mut tmp = dvec!(tmp_size);
                arith::shifted_grand_product(self.storage.as_mut(), &mut tmp)
            }
        }
    };
}

impl_common_poly!(LDE);
impl_common_poly!(LagrangeBasis);
impl_common_poly!(CosetEvaluations);

macro_rules! impl_common_complex_poly {
    ($form:tt) => {
        impl<'a> ComplexPoly<'a, $form> {
            #[allow(dead_code)]
            pub fn from_real(c0: Poly<'a, $form>) -> CudaResult<ComplexPoly<'a, $form>> {
                assert!(c0.is_owned());
                let domain_size = c0.storage.len();
                assert!(domain_size.is_power_of_two());
                // we always want real and imaginary part to be continuous in the memory
                // so do a copy here
                let storage = dvec!(2 * domain_size);
                let parts = storage.into_adjacent_chunks(domain_size);
                assert_eq!(parts.len(), 2);
                let mut parts = parts.into_iter();
                let mut new_c0 = parts.next().unwrap();
                mem::h2d(c0.storage.as_ref(), &mut new_c0)?;
                let mut new_c1 = parts.next().unwrap();
                helpers::set_zero(&mut new_c1)?;

                Ok(Self {
                    c0: Poly::from(new_c0),
                    c1: Poly::from(new_c1),
                })
            }

            #[allow(dead_code)]
            pub fn add_assign<'b>(&mut self, other: &ComplexPoly<'b, $form>) -> CudaResult<()> {
                assert_eq!(self.c0.storage.len(), other.c0.storage.len());
                assert_eq!(self.c1.storage.len(), other.c1.storage.len());
                self.c0.add_assign(&other.c0)?;
                self.c1.add_assign(&other.c1)?;

                Ok(())
            }

            #[allow(dead_code)]
            pub fn add_assign_real<'b>(&mut self, other: &Poly<'b, $form>) -> CudaResult<()> {
                assert_eq!(self.c0.storage.len(), other.storage.len());
                self.c0.add_assign(other)?;

                Ok(())
            }

            #[allow(dead_code)]
            pub fn sub_assign<'b>(&mut self, other: &ComplexPoly<'b, $form>) -> CudaResult<()> {
                assert_eq!(self.c0.storage.len(), other.c0.storage.len());
                assert_eq!(self.c1.storage.len(), other.c1.storage.len());
                self.c0.sub_assign(&other.c0)?;
                self.c1.sub_assign(&other.c1)?;

                Ok(())
            }

            #[allow(dead_code)]
            pub fn sub_assign_real<'b>(&mut self, other: &Poly<'b, $form>) -> CudaResult<()> {
                assert_eq!(self.c0.storage.len(), other.storage.len());
                self.c0.sub_assign(other)?;

                Ok(())
            }

            #[allow(dead_code)]
            pub fn mul_assign<'b>(&mut self, other: &ComplexPoly<'b, $form>) -> CudaResult<()> {
                let non_residue = DF::non_residue()?;

                let mut t0 = self.c0.clone();
                let mut t1 = self.c1.clone();

                t0.mul_assign(&other.c0)?;
                t1.mul_assign(&other.c1)?;
                t1.scale(&non_residue)?;
                t0.add_assign(&t1)?;

                self.c0.mul_assign(&other.c1)?;
                self.c1.mul_assign(&other.c0)?;
                self.c1.add_assign(&self.c0)?;
                mem::d2d(&t0.storage.as_ref(), self.c0.storage.as_mut())?;

                Ok(())
            }

            #[allow(dead_code)]
            pub fn add_constant(&mut self, value: &DExt) -> CudaResult<()> {
                self.c0.add_constant(&value.c0)?;
                self.c1.add_constant(&value.c1)?;

                Ok(())
            }

            #[allow(dead_code)]
            pub fn sub_constant(&mut self, value: &DExt) -> CudaResult<()> {
                self.c0.sub_constant(&value.c0)?;
                self.c1.sub_constant(&value.c1)?;

                Ok(())
            }

            #[allow(dead_code)]
            pub fn scale_real(&mut self, point: &DExt) -> CudaResult<()> {
                // self.c1.storage.copy_from_device_slice(&self.c0.storage)?;
                mem::d2d(self.c0.storage.as_ref(), self.c1.storage.as_mut())?;
                self.c0.scale(&point.c0)?;
                self.c1.scale(&point.c1)?;

                Ok(())
            }

            #[allow(dead_code)]
            pub fn scale(&mut self, point: &DExt) -> CudaResult<()> {
                let non_residue = DF::non_residue()?;

                let mut t0 = self.c0.storage.clone();
                let mut t1 = self.c1.storage.clone();

                arith::scale(t0.as_mut(), &point.c0)?;
                arith::scale(t1.as_mut(), &point.c1)?;
                arith::scale(t1.as_mut(), &non_residue)?;
                arith::add_assign(t0.as_mut(), t1.as_ref())?;

                arith::scale(self.c0.storage.as_mut(), &point.c1)?;
                arith::scale(self.c1.storage.as_mut(), &point.c0)?;
                arith::add_assign(self.c1.storage.as_mut(), self.c0.storage.as_ref())?;

                // self.c0.storage.copy_from_device_slice(&t0)?;
                mem::d2d(&t0.as_ref(), self.c0.storage.as_mut())?;
                Ok(())
            }

            #[allow(dead_code)]
            pub fn negate(&mut self) -> CudaResult<()> {
                self.c0.negate()?;
                self.c1.negate()?;

                Ok(())
            }

            #[allow(dead_code)]
            pub fn inverse(&mut self) -> CudaResult<()> {
                arith::inverse_ef(self.c0.storage.as_mut(), self.c1.storage.as_mut())
            }

            #[allow(dead_code)]
            pub fn shifted_grand_product(&mut self) -> CudaResult<()> {
                let tmp_size = helpers::calculate_tmp_buffer_size_for_grand_product(
                    2 * self.c0.storage.len(),
                )?;
                let mut tmp = dvec!(tmp_size);
                arith::complex_shifted_grand_product(
                    self.c0.storage.as_mut(),
                    self.c1.storage.as_mut(),
                    &mut tmp,
                )?;

                Ok(())
            }

            #[allow(dead_code)]
            pub fn bitreverse(&mut self) -> CudaResult<()> {
                self.c0.bitreverse()?;
                self.c1.bitreverse()?;

                Ok(())
            }
        }
    };
}

impl_common_complex_poly!(LDE);
impl_common_complex_poly!(LagrangeBasis);
impl_common_complex_poly!(CosetEvaluations);
