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

#[derive(Debug, Clone)]
pub struct Undefined;
impl PolyForm for Undefined {}

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
    BorrowedMut(&'a mut [F]),
    Owned(DVec<F>),
}

impl<'a> AsRef<[F]> for PolyStorage<'a> {
    fn as_ref(&self) -> &[F] {
        match self {
            PolyStorage::Borrowed(inner) => *inner,
            PolyStorage::BorrowedMut(inner) => *inner,
            PolyStorage::Owned(inner) => inner,
        }
    }
}
impl<'a> AsMut<[F]> for PolyStorage<'a> {
    fn as_mut(&mut self) -> &mut [F] {
        match self {
            PolyStorage::Borrowed(_) => unimplemented!(),
            PolyStorage::BorrowedMut(inner) => *inner,
            PolyStorage::Owned(inner) => inner,
        }
    }
}

impl<'a> PolyStorage<'a> {
    pub fn len(&self) -> usize {
        match self {
            PolyStorage::Borrowed(inner) => inner.len(),
            PolyStorage::BorrowedMut(inner) => inner.len(),
            PolyStorage::Owned(inner) => inner.len(),
        }
    }

    pub fn into_inner(self) -> DVec<F> {
        match self {
            PolyStorage::Borrowed(_) => unimplemented!(),
            PolyStorage::BorrowedMut(_) => unimplemented!(),
            PolyStorage::Owned(inner) => inner,
        }
    }

    #[allow(dead_code)]
    pub fn copy_from_device_slice(&mut self, other: &[F]) -> CudaResult<()> {
        match self {
            PolyStorage::Borrowed(_) => unimplemented!(),
            PolyStorage::BorrowedMut(_) => unimplemented!(),
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
            PolyStorage::BorrowedMut(inner) => {
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
        if !is_dry_run().unwrap_or(true) {
            match self {
                PolyStorage::Borrowed(inner) => {
                    // TODO: shallow or deep copy?
                    mem::d2d(inner, &mut new).expect("clone");
                }
                PolyStorage::BorrowedMut(inner) => {
                    // TODO: shallow or deep copy?
                    mem::d2d(inner, &mut new).expect("clone");
                }
                PolyStorage::Owned(inner) => {
                    mem::d2d(inner, &mut new).expect("clone");
                }
            };
        }
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
            PolyStorage::BorrowedMut(_) => false,
            PolyStorage::Owned(_) => true,
        }
    }

    #[allow(dead_code)]
    pub fn empty(domain_size: usize) -> CudaResult<Self> {
        let storage = dvec!(domain_size);
        Ok(Self {
            storage: PolyStorage::Owned(storage),
            marker: std::marker::PhantomData,
        })
    }

    #[allow(dead_code)]
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

#[derive(Debug)]
pub struct ComplexPoly<'a, P: PolyForm> {
    pub c0: Poly<'a, P>,
    pub c1: Poly<'a, P>,
}

impl<'a, P: PolyForm> Clone for ComplexPoly<'a, P> {
    fn clone(&self) -> Self {
        // Uses expect, like PolyStorage::clone. We can't return a Result<Self> here unfortunately.
        let mut new = ComplexPoly::<P>::empty(self.domain_size()).expect("empty");
        mem::d2d(self.as_single_slice(), new.as_single_slice_mut()).expect("clone");

        new
    }
}

impl<'a, P: PolyForm> AsSingleSlice for ComplexPoly<'a, P> {
    fn domain_size(&self) -> usize {
        let domain_size = self.c0.domain_size();
        assert!(domain_size.is_power_of_two());
        domain_size
    }

    fn num_polys(&self) -> usize {
        1
    }

    fn as_single_slice(&self) -> &[F] {
        ComplexPoly::<P>::assert_c0_c1_adjacent(&self.c0, &self.c1);
        unsafe {
            let len = 2 * self.c0.domain_size();
            std::slice::from_raw_parts(self.c0.storage.as_ref().as_ptr(), len)
        }
    }

    fn as_single_slice_mut(&mut self) -> &mut [F] {
        ComplexPoly::<P>::assert_c0_c1_adjacent(&self.c0, &self.c1);
        unsafe {
            let len = 2 * self.c0.domain_size();
            std::slice::from_raw_parts_mut(self.c0.storage.as_mut().as_mut_ptr(), len)
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

impl<'a, P: PolyForm> From<&'a mut [F]> for Poly<'a, P> {
    fn from(values: &'a mut [F]) -> Self {
        Poly {
            storage: PolyStorage::BorrowedMut(values),
            marker: std::marker::PhantomData,
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

impl<'a, P: PolyForm> ComplexPoly<'a, P> {
    fn assert_c0_c1_adjacent(c0: &Poly<'a, P>, c1: &Poly<'a, P>) {
        let c0_ptr = c0.storage.as_ref().as_ptr();
        let c0_len = c0.domain_size();
        assert_eq!(c0_len, c1.storage.len());
        unsafe {
            assert_eq!(c0_ptr.add(c0_len), c1.storage.as_ref().as_ptr());
        }
    }

    pub fn new(c0: Poly<'a, P>, c1: Poly<'a, P>) -> Self {
        ComplexPoly::<P>::assert_c0_c1_adjacent(&c0, &c1);
        Self { c0, c1 }
    }

    #[allow(dead_code)]
    pub fn is_owned(&self) -> bool {
        let c0 = match self.c0.storage {
            PolyStorage::Borrowed(_) => false,
            PolyStorage::BorrowedMut(_) => false,
            PolyStorage::Owned(_) => true,
        };
        let c1 = match self.c1.storage {
            PolyStorage::Borrowed(_) => false,
            PolyStorage::BorrowedMut(_) => false,
            PolyStorage::Owned(_) => true,
        };

        c0 && c1
    }

    pub fn empty(domain_size: usize) -> CudaResult<Self> {
        let mut chunks = dvec!(2 * domain_size)
            .into_adjacent_chunks(domain_size)
            .into_iter();
        let c0 = chunks.next().unwrap();
        let c1 = chunks.next().unwrap();
        assert!(chunks.next().is_none());

        Ok(Self {
            c0: Poly::from(c0),
            c1: Poly::from(c1),
        })
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

impl<'a> Poly<'a, LDE> {
    pub fn intt(mut self) -> CudaResult<Poly<'a, MonomialBasis>> {
        ntt::lde_intt(self.storage.as_mut())?;
        Ok(Poly {
            storage: self.storage,
            marker: std::marker::PhantomData,
        })
    }
}

impl<'a> Poly<'a, LagrangeBasis> {
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
}

impl<'a> ComplexPoly<'a, LDE> {
    pub fn intt(self) -> CudaResult<ComplexPoly<'a, MonomialBasis>> {
        let Self { c0, c1 } = self;
        let c0 = c0.intt()?;
        let c1 = c1.intt()?;

        Ok(ComplexPoly { c0, c1 })
    }
}

impl<'a> ComplexPoly<'a, LagrangeBasis> {
    pub fn grand_sum(&self) -> CudaResult<DExt> {
        let sum_c0 = self.c0.grand_sum()?;
        let sum_c1 = self.c1.grand_sum()?;

        Ok(DExt::new(sum_c0, sum_c1))
    }
}

impl<'a> ComplexPoly<'a, MonomialBasis> {
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
                arith::sub_assign(self.storage.as_mut(), other.storage.as_ref())
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
                arith::sub_constant(self.storage.as_mut(), &value)
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
            pub fn from_real(c0: &Poly<'a, $form>) -> CudaResult<ComplexPoly<'a, $form>> {
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
                arith::mul_assign_complex(
                    self.c0.storage.as_mut(),
                    self.c1.storage.as_mut(),
                    other.c0.storage.as_ref(),
                    other.c1.storage.as_ref(),
                )
            }

            #[allow(dead_code)]
            pub fn mul_assign_real<'b>(&mut self, other: &Poly<'b, $form>) -> CudaResult<()> {
                assert_eq!(self.c0.storage.len(), other.storage.len());
                assert_eq!(self.c1.storage.len(), other.storage.len());
                self.c0.mul_assign(&other)?;
                self.c1.mul_assign(&other)?;

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
