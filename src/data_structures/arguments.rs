use crate::data_structures::cache::StorageCache;
use boojum::cs::{implementations::proof::OracleQuery, oracle::TreeHasher, LookupParameters};
use std::ops::{Deref, Range};
use std::rc::Rc;

use super::*;

#[allow(dead_code)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum ArgumentsPolyType {
    Z,
    PartialProduct,
    LookupA,
    LookupB,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct ArgumentsLayout {
    pub num_z_polys: usize,
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
        let num_z_polys = 1;
        let num_variable_cols = trace_layout.num_variable_cols;
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
                        assert!(!share_table_id);
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

        let num_z_polys = num_z_polys * 2;
        let num_partial_products = num_partial_products * 2;
        let num_lookup_a_polys = num_lookup_a_polys * 2;
        let num_lookup_b_polys = num_lookup_b_polys * 2;

        Self {
            num_z_polys,
            num_partial_products,
            num_lookup_a_polys,
            num_lookup_b_polys,
        }
    }

    pub fn num_polys(&self) -> usize {
        self.num_z_polys
            + self.num_partial_products
            + self.num_lookup_a_polys
            + self.num_lookup_b_polys
    }
}

impl GenericStorageLayout for ArgumentsLayout {
    type PolyType = ArgumentsPolyType;

    fn num_polys(&self) -> usize {
        self.num_polys()
    }

    fn poly_range(&self, poly_type: Self::PolyType) -> (Range<usize>, Self) {
        let start = match poly_type {
            ArgumentsPolyType::Z => 0,
            ArgumentsPolyType::PartialProduct => self.num_z_polys,
            ArgumentsPolyType::LookupA => self.num_z_polys + self.num_partial_products,
            ArgumentsPolyType::LookupB => {
                self.num_z_polys + self.num_partial_products + self.num_lookup_a_polys
            }
        };
        let len = match poly_type {
            ArgumentsPolyType::Z => self.num_z_polys,
            ArgumentsPolyType::PartialProduct => self.num_partial_products,
            ArgumentsPolyType::LookupA => self.num_lookup_a_polys,
            ArgumentsPolyType::LookupB => self.num_lookup_b_polys,
        };
        let range = start..start + len;
        let layout = Self {
            num_z_polys: match poly_type {
                ArgumentsPolyType::Z => self.num_z_polys,
                _ => 0,
            },
            num_partial_products: match poly_type {
                ArgumentsPolyType::PartialProduct => self.num_partial_products,
                _ => 0,
            },
            num_lookup_a_polys: match poly_type {
                ArgumentsPolyType::LookupA => self.num_lookup_a_polys,
                _ => 0,
            },
            num_lookup_b_polys: match poly_type {
                ArgumentsPolyType::LookupB => self.num_lookup_b_polys,
                _ => 0,
            },
        };
        (range, layout)
    }
}

pub type GenericArgumentsStorage<P> = GenericStorage<P, ArgumentsLayout>;

pub type ArgumentsCache = StorageCache<ArgumentsLayout, ()>;

pub struct ArgumentsPolynomials<'a, P: PolyForm> {
    pub z_polys: Vec<ComplexPoly<'a, P>>,
    pub partial_products: Vec<ComplexPoly<'a, P>>,
    pub lookup_a_polys: Vec<ComplexPoly<'a, P>>,
    pub lookup_b_polys: Vec<ComplexPoly<'a, P>>,
}

impl<'a, P: PolyForm> ArgumentsPolynomials<'a, P> {
    pub fn new(mut polynomials: Vec<ComplexPoly<'a, P>>, layout: ArgumentsLayout) -> Self {
        let ArgumentsLayout {
            num_z_polys,
            num_partial_products,
            num_lookup_a_polys,
            num_lookup_b_polys,
        } = layout;
        assert_eq!(num_z_polys % 2, 0);
        let num_z_polys = num_z_polys / 2;
        assert_eq!(num_partial_products % 2, 0);
        let num_partial_products = num_partial_products / 2;
        assert_eq!(num_lookup_a_polys % 2, 0);
        let num_lookup_a_polys = num_lookup_a_polys / 2;
        assert_eq!(num_lookup_b_polys % 2, 0);
        let num_lookup_b_polys = num_lookup_b_polys / 2;
        let lookup_b_polys = polynomials.split_off(polynomials.len() - num_lookup_b_polys);
        let lookup_a_polys = polynomials.split_off(polynomials.len() - num_lookup_a_polys);
        let partial_products = polynomials.split_off(polynomials.len() - num_partial_products);
        let z_polys = polynomials.split_off(polynomials.len() - num_z_polys);
        assert!(polynomials.is_empty());
        Self {
            z_polys,
            partial_products,
            lookup_a_polys,
            lookup_b_polys,
        }
    }
}

impl<P: PolyForm> GenericArgumentsStorage<P> {
    pub fn as_polynomials(&self) -> ArgumentsPolynomials<P> {
        let layout = self.layout;
        let polynomials = self.as_complex_polys();
        ArgumentsPolynomials::new(polynomials, layout)
    }

    pub fn as_polynomials_mut(&mut self) -> ArgumentsPolynomials<P> {
        let layout = self.layout;
        let polynomials = self.as_complex_polys_mut();
        ArgumentsPolynomials::new(polynomials, layout)
    }
}

impl GenericArgumentsStorage<CosetEvaluations> {
    pub(crate) fn barycentric_evaluate(
        &self,
        bases: &PrecomputedBasisForBarycentric,
    ) -> CudaResult<Vec<EF>> {
        batch_barycentric_evaluate_ext(self, bases, self.domain_size(), self.num_polys() / 2)
    }
}

impl<'a> LeafSourceQuery for ArgumentsPolynomials<'a, CosetEvaluations> {
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
        assert_eq!(self.z_polys.len(), 1);
        let z_poly = &self.z_polys[0];
        let el = z_poly.c0.storage.clone_el_to_host(row_idx)?;
        values.push(el);
        let el = z_poly.c1.storage.clone_el_to_host(row_idx)?;
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

        if coset_idx >= self.fri_lde_degree {
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
