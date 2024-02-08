use boojum::cs::{implementations::polynomial_storage::SetupBaseStorage, oracle::TreeHasher};
use boojum::worker::Worker;
use std::ops::Range;

use crate::cs::{
    materialize_permutation_cols_from_indexes_into, GpuSetup, PACKED_PLACEHOLDER_BITMASK,
};
use crate::data_structures::cache::{StorageCache, StorageCacheStrategy};
use crate::primitives::helpers::set_by_value;

use super::*;

#[allow(dead_code)]
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum SetupPolyType {
    Permutation,
    Constant,
    Table,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct SetupLayout {
    pub num_permutation_cols: usize,
    pub num_constant_cols: usize,
    pub num_table_cols: usize,
}

impl SetupLayout {
    pub fn from_setup<A: GoodAllocator>(setup: &GpuSetup<A>) -> Self {
        assert!(!setup.variables_hint.is_empty());
        assert!(!setup.constant_columns.is_empty());
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
        assert!(!base_setup.copy_permutation_polys.is_empty());
        assert!(!base_setup.constant_columns.is_empty());
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

impl GenericStorageLayout for SetupLayout {
    type PolyType = SetupPolyType;

    fn num_polys(&self) -> usize {
        self.num_polys()
    }

    fn poly_range(&self, poly_type: Self::PolyType) -> (Range<usize>, Self) {
        let start = match poly_type {
            SetupPolyType::Permutation => 0,
            SetupPolyType::Constant => self.num_permutation_cols,
            SetupPolyType::Table => self.num_permutation_cols + self.num_constant_cols,
        };
        let len = match poly_type {
            SetupPolyType::Permutation => self.num_permutation_cols,
            SetupPolyType::Constant => self.num_constant_cols,
            SetupPolyType::Table => self.num_table_cols,
        };
        let range = start..start + len;
        let layout = Self {
            num_permutation_cols: match poly_type {
                SetupPolyType::Permutation => len,
                _ => 0,
            },
            num_constant_cols: match poly_type {
                SetupPolyType::Constant => len,
                _ => 0,
            },
            num_table_cols: match poly_type {
                SetupPolyType::Table => len,
                _ => 0,
            },
        };
        (range, layout)
    }
}

pub type GenericSetupStorage<P> = GenericStorage<P, SetupLayout>;

impl<P: PolyForm> GenericSetupStorage<P> {
    pub fn as_polynomials(&self) -> SetupPolynomials<P> {
        let SetupLayout {
            num_permutation_cols,
            num_constant_cols,
            num_table_cols,
        } = self.layout;

        let all_polys = self.as_polys();
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
    pub fn fill_from_gpu_setup<A: GoodAllocator>(
        setup: &GpuSetup<A>,
        variable_indexes: &DVec<u32>,
        worker: &Worker,
        mut storage: GenericSetupStorage<Undefined>,
    ) -> CudaResult<Self> {
        let GpuSetup {
            constant_columns,
            lookup_tables_columns,
            variables_hint,
            ..
        } = setup;
        let domain_size = storage.domain_size();
        assert!(domain_size.is_power_of_two());
        for col in constant_columns.iter().chain(lookup_tables_columns.iter()) {
            assert_eq!(col.len(), domain_size);
        }
        let num_copy_permutation_polys = variables_hint.len();
        let size_of_all_copy_perm_polys = num_copy_permutation_polys * domain_size;
        let (copy_permutation_storage, remaining_polys) = storage
            .as_single_slice_mut()
            .split_at_mut(size_of_all_copy_perm_polys);
        assert_eq!(
            remaining_polys.len(),
            (constant_columns.len() + lookup_tables_columns.len()) * domain_size
        );
        materialize_permutation_cols_from_indexes_into(
            copy_permutation_storage,
            variable_indexes,
            num_copy_permutation_polys,
            domain_size,
        )?;
        let mut pending_callbacks = vec![];
        for (dst, src) in remaining_polys
            .chunks_mut(domain_size)
            .zip(constant_columns.iter().chain(lookup_tables_columns.iter()))
        {
            pending_callbacks.push(mem::h2d_buffered(src, dst, domain_size / 2, worker)?);
        }
        get_h2d_stream().synchronize()?;
        drop(pending_callbacks);
        let result = unsafe { storage.transmute() };
        Ok(result)
    }

    pub fn create_from_gpu_setup<A: GoodAllocator>(
        setup: &GpuSetup<A>,
        variable_indexes: &DVec<u32>,
        worker: &Worker,
    ) -> CudaResult<Self> {
        let layout = setup.layout.clone();
        let domain_size = setup.constant_columns[0].len();
        let storage = GenericSetupStorage::allocate(layout, domain_size);
        Self::fill_from_gpu_setup(setup, variable_indexes, worker, storage)
    }

    #[cfg(test)]
    pub fn from_gpu_setup(setup: &GpuSetup<Global>, worker: &Worker) -> CudaResult<Self> {
        let layout = setup.layout.clone();
        let domain_size = setup.constant_columns[0].len();
        assert!(domain_size.is_power_of_two());
        let variable_indexes =
            construct_indexes_from_hint(&setup.variables_hint, domain_size, worker)?;
        let storage = GenericSetupStorage::allocate(layout, domain_size);
        Self::fill_from_gpu_setup(setup, &variable_indexes, worker, storage)
    }

    #[cfg(test)]
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
        let mut storage = GenericSetupStorage::allocate(setup_layout, domain_size);

        for (dst, src) in storage.as_single_slice_mut().chunks_mut(domain_size).zip(
            permutation_cols
                .iter()
                .chain(constant_cols.iter())
                .chain(table_cols.iter()),
        ) {
            mem::h2d(src, dst)?;
        }

        unsafe { Ok(storage.transmute()) }
    }
}

impl GenericSetupStorage<CosetEvaluations> {
    pub(crate) fn barycentric_evaluate<A: GoodAllocator>(
        &self,
        bases: &PrecomputedBasisForBarycentric,
    ) -> CudaResult<Vec<EF, A>> {
        batch_barycentric_evaluate_base(self, bases, self.domain_size(), self.num_polys())
    }
}

pub struct SetupCacheAux {
    tree_cap: Vec<<DefaultTreeHasher as TreeHasher<F>>::Output>,
    pub variable_indexes: DVec<u32>,
    pub witness_indexes: DVec<u32>,
}

pub fn construct_indexes_from_hint<A: GoodAllocator>(
    hint: &Vec<Vec<u32, A>>,
    domain_size: usize,
    worker: &Worker,
) -> CudaResult<DVec<u32>> {
    let stream = get_h2d_stream();
    let mut indexes = dvec![domain_size * hint.len()];
    let mut pending_callbacks = vec![];
    for (h, d) in hint.iter().zip(indexes.chunks_mut(domain_size)) {
        let (d, padding) = d.split_at_mut(h.len());
        let result = mem::h2d_buffered(h, d, domain_size / 2, worker)?;
        pending_callbacks.push(result);
        if !padding.is_empty() {
            set_by_value(padding, PACKED_PLACEHOLDER_BITMASK, stream)?;
        }
    }
    stream.synchronize()?;
    drop(pending_callbacks);
    Ok(indexes)
}

pub type SetupCache = StorageCache<SetupLayout, SetupCacheAux>;

impl SetupCache {
    pub fn new_from_gpu_setup<A: GoodAllocator>(
        strategy: StorageCacheStrategy,
        setup: &GpuSetup<A>,
        fri_lde_degree: usize,
        used_lde_degree: usize,
        worker: &Worker,
    ) -> CudaResult<&'static mut Self> {
        let setup_tree_cap = setup.setup_tree.get_cap();
        if let Some(cache) = _setup_cache_get() {
            if cache.aux.tree_cap.eq(&setup_tree_cap) {
                assert_eq!(cache.fri_lde_degree, fri_lde_degree);
                assert_eq!(cache.used_lde_degree, used_lde_degree);
                assert_eq!(cache.layout, setup.layout);
                println!("reusing setup cache");
                return Ok(cache);
            }
            _setup_cache_reset();
        }
        let layout = setup.layout;
        let domain_size = setup.constant_columns[0].len();
        assert!(domain_size.is_power_of_two());
        let cap_size = setup.setup_tree.cap_size;
        let variable_indexes =
            construct_indexes_from_hint(&setup.variables_hint, domain_size, worker)?;
        let witness_indexes =
            construct_indexes_from_hint(&setup.witnesses_hint, domain_size, worker)?;
        let evaluations =
            GenericSetupStorage::create_from_gpu_setup(setup, &variable_indexes, worker)?;
        let aux = SetupCacheAux {
            tree_cap: setup_tree_cap,
            variable_indexes,
            witness_indexes,
        };
        let mut cache = StorageCache::new_and_initialize(
            strategy,
            layout,
            domain_size,
            fri_lde_degree,
            used_lde_degree,
            cap_size,
            aux,
            evaluations,
        )?;
        let (_, computed_cap) = cache.get_commitment::<DefaultTreeHasher>(cap_size)?;
        if !is_dry_run()? {
            assert_eq!(cache.aux.tree_cap, computed_cap);
        }
        _setup_cache_set(cache);
        Ok(_setup_cache_get().unwrap())
    }
}
