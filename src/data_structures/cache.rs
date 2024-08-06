use boojum::cs::implementations::pow::PoWRunner;
use boojum::cs::implementations::prover::ProofConfig;
use boojum::cs::implementations::transcript::Transcript;
use boojum::cs::implementations::verifier::{VerificationKey, VerificationKeyCircuitGeometry};
use boojum::cs::implementations::witness::WitnessVec;
use boojum::cs::oracle::TreeHasher;
use boojum::worker::Worker;
use era_cudart_sys::CudaError::ErrorMemoryAllocation;
use std::collections::BTreeMap;
use std::ops::Deref;
use std::rc::Rc;

use crate::data_structures::{GenericStorage, GenericStorageLayout};
use crate::oracle::SubTree;
use crate::poly::{CosetEvaluations, LagrangeBasis, MonomialBasis};

use super::*;

#[allow(dead_code)]
#[derive(Clone, Copy, Debug, Hash, Eq, PartialEq, Ord, PartialOrd)]
pub enum StorageCacheStrategy {
    InPlace,
    CacheMonomials,
    CacheMonomialsAndFirstCoset,
    CacheMonomialsAndFriCosets,
    CacheEvaluationsAndMonomials,
    CacheEvaluationsMonomialsAndFirstCoset,
    CacheEvaluationsMonomialsAndFriCosets,
    CacheEvaluationsAndAllCosets,
}

impl StorageCacheStrategy {
    pub fn required_storages_count(&self, fri_lde_degree: usize, used_lde_degree: usize) -> usize {
        match self {
            InPlace => 1,
            CacheMonomials => 1,
            CacheMonomialsAndFirstCoset => 2,
            CacheMonomialsAndFriCosets => 1 + fri_lde_degree,
            CacheEvaluationsAndMonomials => 2,
            CacheEvaluationsMonomialsAndFirstCoset => 3,
            CacheEvaluationsMonomialsAndFriCosets => 2 + fri_lde_degree,
            CacheEvaluationsAndAllCosets => 1 + used_lde_degree,
        }
    }
}

use crate::cs::GpuSetup;
use crate::gpu_proof_config::GpuProofConfig;
use crate::prover::{
    compute_quotient_degree, gpu_prove_from_external_witness_data_with_cache_strategy,
};
use StorageCacheStrategy::*;

pub struct StorageCache<L: GenericStorageLayout, T> {
    pub strategy: StorageCacheStrategy,
    pub layout: L,
    pub domain_size: usize,
    pub fri_lde_degree: usize,
    pub used_lde_degree: usize,
    pub cap_size: usize,
    pub aux: T,
    evaluations: Option<Rc<GenericStorage<LagrangeBasis, L>>>,
    monomials: Option<Rc<GenericStorage<MonomialBasis, L>>>,
    coset_evaluations: BTreeMap<usize, Rc<GenericStorage<CosetEvaluations, L>>>,
    subtrees_and_caps: BTreeMap<usize, (SubTree, Vec<[F; 4]>)>,
    uninitialized_storages: Vec<GenericStorage<Undefined, L>>,
    uninitialized_subtree_nodes: Vec<DVec<F>>,
}

impl<L: GenericStorageLayout, T> StorageCache<L, T> {
    fn required_storages_count(&self) -> usize {
        self.strategy
            .required_storages_count(self.fri_lde_degree, self.used_lde_degree)
    }

    fn allocate(
        strategy: StorageCacheStrategy,
        layout: L,
        domain_size: usize,
        fri_lde_degree: usize,
        used_lde_degree: usize,
        cap_size: usize,
        aux: T,
        will_own_evaluations: bool,
    ) -> Self {
        let mut storages_count = strategy.required_storages_count(fri_lde_degree, used_lde_degree);
        if will_own_evaluations {
            storages_count -= 1;
        }
        let unused_storages = (0..storages_count)
            .map(|_| GenericStorage::allocate(layout, domain_size))
            .collect();
        let nodes_size = 2 * NUM_EL_PER_HASH * domain_size;
        let unused_subtree_nodes = (0..fri_lde_degree).map(|_| dvec!(nodes_size)).collect();
        Self {
            strategy,
            layout,
            domain_size,
            fri_lde_degree,
            used_lde_degree,
            cap_size,
            aux,
            evaluations: None,
            monomials: None,
            coset_evaluations: BTreeMap::new(),
            subtrees_and_caps: BTreeMap::new(),
            uninitialized_storages: unused_storages,
            uninitialized_subtree_nodes: unused_subtree_nodes,
        }
    }

    pub fn new(
        strategy: StorageCacheStrategy,
        layout: L,
        domain_size: usize,
        fri_lde_degree: usize,
        used_lde_degree: usize,
        cap_size: usize,
        aux: T,
    ) -> Self {
        Self::allocate(
            strategy,
            layout,
            domain_size,
            fri_lde_degree,
            used_lde_degree,
            cap_size,
            aux,
            false,
        )
    }

    pub fn new_and_initialize(
        strategy: StorageCacheStrategy,
        layout: L,
        domain_size: usize,
        fri_lde_degree: usize,
        used_lde_degree: usize,
        cap_size: usize,
        aux: T,
        evaluations: GenericStorage<LagrangeBasis, L>,
    ) -> CudaResult<Self> {
        let mut cache = Self::allocate(
            strategy,
            layout,
            domain_size,
            fri_lde_degree,
            used_lde_degree,
            cap_size,
            aux,
            true,
        );
        cache.initialize(Rc::new(evaluations))?;
        Ok(cache)
    }

    fn pop_storage(&mut self) -> GenericStorage<Undefined, L> {
        self.uninitialized_storages.pop().unwrap()
    }

    pub fn get_temp_storage(&self) -> GenericStorage<Undefined, L> {
        GenericStorage::allocate(self.layout, self.domain_size)
    }

    pub fn get_evaluations_storage(&mut self) -> GenericStorage<Undefined, L> {
        assert_eq!(
            self.uninitialized_storages.len(),
            self.required_storages_count()
        );
        self.pop_storage()
    }

    pub fn initialize(
        &mut self,
        evaluations: Rc<GenericStorage<LagrangeBasis, L>>,
    ) -> CudaResult<()> {
        let can_owns_evaluations = Rc::strong_count(&evaluations) == 1;
        let must_own_evaluations = match self.strategy {
            InPlace | CacheMonomials | CacheMonomialsAndFirstCoset | CacheMonomialsAndFriCosets => {
                self.required_storages_count() > self.uninitialized_storages.len()
            }
            CacheEvaluationsAndMonomials
            | CacheEvaluationsMonomialsAndFirstCoset
            | CacheEvaluationsMonomialsAndFriCosets
            | CacheEvaluationsAndAllCosets => false,
        };
        assert!(can_owns_evaluations || !must_own_evaluations);
        let monomials = match self.strategy {
            InPlace | CacheMonomials | CacheMonomialsAndFirstCoset | CacheMonomialsAndFriCosets
                if must_own_evaluations =>
            {
                Rc::into_inner(evaluations).unwrap().into_monomials()?
            }
            InPlace | CacheMonomials | CacheMonomialsAndFirstCoset | CacheMonomialsAndFriCosets => {
                evaluations.fill_monomials(self.pop_storage())?
            }
            CacheEvaluationsAndMonomials
            | CacheEvaluationsMonomialsAndFirstCoset
            | CacheEvaluationsMonomialsAndFriCosets
            | CacheEvaluationsAndAllCosets => {
                let monomials = evaluations.fill_monomials(self.pop_storage())?;
                self.evaluations = Some(evaluations);
                monomials
            }
        };
        let mut monomials = Some(monomials);
        let cosets_count = match self.strategy {
            CacheEvaluationsAndAllCosets => self.used_lde_degree,
            _ => self.fri_lde_degree,
        };
        let coset_cap_size = coset_cap_size(self.cap_size, self.fri_lde_degree);
        for coset_idx in 0..cosets_count {
            let coset = match self.strategy {
                InPlace => monomials
                    .take()
                    .unwrap()
                    .into_coset_evaluations(coset_idx, self.used_lde_degree)?,
                CacheMonomials | CacheEvaluationsAndMonomials => monomials
                    .as_ref()
                    .unwrap()
                    .create_coset_evaluations(coset_idx, self.used_lde_degree)?,
                CacheMonomialsAndFirstCoset | CacheEvaluationsMonomialsAndFirstCoset
                    if coset_idx == 0 =>
                {
                    monomials.as_ref().unwrap().fill_coset_evaluations(
                        coset_idx,
                        self.used_lde_degree,
                        self.pop_storage(),
                    )?
                }
                CacheMonomialsAndFriCosets | CacheEvaluationsMonomialsAndFriCosets
                    if coset_idx < self.fri_lde_degree =>
                {
                    monomials.as_ref().unwrap().fill_coset_evaluations(
                        coset_idx,
                        self.used_lde_degree,
                        self.pop_storage(),
                    )?
                }
                CacheMonomialsAndFirstCoset
                | CacheMonomialsAndFriCosets
                | CacheEvaluationsMonomialsAndFirstCoset
                | CacheEvaluationsMonomialsAndFriCosets => monomials
                    .as_ref()
                    .unwrap()
                    .create_coset_evaluations(coset_idx, self.used_lde_degree)?,
                CacheEvaluationsAndAllCosets => {
                    if coset_idx + 1 == self.used_lde_degree {
                        monomials
                            .take()
                            .unwrap()
                            .into_coset_evaluations(coset_idx, self.used_lde_degree)?
                    } else {
                        monomials.as_ref().unwrap().fill_coset_evaluations(
                            coset_idx,
                            self.used_lde_degree,
                            self.pop_storage(),
                        )?
                    }
                }
            };
            if coset_idx < self.fri_lde_degree {
                let coset_nodes = self.uninitialized_subtree_nodes.pop().unwrap();
                let (subtree, subtree_cap) =
                    coset.build_subtree_for_coset(coset_cap_size, coset_idx, coset_nodes)?;
                self.subtrees_and_caps
                    .insert(coset_idx, (subtree, subtree_cap));
            }

            match self.strategy {
                InPlace => {
                    monomials = Some(coset.into_monomials(coset_idx, self.used_lde_degree)?);
                }
                CacheMonomials | CacheEvaluationsAndMonomials => {}
                CacheMonomialsAndFirstCoset | CacheEvaluationsMonomialsAndFirstCoset
                    if coset_idx == 0 =>
                {
                    self.coset_evaluations.insert(coset_idx, Rc::new(coset));
                }
                CacheMonomialsAndFriCosets | CacheEvaluationsMonomialsAndFriCosets
                    if coset_idx < self.fri_lde_degree =>
                {
                    self.coset_evaluations.insert(coset_idx, Rc::new(coset));
                }
                CacheMonomialsAndFirstCoset
                | CacheMonomialsAndFriCosets
                | CacheEvaluationsMonomialsAndFirstCoset
                | CacheEvaluationsMonomialsAndFriCosets => {}
                CacheEvaluationsAndAllCosets => {
                    self.coset_evaluations.insert(coset_idx, Rc::new(coset));
                }
            };
        }
        match self.strategy {
            CacheEvaluationsAndAllCosets => {}
            _ => {
                self.monomials = Some(Rc::new(monomials.take().unwrap()));
            }
        };
        Ok(())
    }

    pub fn num_polys(&self) -> usize {
        self.layout.num_polys()
    }

    pub fn get_evaluations(&mut self) -> CudaResult<Rc<GenericStorage<LagrangeBasis, L>>> {
        let result = match self.strategy {
            InPlace => {
                if let Some(evaluations) = &self.evaluations {
                    evaluations.clone()
                } else {
                    let monomials = self.get_monomials()?;
                    drop(self.monomials.take());
                    let monomials = Rc::into_inner(monomials).unwrap();
                    let evaluations = Rc::new(monomials.into_evaluations()?);
                    self.evaluations = Some(evaluations.clone());
                    evaluations
                }
            }
            CacheMonomials | CacheMonomialsAndFirstCoset | CacheMonomialsAndFriCosets => {
                Rc::new(self.monomials.as_ref().unwrap().create_evaluations()?)
            }
            _ => self.evaluations.as_ref().unwrap().clone(),
        };
        Ok(result)
    }

    pub fn get_monomials(&mut self) -> CudaResult<Rc<GenericStorage<MonomialBasis, L>>> {
        let result = match self.strategy {
            InPlace => {
                if let Some(monomials) = &self.monomials {
                    monomials.clone()
                } else {
                    let monomials = if let Some(evaluations) = self.evaluations.take() {
                        let evaluations = Rc::into_inner(evaluations).unwrap();
                        evaluations.into_monomials()?
                    } else {
                        let (coset_idx, coset) = self.coset_evaluations.pop_first().unwrap();
                        assert!(self.coset_evaluations.is_empty());
                        let coset = Rc::into_inner(coset).unwrap();
                        coset.into_monomials(coset_idx, self.used_lde_degree)?
                    };
                    let monomials = Rc::new(monomials);
                    self.monomials = Some(monomials.clone());
                    monomials
                }
            }
            CacheEvaluationsAndAllCosets => {
                let (coset_idx, coset) = self.coset_evaluations.first_key_value().unwrap();
                let monomials = coset.create_monomials(*coset_idx, self.used_lde_degree)?;
                Rc::new(monomials)
            }
            _ => self.monomials.as_ref().unwrap().clone(),
        };
        Ok(result)
    }

    pub fn get_coset_evaluations(
        &mut self,
        coset_idx: usize,
    ) -> CudaResult<Rc<GenericStorage<CosetEvaluations, L>>> {
        assert!(coset_idx < self.used_lde_degree);
        let result = match self.strategy {
            InPlace => {
                if let Some(coset) = self.coset_evaluations.get(&coset_idx) {
                    coset.clone()
                } else {
                    let monomials = self.get_monomials()?;
                    drop(self.monomials.take());
                    let monomials = Rc::into_inner(monomials).unwrap();
                    let coset =
                        Rc::new(monomials.into_coset_evaluations(coset_idx, self.used_lde_degree)?);
                    self.coset_evaluations.insert(coset_idx, coset.clone());
                    coset
                }
            }
            CacheMonomialsAndFirstCoset | CacheEvaluationsMonomialsAndFirstCoset
                if coset_idx == 0 =>
            {
                self.coset_evaluations.get(&coset_idx).unwrap().clone()
            }
            CacheMonomialsAndFriCosets | CacheEvaluationsMonomialsAndFriCosets
                if coset_idx < self.fri_lde_degree =>
            {
                self.coset_evaluations.get(&coset_idx).unwrap().clone()
            }
            CacheMonomials
            | CacheMonomialsAndFirstCoset
            | CacheMonomialsAndFriCosets
            | CacheEvaluationsAndMonomials
            | CacheEvaluationsMonomialsAndFirstCoset
            | CacheEvaluationsMonomialsAndFriCosets => Rc::new(
                self.monomials
                    .as_ref()
                    .unwrap()
                    .create_coset_evaluations(coset_idx, self.used_lde_degree)?,
            ),
            CacheEvaluationsAndAllCosets => self.coset_evaluations.get(&coset_idx).unwrap().clone(),
        };
        Ok(result)
    }

    #[allow(dead_code)]
    pub fn get_coset_evaluations_subset(
        &mut self,
        coset_idx: usize,
        subset: L::PolyType,
    ) -> CudaResult<Rc<GenericStorage<CosetEvaluations, L>>> {
        assert!(coset_idx < self.used_lde_degree);
        let result = match self.strategy {
            InPlace | CacheEvaluationsAndAllCosets => self.get_coset_evaluations(coset_idx)?,
            CacheMonomialsAndFirstCoset | CacheEvaluationsMonomialsAndFirstCoset
                if coset_idx == 0 =>
            {
                self.coset_evaluations.get(&coset_idx).unwrap().clone()
            }
            CacheMonomialsAndFriCosets | CacheEvaluationsMonomialsAndFriCosets
                if coset_idx < self.fri_lde_degree =>
            {
                self.coset_evaluations.get(&coset_idx).unwrap().clone()
            }
            CacheMonomials
            | CacheMonomialsAndFirstCoset
            | CacheMonomialsAndFriCosets
            | CacheEvaluationsAndMonomials
            | CacheEvaluationsMonomialsAndFirstCoset
            | CacheEvaluationsMonomialsAndFriCosets => Rc::new(
                self.monomials
                    .as_ref()
                    .unwrap()
                    .create_coset_evaluations_subset(coset_idx, self.used_lde_degree, subset)?,
            ),
        };
        Ok(result)
    }

    pub fn get_commitment<H: TreeHasher<F, Output = [F; 4]>>(
        &mut self,
        cap_size: usize,
    ) -> CudaResult<(Vec<SubTree>, Vec<[F; 4]>)> {
        let fri_lde_degree = self.fri_lde_degree;
        let coset_cap_size = coset_cap_size(cap_size, self.fri_lde_degree);
        let mut subtrees = vec![];
        let mut subtree_caps = vec![];
        for coset_idx in 0..fri_lde_degree {
            let (subtree, subtree_cap) = &self.subtrees_and_caps.get(&coset_idx).unwrap();
            assert_eq!(subtree.cap_size, coset_cap_size);
            let subtree = SubTree::new(
                subtree.nodes.clone(),
                subtree.num_leafs,
                subtree.cap_size,
                coset_idx,
            );
            subtrees.push(subtree);
            subtree_caps.push(subtree_cap.clone());
        }
        let tree_cap = subtree_caps.compute_cap::<H>(&mut subtrees, cap_size)?;
        Ok((subtrees, tree_cap))
    }

    pub fn batch_query_for_coset<H: TreeHasher<F, Output = [F; 4]>, A: GoodAllocator>(
        &mut self,
        coset_idx: usize,
        indexes: &DVec<u32, SmallStaticDeviceAllocator>,
        num_queries: usize,
        domain_size: usize,
        h_all_leaf_elems: &mut Vec<F, A>,
        h_all_proofs: &mut Vec<F, A>,
    ) -> CudaResult<()> {
        let leaf_sources = self.get_coset_evaluations(coset_idx)?;
        let (oracle_data, _) = self.subtrees_and_caps.get(&coset_idx).unwrap();
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
}

#[derive(Clone, Copy, Debug, Hash, Eq, PartialEq, Ord, PartialOrd)]
pub(crate) struct CacheStrategy {
    pub(crate) setup: StorageCacheStrategy,
    pub(crate) trace: StorageCacheStrategy,
    pub(crate) arguments: StorageCacheStrategy,
}

impl CacheStrategy {
    pub(crate) fn get<
        TR: Transcript<F, CompatibleCap = [F; 4]>,
        H: TreeHasher<F, Output = TR::CompatibleCap>,
        POW: PoWRunner,
        A: GoodAllocator,
    >(
        config: &GpuProofConfig,
        external_witness_data: &WitnessVec<F>,
        proof_config: ProofConfig,
        setup: &GpuSetup<A>,
        vk: &VerificationKey<F, H>,
        transcript_params: TR::TransciptParameters,
        worker: &Worker,
    ) -> CudaResult<Self> {
        let cap = &vk.setup_merkle_tree_cap;
        if let Some(strategy) = _strategy_cache_get().get(cap) {
            println!("reusing cache strategy");
            Ok(*strategy)
        } else {
            let strategies =
                Self::get_strategy_candidates(config, &proof_config, setup, &vk.fixed_parameters);
            for (_, strategy) in strategies.iter().copied() {
                _setup_cache_reset();
                dry_run_start();
                let result =
                    gpu_prove_from_external_witness_data_with_cache_strategy::<TR, H, POW, A>(
                        config,
                        external_witness_data,
                        proof_config.clone(),
                        setup,
                        vk,
                        transcript_params.clone(),
                        worker,
                        strategy,
                    );
                _setup_cache_reset();
                let result = result.and(dry_run_stop());
                match result {
                    Ok(_) => {
                        println!("determined cache strategy: {:?}", strategy);
                        _strategy_cache_get().insert(cap.clone(), strategy);
                        return Ok(strategy);
                    }
                    Err(ErrorMemoryAllocation) => {
                        continue;
                    }
                    Err(e) => return Err(e),
                }
            }
            Err(ErrorMemoryAllocation)
        }
    }

    pub(crate) fn get_strategy_candidates<A: GoodAllocator>(
        config: &GpuProofConfig,
        proof_config: &ProofConfig,
        setup: &GpuSetup<A>,
        geometry: &VerificationKeyCircuitGeometry,
    ) -> Vec<((usize, usize), CacheStrategy)> {
        let fri_lde_degree = proof_config.fri_lde_factor;
        let quotient_degree = compute_quotient_degree(&config, &setup.selectors_placement);
        let used_lde_degree = usize::max(quotient_degree, fri_lde_degree);
        let setup_layout = setup.layout;
        let domain_size = geometry.domain_size as usize;
        let lookup_parameters = geometry.lookup_parameters;
        let total_tables_len = geometry.total_tables_len as usize;
        let num_multiplicity_cols =
            lookup_parameters.num_multipicities_polys(total_tables_len, domain_size);
        let trace_layout = TraceLayout {
            num_variable_cols: setup.variables_hint.len(),
            num_witness_cols: setup.witnesses_hint.len(),
            num_multiplicity_cols,
        };
        let arguments_layout = ArgumentsLayout::from_trace_layout_and_lookup_params(
            trace_layout,
            quotient_degree,
            geometry.lookup_parameters,
        );
        let setup_num_polys = setup_layout.num_polys();
        let trace_num_polys = trace_layout.num_polys();
        let arguments_num_polys = arguments_layout.num_polys();
        let setup_strategies = [
            InPlace,
            CacheMonomials,
            CacheMonomialsAndFirstCoset,
            CacheMonomialsAndFriCosets,
            CacheEvaluationsAndMonomials,
            CacheEvaluationsMonomialsAndFirstCoset,
            CacheEvaluationsMonomialsAndFriCosets,
            CacheEvaluationsAndAllCosets,
        ];
        let trace_and_arguments_strategies = [
            InPlace,
            CacheMonomials,
            CacheMonomialsAndFirstCoset,
            CacheMonomialsAndFriCosets,
        ];
        let mut strategies = Vec::new();
        for setup_strategy in setup_strategies.iter().copied() {
            for trace_strategy in trace_and_arguments_strategies.iter().copied() {
                for arguments_strategy in trace_and_arguments_strategies.iter().copied() {
                    let strategy = Self {
                        setup: setup_strategy,
                        trace: trace_strategy,
                        arguments: arguments_strategy,
                    };
                    let setup_cost =
                        strategy.get_setup_cost(fri_lde_degree, used_lde_degree) * setup_num_polys;
                    let proof_cost_setup = strategy
                        .get_proof_cost_setup(fri_lde_degree, used_lde_degree)
                        * setup_num_polys;
                    let proof_cost_trace = strategy
                        .get_proof_cost_trace(fri_lde_degree, used_lde_degree)
                        * trace_num_polys;
                    let proof_cost_arguments = strategy
                        .get_proof_cost_arguments(fri_lde_degree, used_lde_degree)
                        * arguments_num_polys;
                    let proof_cost = proof_cost_setup + proof_cost_trace + proof_cost_arguments;
                    strategies.push(((proof_cost, setup_cost), strategy));
                }
            }
        }
        strategies.sort_by_key(|x| x.0);
        strategies
    }

    fn get_setup_cost(&self, fri_lde_degree: usize, used_lde_degree: usize) -> usize {
        let f = fri_lde_degree;
        let u = used_lde_degree;
        match self.setup {
            InPlace => 1 + 2 * f,
            CacheMonomials
            | CacheMonomialsAndFirstCoset
            | CacheMonomialsAndFriCosets
            | CacheEvaluationsAndMonomials
            | CacheEvaluationsMonomialsAndFirstCoset
            | CacheEvaluationsMonomialsAndFriCosets => 1 + f,
            CacheEvaluationsAndAllCosets => 1 + u,
        }
    }

    fn get_proof_cost_setup(&self, fri_lde_degree: usize, used_lde_degree: usize) -> usize {
        let f = fri_lde_degree;
        let u = used_lde_degree;
        match self.setup {
            InPlace => 2 + 2 * u + 2 + 2 * (f - 1) + 2 * f,
            CacheMonomials => 1 + u + 1 + f + f,
            CacheMonomialsAndFirstCoset => 1 + u - 1 + f - 1 + f - 1,
            CacheMonomialsAndFriCosets => 1 + u - f,
            CacheEvaluationsAndMonomials => u + 1 + f + f,
            CacheEvaluationsMonomialsAndFirstCoset => u - 1 + f - 1 + f - 1,
            CacheEvaluationsMonomialsAndFriCosets => u - f,
            CacheEvaluationsAndAllCosets => 0,
        }
    }

    fn get_proof_cost_trace(&self, fri_lde_degree: usize, used_lde_degree: usize) -> usize {
        let f = fri_lde_degree;
        let u = used_lde_degree;
        match self.trace {
            InPlace => 1 + 2 * f + 1 + 2 * u + 2 + 2 * (f - 1) + 2 * f,
            CacheMonomials => 1 + f + u + 1 + f + f,
            CacheMonomialsAndFirstCoset => 1 + f + u - 1 + f - 1 + f - 1,
            CacheMonomialsAndFriCosets => 1 + f + u - f,
            CacheEvaluationsAndMonomials => 1 + f + u + 1 + f + f,
            CacheEvaluationsMonomialsAndFirstCoset => 1 + f + u - 1 + f - 1 + f - 1,
            CacheEvaluationsMonomialsAndFriCosets => 1 + f + u - f,
            CacheEvaluationsAndAllCosets => 1 + u,
        }
    }

    fn get_proof_cost_arguments(&self, fri_lde_degree: usize, used_lde_degree: usize) -> usize {
        let f = fri_lde_degree;
        let u = used_lde_degree;
        match self.arguments {
            InPlace => 1 + 2 * f + 2 * u - 1 + 1 + 1 + 2 * (f - 1) + 2 * f,
            CacheMonomials => 1 + f + u + 1 + f + f,
            CacheMonomialsAndFirstCoset => 1 + f + u - 1 + f - 1 + f - 1,
            CacheMonomialsAndFriCosets => 1 + f + u - f,
            CacheEvaluationsAndMonomials => 1 + f + u + 1 + f + f,
            CacheEvaluationsMonomialsAndFirstCoset => 1 + f + u - 1 + f - 1 + f - 1,
            CacheEvaluationsMonomialsAndFriCosets => 1 + f + u - f,
            CacheEvaluationsAndAllCosets => 1 + u,
        }
    }
}
