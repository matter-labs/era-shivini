use boojum::{
    cs::{
        implementations::{
            proof::OracleQuery, transcript::Transcript, utils::precompute_twiddles_for_fft,
        },
        oracle::TreeHasher,
    },
    fft::bitreverse_enumeration_inplace,
    field::{FieldExtension, U64Representable},
    worker::Worker,
};

use super::*;

#[derive(Clone)]
pub struct FRIOracle {
    pub nodes: DVec<F>,
    pub num_leafs: usize,
    pub cap_size: usize,
    pub num_elems_per_leaf: usize,
}

impl AsSingleSlice for &FRIOracle {
    fn domain_size(&self) -> usize {
        assert_eq!(2 * NUM_EL_PER_HASH * self.num_leafs, self.nodes.len());
        self.num_leafs
    }

    fn num_polys(&self) -> usize {
        2
    }

    fn as_single_slice(&self) -> &[F] {
        &self.nodes
    }
}

impl FRIOracle {
    pub fn get_tree_cap(&self) -> CudaResult<Vec<[F; 4]>> {
        get_tree_cap_from_nodes(&self.nodes, self.cap_size)
    }
}

impl TreeQuery for FRIOracle {
    fn query<H: TreeHasher<F, Output = [F; 4]>, L: LeafSourceQuery>(
        &self,
        leaf_sources: &L,
        coset_idx: usize,
        lde_degree: usize,
        row_idx: usize,
        domain_size: usize,
    ) -> CudaResult<OracleQuery<F, H>> {
        let Self {
            nodes,
            num_leafs,
            cap_size,
            num_elems_per_leaf,
        } = self;
        let num_elems_per_leaf = *num_elems_per_leaf;
        let cap_size = *cap_size;
        let mut num_leafs = *num_leafs;

        let leaf_elements = leaf_sources.get_leaf_sources(
            coset_idx,
            domain_size,
            lde_degree,
            row_idx,
            num_elems_per_leaf,
        )?;

        let inner_idx = row_idx >> num_elems_per_leaf.trailing_zeros();
        let shift = domain_size.trailing_zeros() - num_elems_per_leaf.trailing_zeros();
        let mut tree_idx = (coset_idx << shift) + inner_idx;

        assert_eq!(nodes.len(), 2 * num_leafs * NUM_EL_PER_HASH);
        let num_polys = 2;
        assert_eq!(leaf_elements.len(), num_polys * num_elems_per_leaf);

        let log_cap = cap_size.trailing_zeros();
        let depth = num_leafs.trailing_zeros();
        let num_layers = depth - log_cap; // ignore cap element(s)

        let mut result = vec![];
        let mut layer_start = 0;

        for _ in 0..num_layers {
            let sbling_idx = tree_idx ^ 1;

            let mut node_hash = [F::ZERO; NUM_EL_PER_HASH];
            for col_idx in 0..NUM_EL_PER_HASH {
                let pos = (layer_start + col_idx * num_leafs) + sbling_idx;
                node_hash[col_idx] = nodes.clone_el_to_host(pos)?;
            }
            result.push(node_hash);

            layer_start += (num_leafs * NUM_EL_PER_HASH);
            num_leafs >>= 1;
            tree_idx >>= 1;
        }

        Ok(OracleQuery {
            leaf_elements,
            proof: result,
        })
    }
}

#[derive(Clone)]
pub struct CodeWord {
    storage: DVec<F>,
    blowup_factor: usize,
    pub(crate) is_base_code_word: bool,
}

impl AsSingleSlice for &CodeWord {
    fn domain_size(&self) -> usize {
        self.length()
    }

    fn num_polys(&self) -> usize {
        2
    }

    fn as_single_slice(&self) -> &[F] {
        &self.storage
    }
}

// for FRI oracles we may store multiple rows in single leaf
// that will allow us to skip to produce oracles for some intermediate layers
impl LeafSourceQuery for CodeWord {
    fn get_leaf_sources(
        &self,
        coset_idx: usize,
        domain_size: usize,
        lde_degree: usize,
        row_idx: usize,
        num_elements_per_leaf: usize,
    ) -> CudaResult<Vec<F>> {
        let mut values = vec![F::ZERO; 2 * num_elements_per_leaf];
        // leaf sources are chunked by num elems per leaf
        // treat first N bits as actual chunk id
        assert_eq!(self.blowup_factor, lde_degree);
        assert!(lde_degree > coset_idx);
        assert_eq!(self.storage.len(), 2 * domain_size * lde_degree);

        assert!(num_elements_per_leaf.is_power_of_two());
        let inner_idx = row_idx >> num_elements_per_leaf.trailing_zeros();
        let inner_start_aligned = inner_idx * num_elements_per_leaf;

        let start = coset_idx * domain_size + inner_start_aligned;
        let end = start + num_elements_per_leaf;
        assert!(end <= self.length());
        assert_eq!(self.storage.len(), 2 * self.length());
        let (c0_storage, c1_storage) = self.storage.split_at(self.length());
        mem::d2h(
            &c0_storage[start..end],
            &mut values[..num_elements_per_leaf],
        )?;
        mem::d2h(
            &c1_storage[start..end],
            &mut values[num_elements_per_leaf..],
        )?;

        Ok(values)
    }
}

pub fn compute_effective_indexes_for_fri_layers(
    fri_holder: &FRICache,
    query_details_for_cosets: &Vec<Vec<u32>>,
) -> CudaResult<Vec<Vec<DVec<u32, SmallStaticDeviceAllocator>>>> {
    let fri_lde_degree = fri_holder.fri_lde_degree;
    let folding_schedule = &fri_holder.folding_schedule;
    assert_eq!(fri_lde_degree, 2);
    let mut query_indexes: Vec<Vec<u32>> = query_details_for_cosets.iter().cloned().collect();
    let mut effective_fri_indexes_for_all = vec![];
    for (layer_idx, (codeword, oracle)) in fri_holder.flatten().into_iter().enumerate() {
        let mut indexes = vec![];
        for coset_idx in 0..fri_lde_degree {
            // codewords store lde values but we need original domain size
            let domain_size = codeword.length() / fri_lde_degree;
            let num_elems_per_leaf = oracle.num_elems_per_leaf;
            let schedule = folding_schedule[layer_idx];
            assert_eq!(num_elems_per_leaf, 1 << schedule);
            assert_eq!(
                fri_lde_degree * domain_size,
                num_elems_per_leaf * oracle.num_leafs
            );
            let query_indexes = &mut query_indexes[coset_idx];
            let mut d_queries = svec!(query_indexes.len());
            let effective_indexes = compute_effective_indexes(
                query_indexes,
                coset_idx,
                domain_size,
                num_elems_per_leaf,
            );
            assert_eq!(effective_indexes.len(), query_indexes.len());
            mem::h2d(&effective_indexes, &mut d_queries)?;
            indexes.push(d_queries);
            query_indexes.iter_mut().for_each(|i| *i >>= schedule);
        }
        effective_fri_indexes_for_all.push(indexes);
    }

    Ok(effective_fri_indexes_for_all)
}

pub(crate) fn compute_effective_indexes(
    indexes: &[u32],
    coset_idx: usize,
    domain_size: usize,
    num_elems_per_leaf: usize,
) -> Vec<u32> {
    assert!(num_elems_per_leaf.is_power_of_two());
    let log2_schedule = num_elems_per_leaf.trailing_zeros();
    indexes
        .iter()
        .map(|&index| (index + (coset_idx * domain_size) as u32) >> log2_schedule)
        .collect()
}

impl CodeWord {
    pub fn new_base_assuming_adjacent(storage: DVec<F>, blowup_factor: usize) -> Self {
        assert_eq!(blowup_factor, 2);
        assert!(storage.len().is_power_of_two());
        Self {
            storage,
            blowup_factor,
            is_base_code_word: true,
        }
    }
    pub fn new_assuming_adjacent(storage: DVec<F>, blowup_factor: usize) -> Self {
        assert_eq!(blowup_factor, 2);
        assert!(storage.len().is_power_of_two());
        Self {
            storage,
            blowup_factor,
            is_base_code_word: false,
        }
    }

    pub fn new(c0: DVec<F>, c1: DVec<F>, blowup_factor: usize) -> CudaResult<Self> {
        assert_eq!(c0.len(), c1.len());
        let len = c0.len();
        assert!(len.is_power_of_two());
        let mut storage = dvec!(2 * len);
        mem::d2d(&c0, &mut storage[..len])?;
        mem::d2d(&c1, &mut storage[len..])?;

        Ok(Self {
            storage,
            blowup_factor,
            is_base_code_word: false,
        })
    }

    pub fn length(&self) -> usize {
        assert_eq!(self.blowup_factor, 2);
        assert!(self.storage.len().is_power_of_two());
        let domain_size = self.storage.len() / 2;
        assert!(domain_size.is_power_of_two());
        domain_size
    }

    pub fn compute_oracle(
        &self,
        cap_size: usize,
        num_elems_per_leaf: usize,
    ) -> CudaResult<FRIOracle> {
        let num_leafs = self.length() / num_elems_per_leaf;
        let mut result = dvec!(2 * NUM_EL_PER_HASH * num_leafs);
        tree::build_tree(
            self.as_single_slice(),
            &mut result,
            self.length(),
            cap_size,
            num_elems_per_leaf,
        )?;

        Ok(FRIOracle {
            nodes: result,
            num_leafs,
            cap_size,
            num_elems_per_leaf,
        })
    }
}

pub struct FoldingOperator {
    coset_inverse: F,
}

impl FoldingOperator {
    pub fn init() -> CudaResult<Self> {
        let coset_inverse = F::multiplicative_generator().inverse().unwrap();

        Ok(Self { coset_inverse })
    }

    pub fn fold_flattened_multiple(
        &mut self,
        codeword: &CodeWord,
        challenges: Vec<DExt>,
    ) -> CudaResult<CodeWord> {
        assert!(codeword.length().is_power_of_two());
        let mut prev = &codeword.storage;

        let mut all_codewords = vec![];
        for challenge in challenges {
            let fold_size = prev.len() >> 1;
            let mut result = dvec!(fold_size);
            arith::fold_flattened(prev, &mut result, self.coset_inverse, &challenge)?;
            all_codewords.push(result);
            prev = &all_codewords.last().unwrap();

            self.coset_inverse.square();
        }

        let last = all_codewords.pop().unwrap();
        Ok(CodeWord::new_assuming_adjacent(
            last,
            codeword.blowup_factor,
        ))
    }
}

pub struct FRICache {
    pub(crate) base_codeword: CodeWord,
    pub(crate) intermediate_codewords: Vec<CodeWord>,
    pub(crate) base_oracle: FRIOracle,
    pub(crate) intermediate_oracles: Vec<FRIOracle>,
    pub(crate) fri_lde_degree: usize,
    pub(crate) folding_schedule: Vec<usize>,
}

impl FRICache {
    pub fn flatten(&self) -> Vec<(&CodeWord, &FRIOracle)> {
        let mut fri_layers = vec![];
        fri_layers.push((&self.base_codeword, &self.base_oracle));
        for l in self
            .intermediate_codewords
            .iter()
            .zip(self.intermediate_oracles.iter())
        {
            fri_layers.push(l)
        }

        fri_layers
    }

    pub fn base_oracle_batch_query<H: TreeHasher<F, Output = [F; 4]>, A: GoodAllocator>(
        &mut self,
        layer_idx: usize,
        d_indexes: &DVec<u32, SmallStaticDeviceAllocator>,
        num_queries: usize,
        domain_size: usize,
        h_all_leaf_elems: &mut Vec<F, A>,
        h_all_proofs: &mut Vec<F, A>,
    ) -> CudaResult<()> {
        assert_eq!(layer_idx, 0);
        assert!(self.base_codeword.is_base_code_word);
        assert_eq!(domain_size, self.base_codeword.length());
        let num_elems_per_leaf = 1 << self.folding_schedule[0];
        assert_eq!(domain_size / num_elems_per_leaf, self.base_oracle.num_leafs);
        batch_query::<H, A>(
            d_indexes,
            num_queries,
            &self.base_codeword,
            2,
            &self.base_oracle,
            self.base_oracle.cap_size,
            domain_size,
            num_elems_per_leaf,
            h_all_leaf_elems,
            h_all_proofs,
        )
    }

    pub fn intermediate_oracle_batch_query<H: TreeHasher<F, Output = [F; 4]>, A: GoodAllocator>(
        &mut self,
        layer_idx: usize,
        d_indexes: &DVec<u32, SmallStaticDeviceAllocator>,
        num_queries: usize,
        domain_size: usize,
        h_all_leaf_elems: &mut Vec<F, A>,
        h_all_proofs: &mut Vec<F, A>,
    ) -> CudaResult<()> {
        assert!(layer_idx < self.folding_schedule.len());
        let current_codeword = &self.intermediate_codewords[layer_idx - 1];
        let current_oracle = &self.intermediate_oracles[layer_idx - 1];
        assert_eq!(current_codeword.is_base_code_word, false);
        assert_eq!(domain_size, current_codeword.length());
        let num_elems_per_leaf = 1 << self.folding_schedule[layer_idx];
        assert_eq!(domain_size / num_elems_per_leaf, current_oracle.num_leafs);
        batch_query::<H, A>(
            d_indexes,
            num_queries,
            current_codeword,
            2,
            current_oracle,
            current_oracle.cap_size,
            domain_size,
            num_elems_per_leaf,
            h_all_leaf_elems,
            h_all_proofs,
        )
    }
}

pub fn compute_fri<T: Transcript<F, CompatibleCap = [F; 4]>, A: GoodAllocator>(
    base_code_word: CodeWord,
    transcript: &mut T,
    folding_schedule: Vec<usize>,
    fri_lde_degree: usize,
    cap_size: usize,
    worker: &Worker,
) -> CudaResult<(FRICache, [Vec<F>; 2])> {
    let full_size = base_code_word.length();
    let degree = full_size / fri_lde_degree;
    let mut final_degree = degree;
    for interpolation_log2 in folding_schedule.iter() {
        let factor = 1usize << interpolation_log2;
        final_degree /= factor;
    }

    assert!(final_degree.is_power_of_two());
    let mut operator = FoldingOperator::init()?;

    let mut intermediate_oracles = vec![];
    let mut intermediate_codewords = vec![];
    let mut prev_code_word = &base_code_word;

    for (layer_idx, log_schedule) in folding_schedule.iter().cloned().enumerate() {
        let num_elems_per_leaf = 1 << log_schedule;
        let num_layers_to_skip = log_schedule;

        assert!(num_elems_per_leaf > 0);
        assert!(num_elems_per_leaf < 1 << 4);

        assert_eq!(prev_code_word.length() % num_elems_per_leaf, 0);
        let current_oracle = prev_code_word.compute_oracle(cap_size, num_elems_per_leaf)?;
        assert_eq!(
            current_oracle.num_leafs,
            prev_code_word.length() / num_elems_per_leaf
        );
        let oracle_cap = current_oracle.get_tree_cap()?;
        intermediate_oracles.push(current_oracle);

        transcript.witness_merkle_tree_cap(&oracle_cap.as_ref());
        let h_challenge = transcript.get_multiple_challenges_fixed::<2>();

        let mut h_challenge = ExtensionField::<F, 2, EXT>::from_coeff_in_base(h_challenge);
        let mut challenge_powers = vec![];

        for _ in 0..num_layers_to_skip {
            challenge_powers.push(h_challenge.clone().into());
            h_challenge.square();
        }

        let folded_code_word =
            operator.fold_flattened_multiple(prev_code_word, challenge_powers)?;
        intermediate_codewords.push(folded_code_word);

        prev_code_word = intermediate_codewords.last().unwrap();
    }

    let first_oracle = intermediate_oracles.drain(0..1).next().unwrap();

    // since last codeword is tiny we can do ifft and asserts on the cpu
    let last_code_word = intermediate_codewords.pop().unwrap();
    let last_code_len = last_code_word.length();
    dbg!(last_code_word.length().trailing_zeros());
    let mut last_code_word_flattened = last_code_word.storage.to_vec_in(A::default())?;
    // FIXME: we can still construct monomials on the device for better stream handling
    synchronize_streams()?;
    assert_eq!(last_code_word_flattened.len(), 2 * last_code_len);
    let mut last_c0 = last_code_word_flattened[..last_code_len].to_vec_in(A::default());
    let mut last_c1 = last_code_word_flattened[last_code_len..].to_vec_in(A::default());

    bitreverse_enumeration_inplace(&mut last_c0);
    bitreverse_enumeration_inplace(&mut last_c1);

    let mut last_coset_inverse = operator.coset_inverse.clone();

    let coset = last_coset_inverse.inverse().unwrap();
    // IFFT our presumable LDE of some low degree poly
    let fft_size = last_c0.len();
    let roots: Vec<F> = precompute_twiddles_for_fft::<_, _, _, true>(fft_size, &worker, &mut ());
    boojum::fft::ifft_natural_to_natural(&mut last_c0, coset, &roots[..fft_size / 2]);
    boojum::fft::ifft_natural_to_natural(&mut last_c1, coset, &roots[..fft_size / 2]);

    assert_eq!(final_degree, fft_size / fri_lde_degree);

    // self-check
    if boojum::config::DEBUG_SATISFIABLE == false {
        for el in last_c0[final_degree..].iter() {
            assert_eq!(*el, F::ZERO);
        }

        for el in last_c1[final_degree..].iter() {
            assert_eq!(*el, F::ZERO);
        }
    }

    // add to the transcript
    transcript.witness_field_elements(&last_c0[..final_degree]);
    transcript.witness_field_elements(&last_c1[..final_degree]);

    // now we should do some PoW and we are good to go

    let monomial_form_0 = last_c0[..(fft_size / fri_lde_degree)].to_vec();
    let monomial_form_1 = last_c1[..(fft_size / fri_lde_degree)].to_vec();
    let fri_holder = FRICache {
        base_codeword: base_code_word,
        base_oracle: first_oracle,
        intermediate_codewords,
        intermediate_oracles,
        folding_schedule,
        fri_lde_degree,
    };
    Ok((fri_holder, [monomial_form_0, monomial_form_1]))
}
