use boojum::{
    cs::{
        implementations::{
            proof::OracleQuery, transcript::Transcript, utils::precompute_twiddles_for_fft,
        },
        oracle::TreeHasher,
    },
    fft::bitreverse_enumeration_inplace,
};

use super::*;

#[derive(Clone)]
pub struct FRIOracle {
    pub nodes: DVec<F>,
    pub num_leafs: usize,
    pub cap_size: usize,
    pub num_elems_per_leaf: usize,
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

            layer_start += num_leafs * NUM_EL_PER_HASH;
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
    c0: DVec<F>,
    c1: DVec<F>,
    blowup_factor: usize,
    #[allow(dead_code)]
    is_base_code_word: bool,
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
        assert_eq!(self.c0.len(), domain_size * lde_degree);
        assert_eq!(self.c1.len(), domain_size * lde_degree);

        let inner_idx = row_idx >> num_elements_per_leaf.trailing_zeros();
        let inner_start_aligned = inner_idx * num_elements_per_leaf;

        let start = coset_idx * domain_size + inner_start_aligned;
        let end = start + num_elements_per_leaf;
        assert!(end <= self.length());

        let c0_values = self.c0.clone_range_to_host(start..end)?;
        values[..num_elements_per_leaf].copy_from_slice(&c0_values);

        let c1_values = self.c1.clone_range_to_host(start..end)?;
        values[num_elements_per_leaf..].copy_from_slice(&c1_values);

        Ok(values)
    }
}

impl CodeWord {
    #[allow(dead_code)]
    pub fn new_base(c0: DVec<F>, c1: DVec<F>, blowup_factor: usize) -> Self {
        assert_eq!(c0.len(), c1.len());
        Self {
            c0,
            c1,
            blowup_factor,
            is_base_code_word: true,
        }
    }

    pub fn new(c0: DVec<F>, c1: DVec<F>, blowup_factor: usize) -> Self {
        assert_eq!(c0.len(), c1.len());
        Self {
            c0,
            c1,
            blowup_factor,
            is_base_code_word: false,
        }
    }

    pub fn length(&self) -> usize {
        self.c0.len()
    }

    fn into_owned_leaf_sources(&self) -> CudaResult<DVec<F>> {
        let len = self.length();
        let mut sources = dvec!(len * 2);
        mem::d2d(&self.c0[..], &mut sources[..len])?;
        mem::d2d(&self.c1[..], &mut sources[len..])?;

        Ok(sources)
    }

    pub fn compute_oracle(
        &self,
        cap_size: usize,
        num_elems_per_leaf: usize,
    ) -> CudaResult<FRIOracle> {
        let num_leafs = self.c0.len() / num_elems_per_leaf;
        let result_len = 2 * NUM_EL_PER_HASH * num_leafs;
        let mut result = dvec!(result_len);
        tree::build_tree(
            &self.into_owned_leaf_sources()?,
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

    pub fn fold_multiple(
        &mut self,
        code_word: &CodeWord,
        challenges: Vec<DExt>,
    ) -> CudaResult<CodeWord> {
        let mut prev_c0 = code_word.c0.clone();
        let mut prev_c1 = code_word.c1.clone();
        assert_eq!(prev_c0.len(), prev_c1.len());
        assert!(prev_c0.len().is_power_of_two());

        for challenge in challenges {
            let coset_inverse: DF = self.coset_inverse.into();
            let fold_size = prev_c0.len() >> 1;
            // FIXME
            let mut folded_c0_values = dvec!(fold_size);
            let mut folded_c1_values = dvec!(fold_size);
            arith::fold(
                &prev_c0,
                &prev_c1,
                &mut folded_c0_values,
                &mut folded_c1_values,
                coset_inverse.clone(),
                challenge.clone(),
            )?;

            prev_c0 = folded_c0_values;
            prev_c1 = folded_c1_values;

            self.coset_inverse.square();
        }

        Ok(CodeWord::new(prev_c0, prev_c1, code_word.blowup_factor))
    }
}

pub fn compute_fri<T: Transcript<F, CompatibleCap = [F; 4]>>(
    base_code_word: CodeWord,
    transcript: &mut T,
    folding_schedule: Vec<usize>,
    lde_degree: usize,
    cap_size: usize,
) -> CudaResult<(FRIOracle, Vec<CodeWord>, Vec<FRIOracle>, [Vec<F>; 2])> {
    let full_size = base_code_word.c0.len();
    let degree = full_size / lde_degree;
    let mut final_degree = degree;
    for interpolation_log2 in folding_schedule.iter() {
        let factor = 1usize << interpolation_log2;
        final_degree /= factor;
    }

    assert!(final_degree > 0);
    let mut operator = FoldingOperator::init()?;

    let mut intermediate_oracles = vec![];
    let mut intermediate_code_words = vec![];
    let mut prev_code_word = &base_code_word;

    for (_layer_idx, log_schedule) in folding_schedule.into_iter().enumerate() {
        let num_elems_per_leaf = 1 << log_schedule;
        let num_layers_to_skip = log_schedule;

        assert!(num_elems_per_leaf > 0);
        assert!(num_elems_per_leaf < 1 << 4);

        assert_eq!(prev_code_word.c0.len() % num_elems_per_leaf, 0);
        let current_oracle = prev_code_word.compute_oracle(cap_size, num_elems_per_leaf)?;
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

        let folded_code_word = operator.fold_multiple(prev_code_word, challenge_powers)?;
        intermediate_code_words.push(folded_code_word);

        prev_code_word = intermediate_code_words.last().unwrap();
    }

    let first_oracle = intermediate_oracles.drain(0..1).next().unwrap();
    let _first_code_word = base_code_word;

    let last_code_word = intermediate_code_words.last().unwrap().clone();
    // since last codeword is tiny we can do ifft and asserts on the cpu
    let mut last_c0 = last_code_word.c0.to_vec()?;
    let mut last_c1 = last_code_word.c1.to_vec()?;

    bitreverse_enumeration_inplace(&mut last_c0);
    bitreverse_enumeration_inplace(&mut last_c1);

    let last_coset_inverse = operator.coset_inverse.clone();

    let coset = last_coset_inverse.inverse().unwrap();
    // IFFT our presumable LDE of some low degree poly
    let fft_size = last_c0.len();
    use boojum::worker::Worker;
    let worker = Worker::new();
    let roots: Vec<F> = precompute_twiddles_for_fft::<_, _, _, true>(fft_size, &worker, &mut ());
    boojum::fft::ifft_natural_to_natural(&mut last_c0, coset, &roots[..fft_size / 2]);
    boojum::fft::ifft_natural_to_natural(&mut last_c1, coset, &roots[..fft_size / 2]);

    assert_eq!(final_degree, fft_size / lde_degree);

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

    let monomial_form_0 = last_c0[..(fft_size / lde_degree)].to_vec();
    let monomial_form_1 = last_c1[..(fft_size / lde_degree)].to_vec();

    Ok((
        first_oracle,
        intermediate_code_words,
        intermediate_oracles,
        [monomial_form_0, monomial_form_1],
    ))
}
