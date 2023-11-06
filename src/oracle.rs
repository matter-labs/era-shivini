use std::rc::Rc;

use crate::primitives::ntt::{batch_bitreverse, batch_ntt, coset_fft_into};
use boojum::{
    cs::{
        implementations::{proof::OracleQuery, transcript::Transcript},
        oracle::TreeHasher,
    },
    field::U64Representable,
};

use super::*;

pub const NUM_EL_PER_HASH: usize = 4;

pub trait LeafSourceQuery {
    fn get_leaf_sources(
        &self,
        coset_idx: usize,
        domain_size: usize,
        lde_factor: usize,
        row_idx: usize,
        num_elems_per_leaf: usize,
    ) -> CudaResult<Vec<F>>;
}

pub trait TreeQuery {
    fn query<H: TreeHasher<F, Output = [F; 4]>, L: LeafSourceQuery>(
        &self,
        leaf_sources: &L,
        coset_idx: usize,
        lde_degree: usize,
        row_idx: usize,
        domain_size: usize,
    ) -> CudaResult<OracleQuery<F, H>>;
}

pub trait ParentTree {
    fn compute_cap<H: TreeHasher<F, Output = [F; 4]>>(
        &mut self,
        subtrees: &mut [SubTree],
        cap_size: usize,
    ) -> CudaResult<Vec<[F; 4]>>;
}

impl ParentTree for Vec<Vec<[F; 4]>> {
    fn compute_cap<H: TreeHasher<F, Output = [F; 4]>>(
        &mut self,
        subtrees: &mut [SubTree],
        cap_size: usize,
    ) -> CudaResult<Vec<[F; 4]>> {
        let mut flattened_caps = vec![];
        for cap in self.iter() {
            flattened_caps.extend_from_slice(cap);
        }
        if flattened_caps.len() == cap_size {
            for subtree in subtrees.iter_mut() {
                subtree.parent = Rc::new(vec![flattened_caps.clone()]);
            }
            return Ok(flattened_caps);
        }

        let num_subtrees = self.len();
        assert!(num_subtrees.is_power_of_two());
        assert_eq!(subtrees.len(), num_subtrees);
        assert!(cap_size.is_power_of_two());
        let num_layers = cap_size.trailing_zeros() - num_subtrees.trailing_zeros();

        let mut prev_layer = &flattened_caps;
        // prepare parent nodes then query them in FRI
        let mut all_layers = vec![];
        all_layers.push(flattened_caps.to_vec());
        for layer_idx in 0..num_layers {
            let mut layer_hashes = vec![];
            for [left, right] in prev_layer.array_chunks::<2>() {
                let node_hash = H::hash_into_node(left, right, layer_idx as usize);
                layer_hashes.push(node_hash);
            }
            all_layers.push(layer_hashes);
            prev_layer = all_layers.last().expect("last");
        }
        assert_eq!(prev_layer.len(), cap_size);

        // link subtrees with parent
        let all_layers = Rc::new(all_layers.clone());
        for subtree in subtrees.iter_mut() {
            subtree.parent = all_layers.clone();
        }
        Ok(prev_layer.to_vec())
    }
}

// We can use move trees back to the cpu while proof is generated.
// This will allow us to leave gpu earlier and do fri queries on the cpu
pub struct SubTree {
    pub parent: Rc<Vec<Vec<[F; 4]>>>,
    pub nodes: DVec<F>,
    pub num_leafs: usize,
    pub cap_size: usize,
    pub tree_idx: usize,
}

impl SubTree {
    pub fn new(nodes: DVec<F>, num_leafs: usize, cap_size: usize, coset_idx: usize) -> Self {
        assert!(num_leafs.is_power_of_two());
        assert!(cap_size.is_power_of_two());
        assert_eq!(nodes.len(), 2 * num_leafs * NUM_EL_PER_HASH);
        SubTree {
            parent: Rc::new(vec![]), // TODO each subtree have access to the parent for querying
            nodes,
            num_leafs,
            cap_size,
            tree_idx: coset_idx,
        }
    }
}

impl AsSingleSlice for SubTree {
    fn domain_size(&self) -> usize {
        self.num_leafs
    }

    fn num_polys(&self) -> usize {
        unreachable!()
    }

    fn as_single_slice(&self) -> &[F] {
        &self.nodes
    }
}

impl AsSingleSlice for &SubTree {
    fn domain_size(&self) -> usize {
        self.num_leafs
    }

    fn num_polys(&self) -> usize {
        unreachable!()
    }

    fn as_single_slice(&self) -> &[F] {
        &self.nodes
    }
}

impl TreeQuery for Vec<SubTree> {
    fn query<H: TreeHasher<F, Output = [F; 4]>, L: LeafSourceQuery>(
        &self,
        leaf_sources: &L,
        coset_idx: usize,
        lde_degree: usize,
        row_idx: usize,
        domain_size: usize,
    ) -> CudaResult<OracleQuery<F, H>> {
        assert!(!self[0].parent.is_empty());
        assert_eq!(self.len(), lde_degree);
        let current_subtree = &self[coset_idx];
        // first create proof for subtree
        // then read remaining cap elems from parent if necessary
        let cap_elems_per_coset = current_subtree.cap_size;
        let cap_size = cap_elems_per_coset * lde_degree;
        let num_leafs = current_subtree.num_leafs;
        let mut subtree_proof = query::<H, _>(
            &current_subtree.nodes,
            leaf_sources,
            coset_idx,
            domain_size,
            lde_degree,
            row_idx,
            num_leafs,
            cap_elems_per_coset,
        )?;

        if cap_size <= lde_degree {
            // add nodes from the parent tree
            let mut idx = coset_idx;
            let mut parent = current_subtree.parent.as_ref().clone();
            let _ = parent.pop().unwrap();
            for layer in parent.iter() {
                let sibling_idx = idx ^ 1;
                subtree_proof.proof.push(layer[sibling_idx]);
                idx >>= 1;
            }
        }

        Ok(subtree_proof)
    }
}

pub fn query<H: TreeHasher<F, Output = [F; 4]>, L: LeafSourceQuery>(
    current_tree: &DVec<F>,
    leaf_sources: &L,
    coset_idx: usize, // we need coset idx because fri queries are LDEs
    domain_size: usize,
    lde_degree: usize,
    row_idx: usize,
    num_leafs: usize,
    cap_elems_per_coset: usize,
) -> CudaResult<OracleQuery<F, H>> {
    let leaf_elements =
        leaf_sources.get_leaf_sources(coset_idx, domain_size, lde_degree, row_idx, 1)?;
    // we are looking for a leaf of the subtree of given coset.
    // we put single element into leaf for non-fri related trees
    // so just use the given leaf idx
    let leaf_idx = row_idx;
    assert!(leaf_idx < num_leafs);
    assert_eq!(current_tree.len(), 2 * num_leafs * NUM_EL_PER_HASH);
    assert!(cap_elems_per_coset.is_power_of_two());
    assert!(num_leafs.is_power_of_two());
    let log_cap = cap_elems_per_coset.trailing_zeros();
    let depth = num_leafs.trailing_zeros();
    let num_layers = depth - log_cap; // ignore cap element(s)

    let mut result = vec![];
    let mut layer_start = 0;
    let mut num_leafs = num_leafs;
    let mut tree_idx = leaf_idx;

    for _ in 0..num_layers {
        let sibling_idx = tree_idx ^ 1;
        let mut node_hash = [F::ZERO; NUM_EL_PER_HASH];
        for col_idx in 0..NUM_EL_PER_HASH {
            let pos = (layer_start + col_idx * num_leafs) + sibling_idx;
            node_hash[col_idx] = current_tree.clone_el_to_host(pos)?;
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

pub fn compute_tree_cap(
    leaf_sources: &[F],
    result: &mut [F],
    source_len: usize,
    cap_size: usize,
    num_elems_per_leaf: usize,
) -> CudaResult<Vec<[F; 4]>> {
    tree::build_tree(
        leaf_sources,
        result,
        source_len,
        cap_size,
        num_elems_per_leaf,
    )?;
    let tree_cap = get_tree_cap_from_nodes(result, cap_size)?;
    // TODO: transfer subtree to the host
    Ok(tree_cap)
}

pub fn get_tree_cap_from_nodes(result: &[F], cap_size: usize) -> CudaResult<Vec<[F; 4]>> {
    let result_len = result.len();
    let actual_cap_len = NUM_EL_PER_HASH * cap_size;
    let cap_start_pos = result_len - 2 * actual_cap_len;
    let cap_end_pos = cap_start_pos + actual_cap_len;
    let range = cap_start_pos..cap_end_pos;
    let len = range.len();

    let mut layer_nodes = vec![F::ZERO; len];
    mem::d2h(&result[range], &mut layer_nodes)?;

    let mut cap_values = vec![];
    for node_idx in 0..cap_size {
        let mut actual = [F::ZERO; NUM_EL_PER_HASH];
        for col_idx in 0..NUM_EL_PER_HASH {
            let idx = col_idx * cap_size + node_idx;
            actual[col_idx] = layer_nodes[idx];
        }
        cap_values.push(actual);
    }
    assert_eq!(cap_values.len(), cap_size);

    Ok(cap_values)
}

pub fn batch_query<H: TreeHasher<F, Output = [F; 4]>, A: GoodAllocator>(
    d_indexes: &DVec<u32, SmallStaticDeviceAllocator>,
    num_queries: usize,
    d_leaf_sources: impl AsSingleSlice,
    num_cols: usize,
    d_oracle_data: impl AsSingleSlice,
    cap_size: usize,
    domain_size: usize,
    num_elems_per_leaf: usize,
    h_all_leaf_elems: &mut Vec<F, A>,
    h_all_proofs: &mut Vec<F, A>,
) -> CudaResult<()> {
    batch_query_leaf_sources(
        d_indexes,
        num_queries,
        d_leaf_sources,
        num_cols,
        domain_size,
        num_elems_per_leaf,
        h_all_leaf_elems,
    )?;
    batch_query_tree::<H, _>(
        d_indexes,
        num_queries,
        d_oracle_data,
        cap_size,
        domain_size,
        num_elems_per_leaf,
        h_all_proofs,
    )?;

    Ok(())
}

pub fn batch_query_tree<H: TreeHasher<F, Output = [F; 4]>, A: GoodAllocator>(
    d_indexes: &DVec<u32, SmallStaticDeviceAllocator>,
    num_queries: usize,
    d_oracle_data: impl AsSingleSlice,
    cap_size: usize,
    domain_size: usize,
    num_elems_per_leaf: usize,
    h_all_proofs: &mut Vec<F, A>,
) -> CudaResult<()> {
    use cudart::slice::DeviceSlice;
    assert_eq!(d_indexes.len(), num_queries);
    assert!(domain_size.is_power_of_two());
    assert!(cap_size.is_power_of_two());
    assert!(num_elems_per_leaf.is_power_of_two());
    let num_leafs = domain_size / num_elems_per_leaf;
    assert_eq!(num_leafs, d_oracle_data.domain_size());
    let num_layers = (num_leafs.trailing_zeros() - cap_size.trailing_zeros()) as usize;
    if num_layers == 0 {
        return Ok(());
    }
    let mut d_all_proofs = dvec!(num_queries * NUM_EL_PER_HASH * num_layers);
    assert!(h_all_proofs.capacity() >= d_all_proofs.len());
    unsafe { h_all_proofs.set_len(d_all_proofs.len()) };
    let (d_indexes_ref, d_oracle_data, mut d_all_proof_elems_ref) = unsafe {
        (
            DeviceSlice::from_slice(&d_indexes),
            DeviceSlice::from_slice(&d_oracle_data.as_single_slice()),
            DeviceSlice::from_mut_slice(&mut d_all_proofs),
        )
    };

    boojum_cuda::poseidon::gather_merkle_paths(
        d_indexes_ref,
        &d_oracle_data,
        &mut d_all_proof_elems_ref,
        num_layers as u32,
        get_stream(),
    )?;
    mem::d2h(&d_all_proofs, &mut h_all_proofs[..])?;

    Ok(())
}

pub fn batch_query_leaf_sources<A: GoodAllocator>(
    d_indexes: &DVec<u32, SmallStaticDeviceAllocator>,
    num_queries: usize,
    d_leaf_sources: impl AsSingleSlice,
    num_cols: usize,
    domain_size: usize,
    num_elems_per_leaf: usize,
    h_all_leaf_elems: &mut Vec<F, A>,
) -> CudaResult<()> {
    use boojum_cuda::device_structures::{DeviceMatrix, DeviceMatrixMut};
    use cudart::slice::DeviceSlice;
    assert_eq!(d_indexes.len(), num_queries);
    assert!(domain_size.is_power_of_two());
    assert_eq!(domain_size, d_leaf_sources.domain_size());
    assert!(num_elems_per_leaf.is_power_of_two());
    assert_eq!(d_leaf_sources.len() % domain_size, 0);
    let num_polys = d_leaf_sources.len() / domain_size;
    // assert_eq!(d_leaf_sources.num_polys(), num_polys);
    assert_eq!(num_polys, num_cols);
    let mut d_all_leaf_elems = dvec!(num_queries * num_polys * num_elems_per_leaf);
    assert!(h_all_leaf_elems.capacity() >= d_all_leaf_elems.len());
    unsafe { h_all_leaf_elems.set_len(d_all_leaf_elems.len()) };
    let (d_indexes_ref, d_leaf_sources, mut d_all_leaf_elems_ref) = unsafe {
        (
            DeviceSlice::from_slice(&d_indexes),
            DeviceMatrix::new(
                DeviceSlice::from_slice(d_leaf_sources.as_single_slice()),
                domain_size,
            ),
            DeviceMatrixMut::new(
                DeviceSlice::from_mut_slice(&mut d_all_leaf_elems),
                num_queries * num_elems_per_leaf,
            ),
        )
    };
    let log_rows_per_index = num_elems_per_leaf.trailing_zeros();
    boojum_cuda::poseidon::gather_rows(
        d_indexes_ref,
        log_rows_per_index,
        &d_leaf_sources,
        &mut d_all_leaf_elems_ref,
        get_stream(),
    )?;
    mem::d2h(&d_all_leaf_elems, &mut h_all_leaf_elems[..])?;

    Ok(())
}

#[test]
fn test_batch_query_for_leaf_sources() -> CudaResult<()> {
    let _ctx = ProverContext::create_14gb()?;
    let domain_size = 1 << 16;
    let lde_degree = 2;
    let num_cols = 2;
    let num_queries = 1 << 10;

    for log_n in 0..4 {
        let num_elems_per_leaf = 1 << log_n;
        print!("running for num elems per leaf {}", num_elems_per_leaf);
        run_batch_query_for_leaf_sources(
            domain_size,
            lde_degree,
            num_cols,
            num_queries,
            num_elems_per_leaf,
        )?;
        println!(" [DONE]");
    }

    Ok(())
}

fn run_batch_query_for_leaf_sources(
    domain_size: usize,
    lde_degree: usize,
    num_cols: usize,
    num_queries: usize,
    num_elems_per_leaf: usize,
) -> CudaResult<()> {
    use crate::prover::construct_single_query_for_leaf_source_from_batch_sources;
    use rand::{Rng, SeedableRng};
    let mut rng = rand::rngs::StdRng::seed_from_u64(42u64);

    assert!(domain_size >= num_queries);

    let mut storage = vec![];
    for _ in 0..num_cols * lde_degree {
        for idx in 0..domain_size {
            storage.push(F::from_u64_unchecked(idx as u64));
        }
    }

    let d_storage = DVec::from_vec(storage)?;
    let codeword = CodeWord::new_base_assuming_adjacent(d_storage, lde_degree);
    assert!(domain_size <= (u32::MAX as usize));
    let mut all_indexes = Vec::with_capacity(lde_degree);
    for _ in 0..lde_degree {
        let indexes: Vec<_> = (0..num_queries)
            .map(|_| rng.gen::<u32>() % domain_size as u32)
            .collect();
        assert_eq!(indexes.len(), num_queries);
        all_indexes.push(indexes);
    }

    for coset_idx in 0..lde_degree {
        let mut h_all_leaf_elems_expected =
            Vec::with_capacity(num_cols * num_queries * num_elems_per_leaf);
        let mut h_all_leaf_elems_actual =
            Vec::with_capacity(num_cols * num_queries * num_elems_per_leaf);
        let coset_indexes = &all_indexes[coset_idx];
        for query_idx in coset_indexes.iter() {
            let expected_query = codeword.get_leaf_sources(
                coset_idx,
                domain_size,
                lde_degree,
                *query_idx as usize,
                num_elems_per_leaf,
            )?;
            h_all_leaf_elems_expected.extend_from_slice(&expected_query);
        }

        let effective_indexes: Vec<_> =
            compute_effective_indexes(coset_indexes, coset_idx, domain_size, num_elems_per_leaf);
        let mut d_effective_indexes = svec!(effective_indexes.len());
        mem::h2d(&effective_indexes, &mut d_effective_indexes)?;

        let num_queries = coset_indexes.len();
        batch_query_leaf_sources(
            &d_effective_indexes,
            num_queries,
            &codeword,
            num_cols,
            domain_size * lde_degree,
            num_elems_per_leaf,
            &mut h_all_leaf_elems_actual,
        )?;

        for (query_idx, expected_chunk) in h_all_leaf_elems_expected
            .chunks(num_elems_per_leaf * num_cols)
            .enumerate()
        {
            let (expected_c0, expected_c1) = expected_chunk.split_at(num_elems_per_leaf);
            let (actual_c0_batch, actual_c1_batch) =
                h_all_leaf_elems_actual.split_at(num_queries * num_elems_per_leaf);
            let start = query_idx * num_elems_per_leaf;
            let end = start + num_elems_per_leaf;
            assert_eq!(expected_c0, &actual_c0_batch[start..end]);
            assert_eq!(expected_c1, &actual_c1_batch[start..end]);
        }

        for (query_idx, expected_chunk) in h_all_leaf_elems_expected
            .chunks(num_elems_per_leaf * num_cols)
            .enumerate()
        {
            let leaf_elems = construct_single_query_for_leaf_source_from_batch_sources(
                &h_all_leaf_elems_actual,
                num_queries,
                query_idx,
                num_cols,
                num_elems_per_leaf,
            );
            assert_eq!(expected_chunk.len(), leaf_elems.len());
            assert_eq!(expected_chunk, &leaf_elems);
        }
    }
    Ok(())
}

#[test]
fn test_batch_query_for_fri_layers() -> CudaResult<()> {
    let _ctx = ProverContext::create_14gb()?;
    let domain_size = 1 << 16;
    let lde_degree = 2;
    let num_cols = 2;
    let num_queries = 1 << 10;
    let cap_size = 4;

    run_batch_query_for_fri_layers(domain_size, lde_degree, num_cols, num_queries, cap_size)?;

    Ok(())
}

fn run_batch_query_for_fri_layers(
    domain_size: usize,
    lde_degree: usize,
    num_cols: usize,
    num_queries: usize,
    cap_size: usize,
) -> CudaResult<()> {
    use crate::prover::construct_single_query_for_leaf_source_from_batch_sources;
    use boojum::worker::Worker;
    use rand::{Rng, SeedableRng};
    let mut rng = rand::rngs::StdRng::seed_from_u64(42u64);

    assert!(domain_size >= num_queries);

    let h_values = (0..num_cols * domain_size)
        .map(|idx| F::from_u64_unchecked(idx as u64))
        .collect::<Vec<_>>();
    let mut d_values = DVec::from_vec(h_values)?;
    let mut d_storage = dvec!(d_values.len() * lde_degree);
    batch_ntt(&mut d_values, false, true, domain_size, num_cols)?;
    batch_bitreverse(&mut d_values, domain_size)?;
    for (v, s) in d_values
        .chunks(domain_size)
        .zip(d_storage.chunks_mut(domain_size * lde_degree))
    {
        for (coset_idx, c) in s.chunks_mut(domain_size).enumerate() {
            coset_fft_into(
                v,
                c,
                bitreverse_index(coset_idx, lde_degree.trailing_zeros() as usize),
                lde_degree,
            )?;
        }
    }
    let base_codeword = CodeWord::new_base_assuming_adjacent(d_storage, lde_degree);
    let folding_schedule = vec![2, 2, 1];

    let (fri_holder, _) = compute_fri::<_, Global>(
        base_codeword.clone(),
        &mut DefaultTranscript::new(()),
        folding_schedule.clone(),
        lde_degree,
        cap_size,
        &Worker::new(),
    )?;

    assert!(domain_size <= (u32::MAX as usize));
    let mut all_indexes = Vec::with_capacity(lde_degree);
    for _ in 0..lde_degree {
        let indexes: Vec<_> = (0..num_queries)
            .map(|_| rng.gen::<u32>() % domain_size as u32)
            .collect();
        assert_eq!(indexes.len(), num_queries);
        all_indexes.push(indexes);
    }
    let mut original_indexes = all_indexes.to_vec();
    let mut domain_size = domain_size;

    for (layer_idx, (codeword, oracle)) in fri_holder.flatten().into_iter().enumerate() {
        dbg!(layer_idx);
        let num_elems_per_leaf = oracle.num_elems_per_leaf;
        assert_eq!(1 << folding_schedule[layer_idx], num_elems_per_leaf);
        for coset_idx in 0..lde_degree {
            let mut h_all_leaf_elems_expected =
                Vec::with_capacity(num_cols * num_queries * num_elems_per_leaf);
            let mut h_all_leaf_elems_actual =
                Vec::with_capacity(num_cols * num_queries * num_elems_per_leaf);
            let coset_indexes = &original_indexes[coset_idx];
            for query_idx in coset_indexes.iter().cloned() {
                let expected_query = codeword.get_leaf_sources(
                    coset_idx,
                    domain_size,
                    lde_degree,
                    query_idx as usize,
                    num_elems_per_leaf,
                )?;
                h_all_leaf_elems_expected.extend_from_slice(&expected_query);
            }

            let effective_indexes = compute_effective_indexes(
                coset_indexes,
                coset_idx,
                domain_size,
                num_elems_per_leaf,
            );
            let mut d_effective_indexes = svec!(effective_indexes.len());
            mem::h2d(&effective_indexes, &mut d_effective_indexes)?;

            let num_queries = coset_indexes.len();
            batch_query_leaf_sources(
                &d_effective_indexes,
                num_queries,
                codeword,
                num_cols,
                domain_size * lde_degree,
                num_elems_per_leaf,
                &mut h_all_leaf_elems_actual,
            )?;

            for (query_idx, expected_chunk) in h_all_leaf_elems_expected
                .chunks(num_elems_per_leaf * lde_degree)
                .enumerate()
            {
                let (expected_c0, expected_c1) = expected_chunk.split_at(num_elems_per_leaf);
                let (actual_c0_batch, actual_c1_batch) =
                    h_all_leaf_elems_actual.split_at(num_queries * num_elems_per_leaf);
                let start = query_idx * num_elems_per_leaf;
                let end = start + num_elems_per_leaf;
                assert_eq!(&expected_c0[..], &actual_c0_batch[start..end]);
                assert_eq!(&expected_c1[..], &actual_c1_batch[start..end]);
            }

            for (query_idx, expected_chunk) in h_all_leaf_elems_expected
                .chunks(num_elems_per_leaf * num_cols)
                .enumerate()
            {
                let leaf_elems = construct_single_query_for_leaf_source_from_batch_sources(
                    &h_all_leaf_elems_actual,
                    num_queries,
                    query_idx,
                    num_cols,
                    num_elems_per_leaf,
                );
                assert_eq!(expected_chunk.len(), leaf_elems.len());
                assert_eq!(expected_chunk, &leaf_elems);
            }
        }

        let log2_schedule = num_elems_per_leaf.trailing_zeros() as usize;
        domain_size >>= log2_schedule;
        for indexes in original_indexes.iter_mut() {
            for index in indexes.iter_mut() {
                *index >>= log2_schedule;
            }
        }
    }

    Ok(())
}

#[test]
fn test_batch_query_for_merkle_paths() -> CudaResult<()> {
    let _ctx = ProverContext::create_14gb()?;
    let domain_size = 1 << 4;
    let lde_degree = 2;
    let num_cols = 2;
    let num_queries = 1 << 1;
    let cap_size = 2;

    run_batch_query_for_merkle_paths(domain_size, lde_degree, num_cols, num_queries, cap_size)?;

    Ok(())
}

fn run_batch_query_for_merkle_paths(
    domain_size: usize,
    lde_degree: usize,
    num_cols: usize,
    num_queries: usize,
    cap_size: usize,
) -> CudaResult<()> {
    use crate::prover::construct_single_query_for_merkle_path_from_batch_sources;
    use boojum::worker::Worker;
    use rand::{Rng, SeedableRng};
    let mut rng = rand::rngs::StdRng::seed_from_u64(42 as u64);

    assert!(domain_size >= num_queries);

    let h_values = (0..num_cols * domain_size)
        .map(|idx| F::from_u64_unchecked(idx as u64))
        .collect::<Vec<_>>();
    let mut d_values = DVec::from_vec(h_values)?;
    let mut d_storage = dvec!(d_values.len() * lde_degree);
    batch_ntt(&mut d_values, false, true, domain_size, num_cols)?;
    batch_bitreverse(&mut d_values, domain_size)?;
    for (v, s) in d_values
        .chunks(domain_size)
        .zip(d_storage.chunks_mut(domain_size * lde_degree))
    {
        for (coset_idx, c) in s.chunks_mut(domain_size).enumerate() {
            coset_fft_into(
                v,
                c,
                bitreverse_index(coset_idx, lde_degree.trailing_zeros() as usize),
                lde_degree,
            )?;
        }
    }
    let base_codeword = CodeWord::new_base_assuming_adjacent(d_storage, lde_degree);
    let folding_schedule = vec![1, 1];

    let (fri_holder, _) = compute_fri::<_, Global>(
        base_codeword.clone(),
        &mut DefaultTranscript::new(()),
        folding_schedule.clone(),
        lde_degree,
        cap_size,
        &Worker::new(),
    )?;

    assert!(domain_size <= (u32::MAX as usize));
    let mut all_indexes = Vec::with_capacity(lde_degree);
    for _ in 0..lde_degree {
        let indexes: Vec<_> = (0..num_queries)
            .map(|_| rng.gen::<u32>() % domain_size as u32)
            .collect();
        assert_eq!(indexes.len(), num_queries);
        all_indexes.push(indexes);
    }
    let mut domain_size = domain_size;

    for (layer_idx, (codeword, oracle)) in fri_holder.flatten().into_iter().enumerate() {
        dbg!(layer_idx);
        let num_elems_per_leaf = oracle.num_elems_per_leaf;
        assert_eq!(1 << folding_schedule[layer_idx], num_elems_per_leaf);
        assert_eq!(lde_degree * domain_size, codeword.length());
        let num_leafs = lde_degree * domain_size / num_elems_per_leaf;
        assert_eq!(num_leafs, oracle.num_leafs);
        let num_layers = num_leafs.trailing_zeros() as usize;
        let layers_to_skip = cap_size.trailing_zeros() as usize;
        let num_actual_layers = num_layers - layers_to_skip;
        for coset_idx in 0..lde_degree {
            let mut h_all_proof_elems_expected =
                Vec::with_capacity(num_actual_layers * num_queries);
            let mut h_all_proof_elems_actual =
                Vec::with_capacity(num_actual_layers * num_queries * NUM_EL_PER_HASH);
            let coset_indexes = &all_indexes[coset_idx];
            assert_eq!(num_queries, coset_indexes.len());
            for query_idx in coset_indexes.iter().cloned() {
                let expected_query = oracle.query::<DefaultTreeHasher, _>(
                    codeword,
                    coset_idx,
                    lde_degree,
                    query_idx as usize,
                    domain_size,
                )?;
                dbg!(query_idx);
                dbg!(&expected_query.proof);
                h_all_proof_elems_expected.extend_from_slice(&expected_query.proof);
            }
            assert_eq!(
                h_all_proof_elems_expected.len(),
                h_all_proof_elems_expected.capacity()
            );

            let effective_indexes = compute_effective_indexes(
                coset_indexes,
                coset_idx,
                domain_size,
                num_elems_per_leaf,
            );
            dbg!(&effective_indexes);
            let mut d_effective_indexes = svec!(effective_indexes.len());
            mem::h2d(&effective_indexes, &mut d_effective_indexes)?;

            batch_query_tree::<DefaultTreeHasher, _>(
                &d_effective_indexes,
                num_queries,
                oracle,
                cap_size,
                domain_size * lde_degree,
                num_elems_per_leaf,
                &mut h_all_proof_elems_actual,
            )?;

            assert_eq!(
                h_all_proof_elems_actual.len(),
                h_all_proof_elems_actual.capacity()
            );
            dbg!(oracle.nodes.to_vec().unwrap());
            dbg!(&h_all_proof_elems_expected);
            dbg!(&h_all_proof_elems_actual);

            for (query_idx, expected_chunk) in h_all_proof_elems_expected
                .chunks(num_actual_layers)
                .enumerate()
            {
                let actual_chunk = construct_single_query_for_merkle_path_from_batch_sources(
                    &h_all_proof_elems_actual,
                    cap_size,
                    num_queries,
                    query_idx,
                    num_elems_per_leaf,
                    domain_size * lde_degree,
                );
                assert_eq!(actual_chunk.len(), num_actual_layers);
                for (expected, actual) in expected_chunk.iter().zip(actual_chunk.iter()) {
                    assert_eq!(expected, actual);
                }
            }
        }

        let log2_schedule = num_elems_per_leaf.trailing_zeros() as usize;
        domain_size >>= log2_schedule;
        for coset_idxes in all_indexes.iter_mut() {
            for index in coset_idxes.iter_mut() {
                *index >>= log2_schedule;
            }
        }
    }

    Ok(())
}
