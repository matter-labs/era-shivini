use std::rc::Rc;

use boojum::cs::{implementations::proof::OracleQuery, oracle::TreeHasher};

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
        SubTree {
            parent: Rc::new(vec![]), // TODO each subtree have access to the parent for querying
            nodes,
            num_leafs,
            cap_size,
            tree_idx: coset_idx,
        }
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
        // then read remaning cap elems from parent if necessary
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
                let sbling_idx = idx ^ 1;
                subtree_proof.proof.push(layer[sbling_idx]);
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
        let sbling_idx = tree_idx ^ 1;
        let mut node_hash = [F::ZERO; NUM_EL_PER_HASH];
        for col_idx in 0..NUM_EL_PER_HASH {
            let pos = (layer_start + col_idx * num_leafs) + sbling_idx;
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
    let num_el_per_hash = 4;
    let result_len = result.len();
    let actual_cap_len = num_el_per_hash * cap_size;
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
