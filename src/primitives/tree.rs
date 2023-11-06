use super::*;

type P = boojum_cuda::poseidon::Poseidon2;

pub fn build_tree(
    leaf_sources: &[F],
    result: &mut [F],
    source_len: usize,
    cap_size: usize,
    num_elems_per_leaf: usize,
) -> CudaResult<()> {
    assert!(!leaf_sources.is_empty());
    let num_leafs = source_len / num_elems_per_leaf;
    let log_cap = cap_size.trailing_zeros();
    let depth = num_leafs.trailing_zeros();
    let num_layers = depth - log_cap + 1;
    let (leaf_sources, result) = unsafe {
        (
            DeviceSlice::from_slice(&leaf_sources[..]),
            DeviceSlice::from_mut_slice(result),
        )
    };
    boojum_cuda::poseidon::build_merkle_tree::<P>(
        leaf_sources,
        result,
        num_elems_per_leaf.trailing_zeros(),
        get_stream(),
        num_layers as u32,
    )
}

pub(crate) const POSEIDON_RATE: usize = 8;

pub fn build_leaves_from_chunk(
    leaf_sources: &[F],
    result: &mut [F],
    domain_size: usize,
    load_intermediate: bool,
    store_intermediate: bool,
) -> CudaResult<()> {
    let (d_values, d_result) = unsafe {
        (
            DeviceSlice::from_slice(leaf_sources),
            DeviceSlice::from_mut_slice(result),
        )
    };
    boojum_cuda::poseidon::build_merkle_tree_leaves::<P>(
        d_values,
        d_result,
        0,
        load_intermediate,
        store_intermediate,
        get_stream(),
    )?;

    Ok(())
}

pub fn build_tree_nodes(
    leaf_hashes: &[F],
    result: &mut [F],
    domain_size: usize,
    cap_size: usize,
) -> CudaResult<()> {
    assert!(!leaf_hashes.is_empty());
    let num_sources = leaf_hashes.len() / domain_size;
    let num_leafs = domain_size;
    let log_cap = cap_size.trailing_zeros();
    let depth = num_leafs.trailing_zeros();
    let num_layers = depth - log_cap + 1;
    let (leaf_sources, result) = unsafe {
        (
            DeviceSlice::from_slice(&leaf_hashes[..]),
            DeviceSlice::from_mut_slice(result),
        )
    };
    boojum_cuda::poseidon::build_merkle_tree_nodes::<P>(
        leaf_sources,
        result,
        num_layers as u32,
        get_stream(),
    )
}
