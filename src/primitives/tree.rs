use super::*;

pub fn build_tree(
    leaf_sources: &[F],
    result: &mut [F],
    source_len: usize,
    cap_size: usize,
    num_elems_per_leaf: usize,
) -> CudaResult<()> {
    assert!(!leaf_sources.is_empty());
    let _num_sources = leaf_sources.len() / source_len;
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
    boojum_cuda::poseidon::build_merkle_tree::<boojum_cuda::poseidon::Poseidon2>(
        leaf_sources,
        result,
        num_elems_per_leaf.trailing_zeros(),
        get_stream(),
        num_layers as u32,
    )
}
