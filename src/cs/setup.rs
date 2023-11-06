use boojum::{
    config::{CSConfig, CSSetupConfig},
    cs::{
        implementations::{
            fast_serialization::MemcopySerializable,
            hints::{DenseVariablesCopyHint, DenseWitnessCopyHint},
            polynomial_storage::SetupBaseStorage,
            reference_cs::CSReferenceAssembly,
            setup::TreeNode,
            utils::make_non_residues,
        },
        oracle::merkle_tree::MerkleTreeWithCap,
        Place, Variable, Witness,
    },
    field::U64Representable,
    worker::Worker,
};
use boojum_cuda::ops_complex::pack_variable_indexes;
use cudart::slice::{CudaSlice, CudaSliceMut, DeviceSlice};

use super::*;
pub(crate) const PACKED_PLACEHOLDER_BITMASK: u32 = 1 << 31;

#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct GpuSetup<A: GoodAllocator> {
    #[serde(serialize_with = "boojum::utils::serialize_vec_vec_with_allocators")]
    #[serde(deserialize_with = "boojum::utils::deserialize_vec_vec_with_allocators")]
    pub constant_columns: Vec<Vec<F, A>>,
    #[serde(serialize_with = "boojum::utils::serialize_vec_vec_with_allocators")]
    #[serde(deserialize_with = "boojum::utils::deserialize_vec_vec_with_allocators")]
    pub lookup_tables_columns: Vec<Vec<F, A>>,
    #[serde(serialize_with = "boojum::utils::serialize_vec_vec_with_allocators")]
    #[serde(deserialize_with = "boojum::utils::deserialize_vec_vec_with_allocators")]
    pub variables_hint: Vec<Vec<u32, A>>,
    #[serde(serialize_with = "boojum::utils::serialize_vec_vec_with_allocators")]
    #[serde(deserialize_with = "boojum::utils::deserialize_vec_vec_with_allocators")]
    pub witnesses_hint: Vec<Vec<u32, A>>,
    pub table_ids_column_idxes: Vec<usize>,
    pub selectors_placement: TreeNode,
    pub setup_tree: MerkleTreeWithCap<F, DefaultTreeHasher>,
    pub layout: SetupLayout,
}

pub fn transform_indexes_on_device<A: GoodAllocator, T>(
    variables_hint: Vec<Vec<T>>,
) -> CudaResult<Vec<Vec<u32, A>>> {
    if variables_hint.is_empty() {
        return Ok(vec![]);
    }
    // we want to keep size of the hints as small as possible
    // that's why we expect them to be "unpadded" and padding will be
    // applied during materializiation of the corresponding values
    // e.g permutation cols, variable cols etc.
    let num_cols = variables_hint.len();
    let total_size = variables_hint.iter().map(|col| col.len()).sum();

    assert_eq!(std::mem::size_of::<T>(), std::mem::size_of::<u64>());

    let mut transformed_hints = Vec::with_capacity(num_cols);
    for col in variables_hint.iter() {
        assert!(col.len() as u32 & PACKED_PLACEHOLDER_BITMASK != 0);
        let mut new = Vec::with_capacity_in(col.len(), A::default());
        unsafe { new.set_len(col.len()) }
        transformed_hints.push(new);
    }

    let alloc = _alloc().clone();

    let mut original_variables =
        DVec::<u64, StaticDeviceAllocator>::with_capacity_in(total_size, alloc.clone());
    let mut transformed_variables =
        DVec::<u32, StaticDeviceAllocator>::with_capacity_in(total_size, alloc);
    // this is a transfer between same shape buffers
    // we have avoided to flatten source values whereas destination is already flattened
    let mut start = 0;
    for src in variables_hint.iter() {
        let src = unsafe { std::slice::from_raw_parts(src.as_ptr() as *const _, src.len()) };
        let end = start + src.len();
        let dst = &mut original_variables[start..end];
        mem::h2d(src, dst)?;
        start = end;
    }

    let (d_variables, d_variables_transformed) = unsafe {
        (
            DeviceSlice::from_slice(&original_variables[..]),
            DeviceSlice::from_mut_slice(&mut transformed_variables[..]),
        )
    };

    pack_variable_indexes(d_variables, d_variables_transformed, get_stream())?;

    let mut start = 0;
    for dst in transformed_hints.iter_mut() {
        let end = start + dst.len();
        let src = &transformed_variables[start..end];
        mem::d2h(src, dst)?;
        start = end;
    }

    Ok(transformed_hints)
}

pub fn transform_variable_indexes<A: GoodAllocator>(
    hints: Vec<Vec<Variable>>,
    worker: &Worker,
) -> Vec<Vec<u32, A>> {
    let num_cols = hints.len();
    assert!(num_cols > 0);
    let mut transformed_hints = Vec::with_capacity(num_cols);
    for col in hints.iter() {
        assert_eq!(col.len() as u32 & PACKED_PLACEHOLDER_BITMASK, 0);
        let mut new = Vec::with_capacity_in(col.len(), A::default());
        unsafe {
            new.set_len(col.len());
        }
        transformed_hints.push(new);
    }
    assert_eq!(std::mem::size_of::<Variable>(), std::mem::size_of::<u64>());

    worker.scope(hints.len(), |scope, chunk_size| {
        for (src_cols_chunk, dst_cols_chunk) in hints
            .chunks(chunk_size)
            .zip(transformed_hints.chunks_mut(chunk_size))
        {
            assert_eq!(src_cols_chunk.len(), dst_cols_chunk.len());
            scope.spawn(move |_| {
                for (src_col, dst_col) in src_cols_chunk.iter().zip(dst_cols_chunk.iter_mut()) {
                    assert_eq!(src_col.len(), dst_col.len());
                    for (src, dst) in src_col.iter().zip(dst_col.iter_mut()) {
                        if src.is_placeholder() {
                            *dst = PACKED_PLACEHOLDER_BITMASK;
                        } else {
                            *dst = src.as_variable_index();
                        }
                    }
                }
            })
        }
    });

    transformed_hints
}

pub fn transform_witness_indexes<A: GoodAllocator>(
    hints: Vec<Vec<Witness>>,
    worker: &Worker,
) -> Vec<Vec<u32, A>> {
    let num_cols = hints.len();
    if num_cols == 0 {
        return vec![];
    }
    let mut transformed_hints = Vec::with_capacity(num_cols);
    for col in hints.iter() {
        // still good to check max number of witness values can't fit into u32
        assert_eq!(col.len() as u32 & PACKED_PLACEHOLDER_BITMASK, 0);
        let mut new = Vec::with_capacity_in(col.len(), A::default());
        unsafe {
            new.set_len(col.len());
        }
        transformed_hints.push(new);
    }
    assert_eq!(std::mem::size_of::<Variable>(), std::mem::size_of::<u64>());

    worker.scope(hints.len(), |scope, chunk_size| {
        for (src_cols_chunk, dst_cols_chunk) in hints
            .chunks(chunk_size)
            .zip(transformed_hints.chunks_mut(chunk_size))
        {
            assert_eq!(src_cols_chunk.len(), dst_cols_chunk.len());
            scope.spawn(move |_| {
                for (src_col, dst_col) in src_cols_chunk.iter().zip(dst_cols_chunk.iter_mut()) {
                    assert_eq!(src_col.len(), dst_col.len());
                    for (src, dst) in src_col.iter().zip(dst_col.iter_mut()) {
                        if src.is_placeholder() {
                            *dst = PACKED_PLACEHOLDER_BITMASK;
                        } else {
                            *dst = src.as_witness_index() as u32;
                        }
                    }
                }
            })
        }
    });

    transformed_hints
}

impl<A: GoodAllocator> GpuSetup<A> {
    pub fn from_setup_and_hints<
        P: boojum::field::traits::field_like::PrimeFieldLikeVectorized<Base = F>,
    >(
        base_setup: SetupBaseStorage<F, P, Global, Global>,
        setup_tree: MerkleTreeWithCap<F, DefaultTreeHasher>,
        variables_hint: DenseVariablesCopyHint,
        witnesses_hint: DenseWitnessCopyHint,
        worker: &Worker,
    ) -> CudaResult<Self> {
        assert_eq!(
            variables_hint.maps.len(),
            base_setup.copy_permutation_polys.len()
        );
        let domain_size = base_setup.copy_permutation_polys[0].domain_size();
        let layout = SetupLayout::from_base_setup_and_hints(&base_setup);

        let SetupBaseStorage {
            constant_columns,
            lookup_tables_columns,
            table_ids_column_idxes,
            selectors_placement,
            ..
        } = base_setup;

        let transformed_variables_hints = transform_variable_indexes(variables_hint.maps, worker);
        let transformed_witnesses_hints = transform_witness_indexes(witnesses_hint.maps, worker);

        let mut constant_cols_in = Vec::with_capacity(constant_columns.len());
        for src in constant_columns.iter() {
            let mut new = Vec::with_capacity_in(src.domain_size(), A::default());
            new.extend_from_slice(P::slice_into_base_slice(&src.storage));
            constant_cols_in.push(new);
        }

        let mut lookup_tables_columns_in = Vec::with_capacity(lookup_tables_columns.len());
        for src in lookup_tables_columns.iter() {
            let mut new = Vec::with_capacity_in(src.domain_size(), A::default());
            new.extend_from_slice(P::slice_into_base_slice(&src.storage));
            lookup_tables_columns_in.push(new);
        }

        Ok(Self {
            constant_columns: constant_cols_in,
            lookup_tables_columns: lookup_tables_columns_in,
            table_ids_column_idxes,
            selectors_placement,
            variables_hint: transformed_variables_hints,
            witnesses_hint: transformed_witnesses_hints,
            setup_tree,
            layout,
        })
    }
}

pub fn calculate_tmp_buffer_size(
    num_cells: usize,
    block_size_in_bytes: usize,
) -> CudaResult<usize> {
    let tmp_storage_size_in_bytes =
        boojum_cuda::ops_complex::get_generate_permutation_matrix_temp_storage_bytes(num_cells)?;

    let mut num_blocks_for_tmp_storage = tmp_storage_size_in_bytes / block_size_in_bytes;
    if tmp_storage_size_in_bytes % block_size_in_bytes != 0 {
        num_blocks_for_tmp_storage += 1;
    }
    let tmp_storage_size = num_blocks_for_tmp_storage * block_size_in_bytes;

    Ok(tmp_storage_size)
}

fn materialize_non_residues(
    num_cols: usize,
    domain_size: usize,
) -> CudaResult<DVec<F, SmallStaticDeviceAllocator>> {
    let mut non_residues = Vec::with_capacity(num_cols);
    non_residues.push(F::ONE);
    non_residues.extend_from_slice(&make_non_residues::<F>(num_cols - 1, domain_size));
    let mut d_non_residues = svec!(num_cols);
    mem::h2d(&non_residues, &mut d_non_residues)?;

    Ok(d_non_residues)
}

pub fn materialize_permutation_cols_from_transformed_hints_into<'a, A: GoodAllocator>(
    d_result: &mut [F],
    variables_hint: &Vec<Vec<u32, A>>, // TODO: reuse variables from variable assignment
    domain_size: usize,
) -> CudaResult<()> {
    assert!(variables_hint.is_empty() == false);
    assert!(domain_size.is_power_of_two());

    let num_cols = variables_hint.len();
    let num_cells = num_cols * domain_size;
    assert_eq!(d_result.len(), num_cells);

    let alloc = _alloc().clone();
    // FIXME: although it fails with actual number of bytes, it works with padded value
    let tmp_storage_size_in_bytes =
        calculate_tmp_buffer_size(num_cells, alloc.block_size_in_bytes())?;

    let mut d_tmp_storage: DVec<u8> = dvec!(tmp_storage_size_in_bytes);
    let mut d_variables_transformed =
        DVec::<u32, StaticDeviceAllocator>::with_capacity_in(num_cells, alloc);

    // this is a transfer between different shape buffers
    // we have avoided to flatten source values whereas destination is already flattened
    let mut d_bitmask = dvec!(1);
    mem::h2d(&[PACKED_PLACEHOLDER_BITMASK], &mut d_bitmask)?;
    let d_bitmask = &d_bitmask[0];
    for (src, dst) in variables_hint
        .iter()
        .zip(d_variables_transformed.chunks_mut(domain_size))
    {
        assert_eq!(src.len() as u32 & PACKED_PLACEHOLDER_BITMASK, 0);
        let (actual, padding) = dst.split_at_mut(src.len());
        mem::h2d(src, actual)?;
        // don't forget to mark padding space as placeholder
        if padding.is_empty() == false {
            helpers::set_value_generic(padding, &d_bitmask)?;
        }
    }
    let d_non_residues = materialize_non_residues(num_cols, domain_size)?;

    assert_eq!(d_result.len(), num_cells);
    let (d_variables_transformed, d_tmp_storage, d_result_ref, d_non_residues) = unsafe {
        (
            DeviceSlice::from_mut_slice(&mut d_variables_transformed[..]),
            DeviceSlice::from_mut_slice(&mut d_tmp_storage[..]),
            DeviceSlice::from_mut_slice(&mut d_result[..]),
            DeviceSlice::from_slice(&d_non_residues[..]),
        )
    };
    boojum_cuda::ops_complex::generate_permutation_matrix(
        d_tmp_storage,
        d_variables_transformed,
        d_non_residues,
        d_result_ref,
        get_stream(),
    )?;

    Ok(())
}
