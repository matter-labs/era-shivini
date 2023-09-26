use boojum::{
    cs::{
        implementations::{
            hints::DenseVariablesCopyHint, polynomial_storage::SetupBaseStorage, setup::TreeNode,
            utils::make_non_residues,
        },
        oracle::merkle_tree::MerkleTreeWithCap,
        traits::GoodAllocator,
        Variable,
    },
    worker::Worker,
};
use boojum_cuda::ops_complex::pack_variable_indexes;
use cudart::slice::{CudaSlice, DeviceSlice};

use super::*;

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
    pub table_ids_column_idxes: Vec<usize>,
    pub selectors_placement: TreeNode,
    setup_tree: MerkleTreeWithCap<F, DefaultTreeHasher>,
    pub layout: SetupLayout,
}

pub fn transform_variable_indexes_on_host(
    variables: &Vec<Vec<Variable>>,
    worker: &Worker,
) -> Vec<Vec<u32>> {
    let num_cols = variables.len();
    assert!(num_cols > 0);
    let domain_size = variables[0].len();
    let _num_cells = num_cols * domain_size;

    for col in variables[1..].iter() {
        assert_eq!(col.len(), domain_size);
    }

    assert_eq!(std::mem::size_of::<Variable>(), std::mem::size_of::<u64>());

    let mut transformed_hints = vec![vec![0u32; domain_size]; num_cols];

    worker.scope(variables.len(), |_scope, chunk_size| {
        for (src_chunk, dst_chunk) in variables
            .chunks(chunk_size)
            .zip(transformed_hints.chunks_mut(chunk_size))
        {
            for (src_col, dst_col) in src_chunk.iter().zip(dst_chunk.iter_mut()) {
                assert_eq!(src_col.len(), dst_col.len());
                for (src, dst) in src_col.iter().zip(dst_col.iter_mut()) {
                    if src.is_placeholder() {
                        continue;
                    }
                    *dst = src.as_variable_index();
                }
            }
        }
    });

    transformed_hints
}

pub fn transform_variable_indexes_on_device<A: GoodAllocator>(
    variables_hint: &Vec<Vec<Variable>>,
) -> CudaResult<Vec<Vec<u32, A>>> {
    let num_cols = variables_hint.len();
    let domain_size = variables_hint[0].len();
    let num_cells = num_cols * domain_size;

    for col in variables_hint.iter() {
        assert_eq!(col.len(), domain_size);
    }

    assert_eq!(std::mem::size_of::<Variable>(), std::mem::size_of::<u64>());

    let mut transformed_hints = Vec::with_capacity(num_cols);
    for _ in 0..num_cols {
        let mut col = Vec::with_capacity_in(domain_size, A::default());
        unsafe { col.set_len(domain_size) }
        transformed_hints.push(col);
    }

    let alloc = _alloc().clone();

    let mut original_variables =
        DVec::<u64, VirtualMemoryManager>::with_capacity_in(num_cells, alloc.clone());
    let mut transformed_variables =
        DVec::<u32, VirtualMemoryManager>::with_capacity_in(num_cells, alloc);

    for (src, dst) in variables_hint
        .iter()
        .zip(original_variables.chunks_mut(domain_size))
    {
        let src = unsafe {
            assert_eq!(src.len(), domain_size);
            std::slice::from_raw_parts(src.as_ptr() as *const _, domain_size)
        };
        mem::h2d(src, dst)?;
    }

    let (d_variables, d_variables_transformed) = unsafe {
        (
            DeviceSlice::from_slice(&original_variables[..]),
            DeviceSlice::from_mut_slice(&mut transformed_variables[..]),
        )
    };

    pack_variable_indexes(d_variables, d_variables_transformed, get_stream())?;

    for (src, dst) in transformed_variables
        .chunks(domain_size)
        .zip(transformed_hints.iter_mut())
    {
        mem::d2h(src, dst)?;
    }

    Ok(transformed_hints)
}

impl<A: GoodAllocator> GpuSetup<A> {
    pub fn from_setup_and_hints<
        P: boojum::field::traits::field_like::PrimeFieldLikeVectorized<Base = F>,
    >(
        base_setup: SetupBaseStorage<F, P, Global, Global>,
        setup_tree: MerkleTreeWithCap<F, DefaultTreeHasher>,
        mut variables_hint: DenseVariablesCopyHint,
    ) -> CudaResult<Self> {
        assert!(variables_hint.maps.len() < u32::MAX as usize);
        assert_eq!(
            variables_hint.maps.len(),
            base_setup.copy_permutation_polys.len()
        );
        let domain_size = base_setup.copy_permutation_polys[0].domain_size();
        for col in variables_hint.maps.iter_mut() {
            assert!(col.len() <= domain_size);
            col.resize(domain_size, Variable::placeholder());
        }

        let layout = SetupLayout::from_base_setup_and_hints(&base_setup);

        let SetupBaseStorage {
            constant_columns,
            lookup_tables_columns,
            table_ids_column_idxes,
            selectors_placement,
            ..
        } = base_setup;
        let _worker = &Worker::new();
        // TODO: use host based index tranformation
        // let transformed_hints = transform_variable_indexes_on_host(&variables_hint.maps, worker);
        let transformed_hints = transform_variable_indexes_on_device(&variables_hint.maps)?;
        synchronize_streams()?;

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
            variables_hint: transformed_hints,
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
) -> CudaResult<DVec<F, SmallVirtualMemoryManager>> {
    let mut non_residues = Vec::with_capacity(num_cols);
    non_residues.push(F::ONE);
    non_residues.extend_from_slice(&make_non_residues::<F>(num_cols - 1, domain_size));
    let mut d_non_residues = svec!(num_cols);
    mem::h2d(&non_residues, &mut d_non_residues)?;

    Ok(d_non_residues)
}

pub fn materialize_permutation_cols_from_transformed_hints_into<'a, A: GoodAllocator>(
    d_result: &mut [F],
    variables_hint: &Vec<Vec<u32, A>>,
) -> CudaResult<()> {
    assert!(variables_hint.is_empty() == false);

    let domain_size = variables_hint[0].len();
    assert!(domain_size.is_power_of_two());
    for col in variables_hint.iter() {
        assert_eq!(col.len(), domain_size);
    }
    let alloc = _alloc().clone();

    let num_cols = variables_hint.len();
    let num_cells = num_cols * domain_size;
    // FIXME: although it fails with actual number of bytes, it works with padded value
    let tmp_storage_size_in_bytes =
        calculate_tmp_buffer_size(num_cells, alloc.block_size_in_bytes())?;

    let mut d_tmp_storage: DVec<u8> = dvec!(tmp_storage_size_in_bytes);
    let mut d_variables_transformed =
        DVec::<u32, VirtualMemoryManager>::with_capacity_in(num_cells, alloc);

    if is_adjacent(&variables_hint) {
        let flattened_variables = unsafe {
            std::slice::from_raw_parts(variables_hint[0].as_ptr() as *const u32, num_cells)
        };
        mem::h2d(flattened_variables, &mut d_variables_transformed)?;
    } else {
        for (src, dst) in variables_hint
            .iter()
            .zip(d_variables_transformed.chunks_mut(domain_size))
        {
            mem::h2d(src, dst)?;
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

pub fn is_adjacent<T, A: GoodAllocator>(data: &[Vec<T, A>]) -> bool {
    assert!(!data.is_empty());
    let mut prev_ptr = data[0].as_ptr();
    let mut prev_len = data[0].len();
    for item in data[1..].iter() {
        unsafe {
            if std::ptr::eq(prev_ptr.add(prev_len), item.as_ptr()) == false {
                return false;
            }
        }

        prev_ptr = item.as_ptr();
        prev_len = item.len();
    }
    dbg!("columns are adjacent");
    true
}
