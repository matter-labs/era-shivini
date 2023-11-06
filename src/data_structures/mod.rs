use super::*;

mod storage;
pub use storage::*;

mod trace;
pub use trace::*;

mod setup;
pub use setup::*;

mod arguments;
pub use arguments::*;

mod tree;
pub use tree::*;

use boojum::cs::traits::GoodAllocator;
use boojum_cuda::device_structures::{DeviceMatrix, DeviceMatrixMut};
use cudart::{event::CudaEventCreateFlags, slice::DeviceSlice};

pub trait AsSingleSlice {
    fn domain_size(&self) -> usize;
    fn num_polys(&self) -> usize;
    fn as_single_slice(&self) -> &[F];
    fn as_single_slice_mut(&mut self) -> &mut [F] {
        unreachable!()
    }
    fn len(&self) -> usize {
        self.as_single_slice().len()
    }
}

pub(crate) fn coset_cap_size(cap_size: usize, lde_degree: usize) -> usize {
    let coset_cap_size = if cap_size < lde_degree {
        1
    } else {
        assert!(cap_size.is_power_of_two());
        1 << (cap_size.trailing_zeros() - lde_degree.trailing_zeros())
    };

    coset_cap_size
}

// (slice, domain size, num polys)
// barycentric evaluations of lookup
impl AsSingleSlice for (&[F], usize, usize) {
    fn domain_size(&self) -> usize {
        self.1
    }

    fn num_polys(&self) -> usize {
        self.2
    }

    fn as_single_slice(&self) -> &[F] {
        // we only use these complex poly storage of the quotient chunks
        assert_eq!(self.0.len(), 2 * self.num_polys() * self.domain_size());
        self.0
    }

    fn as_single_slice_mut(&mut self) -> &mut [F] {
        unimplemented!()
    }
}
