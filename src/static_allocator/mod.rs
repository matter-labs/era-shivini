use super::*;
mod device;
mod host;

pub use device::*;
pub use host::*;

pub trait StaticAllocator: GoodAllocator {}
pub trait SmallStaticAllocator: StaticAllocator {}
