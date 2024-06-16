use super::*;
mod device;
mod host;

pub use device::*;
pub use host::*;

pub trait StaticAllocator: GoodAllocator {}
#[allow(dead_code)]
pub trait SmallStaticAllocator: StaticAllocator {}
