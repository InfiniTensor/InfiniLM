mod blob;
mod fmt;
mod macros;

use nn::Tensor;
use operators::TensorLayout;

pub(crate) use blob::{Blob, Data};
pub(crate) use fmt::fmt;
pub(crate) use macros::*;

pub(crate) fn layout<T, const N: usize>(t: &Tensor<T, N>) -> TensorLayout {
    TensorLayout {
        dt: t.dt(),
        layout: t.layout().to_inline_size(),
    }
}

#[inline(always)]
pub(crate) fn offset_ptr<T, const N: usize>(t: &Tensor<*const T, N>) -> *const T {
    unsafe { t.get().byte_offset(t.layout().offset()) }
}
