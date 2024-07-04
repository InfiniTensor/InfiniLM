use common::utok;
use operators::cndrv::{DevByte, Queue};
use std::ops::{Deref, DerefMut};
use tensor::Tensor;

pub fn gather<T, U, I>(x: &mut Tensor<T>, table: &Tensor<U>, tokens: I, queue: &Queue)
where
    T: DerefMut<Target = [DevByte]>,
    U: Deref<Target = [u8]>,
    I: IntoIterator<Item = utok>,
{
    let &[_, d] = x.shape() else { panic!() };

    debug_assert_eq!(x.data_layout(), table.data_layout());
    debug_assert_eq!(table.shape().len(), 2);
    debug_assert_eq!(table.shape()[1], d);
    debug_assert!(x.is_contiguous());
    debug_assert!(table.is_contiguous());
    let d = d as usize * x.data_layout().nbytes();

    let x = &mut **x.physical_mut();
    let table = table.as_slice();
    for (i, t) in tokens.into_iter().enumerate() {
        let dst = &mut x[d * i..][..d];
        let src = &table[d * t as usize..][..d];
        queue.memcpy_h2d(dst, src);
    }
}
