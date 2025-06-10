use super::{Handle, Operator};
use crate::utils::destruct;
use nn::{Arg, Tensor, digit_layout::types};
use operators::{
    cuda::{Stream, VirByte},
    nccl::ReduceType,
};

pub struct AllReduce;

impl Operator for AllReduce {
    fn launch<'a, const N: usize>(
        handle: &mut Handle,
        arg: Option<nn::Arg>,
        inputs: impl IntoIterator<Item = Tensor<*const VirByte, N>>,
        outputs: impl IntoIterator<Item = Tensor<*const VirByte, N>>,
        stream: &Stream,
    ) {
        destruct!([y] = outputs);
        destruct!([x] = inputs);
        let Some(Arg::Str(op)) = arg else { panic!() };

        assert_eq!(y.dt(), x.dt());
        assert_eq!(y.shape(), x.shape());
        let y = y.transform(|layout| layout.merge_be(0, layout.ndim()).unwrap());
        let x = x.transform(|layout| layout.merge_be(0, layout.ndim()).unwrap());

        let dt = y.dt();
        let &[len] = y.shape() else { unreachable!() };
        assert_eq!(dt, types::F16);
        assert_eq!(op, "sum");

        let len = len * dt.nbytes();
        let dst = unsafe { std::slice::from_raw_parts_mut(y.take().cast_mut().cast(), len) };
        let src = unsafe { std::slice::from_raw_parts(x.take().cast(), len) };
        handle
            .comm
            .as_ref()
            .unwrap()
            .all_reduce(dst, Some(src), dt, ReduceType::hcclSum, stream);
    }
}
