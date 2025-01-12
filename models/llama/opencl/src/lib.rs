#![cfg(detected)]

use common::Contiguous;
use llama::{BlkWeight, LlamaStorage, Tensor, WeightLoader};
use operators::{
    all_reduce::{AllReduce, NonAllReduce},
    clrt::{CommandQueue, Invalid, SvmBlob, SvmByte},
    opencl::ClDevice,
    random_sample::opencl::Operator as RandomSampleCl,
    rearrange::opencl::Operator as Rearrange,
    ByteOf, QueueOf, TopoNode,
};
use std::{
    marker::PhantomData,
    ops::{Deref, RangeBounds},
    ptr::copy_nonoverlapping,
};

pub struct Operators<N = ClDevice, R = NonAllReduce<ClDevice, Rearrange>>(PhantomData<(N, R)>);

pub type RandomSample = llama::RandomSample<ClDevice, RandomSampleCl>;

#[repr(transparent)]
pub struct Weights(LlamaStorage<SvmBlob>);

macro_rules! op {
    ($name:ident) => {
        operators::$name::opencl::Operator
    };
}

impl<N, R> llama::Operators for Operators<N, R>
where
    N: TopoNode<ClDevice>,
    R: AllReduce<ClDevice, N>,
{
    type Hardware = ClDevice;
    type TopoNode = N;
    type RmsNorm = op!(rms_norm);
    type MatMul = op!(mat_mul);
    type Rope = op!(rope);
    type AttnKVCached = op!(attention_kv_cached);
    type Swiglu = op!(swiglu);
    type Rearrange = op!(rearrange);
    type AllReduce = R;

    fn debug<T>(tensor: &Tensor<T>, queue: &QueueOf<Self::Hardware>)
    where
        T: Deref<Target = [ByteOf<Self::Hardware>]>,
    {
        let tensor = tensor.as_ref().map(|s| queue.map(s));
        println!("{tensor}");
        queue.unmap(tensor.take())
    }

    fn memcpy_d2h<T: Copy>(
        dst: &mut [T],
        src: &[ByteOf<Self::Hardware>],
        queue: &QueueOf<Self::Hardware>,
    ) {
        assert_eq!(size_of_val(dst), size_of_val(src));
        let svm = queue.map(src);
        unsafe { copy_nonoverlapping(svm.as_ptr(), dst.as_mut_ptr().cast::<u8>(), dst.len()) }
        queue.unmap(svm)
    }
}

impl Weights {
    pub fn new(
        model: &LlamaStorage<&'_ [u8]>,
        range: impl RangeBounds<usize> + Clone,
        count: usize,
        queue: &CommandQueue,
    ) -> Self {
        let mut meta = model.meta.clone();
        meta.distribute(range.clone(), count);

        let ctx = queue.ctx();
        let from_host = |s: &[u8]| {
            let mut blob = ctx.malloc::<u8>(s.len());
            let mut mem = queue.map_mut(&mut blob, Invalid);
            unsafe { mem.write_only_slice().copy_from_slice(s) };
            queue.unmap(mem);
            blob
        };

        Self(LlamaStorage {
            meta,
            token_embd: from_host(model.token_embd),
            output_norm: from_host(model.output_norm),
            output: from_host(model.output),
            blocks: model
                .blocks
                .iter()
                .map(|blk| {
                    blk.distribute(&model.meta, range.clone(), count, |size| {
                        queue.map_blob(queue.ctx().malloc::<u8>(size))
                    })
                })
                .map(|blk| {
                    blk.map(|c| match c {
                        Contiguous::Borrowed(s) => from_host(s),
                        Contiguous::Owned(m) => queue.unmap_blob(m),
                    })
                })
                .collect(),
        })
    }
}

impl WeightLoader for Weights {
    type Hardware = ClDevice;
    type Weight<'s>
        = &'s [SvmByte]
    where
        Self: 's;

    #[inline]
    fn load_blk(
        &self,
        which: BlkWeight,
        iblk: usize,
        _queue: &QueueOf<Self::Hardware>,
    ) -> Self::Weight<'_> {
        let blk = &self.0.blocks[iblk];
        match which {
            BlkWeight::AttnNorm => &blk.attn_norm,
            BlkWeight::AttnQKV => &blk.attn_qkv,
            BlkWeight::AttnO => &blk.attn_o,
            BlkWeight::FfnNorm => &blk.ffn_norm,
            BlkWeight::FfnGateInp => &blk.ffn_gate_inp,
            BlkWeight::FfnGateUp => &blk.ffn_gate_up,
            BlkWeight::FfnDown => &blk.ffn_down,
        }
    }

    fn load_moe<'a>(
        &'a self,
        which: BlkWeight,
        iblk: usize,
        _iexp: usize,
        _queue: &'a QueueOf<Self::Hardware>,
    ) -> Self::Weight<'a> {
        let _blk = &self.0.blocks[iblk];
        match which {
            BlkWeight::FfnGateUp | BlkWeight::FfnDown => todo!(),
            _ => unreachable!(),
        }
    }

    #[inline]
    fn output_norm(&self, _queue: &QueueOf<Self::Hardware>) -> Self::Weight<'_> {
        &self.0.output_norm
    }

    #[inline]
    fn output(&self, _queue: &QueueOf<Self::Hardware>) -> Self::Weight<'_> {
        &self.0.output
    }
}

#[cfg(test)]
mod infer;
