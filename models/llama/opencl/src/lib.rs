#![cfg(detected)]

use common::{Distribution, WeightMemCalculator};
use llama::{LlamaBlkStorage, LlamaBlkWeight, LlamaStorage, Tensor, WeightLoader};
use operators::{
    all_reduce::{AllReduce, NonAllReduce},
    clrt::{Context, SvmBlob, SvmByte},
    opencl::ClDevice,
    random_sample::opencl::Operator as RandomSampleCl,
    rearrange::opencl::Operator as Rearrange,
    Blob, ByteOf, QueueOf, TopoNode,
};
use std::{
    iter::zip,
    marker::PhantomData,
    ops::{Deref, Range},
    ptr::copy_nonoverlapping,
};

pub struct Operators<N = ClDevice, R = NonAllReduce<ClDevice, Rearrange>>(PhantomData<(N, R)>);

pub type RandomSample = llama::RandomSample<ClDevice, RandomSampleCl>;

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

pub struct Weights {
    nexp: usize,
    mem: SvmBlob,
    blks: Box<[LlamaBlkStorage<Range<usize>>]>,
    output_norm: Range<usize>,
    output: Range<usize>,
}

impl Weights {
    pub fn new(model: &LlamaStorage<&[u8]>, dist: Distribution, ctx: &Context) -> Self {
        let LlamaStorage {
            meta,
            output_norm,
            output,
            blocks,
            ..
        } = model;

        let mut calculator = WeightMemCalculator::new(size_of::<usize>());
        let meta_dist = meta.distribute(dist);
        let blk_size = meta_dist.blk();
        let off_blks = (0..meta_dist.nblk)
            .map(|_| {
                blk_size
                    .clone()
                    .into_vec()
                    .into_iter()
                    .map(|(which, size)| (which, calculator.push(size)))
                    .collect::<LlamaBlkStorage<_>>()
            })
            .collect::<Vec<_>>();
        let off_output_norm = calculator.push(output_norm.len());
        let off_output = calculator.push(output.len());

        let mut mem = ctx.malloc::<u8>(calculator.size());
        let queue = ctx.queue();

        for (blk, off) in zip(blocks, off_blks.clone()) {
            let blk = blk.clone().into_vec();
            let off = off.into_vec();
            for ((which, data), (which_, off)) in zip(blk, off) {
                assert_eq!(which, which_);
                if off.is_empty() {
                    continue;
                }
                let data = meta.distribute_data(which, data, dist, Blob::new);
                let mut map = queue.map_mut(&mut mem[off], false);
                map.copy_from_slice(&data);
                queue.unmap(map)
            }
        }
        let mut map = queue.map_mut(&mut mem[off_output_norm.clone()], false);
        map.copy_from_slice(output_norm);
        queue.unmap(map);
        let mut map = queue.map_mut(&mut mem[off_output.clone()], false);
        map.copy_from_slice(output);
        queue.unmap(map);

        Self {
            nexp: meta.nexp,
            mem,
            blks: off_blks.into_boxed_slice(),
            output_norm: off_output_norm,
            output: off_output,
        }
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
        which: LlamaBlkWeight,
        iblk: usize,
        _queue: &QueueOf<Self::Hardware>,
    ) -> Self::Weight<'_> {
        let LlamaBlkStorage {
            attn_norm,
            attn_qkv,
            attn_qkv_bias,
            attn_o,
            ffn_norm,
            ffn_gate_inp,
            ffn_gate_up,
            ffn_down,
        } = &self.blks[iblk];

        use LlamaBlkWeight as W;
        #[rustfmt::skip]
        let range = match which {
            W::AttnNorm    => attn_norm    ,
            W::AttnQKV     => attn_qkv     ,
            W::AttnQKVBias => attn_qkv_bias,
            W::AttnO       => attn_o       ,
            W::FfnNorm     => ffn_norm     ,
            W::FfnGateInp  => ffn_gate_inp ,
            W::FfnGateUp   => ffn_gate_up  ,
            W::FfnDown     => ffn_down     ,
        };
        &self.mem[range.clone()]
    }

    fn load_moe<'a>(
        &'a self,
        which: LlamaBlkWeight,
        iblk: usize,
        iexp: usize,
        _queue: &'a QueueOf<Self::Hardware>,
    ) -> Self::Weight<'a> {
        let LlamaBlkStorage {
            ffn_gate_up,
            ffn_down,
            ..
        } = &self.blks[iblk];

        let range = match which {
            LlamaBlkWeight::FfnGateUp => ffn_gate_up,
            LlamaBlkWeight::FfnDown => ffn_down,
            _ => unreachable!(),
        };
        let w = &self.mem[range.clone()];
        let one = w.len() / self.nexp;
        &w[iexp * one..][..one]
    }

    #[inline]
    fn output_norm(&self, _queue: &QueueOf<Self::Hardware>) -> Self::Weight<'_> {
        &self.mem[self.output_norm.clone()]
    }

    #[inline]
    fn output(&self, _queue: &QueueOf<Self::Hardware>) -> Self::Weight<'_> {
        &self.mem[self.output.clone()]
    }
}

#[cfg(test)]
mod infer;
