#![cfg(any(use_nvidia, use_iluvatar))]

use common::{Contiguous, Distribution, Slab, WeightMemCalculator};
use llama::{LlamaBlkStorage, LlamaBlkWeight, LlamaStorage, Tensor, WeightLoader};
use log::trace;
use operators::{
    all_reduce::{AllReduce, NonAllReduce},
    cuda::{memcpy_d2h, CurrentCtx, DevByte, DevMem, Gpu},
    random_sample::cuda::Operator as RandomSampleGpu,
    rearrange::cuda::Operator as Rearrange,
    ByteOf, QueueOf, TopoNode,
};
use std::{
    collections::VecDeque,
    iter::zip,
    marker::PhantomData,
    ops::{Deref, Range},
    time::Instant,
};

pub struct Operators<N = Gpu, R = NonAllReduce<Gpu, Rearrange>>(PhantomData<(N, R)>);

pub type RandomSample = llama::RandomSample<Gpu, RandomSampleGpu>;

macro_rules! op {
    ($name:ident) => {
        operators::$name::cuda::Operator
    };
}

impl<N, R> llama::Operators for Operators<N, R>
where
    N: TopoNode<Gpu>,
    R: AllReduce<Gpu, N>,
{
    type Hardware = Gpu;
    type TopoNode = N;
    type RmsNorm = op!(rms_norm);
    type MatMul = op!(mat_mul);
    type Rope = op!(rope);
    type AttnKVCached = op!(attention_kv_cached);
    type Swiglu = op!(swiglu);
    type Rearrange = op!(rearrange);
    type AllReduce = R;

    fn debug<T>(tensor: &Tensor<T>, _queue: &QueueOf<Self::Hardware>)
    where
        T: Deref<Target = [ByteOf<Self::Hardware>]>,
    {
        let tensor = tensor.as_ref().map(|s| {
            let mut host = vec![0u8; s.len()];
            memcpy_d2h(&mut host, s);
            host
        });
        println!("{tensor}")
    }

    fn memcpy_d2h<T: Copy>(
        dst: &mut [T],
        src: &[ByteOf<Self::Hardware>],
        _queue: &QueueOf<Self::Hardware>,
    ) {
        memcpy_d2h(dst, src)
    }
}

pub struct Weights<'ctx> {
    nexp: usize,
    mem: DevMem<'ctx>,
    blks: Box<[LlamaBlkStorage<Range<usize>>]>,
    output_norm: Range<usize>,
    output: Range<usize>,
}

impl<'ctx> Weights<'ctx> {
    pub fn new(model: &LlamaStorage<&[u8]>, dist: Distribution, ctx: &'ctx CurrentCtx) -> Self {
        let LlamaStorage {
            meta,
            output_norm,
            output,
            blocks,
            ..
        } = model;

        let align = ctx.dev().alignment();
        let mut calculator = WeightMemCalculator::new(if align != 0 { align } else { 1024 });
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
        let mut slab = Slab::<usize, _>::new();
        let mut queue = VecDeque::new();
        let stream = ctx.stream();

        macro_rules! host {
            ($l:expr) => {
                slab.take(&$l).unwrap_or_else(|| ctx.malloc_host::<u8>($l))
            };
        }

        for (blk, off) in zip(blocks, off_blks.clone()) {
            let blk = blk.clone().into_vec();
            let off = off.into_vec();
            for ((which, data), (which_, off)) in zip(blk, off) {
                assert_eq!(which, which_);
                if off.is_empty() {
                    continue;
                }
                let data = meta.distribute_data(which, data, dist, |l| host!(l));
                let data = match data {
                    Contiguous::Borrowed(data) => {
                        let mut mem = host!(data.len());
                        mem.copy_from_slice(data);
                        mem
                    }
                    Contiguous::Owned(data) => data,
                };
                stream.memcpy_h2d(&mut mem[off], &data);
                queue.push_back((stream.record(), Instant::now(), data))
            }
            while let Some((event, _, _)) = queue.front() {
                if event.is_complete() {
                    let (_, time, data) = queue.pop_front().unwrap();
                    trace!("{:>16}bytes copied in {:?}", data.len(), time.elapsed());
                    slab.put(data.len(), data)
                } else {
                    break;
                }
            }
        }
        let mut host_ = ctx.malloc_host::<u8>(output_norm.len());
        host_.copy_from_slice(output_norm);
        stream.memcpy_h2d(&mut mem[off_output_norm.clone()], &host_);
        let mut host_ = ctx.malloc_host::<u8>(output.len());
        host_.copy_from_slice(output);
        stream.memcpy_h2d(&mut mem[off_output.clone()], &host_);
        stream.synchronize();

        Self {
            nexp: meta.nexp,
            mem,
            blks: off_blks.into_boxed_slice(),
            output_norm: off_output_norm,
            output: off_output,
        }
    }
}

impl WeightLoader for Weights<'_> {
    type Hardware = Gpu;

    type Weight<'s>
        = &'s [DevByte]
    where
        Self: 's;

    fn load_blk<'a>(
        &'a self,
        which: LlamaBlkWeight,
        iblk: usize,
        _queue: &'a QueueOf<Self::Hardware>,
    ) -> Self::Weight<'a> {
        let off = &self.blks[iblk];
        use LlamaBlkWeight as W;
        #[rustfmt::skip]
        let off = match which {
            W::AttnNorm    => &off.attn_norm    ,
            W::AttnQKV     => &off.attn_qkv     ,
            W::AttnQKVBias => &off.attn_qkv_bias,
            W::AttnO       => &off.attn_o       ,
            W::FfnNorm     => &off.ffn_norm     ,
            W::FfnGateInp  => &off.ffn_gate_inp ,
            W::FfnGateUp   => &off.ffn_gate_up  ,
            W::FfnDown     => &off.ffn_down     ,
        };
        &self.mem[off.clone()]
    }

    fn load_moe<'a>(
        &'a self,
        which: LlamaBlkWeight,
        iblk: usize,
        iexp: usize,
        _queue: &'a QueueOf<Self::Hardware>,
    ) -> Self::Weight<'a> {
        let off = &self.blks[iblk];
        use LlamaBlkWeight as W;
        #[rustfmt::skip]
        let off = match which {
            W::FfnGateUp => &off.ffn_gate_up,
            W::FfnDown   => &off.ffn_down   ,
            _            => unreachable!()  ,
        };
        let w = &self.mem[off.clone()];
        let one = w.len() / self.nexp;
        &w[iexp * one..][..one]
    }

    fn output_norm<'a>(&'a self, _queue: &'a QueueOf<Self::Hardware>) -> Self::Weight<'a> {
        &self.mem[self.output_norm.clone()]
    }

    fn output<'a>(&'a self, _queue: &'a QueueOf<Self::Hardware>) -> Self::Weight<'a> {
        &self.mem[self.output.clone()]
    }
}

#[cfg(test)]
mod infer;

#[cfg(test)]
mod web;

#[cfg(all(test, use_nccl))]
mod nccl_parallel;
