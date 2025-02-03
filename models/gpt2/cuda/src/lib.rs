#![cfg(any(use_nvidia, use_iluvatar))]
use common::Contiguous;
use common::{Distribution, Slab, WeightMemCalculator};
use gpt2::{GPT2BlkStorage, GPT2Storage, Tensor, WeightLoader};
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
    ops::{Deref, Range},
};
use std::{marker::PhantomData, time::Instant};

pub struct Operators<N = Gpu, R = NonAllReduce<Gpu, Rearrange>>(PhantomData<(N, R)>);

pub type RandomSample = gpt2::RandomSample<Gpu, RandomSampleGpu>;

macro_rules! op {
    ($name:ident) => {
        operators::$name::cuda::Operator
    };
}

impl<N, R> gpt2::Operators for Operators<N, R>
where
    N: TopoNode<Gpu>,
    R: AllReduce<Gpu, N>,
{
    type Hardware = Gpu;
    type TopoNode = N;
    type AddRows = op!(add_rows);
    type LayerNorm = op!(layer_norm);
    type MatMul = op!(mat_mul);
    type AttnKVCached = op!(attention_kv_cached);
    type Gelu = op!(gelu);
    type Add = op!(add);
    type Rearrange = op!(rearrange);
    type AllReduce = R;

    fn debug<T>(tensor: &Tensor<T>)
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
    mem: DevMem<'ctx>,
    blks: Box<[GPT2BlkStorage<Range<usize>>]>,
    output_norm_w: Range<usize>,
    output_norm_b: Range<usize>,
    pos_embd: Range<usize>,
    output: Range<usize>,
}

impl<'ctx> Weights<'ctx> {
    pub fn new(model: &GPT2Storage<&[u8]>, dist: Distribution, ctx: &'ctx CurrentCtx) -> Self {
        let GPT2Storage {
            meta,
            output,
            blocks,
            token_embd,
            pos_embd,
            output_norm_b,
            output_norm_w,
        } = model;

        let mut calculator = WeightMemCalculator::new(ctx.dev().alignment());
        let meta_dist = meta.distribute(dist);
        let blk_size = meta_dist.blk();
        let off_blks = (0..meta_dist.nblk)
            .map(|_| {
                blk_size
                    .clone()
                    .into_vec()
                    .into_iter()
                    .map(|(which, size)| (which, calculator.push(size)))
                    .collect::<GPT2BlkStorage<_>>()
            })
            .collect::<Vec<_>>();
        let off_token_embd = calculator.push(token_embd.len());
        let off_pos_embd = calculator.push(pos_embd.len());
        let off_output_norm_b = calculator.push(output_norm_b.len());
        let off_output_norm_w = calculator.push(output_norm_w.len());
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
        stream.memcpy_h2d(&mut mem[off_token_embd.clone()], token_embd);
        stream.memcpy_h2d(&mut mem[off_pos_embd.clone()], pos_embd);
        stream.memcpy_h2d(&mut mem[off_output_norm_b.clone()], output_norm_b);
        stream.memcpy_h2d(&mut mem[off_output_norm_w.clone()], output_norm_w);
        stream.memcpy_h2d(&mut mem[off_output.clone()], output);

        Self {
            mem,
            blks: off_blks.into_boxed_slice(),
            output_norm_w: off_output_norm_w,
            output_norm_b: off_output_norm_b,
            pos_embd: off_pos_embd,
            output: off_output,
        }
    }
}

impl WeightLoader for Weights<'_> {
    type Hardware = Gpu;

    type Memory<'s>
        = &'s [DevByte]
    where
        Self: 's;

    fn load_blk(
        &self,
        which: gpt2::BlkWeight,
        iblk: usize,
        _queue: &QueueOf<Self::Hardware>,
    ) -> [Self::Memory<'_>; 2] {
        let off = &self.blks[iblk];
        use gpt2::BlkWeight;
        #[rustfmt::skip]
        let [w,b] = match which {
            BlkWeight::AttnNorm =>  [&off.attn_norm_w, &off.attn_norm_b],
            BlkWeight::AttnQKV  =>  [&off.attn_qkv_w, &off.attn_qkv_b],
            BlkWeight::AttnO    =>  [&off.attn_o_w, &off.attn_o_b],
            BlkWeight::FfnNorm  =>  [&off.ffn_norm_w, &off.ffn_norm_b],
            BlkWeight::FfnUp    =>  [&off.ffn_up_w, &off.ffn_up_b],
            BlkWeight::FfnDown  =>  [&off.ffn_down_w, &off.ffn_down_b],
        };
        [&self.mem[w.clone()], &self.mem[b.clone()]]
    }

    fn output_norm(&self, _queue: &QueueOf<Self::Hardware>) -> [Self::Memory<'_>; 2] {
        [
            &self.mem[self.output_norm_w.clone()],
            &self.mem[self.output_norm_b.clone()],
        ]
    }

    fn output(&self, _queue: &QueueOf<Self::Hardware>) -> Self::Memory<'_> {
        &self.mem[self.output.clone()]
    }

    fn pos_embd<'a>(&'a self, _queue: &'a QueueOf<Self::Hardware>) -> Self::Memory<'a> {
        &self.mem[self.pos_embd.clone()]
    }
}

#[cfg(test)]
mod infer;
