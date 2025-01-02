#![cfg(any(use_nvidia, use_iluvatar))]

use common::{Contiguous, Slab};
use llama::{BlkWeight, LlamaBlkStorage, LlamaStorage, Tensor, WeightLoader};
use log::trace;
use operators::{
    all_reduce::{AllReduce, NonAllReduce},
    cuda::{memcpy_d2h, AsRaw, CurrentCtx, DevByte, DevMem, Event, Gpu, HostMem, Stream},
    random_sample::cuda::Operator as RandomSampleGpu,
    rearrange::cuda::Operator as Rearrange,
    ByteOf, QueueOf, TopoNode,
};
use std::{
    cell::{RefCell, RefMut},
    marker::PhantomData,
    mem::replace,
    ops::{Deref, RangeBounds},
    rc::Rc,
    time::Instant,
};

pub struct Operators<N = Gpu, R = NonAllReduce<Gpu, Rearrange>>(PhantomData<(N, R)>);

pub type RandomSample = llama::RandomSample<Gpu, RandomSampleGpu>;

pub struct Weights<'ctx> {
    nexp: usize,
    blks: LlamaBlkStorage<Cache<'ctx>>,
    output_norm: DevMem<'ctx>,
    output: DevMem<'ctx>,
}

pub enum Cache<'ctx> {
    Static(Box<[DevMem<'ctx>]>),
    Rolling {
        stream: Rc<Stream<'ctx>>,
        host: Box<[HostMem<'ctx>]>,
        dev: RefCell<RollCache<'ctx>>,
    },
}

pub struct RollCache<'ctx> {
    global_idx: usize,
    local_idx: usize,
    nblk: usize,
    cache: Box<[(DevMem<'ctx>, Event<'ctx>)]>,
}

impl<'ctx> RollCache<'ctx> {
    pub fn new(nblk: usize, cache: Box<[(DevMem<'ctx>, Event<'ctx>)]>) -> Self {
        Self {
            global_idx: 0,
            local_idx: 0,
            nblk,
            cache,
        }
    }

    pub fn first_event(&self) -> &Event<'ctx> {
        let (_, ref event) = self.cache[self.local_idx];
        event
    }
}

pub enum WeightResult<'s, 'ctx> {
    RollCached {
        roll_cache: RefMut<'s, RollCache<'ctx>>,
        load_stream: &'s Stream<'ctx>,
        host: &'s [HostMem<'ctx>],
        compute_stream: &'s Stream<'s>,
    },
    Borrowed(&'s [DevByte]),
}

impl Deref for WeightResult<'_, '_> {
    type Target = [DevByte];

    fn deref(&self) -> &Self::Target {
        match self {
            WeightResult::RollCached { roll_cache, .. } => {
                &roll_cache.cache[roll_cache.local_idx].0
            }
            WeightResult::Borrowed(dev_mem) => dev_mem,
        }
    }
}

impl Drop for WeightResult<'_, '_> {
    fn drop(&mut self) {
        match self {
            WeightResult::RollCached {
                roll_cache,
                load_stream,
                host,
                compute_stream,
            } => {
                // wait for the compute to finish
                load_stream.wait_for(&compute_stream.record());

                let next_load_idx =
                    (roll_cache.global_idx + roll_cache.cache.len()) % roll_cache.nblk;
                let host = &host[next_load_idx];

                roll_cache.global_idx = (roll_cache.global_idx + 1) % roll_cache.nblk;

                let start_idx = roll_cache.local_idx;
                let (dev_mem, event) = &mut roll_cache.cache[start_idx];
                assert!(dev_mem.len() == host.len());
                load_stream.memcpy_h2d(dev_mem, host);
                *event = load_stream.record();

                roll_cache.local_idx = (roll_cache.local_idx + 1) % roll_cache.cache.len();
            }
            WeightResult::Borrowed(_) => {}
        }
    }
}

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

impl<'blk> Weights<'blk> {
    pub fn new(
        model: &LlamaStorage<&'_ [u8]>,
        range: impl RangeBounds<usize> + Clone,
        count: usize,
        pool_size: usize,
        ctx: &'blk CurrentCtx,
    ) -> Self {
        assert!(pool_size > 0);
        let stream = Rc::new(ctx.stream());
        let igpu = unsafe { ctx.dev().as_raw() };
        let mut slab = Slab::new();
        let blks = if pool_size < model.meta.nblk {
            let mut blks_host = model.blocks[0]
                .as_ref()
                .map(|_| Vec::with_capacity(model.meta.nblk));
            for (iblk, blk) in model.blocks.iter().enumerate() {
                let time = Instant::now();
                let blk = blk
                    .distribute(&model.meta, range.clone(), count, |len| {
                        ctx.malloc_host::<u8>(len)
                    })
                    .map(|host| match host {
                        Contiguous::Borrowed(host) => {
                            let mut ans = ctx.malloc_host::<u8>(host.len());
                            ans.copy_from_slice(host);
                            ans
                        }
                        Contiguous::Owned(host) => host,
                    });

                macro_rules! push {
                    ($( $ident:ident )+ ) => {
                        $({ blks_host.$ident.push(blk.$ident); })+
                    };
                }
                push! {
                    attn_norm
                    attn_qkv
                    attn_o
                    ffn_norm
                    ffn_gate_up
                    ffn_down
                }
                trace!("blk{iblk} loaded to gpu{igpu} in {:?}", time.elapsed())
            }
            blks_host.map(|vec| {
                let roll_cache = vec
                    .iter()
                    .take(pool_size)
                    .map(|host| (ctx.from_host(host), stream.record()))
                    .collect::<Box<_>>();
                Cache::Rolling {
                    stream: stream.clone(),
                    host: vec.into_boxed_slice(),
                    dev: RefCell::new(RollCache::new(model.meta.nblk, roll_cache)),
                }
            })
        } else {
            let mut loader = None;
            let mut blks_dev = model.blocks[0]
                .as_ref()
                .map(|_| Vec::with_capacity(model.meta.nblk));
            for (iblk, blk) in model.blocks.iter().enumerate() {
                let blk = blk.distribute(&model.meta, range.clone(), count, |size| {
                    slab.take(&size)
                        .unwrap_or_else(|| ctx.malloc_host::<u8>(size))
                });
                let loader = loader
                    .get_or_insert_with(|| blk.as_ref().map(|s| H2DLoader::new(s.len(), &stream)));

                macro_rules! load {
                    ($( $ident:ident )+ ) => {
                        $(
                            let (dev, host) = loader.$ident.load(blk.$ident, &stream);
                            if let Some(host) = host {
                                slab.put(host.len(), host)
                            }
                            blks_dev.$ident.push(dev);
                        )+
                    };
                }
                let time = Instant::now();
                load! {
                    attn_norm
                    attn_qkv
                    attn_o
                    ffn_norm
                    ffn_gate_inp
                    ffn_gate_up
                    ffn_down
                }
                trace!("blk{iblk} loaded to gpu{igpu} in {:?}", time.elapsed())
            }
            blks_dev.map(|vec| Cache::Static(vec.into_boxed_slice()))
        };

        Self {
            nexp: model.meta.nexp,
            blks,
            output_norm: ctx.from_host(model.output_norm),
            output: ctx.from_host(model.output),
        }
    }
}

struct H2DLoader<'ctx> {
    event: Event<'ctx>,
    host: HostMem<'ctx>,
    dev: DevMem<'ctx>,
}

impl<'ctx> H2DLoader<'ctx> {
    fn new(size: usize, stream: &Stream<'ctx>) -> Self {
        Self {
            event: stream.record(),
            host: stream.ctx().malloc_host::<u8>(size),
            dev: stream.ctx().malloc::<u8>(size),
        }
    }

    fn load(
        &mut self,
        host: Contiguous<HostMem<'ctx>>,
        stream: &Stream<'ctx>,
    ) -> (DevMem<'ctx>, Option<HostMem<'ctx>>) {
        self.event.synchronize();
        let cache = match host {
            Contiguous::Borrowed(host) => {
                self.host.copy_from_slice(host);
                None
            }
            Contiguous::Owned(host) => Some(replace(&mut self.host, host)),
        };
        stream.memcpy_h2d(&mut self.dev, &self.host);
        self.event = stream.record();
        (
            replace(&mut self.dev, stream.ctx().malloc::<u8>(self.host.len())),
            cache,
        )
    }
}

impl<'ctx> WeightLoader for Weights<'ctx> {
    type Hardware = Gpu;
    type Weight<'s>
        = WeightResult<'s, 'ctx>
    where
        Self: 's;

    #[inline]
    fn load_blk<'s>(
        &'s self,
        which: BlkWeight,
        iblk: usize,
        queue: &'s QueueOf<Self::Hardware>,
    ) -> Self::Weight<'s> {
        let cache = match which {
            BlkWeight::AttnNorm => &self.blks.attn_norm,
            BlkWeight::AttnQKV => &self.blks.attn_qkv,
            BlkWeight::AttnO => &self.blks.attn_o,
            BlkWeight::FfnNorm => &self.blks.ffn_norm,
            BlkWeight::FfnGateInp => &self.blks.ffn_gate_inp,
            BlkWeight::FfnGateUp => &self.blks.ffn_gate_up,
            BlkWeight::FfnDown => &self.blks.ffn_down,
        };

        match cache {
            Cache::Static(dev) => WeightResult::Borrowed(&dev[iblk]),
            Cache::Rolling { stream, host, dev } => {
                let roll_cache = dev.borrow_mut();
                queue.wait_for(roll_cache.first_event());
                assert!(iblk == roll_cache.global_idx);
                WeightResult::RollCached {
                    roll_cache,
                    load_stream: stream,
                    host,
                    compute_stream: queue,
                }
            }
        }
    }

    fn load_moe<'a>(
        &'a self,
        which: BlkWeight,
        iblk: usize,
        iexp: usize,
        _queue: &'a QueueOf<Self::Hardware>,
    ) -> Self::Weight<'a> {
        let cache = match which {
            BlkWeight::FfnGateUp => &self.blks.ffn_gate_up,
            BlkWeight::FfnDown => &self.blks.ffn_down,
            _ => unreachable!(),
        };
        match cache {
            Cache::Static(dev) => {
                let w = &dev[iblk];
                let one = w.len() / self.nexp;
                WeightResult::Borrowed(&w[iexp * one..][..one])
            }
            Cache::Rolling { .. } => todo!(),
        }
    }

    #[inline]
    fn output_norm(&self, _queue: &QueueOf<Self::Hardware>) -> Self::Weight<'_> {
        WeightResult::Borrowed(&self.output_norm)
    }

    #[inline]
    fn output(&self, _queue: &QueueOf<Self::Hardware>) -> Self::Weight<'_> {
        WeightResult::Borrowed(&self.output)
    }
}

#[cfg(test)]
mod infer;

#[cfg(all(test, use_nccl))]
mod nccl_parallel;
