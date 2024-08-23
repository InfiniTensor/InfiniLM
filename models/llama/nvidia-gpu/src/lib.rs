#![cfg(detected_cuda)]

mod resource;

#[macro_use]
extern crate log;

use causal_lm::{
    CausalLM, ChatTemplate, DecodingMeta, FromGGuf, Model, QueryContext, SampleMeta, Tokenizer,
};
use common::{map_files, upos, utok, Blob, GGufModel};
use common_nv::{
    cuda::{memcpy_d2h, AsRaw, DevByte, DevMem, Stream},
    slice, udim, Gpu, Kernels, KernelsA as _, KernelsB as _, NvidiaKernels, Tensor,
};
use cuda::{
    ContextResource, ContextSpore, DevMemSpore, Device, EventSpore, HostMemSpore, StreamSpore,
};
use llama::{duplicate_cache, LlamaBlk, LlamaMeta, LlamaModel, QueueOf, SliceOn};
use resource::Resource;
use std::{
    cell::RefCell,
    collections::VecDeque,
    iter::repeat,
    mem::ManuallyDrop,
    ops::Deref,
    path::Path,
    rc::Rc,
    slice::{from_raw_parts, from_raw_parts_mut},
    sync::{Arc, Mutex, MutexGuard},
    time::Instant,
};

pub use common_nv::{cuda, synchronize};
pub use resource::Cache;

pub struct Transformer(ManuallyDrop<Internal>);

struct Internal {
    resource: Arc<Resource>,
    transfer: StreamSpore,
    kernels: NvidiaKernels,
    sample_workspace: DevMemSpore,

    meta: LlamaMeta,
    token_embed: HostMemSpore,
    output_norm: DevMemSpore,
    output: DevMemSpore,
    blocks: Box<[LlamaBlk<HostMemSpore>]>,
    pool: Mutex<VecDeque<(LlamaBlk<DevMemSpore>, EventSpore)>>,
}

pub struct ModelLoadMeta {
    pub device: Device,
    pub load_layers: usize,
}

impl ModelLoadMeta {
    #[inline]
    pub fn load_all_to(n: i32) -> Self {
        Self {
            device: Device::new(n),
            load_layers: usize::MAX,
        }
    }
}

impl Model for Transformer {
    type Config = ModelLoadMeta;
    type Error = ();

    #[inline]
    fn load(
        gguf: impl AsRef<Path>,
        Self::Config {
            device,
            load_layers,
        }: Self::Config,
    ) -> Result<FromGGuf<Self>, Self::Error> {
        let time = Instant::now();

        let _files = map_files(gguf);
        let gguf = GGufModel::read(_files.iter().map(|f| &**f));

        let tokenizer = Tokenizer::from_gguf(&gguf);
        let chat_template = ChatTemplate::from_gguf(&gguf, &tokenizer);
        let llama = LlamaModel::from_gguf(&gguf);
        let LlamaMeta {
            dt_norm,
            dt_mat,
            nblk,
            nh,
            dh,
            dvoc,
            ..
        } = llama.meta;

        info!("load host: {:?}", time.elapsed());

        let resource = Arc::new(Resource::new(&device));
        device.set_mempool_threshold(u64::MAX);

        // 异步编译 CUDA
        let kernels = std::thread::spawn(move || {
            info!("CUDA kernels compiling");
            let ans = NvidiaKernels::new(&[device], dt_norm, dt_mat, nh * dh, dvoc);
            info!("CUDA kernels compiled");
            ans
        });

        let model = resource.clone().apply(|compute| {
            let ctx = compute.ctx();
            let transfer = ctx.stream();

            let page_lock = |s: &[u8]| {
                let mut host = ctx.malloc_host::<u8>(s.len());
                host.copy_from_slice(s);
                host.sporulate()
            };
            let from_host = |u: &[u8]| transfer.from_host(u).sporulate();

            let token_embed = page_lock(llama.token_embed);
            let output_norm = from_host(llama.output_norm);
            let output = from_host(llama.output);
            let blocks = llama
                .blocks
                .iter()
                .map(|l| l.as_ref().map(|s| page_lock(s)))
                .collect::<Box<_>>();
            let pool = blocks
                .iter()
                .take(load_layers.min(nblk))
                .map(|l| {
                    (
                        l.as_ref().map(|s| from_host(s)),
                        transfer.record().sporulate(),
                    )
                })
                .collect();
            let kernels = kernels.join().unwrap();
            let sample_workspace = kernels.sample_workspace(compute).sporulate();

            Self(ManuallyDrop::new(Internal {
                resource,
                transfer: transfer.sporulate(),
                kernels,
                sample_workspace,

                meta: llama.meta,
                token_embed,
                output_norm,
                output,
                blocks,
                pool: Mutex::new(pool),
            }))
        });

        Ok(FromGGuf {
            model,
            tokenizer,
            chat_template,
        })
    }
}

impl Drop for Transformer {
    fn drop(&mut self) {
        let Internal {
            resource,
            transfer,
            sample_workspace,
            token_embed,
            output_norm,
            output,
            blocks,
            pool,
            ..
        } = unsafe { ManuallyDrop::take(&mut self.0) };
        resource.apply(|compute| {
            let ctx = compute.ctx();
            transfer.sprout(ctx);
            sample_workspace.sprout(ctx);
            token_embed.sprout(ctx);
            output_norm.sprout(ctx).drop_on(compute);
            output.sprout(ctx).drop_on(compute);
            for blk in blocks {
                blk.attn_norm.sprout(ctx);
                blk.attn_qkv.sprout(ctx);
                blk.attn_o.sprout(ctx);
                blk.ffn_norm.sprout(ctx);
                blk.ffn_gate_up.sprout(ctx);
                blk.ffn_down.sprout(ctx);
            }
            for (blk, event) in std::mem::take(&mut *pool.lock().unwrap()) {
                blk.attn_norm.sprout(ctx).drop_on(compute);
                blk.attn_qkv.sprout(ctx).drop_on(compute);
                blk.attn_o.sprout(ctx).drop_on(compute);
                blk.ffn_norm.sprout(ctx).drop_on(compute);
                blk.ffn_gate_up.sprout(ctx).drop_on(compute);
                blk.ffn_down.sprout(ctx).drop_on(compute);
                event.sprout(ctx);
            }
        });
    }
}

impl Transformer {
    #[inline]
    fn cache(&self, len: usize) -> Cache {
        Cache::new(&self.0.resource, len)
    }

    #[inline]
    fn tensor(&self, shape: &[udim]) -> Tensor<Cache> {
        Tensor::alloc(self.0.meta.dt_mat, shape, |len| self.cache(len))
    }
}

impl CausalLM for Transformer {
    type Storage = Cache;

    #[inline]
    fn max_seq_len(&self) -> upos {
        self.0.meta.dctx as _
    }
    #[inline]
    fn bos_token(&self) -> utok {
        1
    }
    #[inline]
    fn eos_token(&self) -> utok {
        2
    }

    fn new_cache(&self) -> Tensor<Self::Storage> {
        self.0.meta.new_cache(|len| self.cache(len))
    }

    fn duplicate_cache(&self, cache: &Tensor<Self::Storage>, pos: upos) -> Tensor<Self::Storage> {
        duplicate_cache(
            cache,
            pos,
            |len| self.cache(len),
            |dst, src| {
                self.0.resource.apply(|stream| {
                    let ctx = stream.ctx();
                    self.0.kernels.reform(
                        &mut dst.map_physical(|u| &mut **u.mem.sprout_mut(ctx)),
                        &src.map_physical(|u| &**u.mem.sprout_ref(ctx)),
                        stream,
                    );
                })
            },
        )
    }

    fn token_embed(&self, queries: impl IntoIterator<Item = utok>) -> Tensor<Self::Storage> {
        let LlamaMeta {
            dt_mat,
            nh,
            dh,
            dvoc,
            ..
        } = self.0.meta;
        let d = (nh * dh) as udim;
        let tokens = queries.into_iter().collect::<Vec<_>>();
        let nt = tokens.len() as udim;

        let mut x = self.tensor(&[nt, d]);
        let token_embed = Tensor::new(dt_mat, &[dvoc as _, d], &*self.0.token_embed);
        self.0.resource.apply(|compute| {
            self.0.kernels.gather(
                &mut x
                    .as_mut()
                    .map_physical(|u| &mut **u.mem.sprout_mut(compute.ctx())),
                &token_embed,
                tokens,
                compute,
            )
        });
        x
    }

    fn forward<'a>(
        &self,
        queries: impl IntoIterator<Item = QueryContext<'a, Self::Storage>>,
        token_embedded: Tensor<Self::Storage>,
    ) -> Tensor<Self::Storage>
    where
        Self: 'a,
    {
        self.0.resource.apply(|compute| {
            let ctx = compute.ctx();
            let transfer = self.0.transfer.sprout_ref(ctx);
            let stream = ComputeStream {
                meta: &self.0.meta,
                kernels: &self.0.kernels,
                compute,
                transfer,
                host: &self.0.blocks,
                dev: Rc::new(RefCell::new(self.0.pool.lock().unwrap())),
            };
            <ComputeStream as llama::ComputeStream>::forward(&stream, queries, token_embedded)
        })
    }

    fn decode(
        &self,
        decoding: impl IntoIterator<Item = DecodingMeta>,
        mut hidden_state: Tensor<Self::Storage>,
    ) -> Tensor<Self::Storage> {
        let LlamaMeta {
            dt_norm,
            dt_mat,
            nh,
            dh,
            dvoc,
            epsilon,
            ..
        } = self.0.meta;
        let d = (nh * dh) as udim;

        self.0.resource.apply(|compute| {
            let ctx = compute.ctx();
            let mut x = hidden_state
                .as_mut()
                .map_physical(|u| &mut **u.mem.sprout_mut(ctx));
            let range =
                DecodingMeta::select(&mut x, decoding, |dst, src| compute.memcpy_d2d(dst, src));
            if range.is_empty() {
                return self.tensor(&[0, d]);
            }

            let lm_layernorm = Tensor::new(dt_norm, &[d], &**self.0.output_norm.sprout_ref(ctx));
            let lm_head = Tensor::new(dt_mat, &[dvoc as _, d], &**self.0.output.sprout_ref(ctx))
                .transpose(&[1, 0]);

            let mut x = x.slice(&[slice![range.start => range.end], slice![=>]]);
            let mut logits = self.tensor(&[x.shape()[0], lm_head.shape()[1]]);

            // 复制一个 x 以实现原地归一化
            let x_ = x
                .as_ref()
                .map_physical(|u| unsafe { from_raw_parts(u.as_ptr(), u.len()) });
            self.0
                .kernels
                .rms_norm(&mut x, &x_, &lm_layernorm, epsilon, compute);
            self.0.kernels.mat_mul(
                &mut logits
                    .as_mut()
                    .map_physical(|u| &mut **u.mem.sprout_mut(ctx)),
                0.,
                &x,
                &lm_head,
                1.,
                compute,
            );

            logits
        })
    }

    fn sample(
        &self,
        args: impl IntoIterator<Item = SampleMeta>,
        logits: Tensor<Self::Storage>,
    ) -> Vec<utok> {
        let workspace_ptr = unsafe { self.0.sample_workspace.as_raw() };
        let workspace_len = self.0.sample_workspace.len();
        self.0.resource.apply(|compute| {
            let workspace =
                unsafe { from_raw_parts_mut(workspace_ptr as *mut DevByte, workspace_len) };
            self.0.kernels.sample(
                self.0.meta.dvoc as _,
                args.into_iter()
                    .flat_map(|meta| repeat(meta.args).take(meta.num_decode)),
                logits.take_physical().mem.sprout_ref(compute.ctx()),
                workspace,
                compute,
            )
        })
    }
}

struct ComputeStream<'a> {
    meta: &'a LlamaMeta,
    kernels: &'a NvidiaKernels,
    compute: &'a Stream<'a>,
    transfer: &'a Stream<'a>,
    host: &'a [LlamaBlk<HostMemSpore>],
    dev: DevMemPool<'a>,
}

type DevMemPool<'a> = Rc<RefCell<MutexGuard<'a, VecDeque<(LlamaBlk<DevMemSpore>, EventSpore)>>>>;

impl<'a> llama::ComputeStream for ComputeStream<'a> {
    type Handle = Gpu;
    type Storage = Cache;
    type Buf<'m> = DevMem<'m>;
    type Pos<'m> = DevMem<'m>;

    #[inline]
    fn malloc(&self, len: usize) -> Self::Buf<'_> {
        self.compute.malloc::<u8>(len)
    }
    #[inline]
    fn free(&self, mem: Self::Buf<'_>) {
        mem.drop_on(self.compute);
    }
    #[inline]
    fn map_pos<'b>(&self, pos: &'b [u32]) -> Self::Pos<'b>
    where
        Self: 'b,
    {
        self.compute.from_host(pos)
    }
    #[inline]
    fn free_pos(&self, mem: Self::Pos<'_>) {
        mem.drop_on(self.compute);
    }
    #[inline]
    fn map_storage<'b>(&'b self, storage: &'b mut Self::Storage) -> &'b mut SliceOn<Self::Handle> {
        storage.mem.sprout_mut(self.compute.ctx())
    }
    #[inline]
    fn meta(&self) -> &LlamaMeta {
        self.meta
    }
    #[inline]
    fn kernels(&self) -> &impl Kernels<Self::Handle> {
        self.kernels
    }
    #[inline]
    fn queue(&self) -> &QueueOf<Self::Handle> {
        self.compute
    }

    fn debug<T>(tensor: &Tensor<T>)
    where
        T: Deref<Target = SliceOn<Self::Handle>>,
    {
        println!(
            "{}",
            tensor.as_ref().map_physical(|s| {
                let mut host = Blob::new(s.len());
                memcpy_d2h(&mut host, s);
                host
            })
        );
    }

    fn layers(
        &self,
    ) -> impl Iterator<Item = impl llama::LLamaLayer<Byte = <Self::Handle as llama::Handle>::Byte>>
    {
        Iter {
            meta: self.meta,
            host: self.host,
            pool: self.dev.clone(),
            compute: self.compute,
            transfer: self.transfer,
            layer: 0,
        }
    }
}

struct Iter<'a> {
    meta: &'a LlamaMeta,
    host: &'a [LlamaBlk<HostMemSpore>],
    pool: DevMemPool<'a>,
    compute: &'a Stream<'a>,
    transfer: &'a Stream<'a>,
    layer: usize,
}

impl<'a> Iterator for Iter<'a> {
    type Item = LayerLoader<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.layer >= self.host.len() {
            return None;
        }

        let mut pool = self.pool.borrow_mut();
        let load = if pool.len() < self.host.len() {
            Some((self.layer + pool.len()) % self.host.len())
        } else {
            None
        };
        self.layer += 1;

        let (s, event) = pool.pop_front().unwrap();
        let ctx = self.compute.ctx();
        self.compute.wait_for(&event.sprout(ctx));

        Some(Self::Item {
            meta: self.meta,
            host: self.host,
            pool: self.pool.clone(),
            load,
            transfer: self.transfer,
            storage: Some(s),
        })
    }
}

struct LayerLoader<'a> {
    meta: &'a LlamaMeta,
    host: &'a [LlamaBlk<HostMemSpore>],
    pool: DevMemPool<'a>,
    load: Option<usize>,
    transfer: &'a Stream<'a>,
    storage: Option<LlamaBlk<DevMemSpore>>,
}

macro_rules! access {
    ($self:expr, $name:ident) => {
        &**$self
            .storage
            .as_ref()
            .unwrap()
            .$name
            .sprout_ref($self.transfer.ctx())
    };
}
impl<'a> llama::LLamaLayer for LayerLoader<'a> {
    type Byte = DevByte;
    type Storage<'m> = &'m[DevByte] where Self: 'm;

    #[inline]
    fn att_layernorm(&self) -> Tensor<Self::Storage<'_>> {
        let &LlamaMeta {
            dt_norm, nh, dh, ..
        } = self.meta;
        Tensor::new(dt_norm, &[(nh * dh) as _], access!(self, attn_norm))
    }
    #[inline]
    fn att_qkv(&self) -> Tensor<Self::Storage<'_>> {
        let &LlamaMeta {
            dt_mat,
            nh,
            nkvh,
            dh,
            ..
        } = self.meta;
        Tensor::new(
            dt_mat,
            &[((nh + nkvh + nkvh) * dh) as _, (nh * dh) as _],
            access!(self, attn_qkv),
        )
        .transpose(&[1, 0])
    }
    #[inline]
    fn att_o(&self) -> Tensor<Self::Storage<'_>> {
        let &LlamaMeta { dt_mat, nh, dh, .. } = self.meta;
        Tensor::new(
            dt_mat,
            &[(nh * dh) as _, (nh * dh) as _],
            access!(self, attn_o),
        )
        .transpose(&[1, 0])
    }
    #[inline]
    fn mlp_layernorm(&self) -> Tensor<Self::Storage<'_>> {
        let &LlamaMeta {
            dt_norm, nh, dh, ..
        } = self.meta;
        Tensor::new(dt_norm, &[(nh * dh) as _], access!(self, ffn_norm))
    }
    #[inline]
    fn mlp_gate_up(&self) -> Tensor<Self::Storage<'_>> {
        let &LlamaMeta {
            dt_mat, nh, dh, di, ..
        } = self.meta;
        Tensor::new(
            dt_mat,
            &[(di + di) as _, (nh * dh) as _],
            access!(self, ffn_gate_up),
        )
        .transpose(&[1, 0])
    }
    #[inline]
    fn mlp_down(&self) -> Tensor<Self::Storage<'_>> {
        let &LlamaMeta {
            dt_mat, nh, dh, di, ..
        } = self.meta;
        Tensor::new(dt_mat, &[(nh * dh) as _, di as _], access!(self, ffn_down)).transpose(&[1, 0])
    }
}

impl Drop for LayerLoader<'_> {
    fn drop(&mut self) {
        let mut lll = self.storage.take().unwrap();
        if let Some(load) = self.load {
            macro_rules! exchange {
                ($($name:ident)+) => {
                    $(
                        let host = &self.host[load].$name;
                        let mut dev = lll.$name.sprout_mut(self.transfer.ctx());
                        self.transfer.memcpy_h2d(&mut dev, host);
                    )+
                };
            }
            exchange! {
                attn_norm
                attn_qkv
                attn_o
                ffn_norm
                ffn_gate_up
                ffn_down
            }
        }
        self.pool
            .borrow_mut()
            .push_back((lll, self.transfer.record().sporulate()));
    }
}

#[test]
fn test_infer() {
    if let Err(cuda::NoDevice) = cuda::init() {
        return;
    }
    let device = cuda::Device::new(0);
    causal_lm::test_impl::<Transformer>(
        ModelLoadMeta {
            device,
            load_layers: 20,
        },
        100,
        "Once upon a time,",
    );
}
