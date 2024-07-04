#![cfg(detected_neuware)]

mod resource;

use causal_lm::{CausalLM, DecodingMeta, Model, QueryContext, SampleMeta};
use common::{upos, utok, Blob, FileLoadError};
use common_cn::{sample_cpu, slice, KernelsA, KernelsB, cndrv::{memcpy_d2h, Context, ContextResource, ContextSpore, CurrentCtx, DevByte, DevMem, DevMemSpore, Device, HostMemSpore, Queue, QueueSpore}, rustTensor as Tensor, CambriconKernels, DeviceEnum, Kernels, Mlu};
use std::{iter::repeat, mem::ManuallyDrop, ops::Deref, path::Path, slice::from_raw_parts, sync::Arc};
use llama::{ComputeConst, InferenceConfig, LayerStorage, SliceOn, Weight};

use digit_layout::types::F16;
pub use common_cn::{cndrv, synchronize};
pub use resource::Cache;

pub struct Transformer(ManuallyDrop<Internal>);
pub struct Internal {
    config: InferenceConfig,

    resource: Arc<Context>,
    compute: QueueSpore,
    kernels: CambriconKernels,

    embed_tokens: Tensor<HostMemSpore>,
    layers: Vec<LayerStorage<DevMemSpore>>,
    lm_layernorm: Tensor<DevMemSpore>,
    lm_head: Tensor<DevMemSpore>,
}

impl Model for Transformer {
    type Meta = Device;
    type Error = FileLoadError;

    #[inline]
    fn load(model_dir: impl AsRef<Path>, meta: Self::Meta ) -> Result<Self, Self::Error> {
        // let time = Instant::now();
        let host = llama::Storage::load_safetensors(model_dir)?;
        // info!("load host: {:?}", time.elapsed());
        let resource = Arc::new(meta.acquire_shared());
        resource.apply(|ctx| {
            let page_lock = |u: &Weight| {
                let mut host = ctx.malloc_host::<u8>(u.len());
                host.copy_from_slice(u);
                host.sporulate()
            };

            Ok(Self (ManuallyDrop::new(Internal {
                resource: resource.clone(),
                compute: ctx.queue().sporulate(),
                kernels: CambriconKernels::new(
                    DeviceEnum::DevCambriconMlu
                ),
                embed_tokens: host
                    .embed_tokens
                    .as_ref()
                    .map_physical(page_lock),
                layers: host
                .layers
                    .iter()
                    .map(|l| l.map(|u| ctx.from_host(&u).sporulate()))
                    .collect::<Vec<_>>(),
                lm_layernorm: host
                    .lm_layernorm
                    .map_physical(|u| ctx.from_host(&u).sporulate()),
                lm_head: host
                    .lm_head
                    .map_physical(|u| ctx.from_host(&u).sporulate()),
    
                config: host.config,
            })   ))     
        })

    }
}

impl Transformer {
    #[inline]
    fn cache(&self, len: usize) -> Cache {
        Cache::new(&self.0.resource, len)
    }

    #[inline]
    fn tensor(&self, shape: &[u32]) -> Tensor<Cache> {
        Tensor::alloc(self.0.config.dt, shape, |len| self.cache(len))
    }
}

impl CausalLM for Transformer {
    type Storage = Cache;

    #[inline]
    fn max_seq_len(&self) -> upos {
        self.0.config.max_seq_len
    }

    #[inline]
    fn eos_token(&self) -> utok {
        self.0.config.eos_token
    }

    fn new_cache(&self) -> Tensor<Self::Storage> {
        self.0.config.new_cache(|len| self.cache(len))
    }

    fn duplicate_cache(&self, cache: &Tensor<Self::Storage>, pos: upos) -> Tensor<Self::Storage> {
        self.0.config.duplicate_cache(
            cache,
            pos,
            |len| self.cache(len),
            |dst, src| {
                self.0.resource.apply(|ctx| {
                    self.0.kernels.reform(
                        &mut dst.map_physical(|u| &mut **u.mem.sprout_mut(ctx)),
                        &src.map_physical(|u| &**u.mem.sprout_ref(ctx)),
                        &ctx.queue(),
                    );
                })
            },
        )        
    }

    fn token_embed(&self, queries: impl IntoIterator<Item = utok>) -> Tensor<Self::Storage> {
        let tokens = queries.into_iter().collect::<Vec<_>>();
        let nt = tokens.len() as u32;
        let d = self.0.config.d;

        let mut x = self.tensor(&[nt, d]);
        self.0.resource.apply(|ctx| {
            self.0.kernels.gather(
                &mut x
                    .as_mut()
                    .map_physical(|u| &mut **u.mem.sprout_mut(ctx)),
                &self.0.embed_tokens.as_ref().map_physical(|u| &**u),
                tokens,
                &ctx.queue(),
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
        self.0.resource.apply(|ctx| {
            let stream = ComputeStream {
                nh: self.0.config.nh,
                nkvh: self.0.config.nkvh,
                di: self.0.config.di,
                epsilon: self.0.config.epsilon,
                theta: self.0.config.theta,
                kernels: &self.0.kernels,
                compute: self.0.compute.sprout_ref(ctx),
                layers: &self.0.layers,
            };
            <ComputeStream as llama::ComputeStream>::forward(&stream, queries, token_embedded)
        })
    }

    fn decode(
        &self,
        decoding: impl IntoIterator<Item = DecodingMeta>,
        mut hidden_state: Tensor<Self::Storage>,
    ) -> Tensor<Self::Storage> {
        self.0.resource.apply(|ctx| {
            let compute = self.0.compute.sprout_ref(ctx);
            let mut x = hidden_state
                .as_mut()
                .map_physical(|u| &mut **u.mem.sprout_mut(ctx));
            let range =
                DecodingMeta::select(&mut x, decoding, |dst, src| compute.memcpy_d2d(dst, src));
            if range.is_empty() {
                return self.tensor(&[0, self.0.config.d]);
            }

            let lm_layernorm = self
                .0
                .lm_layernorm
                .as_ref()
                .map_physical(|u| &**u.sprout_ref(ctx));
            let lm_head = self
                .0
                .lm_head
                .as_ref()
                .map_physical(|u| &**u.sprout_ref(ctx));

            let mut x = x.slice(&[slice![range.start => range.end], slice![=>]]);
            let mut logits = self.tensor(&[x.shape()[0], lm_head.shape()[1]]);

            // 复制一个 x 以实现原地归一化
            let x_ = x
                .as_ref()
                .map_physical(|u| unsafe { from_raw_parts(u.as_ptr(), u.len()) });
            self.0
                .kernels
                .rms_norm(&mut x, &x_, &lm_layernorm, self.0.config.epsilon, compute);
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
        assert_eq!(logits.data_layout(), F16);
        let &[_nt, voc] = logits.shape() else {
            panic!()
        };
        let voc = voc as usize;

        self.0.resource.apply(|ctx| {
            let compute = self.0.compute.sprout_ref(ctx);
            sample_cpu(
                args.into_iter()
                    .flat_map(|meta| repeat(meta.args).take(meta.num_decode))
                    .enumerate(),
                logits.take_physical().mem.sprout_ref(ctx),
                voc,
                compute,
            )
        })
    }
}

impl Drop for Transformer {
    #[inline]
    fn drop(&mut self) {
        let Internal {
            config: _,
            resource,
            compute,
            kernels: _,
            embed_tokens,
            layers,
            lm_layernorm,
            lm_head,
        } = unsafe { ManuallyDrop::take(&mut self.0) };
        resource.apply(|ctx| {
            compute.sprout(ctx);
            embed_tokens.take_physical().sprout(ctx);
            lm_layernorm.take_physical().sprout(ctx);
            lm_head.take_physical().sprout(ctx);
            for layer in layers {
                layer.att_layernorm.take_physical().sprout(ctx);
                layer.att_qkv.take_physical().sprout(ctx);
                layer.att_o.take_physical().sprout(ctx);
                layer.mlp_layernorm.take_physical().sprout(ctx);
                layer.mlp_gate_up.take_physical().sprout(ctx);
                layer.mlp_down.take_physical().sprout(ctx);
            }
        });
    }
}

struct ComputeStream<'a> {
    nh: u32,
    nkvh: u32,
    di: u32,
    epsilon: f32,
    theta: f32,
    kernels: &'a CambriconKernels,
    compute: &'a Queue<'a>,
    layers: &'a [LayerStorage<DevMemSpore>],
}

impl<'a> llama::ComputeStream for ComputeStream<'a> {
    type Handle = Mlu;
    type Storage = Cache;
    type Buf<'m> = DevMem<'m>;
    type Pos<'m> = DevMem<'m>;

    #[inline]
    fn malloc(&self, len: usize) -> Self::Buf<'_> {
        self.compute.ctx().malloc::<u8>(len)
    }
    #[inline]
    fn map_pos<'b>(&self, pos: &'b [u32]) -> Self::Pos<'b>
    where
        Self: 'b,
    {
        self.compute.ctx().from_host(pos)
    }
    #[inline]
    fn map_storage<'b>(&'b self, storage: &'b mut Self::Storage) -> &'b mut SliceOn<Self::Handle> {
        storage.mem.sprout_mut(self.compute.ctx())
    }
    #[inline]
    fn kernels(&self) -> &impl Kernels<Self::Handle> {
        self.kernels
    }
    #[inline]
    fn queue(&self) -> &llama::QueueOf<Self::Handle> {
        self.compute
    }
    #[inline]
    fn constant(&self) -> ComputeConst {
        ComputeConst {
            nh: self.nh,
            nkvh: self.nkvh,
            di: self.di,
            epsilon: self.epsilon,
            theta: self.theta,
        }
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
        self.layers.iter().map(|l|LlamaLayer( self.queue().ctx(), l))
    }
}

macro_rules! access {
    ($self:expr, $name:ident) => {
        $self
            .1
            .$name
            .as_ref()
            .map_physical(|p|&**p.sprout_ref(&$self.0))
    };
}

struct LlamaLayer<'a>(&'a CurrentCtx,&'a LayerStorage<DevMemSpore>);

impl<'a> llama::LLamaLayer for LlamaLayer<'a> {
    type Byte = DevByte;
    type Storage<'m> = &'m[DevByte] where Self: 'm;

    fn att_layernorm(&self) -> Tensor<Self::Storage<'_>> {
        access!(self, att_layernorm)
    }
    fn att_qkv(&self) -> Tensor<Self::Storage<'_>> {
        access!(self, att_qkv)
    }
    fn att_o(&self) -> Tensor<Self::Storage<'_>> {
        access!(self, att_o)
    }
    fn mlp_layernorm(&self) -> Tensor<Self::Storage<'_>> {
        access!(self, mlp_layernorm)
    }
    fn mlp_gate_up(&self) -> Tensor<Self::Storage<'_>> {
        access!(self, mlp_gate_up)
    }
    fn mlp_down(&self) -> Tensor<Self::Storage<'_>> {
        access!(self, mlp_down)
    }
}
