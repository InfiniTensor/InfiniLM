use super::{KVCache, model::ModelExec};
use crate::{exec::upos, handle::Handle, memory::MemPages};
use log::debug;
use nn::{
    Distribution, Graph, GraphBuilder, LLaMA, NNGraph, Tensor, TensorMeta, digit_layout::types, op,
};
use operators::{
    attention_kv_cached::cuda::Operator as Attn,
    cuda::{DevByte, Stream, VirByte, VirMem},
};
use std::{
    collections::BTreeMap,
    num::{NonZero, NonZeroUsize},
    sync::{Arc, Barrier, Mutex},
    time::Instant,
};
use tokeneer::utok;

#[derive(Clone)]
pub(crate) struct Req<Cache> {
    pub kv_cache: Cache,
    pub pos: usize,
    pub seq: usize,
}

pub(crate) struct ModelGroup<'ctx> {
    internal: Internal<'ctx>,
    attn: Attn,
    pages: MemPages,
    _weight: VirMem,
}

#[derive(Clone)]
pub(super) struct ModelGroupConfig<T> {
    pub static_model_keys: T,
    pub dyn_cache_size: usize,
    pub use_cuda_graph: bool,
}

impl<'ctx> ModelGroup<'ctx> {
    pub fn new<T: IntoIterator<Item = usize>>(
        llama: LLaMA<Tensor<&[u8], 2>>,
        dist: Distribution,

        config: ModelGroupConfig<T>,

        attn: Attn,
        handle: &mut Handle<'ctx>,
        barrier: Option<&Barrier>,
    ) -> Self {
        let ModelGroupConfig {
            static_model_keys,
            mut dyn_cache_size,
            use_cuda_graph,
        } = config;

        // 构建计算图
        let NNGraph(Graph { topo, nodes, edges }) = builder()
            .build(
                llama.tensor_parallel(dist),
                [
                    TensorMeta::new(types::U32, ["n_tok".into()]),
                    TensorMeta::new(types::U32, ["n_tok".into()]),
                ],
            )
            .unwrap();
        // 加载权重
        let dev = handle.ctx.dev();
        let mut pages = MemPages::new(dev);
        let (_weight, edges) = pages.load_weight(&dev, edges);
        // 构建 cuda graph
        let graph = NNGraph(Graph { topo, nodes, edges });
        debug!("compiling model group @{}", dev.index());
        let static_models = if use_cuda_graph {
            let time = Instant::now();
            let models = static_model_keys
                .into_iter()
                .map(|n_tok| {
                    if let Some(b) = barrier {
                        b.wait();
                    }
                    let key = NonZeroUsize::new(n_tok).unwrap();
                    let exec = ModelExec::new(graph.clone(), n_tok, handle, &mut pages, true);
                    (key, exec)
                })
                .collect::<BTreeMap<_, _>>();
            debug!(
                "group ({} models) compiled @{} in {:.02?}",
                models.len(),
                dev.index(),
                time.elapsed(),
            );
            models
        } else {
            dyn_cache_size += static_model_keys.into_iter().count();
            Default::default()
        };

        let models_with_one_dyn = Internal::new(graph, static_models, dyn_cache_size);
        Self {
            internal: models_with_one_dyn,
            attn,
            pages,
            _weight,
        }
    }

    pub fn load_inputs(
        &mut self,
        handle: &mut Handle<'ctx>,
        len: usize,
        tok: &[utok],
        pos: &[upos],
        stream: &Stream<'ctx>,
    ) -> (NonZeroUsize, &mut [DevByte]) {
        let key = self.internal.get_key(NonZeroUsize::new(len).unwrap());
        let model = self.internal.map_exec(key, handle, &mut self.pages, stream);
        stream.memcpy_h2d(model.tok_buf(), &tok[..key.get()]);
        stream.memcpy_h2d(model.pos_buf(), &pos[..key.get()]);
        (key, model.tok_buf())
    }

    #[cfg(nccl)]
    pub fn share_inputs(
        &mut self,
        key: NonZeroUsize,
        handle: &mut Handle<'ctx>,
        stream: &Stream<'ctx>,
    ) {
        let model = self.internal.map_exec(key, handle, &mut self.pages, stream);
        if let Some(comm) = &handle.comm {
            comm.broadcast(model.tok_buf(), None, 0, stream);
            comm.broadcast(model.pos_buf(), None, 0, stream);
        }
    }

    pub fn launch(
        &mut self,
        key: NonZeroUsize,
        reqs: &[Req<Arc<[Mutex<KVCache>]>>],
        handle: &mut Handle,
        stream: &Stream<'ctx>,
    ) -> Tensor<*const VirByte, 2> {
        let Self {
            internal,
            attn,
            pages,
            ..
        } = self;

        let mut reqs = reqs
            .iter()
            .map(|req| Req {
                kv_cache: req.kv_cache[handle.rank()].lock().unwrap(),
                pos: req.pos,
                seq: req.seq,
            })
            .collect::<Vec<_>>();
        let reqs = reqs
            .iter_mut()
            .map(|req| {
                req.kv_cache.update(req.pos + req.seq, pages);
                Req {
                    kv_cache: req.kv_cache.as_tensor(),
                    pos: req.pos,
                    seq: req.seq,
                }
            })
            .collect::<Vec<_>>();

        internal
            .get_mut(&key)
            .unwrap()
            .launch(attn, handle, &reqs, stream)
    }
}

struct Internal<'ctx> {
    static_models: BTreeMap<NonZeroUsize, ModelExec<'ctx>>,
    dyn_model_cache: lru::LruCache<NonZeroUsize, ModelExec<'ctx>>,
    mapped: Option<NonZeroUsize>,
    graph: NNGraph<Tensor<*const VirByte, 2>>,
}

impl<'ctx> Internal<'ctx> {
    fn new(
        graph: NNGraph<Tensor<*const VirByte, 2>>,
        static_models: BTreeMap<NonZeroUsize, ModelExec<'ctx>>,
        dyn_cache_size: usize,
    ) -> Self {
        const ONE: NonZeroUsize = NonZeroUsize::new(1).unwrap();
        Self {
            static_models,
            dyn_model_cache: lru::LruCache::new(NonZeroUsize::new(dyn_cache_size).unwrap_or(ONE)),
            mapped: None,
            graph,
        }
    }

    /// 获取不小于 `len` 的最小模型索引，如果不存在，则返回 `len`。
    fn get_key(&self, len: NonZero<usize>) -> NonZeroUsize {
        self.static_models
            .range(len..)
            .next()
            .map_or(len, |(k, _)| *k)
    }

    fn get_mut(&mut self, key: &NonZero<usize>) -> Option<&mut ModelExec<'ctx>> {
        self.static_models
            .get_mut(key)
            .or_else(|| self.dyn_model_cache.get_mut(key))
    }

    fn map_exec(
        &mut self,
        key: NonZero<usize>,
        handle: &mut Handle<'ctx>,
        pages: &mut MemPages,
        stream: &Stream<'ctx>,
    ) -> &mut ModelExec<'ctx> {
        // 检查当前映射的模型
        if let Some(mapped) = self.mapped {
            if mapped == key {
                return self.get_mut(&key).unwrap();
            }
            // 当前映射的模型不是要映射的模型，解映射
            if let Some(mapped) = self.get_mut(&mapped) {
                stream.synchronize();
                mapped.unmap(pages)
            }
        }
        let Self {
            static_models,
            dyn_model_cache,
            mapped,
            graph,
        } = self;
        // 更新记录
        *mapped = Some(key);
        // 查找或新建模型
        let model = static_models.get_mut(&key).unwrap_or_else(|| {
            dyn_model_cache.get_or_insert_mut(key, || {
                log::info!("create modelExec for key {}", key.get());
                ModelExec::new(graph.clone(), key.get(), handle, pages, false)
            })
        });
        // 建立映射
        model.map(pages);
        model
    }
}

fn builder() -> GraphBuilder {
    let mut ans = GraphBuilder::default();
    ans.register_op("embedding", op::embedding::Embedding)
        .register_op("rms-norm", op::normalization::RmsNorm)
        .register_op("linear", op::linear::Linear)
        .register_op("rope", op::rope::Rope)
        .register_op("attention", op::attention::Attention)
        .register_op("swiglu", op::activation::SwiGLU)
        .register_op("concat", op::concat::Concat)
        .register_op("split", op::split::Split)
        .register_op("all-reduce", op::all_reduce::AllReduce);
    ans
}
