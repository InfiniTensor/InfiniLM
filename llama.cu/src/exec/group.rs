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

// TODO 这个有一个问题，n_toks 可以很长，可能会导致超出 engine 中的'let mut pre_kv_pairs = ctx.malloc::<KVPair>(max_tok);'
struct ModelsWithOneDyn<'ctx> {
    models: BTreeMap<NonZeroUsize, ModelExec<'ctx>>,
    dyn_model: Option<(NonZeroUsize, ModelExec<'ctx>)>,
    mapped: Option<NonZeroUsize>,
    graph: NNGraph<Tensor<*const VirByte, 2>>,
}

impl<'ctx> ModelsWithOneDyn<'ctx> {
    pub fn new(
        graph: NNGraph<Tensor<*const VirByte, 2>>,
        models: BTreeMap<NonZeroUsize, ModelExec<'ctx>>,
    ) -> Self {
        Self {
            models,
            dyn_model: None,
            mapped: None,
            graph,
        }
    }

    // 获取大于等于len的第一个key，如果len大于所有key，则返回len，
    // TODO 可能需要优化策略，防止多次建图
    pub fn get_key(&self, len: NonZero<usize>) -> NonZeroUsize {
        self.models
            .range(len..)
            .next()
            .map(|(k, _)| *k)
            .unwrap_or(len)
    }

    pub fn get_mut(&mut self, key: NonZero<usize>) -> Option<&mut ModelExec<'ctx>> {
        self.models.get_mut(&key).or_else(|| {
            self.dyn_model
                .as_mut()
                .filter(|(k, _)| *k == key)
                .map(|(_, m)| m)
        })
    }

    pub fn map_exec(
        &mut self,
        key: NonZero<usize>,
        handle: &mut Handle<'ctx>,
        pages: &mut MemPages,
        stream: &Stream<'ctx>,
    ) {
        // 检查当前映射的模型
        if let Some(mapped) = self.mapped {
            if mapped == key {
                return;
            }
            // 当前映射的模型不是要映射的模型，解映射
            stream.synchronize();
            self.get_mut(mapped).unwrap().unmap(pages)
        }
        // 建立映射
        if self.models.get_mut(&key).map(|m| m.map(pages)).is_none() {
            log::info!("create modelExec for key {}", key.get());
            let mut exec = ModelExec::new(self.graph.clone(), key.get(), handle, pages, false);
            exec.map(pages);
            let _ = self.dyn_model.replace((key, exec));
        }
        // 更新记录
        self.mapped = Some(key)
    }
}

#[derive(Clone)]
pub(crate) struct Req<Cache> {
    pub kv_cache: Cache,
    pub pos: usize,
    pub seq: usize,
}

pub(crate) struct ModelGroup<'ctx> {
    models_with_one_dyn: ModelsWithOneDyn<'ctx>,
    attn: Attn,
    pages: MemPages,
    _weight: VirMem,
}

impl<'ctx> ModelGroup<'ctx> {
    pub fn new(
        llama: LLaMA<Tensor<&[u8], 2>>,
        dist: Distribution,

        attn: Attn,
        n_toks: impl IntoIterator<Item = usize>,
        handle: &mut Handle<'ctx>,
        barrier: Option<&Barrier>,
        use_cuda_graph: bool,
    ) -> Self {
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
        let time = Instant::now();
        let models = n_toks
            .into_iter()
            .map(|n_tok| {
                if let Some(b) = barrier {
                    b.wait();
                }
                let key = NonZeroUsize::new(n_tok).unwrap();
                let exec = ModelExec::new(graph.clone(), n_tok, handle, &mut pages, use_cuda_graph);
                (key, exec)
            })
            .collect::<BTreeMap<_, _>>();
        debug!(
            "group ({} models) compiled @{} in {:.02?}",
            models.len(),
            dev.index(),
            time.elapsed(),
        );
        let models_with_one_dyn = ModelsWithOneDyn::new(graph, models);
        Self {
            models_with_one_dyn,
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
        let key = self
            .models_with_one_dyn
            .get_key(NonZeroUsize::new(len).unwrap());
        self.models_with_one_dyn
            .map_exec(key, handle, &mut self.pages, stream);

        let model = self.models_with_one_dyn.get_mut(key).unwrap();
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
        self.models_with_one_dyn
            .map_exec(key, handle, &mut self.pages, stream);
        if let Some(comm) = &handle.comm {
            let model = self.models_with_one_dyn.get_mut(key).unwrap();
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
            models_with_one_dyn,
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

        let model = models_with_one_dyn.get_mut(key).unwrap();

        model.launch(attn, handle, &reqs, stream)
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
