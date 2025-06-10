use super::{
    Command, Output, Request, Session,
    engine_manager::{EngineManager, Round},
    group::{ModelGroup, Req},
    kv_cache::KVCache,
    output_head::OutputHead,
};
use crate::{
    exec::{group::ModelGroupConfig, upos},
    handle::Handle,
    op::{FastEmbedding, random_sample::KVPair},
};
use nn::{Distribution, LLaMA, Tensor};
use operators::{
    Operator,
    attention_kv_cached::cuda::Operator as Attn,
    cuda::{ContextResource, CurrentCtx, Device, Event, Gpu, HostMem},
};
use std::{
    ffi::c_int,
    iter::zip,
    marker::PhantomData,
    num::NonZeroUsize,
    ops::Deref,
    sync::{
        Arc, Barrier, Mutex, RwLock,
        mpsc::{Receiver, Sender},
    },
};
use tokeneer::utok;

#[cfg(nccl)]
use operators::nccl::{Communicator, CommunicatorGroup};

// 目前在有prompt的情况下，state.seq 的长度代表prompt还有多少未prefill，也就是 `prompt[prompt.len() - state.seq..]` 代表未prefill的prompt
pub(super) struct SessionStub {
    pub session: Session,
    pub state: State,
    pub prompt: Option<Box<[utok]>>,
}

#[derive(Clone, Copy)]
pub(super) struct State {
    pub seq: usize,
    pub out: usize,
    pub remain_steps: usize,
}

impl Request {
    pub(super) fn into_stub(self) -> SessionStub {
        let Request {
            session,
            prompt,
            out,
            max_steps,
        } = self;
        SessionStub {
            session,
            state: State {
                seq: prompt.len(),
                out,
                remain_steps: max_steps,
            },
            prompt: Some(prompt),
        }
    }
}

const NTOKS: [usize; 7] = [1, 8, 32, 64, 128, 256, 1024];
const CHUNKED_PREFILL_LEN: Option<usize> = Some(32);
//TODO 该常量应该放在哪比较合适
const MAX_TOKS: usize = 1024;

pub(crate) fn engine(
    llama: LLaMA<Tensor<&[u8], 2>>,
    gpus: &[c_int],
    commands: Receiver<Command>,
    outputs: Sender<Output>,
    use_cuda_graph: bool,
) {
    if let &[dev] = gpus {
        return mono(llama, Device::new(dev), commands, outputs, use_cuda_graph);
    }

    #[cfg(not(nccl))]
    unreachable!();

    #[cfg(nccl)]
    {
        let mut comms = CommunicatorGroup::new(gpus).into_vec().into_iter();
        let first = comms.next().unwrap();

        let mut llama = llama;
        let output_head = llama.output_head.take().unwrap();
        let worker = Worker {
            dev: first.device(),
            dist: Distribution {
                start: 0,
                len: 1,
                total: gpus.len(),
            },
            config: ModelGroupConfig {
                static_model_keys: NTOKS,
                dyn_cache_size: 1,
                use_cuda_graph,
            },
            max_toks: MAX_TOKS,
            barrier: Some(Arc::new(Barrier::new(gpus.len()))),
            task_box: Default::default(),
            chunked_prefill_len: CHUNKED_PREFILL_LEN,
        };

        std::thread::scope(|s| {
            let _threads = comms
                .map(|comm| {
                    let dist = Distribution::new(comm.rank(), 1, gpus.len());
                    let worker = Worker {
                        dev: comm.device(),
                        dist,
                        ..worker.clone()
                    };
                    let llama = llama.clone();
                    s.spawn(move || worker.work(llama, comm))
                })
                .collect::<Vec<_>>();

            worker.lead(llama, output_head, commands, outputs, |ctx| {
                Handle::with_comm(ctx, first)
            })
        })
    }
}

fn mono(
    mut llama: LLaMA<Tensor<&[u8], 2>>,
    dev: Device,
    commands: Receiver<Command>,
    outputs: Sender<Output>,
    use_cuda_graph: bool,
) {
    let output_head = llama.output_head.take().unwrap();
    Worker {
        dev,
        dist: Distribution {
            start: 0,
            len: 1,
            total: 1,
        },
        config: ModelGroupConfig {
            static_model_keys: NTOKS,
            dyn_cache_size: 1,
            use_cuda_graph,
        },
        max_toks: MAX_TOKS,
        barrier: None,
        task_box: Default::default(),
        chunked_prefill_len: CHUNKED_PREFILL_LEN,
    }
    .lead(llama, output_head, commands, outputs, |ctx| {
        Handle::new(ctx)
    })
}

#[derive(Clone)]
struct Worker<T> {
    dev: Device,
    dist: Distribution,
    config: ModelGroupConfig<T>,
    max_toks: usize,
    barrier: Option<Arc<Barrier>>,
    task_box: TaskBox,
    chunked_prefill_len: Option<usize>,
}

type TaskBox = Arc<RwLock<Option<Task>>>;

#[cfg_attr(not(nccl), allow(dead_code))]
struct Task {
    key: NonZeroUsize,
    reqs: Vec<Req<Arc<[Mutex<KVCache>]>>>,
}

impl<T: IntoIterator<Item = usize>> Worker<T> {
    fn lead(
        self,
        llama: LLaMA<Tensor<&[u8], 2>>,
        output_head: nn::OutputHead<Tensor<&[u8], 2>>,
        commands: Receiver<Command>,
        outputs: Sender<Output>,
        handle: impl FnOnce(&CurrentCtx) -> Handle,
    ) {
        let Self {
            dev,
            dist,
            config,
            max_toks,
            barrier,
            task_box,
            chunked_prefill_len,
        } = self;

        dev.set_mempool_threshold(u64::MAX);
        let gpu = Gpu::new(dev.retain_primary(), Default::default());
        let attn = Attn::new(&gpu);
        gpu.apply(|ctx| {
            let mut manager = EngineManager::new(chunked_prefill_len);
            let mut handle = handle(ctx);
            let mut models =
                ModelGroup::new(llama, dist, config, attn, &mut handle, barrier.as_deref());

            let mut output_head = OutputHead::new(output_head, ctx);

            let max_tok = max_toks;
            let mut fast_embd = FastEmbedding::new(max_tok, ctx);
            let mut pre_kv_pairs = ctx.malloc::<KVPair>(max_tok);

            let stream = ctx.stream();
            let len = max_toks;
            const BUF_LEVEL: usize = 3;
            let mut events: [Event; BUF_LEVEL] = std::array::from_fn(|_| stream.record());
            let mut tok_buf = BufN::<utok>::new(len, BUF_LEVEL, ctx);
            let mut pos_buf = BufN::<upos>::new(len, BUF_LEVEL, ctx);
            let mut out_idx_buf = BufN::<utok>::new(len, BUF_LEVEL, ctx);
            let mut fast_embd_buf = BufN::<(utok, utok)>::new(len, BUF_LEVEL, ctx);

            if outputs.send(Output::Ready).is_ok() {
                while manager.receive(&commands, &outputs).is_ok() {
                    // 组织请求
                    let Round {
                        overflow,
                        tokens,
                        reqs,
                        sample,
                        output,
                        fast_map,
                        finished,
                    } = manager.prepare();
                    if !overflow.is_empty()
                        && outputs.send(Output::Overflow(overflow.into())).is_err()
                    {
                        break;
                    }
                    if tokens.is_empty() {
                        assert!(
                            reqs.is_empty()
                                && sample.is_empty()
                                && output.is_empty()
                                && fast_map.is_empty()
                                && finished.is_empty()
                        );
                        continue;
                    }
                    let out_idx = out_idx(&reqs, output.iter().map(|(_, len)| *len));
                    events[out_idx_buf.index()].synchronize();
                    tok_buf.save(&tokens);
                    pos_buf.save(&pos(&reqs));
                    out_idx_buf.save(&out_idx);
                    fast_embd_buf.save(&fast_map);
                    events[out_idx_buf.index()] = stream.record();
                    // 加载输入
                    let (key, tok) = models.load_inputs(
                        &mut handle,
                        tokens.len(),
                        &*tok_buf,
                        &*pos_buf,
                        &stream,
                    );
                    // 快速启动路径
                    fast_embd.launch(
                        tok,
                        &pre_kv_pairs,
                        &fast_embd_buf[..fast_map.len()],
                        &mut handle,
                        &stream,
                    );
                    // 通知协处理单元
                    #[cfg(nccl)]
                    if let Some(barrier) = &barrier {
                        *task_box.write().unwrap() = Some(Task {
                            key,
                            reqs: reqs.clone(),
                        });
                        barrier.wait();
                        models.share_inputs(key, &mut handle, &stream);
                    }
                    // 推理
                    let x = models.launch(key, &reqs, &mut handle, &stream);

                    // 如果没有输出，则跳过
                    if !out_idx.is_empty() {
                        let (output, sample): (Vec<_>, Vec<_>) = output
                            .iter()
                            .zip(sample.iter())
                            .filter_map(|((id, len), sample_arg)| {
                                if *len > 0 {
                                    Some(((*id, *len), sample_arg))
                                } else {
                                    None
                                }
                            })
                            .unzip();

                        let kv_pairs = output_head.launch(
                            x,
                            &out_idx_buf[..out_idx.len()],
                            sample,
                            &mut handle,
                            &stream,
                        );
                        stream.memcpy_d2d(&mut pre_kv_pairs[..kv_pairs.len()], &kv_pairs);

                        let output = Output::Complete {
                            output: output.into(),
                            kv_pair: kv_pairs.sporulate(),
                            event: stream.record().sporulate(),
                            finished: finished.into(),
                        };
                        if outputs.send(output).is_err() {
                            break;
                        }
                    }
                }
            }
            // 通知协处理单元退出
            if let Some(barrier) = &barrier {
                let _ = task_box.write().unwrap().take();
                barrier.wait();
            }
            // 送回存储的会话信息
            for stub in manager.into_stubs() {
                if outputs.send(Output::Removed(stub.session)).is_err() {
                    break;
                }
            }
        })
    }

    #[cfg(nccl)]
    fn work(self, llama: LLaMA<Tensor<&[u8], 2>>, comm: Communicator) {
        let Self {
            dev,
            dist,
            config,
            max_toks: _max_toks,
            barrier,
            task_box,
            ..
        } = self;

        dev.set_mempool_threshold(u64::MAX);
        let gpu = Gpu::new(dev.retain_primary(), Default::default());
        let attn = Attn::new(&gpu);
        let barrier = barrier.unwrap();
        gpu.apply(|ctx| {
            let mut handle = Handle::with_comm(ctx, comm);
            let mut models =
                ModelGroup::new(llama, dist, config, attn, &mut handle, Some(&barrier));

            let stream = ctx.stream();
            loop {
                barrier.wait();
                match &*task_box.read().unwrap() {
                    Some(Task { key, reqs }) => {
                        models.share_inputs(*key, &mut handle, &stream);
                        models.launch(*key, reqs, &mut handle, &stream);
                    }
                    None => break,
                }
            }
        })
    }
}

fn pos<T>(reqs: &[Req<T>]) -> Vec<upos> {
    reqs.iter()
        .flat_map(|req| req.pos..req.pos + req.seq)
        .map(|x| x as _)
        .collect()
}

fn out_idx<T>(reqs: &[Req<T>], outs: impl IntoIterator<Item = usize>) -> Vec<utok> {
    let mut out_idx = Vec::new();

    let mut itok = 0;
    for (req, out) in zip(reqs, outs) {
        for i in req.seq - out..req.seq {
            out_idx.push((itok + i) as _)
        }
        itok += req.seq
    }

    out_idx
}

struct BufN<'ctx, T> {
    buf: HostMem<'ctx>,
    index: usize,
    level: usize,
    _phantom: PhantomData<T>,
}

impl<'ctx, T: Copy> BufN<'ctx, T> {
    fn new(len: usize, level: usize, ctx: &'ctx CurrentCtx) -> Self {
        Self {
            buf: ctx.malloc_host::<T>(len * level),
            index: 0,
            level,
            _phantom: PhantomData,
        }
    }
}

impl<T: Copy> BufN<'_, T> {
    fn save(&mut self, data: &[T]) {
        let data = unsafe { std::slice::from_raw_parts(data.as_ptr().cast(), size_of_val(data)) };

        if self.index + 1 == self.level {
            self.index = 0
        } else {
            self.index += 1
        }

        let piece = self.buf.len() / self.level;
        let (data_, padding) = self.buf[self.index * piece..][..piece].split_at_mut(data.len());
        data_.copy_from_slice(data);
        padding.fill(0)
    }

    const fn index(&self) -> usize {
        self.index
    }
}

impl<T> Deref for BufN<'_, T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        let piece = self.buf.len() / self.level;
        let (&[], piece, &[]) =
            (unsafe { self.buf[self.index * piece..][..piece].align_to::<T>() })
        else {
            unreachable!()
        };
        piece
    }
}
