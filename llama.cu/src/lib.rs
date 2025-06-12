mod batch;
mod exec;
mod handle;
mod load;
mod memory;
mod model;
mod op;
mod utils;

use crate::{
    exec::{Command, KVCache, Output, engine},
    memory::MemPages,
    model::{ChatTemplate, GGufModel, map_files},
    utils::meta,
};
use exec::Request;
use ggus::GGufMetaMapExt;
use log::info;
use nn::Tensor;
use operators::cuda::{self, Device};
use std::{
    collections::{BTreeMap, HashSet},
    ffi::c_int,
    path::Path,
    sync::{
        Arc, Mutex, OnceLock,
        mpsc::{self, Receiver, RecvTimeoutError, Sender, TryRecvError},
    },
    time::{Duration, Instant},
};
use tokeneer::{Bpe, Tokeneer};

pub use crate::op::random_sample::SampleArgs;
pub use batch::{Cache, Session, SessionId};
pub use model::Message;
pub use tokeneer::{TextBuf, utok};

pub struct Service {
    handle: Option<(Receiver<Output>, std::thread::JoinHandle<()>)>,
    terminal: Terminal,
    forbid: HashSet<SessionId>,
}

#[derive(Clone)]
pub struct Terminal {
    sender: Sender<Command>,
    cache_parts: Box<[(Device, usize)]>,
    components: Arc<OnceLock<ModelComponents>>,
}

pub enum ReturnReason {
    Finish,
    Overflow,
}

#[derive(Default)]
pub struct Received {
    pub sessions: Vec<(Session<CacheParts>, ReturnReason)>,
    pub outputs: BTreeMap<SessionId, Vec<utok>>,
}

#[derive(Clone)]
#[repr(transparent)]
pub struct CacheParts(pub(crate) Arc<[Mutex<KVCache>]>);

struct ModelComponents {
    tokenizer: Tokeneer<Bpe>,
    chat_template: Option<ChatTemplate>,
    cache_template: Tensor<usize, 2>,
    eos: utok,
}

impl Service {
    pub fn new(model: impl AsRef<Path>, gpus: &[c_int], use_cuda_grpah: bool) -> Self {
        info!("start inference @gpu{gpus:?}");
        // 创建调度通道
        let (outputs, receiver) = mpsc::channel();
        let (sender, commands) = mpsc::channel();
        // 从文件加载权重
        let maps = map_files(model);
        let gpus = gpus.to_vec();
        let gpus_ = gpus.clone();
        // 启动推理引擎
        assert!(cuda::init().is_ok());
        let once = Arc::new(OnceLock::new());
        let once_ = once.clone();
        let handle = std::thread::spawn(move || {
            let mut gguf = GGufModel::read(maps.iter().map(|x| &**x));
            gguf.insert_sin_cos();

            let tokenizer = Bpe::from_gguf(&gguf);
            let chat_template = gguf.chat_template(&tokenizer);
            let cache_template = gguf.kv_cache();
            let eos = meta![gguf => tokenizer_ggml_eos_token_id];

            once_.get_or_init(|| ModelComponents {
                tokenizer,
                chat_template,
                cache_template,
                eos,
            });
            drop(once_);

            let llama = gguf.llama();
            engine(llama, &gpus_, commands, outputs, use_cuda_grpah)
        });
        once.wait();
        assert!(matches!(receiver.recv().unwrap(), Output::Ready));
        info!("ready for inference");
        Self {
            handle: Some((receiver, handle)),
            terminal: Terminal {
                sender,
                cache_parts: gpus.iter().map(|&i| (Device::new(i), 1)).collect(),
                components: once,
            },
            forbid: Default::default(),
        }
    }

    pub const fn terminal(&self) -> &Terminal {
        &self.terminal
    }

    pub fn recv(&mut self, timeout: Duration) -> Received {
        let time = Instant::now();
        let mut received = Received::default();
        match self.handle.as_ref().unwrap().0.recv_timeout(timeout) {
            Ok(output) => self.handle_output(output, &mut received),
            Err(RecvTimeoutError::Timeout) => return received,
            Err(RecvTimeoutError::Disconnected) => unreachable!(),
        }
        self.recv_all(timeout.saturating_sub(time.elapsed()), &mut received);
        received
    }

    pub fn try_recv(&mut self) -> Received {
        let mut received = Received::default();
        self.recv_all(Duration::MAX, &mut received);
        received
    }

    fn recv_all(&mut self, timeout: Duration, received: &mut Received) {
        let time = Instant::now();
        loop {
            match self.handle.as_ref().unwrap().0.try_recv() {
                Ok(output) => self.handle_output(output, received),
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => unreachable!(),
            }
            if time.elapsed() >= timeout {
                break;
            }
        }
    }

    fn handle_output(&mut self, output: Output, received: &mut Received) {
        match output {
            Output::Overflow(overflow) => {
                for s in overflow {
                    self.forbid.remove(&s.id);
                    received.sessions.push((s, ReturnReason::Overflow))
                }
            }
            Output::Removed(s) => {
                self.forbid.remove(&s.id);
                received.sessions.push((s, ReturnReason::Finish))
            }
            Output::Complete {
                output,
                kv_pair,
                event,
                finished,
            } => {
                let components = self.terminal.components.wait();
                let outputs = self.terminal.cache_parts[0]
                    .0
                    .retain_primary()
                    .apply(|ctx| exec::decode(output, kv_pair, event, &ctx.stream()));
                for (id, mut toks) in outputs {
                    if self.forbid.contains(&id) {
                        continue;
                    }
                    if let Some((len, _)) =
                        toks.iter().enumerate().find(|(_, t)| **t == components.eos)
                    {
                        toks.truncate(len);
                        assert!(self.forbid.insert(id));
                        self.terminal.sender.send(Command::Remove(id)).unwrap()
                    }
                    received.outputs.entry(id).or_default().extend(toks)
                }
                for s in finished {
                    self.forbid.remove(&s.id);
                    received.sessions.push((s, ReturnReason::Finish))
                }
            }
            Output::Ready => unreachable!(),
        }
    }
}

impl Drop for Service {
    fn drop(&mut self) {
        let (receiver, handle) = self.handle.take().unwrap();
        let Terminal {
            sender,
            cache_parts,
            ..
        } = &self.terminal;
        sender.send(Command::ShutDown).unwrap();
        handle.join().unwrap();
        cache_parts[0]
            .0
            .retain_primary()
            .apply(|ctx| receiver.into_iter().for_each(|output| output.drop_on(ctx)))
    }
}

impl Terminal {
    pub fn new_cache(&self) -> Cache<CacheParts> {
        let template = &self.components.wait().cache_template;
        let parts = &self.cache_parts;
        let total = parts.iter().map(|(_, len)| len).sum::<usize>();
        let parts = parts
            .iter()
            .map(|(dev, len)| KVCache::new(template, *len, total, &MemPages::new(*dev)));
        Cache {
            cache: CacheParts(parts.map(Mutex::new).collect()),
            capacity: template.shape()[0],
            len: 0,
        }
    }

    pub fn render(&self, msgs: &[Message]) -> String {
        self.components
            .wait()
            .chat_template
            .as_ref()
            .unwrap()
            .render(msgs, true)
            .unwrap()
    }

    pub fn tokenize(&self, text: &str) -> Vec<utok> {
        self.components.wait().tokenizer.encode(text)
    }

    pub fn start(&self, session: Session<CacheParts>, tokens: &[utok], max_steps: usize) -> bool {
        assert_ne!(max_steps, 0, "Cannot decode 0 step");
        self.sender
            .send(Command::Insert(Request {
                session,
                prompt: tokens.to_vec().into(),
                out: 1,
                max_steps,
            }))
            .is_ok()
    }

    pub fn stop(&self, id: SessionId) -> bool {
        self.sender.send(Command::Remove(id)).is_ok()
    }

    pub fn encode(&self, text: &str) -> Vec<utok> {
        self.components.wait().tokenizer.encode(text)
    }

    pub fn decode(&self, tokens: &[utok], buf: &mut TextBuf) -> String {
        self.components.wait().tokenizer.decode(tokens, buf)
    }
}
