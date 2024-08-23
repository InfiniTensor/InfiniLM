#![deny(warnings)]

mod session;
mod session_manager;

use causal_lm::{CausalLM, SampleArgs};
use chat_template::ChatTemplate;
use session::{Dispatcher, Generator};
use std::{
    fmt::{self, Debug},
    fs::File,
    path::Path,
    sync::Arc,
};
use tokio::task::JoinHandle;

pub use chat_template::Message;
pub use session::{BusySession, ChatError, Session};
pub use session_manager::{SessionError, SessionManager};

/// 对话服务。
pub struct Service<M: CausalLM> {
    component: Arc<ServiceComponent<M>>,
    pub default_sample: SampleArgs,
}

/// 服务中不变的组件，将在所有会话之间共享。
///
/// 推理线程的生命周期与这个组件绑定。
struct ServiceComponent<M: CausalLM> {
    handle: Arc<Dispatcher<M>>,
    tokenizer: Box<dyn Tokenize + Send + Sync>,
    normalizer: Box<dyn Normalizer + Send + Sync>,
    template: ChatTemplate,
    bos: String,
    #[allow(unused)]
    eos: String,
}

impl<M: CausalLM> Drop for ServiceComponent<M> {
    #[inline]
    fn drop(&mut self) {
        // 停止推理任务
        self.handle.stop();
    }
}

impl<M> Service<M>
where
    M: CausalLM + Send + Sync + 'static,
    M::Storage: Send,
    M::Error: Debug,
{
    pub fn load(model_dir: impl AsRef<Path>, meta: M::Meta) -> (Self, JoinHandle<()>) {
        let handle = Arc::new(Dispatcher::from(M::load(&model_dir, meta).unwrap()));
        let tokenizer = tokenizer(&model_dir);
        let normalizer = normalizer(&model_dir);
        let template = template(model_dir);
        (
            Self {
                component: Arc::new(ServiceComponent {
                    handle: handle.clone(),
                    bos: tokenizer.decode(handle.model.bos_token()).into(),
                    eos: tokenizer.decode(handle.model.eos_token()).into(),
                    tokenizer,
                    normalizer,
                    template,
                }),
                default_sample: Default::default(),
            },
            tokio::task::spawn_blocking(move || handle.run()),
        )
    }
}

impl<M: CausalLM> Service<M> {
    /// 从对话服务启动一个会话。
    #[inline]
    pub fn launch(&self) -> Session<M> {
        let mut session: Session<M> = self.component.clone().into();
        session.sample = self.default_sample;
        session
    }

    /// 从对话服务启动一个文本生成器。
    #[inline]
    pub fn generate(&self, prompt: impl fmt::Display, sample: Option<SampleArgs>) -> Generator<M> {
        let sample = sample.unwrap_or(self.default_sample);
        Generator::new(self.component.clone(), prompt, sample)
    }
}

#[test]
fn test() {
    use colored::{Color, Colorize};
    use std::{io::Write, iter::zip};
    use tokio::{runtime::Builder, task::JoinSet};

    let Some(model_dir) = common::test_model::find() else {
        return;
    };
    println!("model_dir: {}", model_dir.display());

    let runtime = Builder::new_current_thread().build().unwrap();
    let _rt = runtime.enter();

    let (service, _handle) = Service::<llama_cpu::Transformer>::load(model_dir, ());

    let mut set = JoinSet::new();
    let tasks = vec![
        ("Say \"Hi\" to me.", Color::Yellow),
        ("Hi", Color::Red),
        ("Where is the capital of France?", Color::Green),
    ];

    let sessions = tasks.iter().map(|_| service.launch()).collect::<Vec<_>>();

    for ((prompt, color), mut session) in zip(tasks, sessions) {
        set.spawn(async move {
            session.extend(&[Message {
                role: "user",
                content: prompt,
            }]);
            let mut busy = session.chat();
            while let Some(s) = busy.decode().await {
                print!("{}", s.color(color));
                std::io::stdout().flush().unwrap();
            }
        });
    }

    runtime.block_on(async { while set.join_next().await.is_some() {} });
    runtime.shutdown_background();
}

fn template(model_dir: impl AsRef<Path>) -> ChatTemplate {
    let template = if model_dir
        .as_ref()
        .display()
        .to_string()
        .to_ascii_lowercase()
        .contains("tinyllama")
    {
        "\
{%- for message in messages -%}
    {%- if message['role'] == 'user' -%}
        {{ '<|user|>\n' + message['content'] + eos_token }}
    {%- elif message['role'] == 'system' -%}
        {{ '<|system|>\n' + message['content'] + eos_token }}
    {%- elif message['role'] == 'assistant' -%}
        {{ '<|assistant|>\n' + message['content'] + eos_token }}
    {%- endif -%}
    {%- if loop.last and add_generation_prompt -%}
        {{ '<|assistant|>\n' }}
    {%- endif -%}
{%- endfor -%}"
    } else {
        "\
{%- for message in messages -%}
    {%- if message['role'] == 'user' -%}
        {{ bos_token + '<用户>' + message['content'].strip() + eos_token }}
    {%- elif message['role'] == 'assistant' -%}
        {{ bos_token + '<AI>' + message['content'].strip() + eos_token }}
    {%- else -%}
        {{ bos_token + message['content'].strip() + eos_token }}
    {%- endif -%}
    {%- if loop.last and add_generation_prompt -%}
        {{ bos_token + '<AI>' }}
    {%- endif -%}
{%- endfor -%}"
    };
    ChatTemplate::new(template.into())
}

fn normalizer(model_dir: impl AsRef<Path>) -> Box<dyn Normalizer + Send + Sync> {
    if model_dir.as_ref().join("tokenizer.model").is_file() {
        return Box::new(BPECommonNormalizer {});
    }
    if model_dir.as_ref().join("vocabs.txt").is_file() {
        return Box::new(());
    }
    panic!("Tokenizer file not found");
}

fn tokenizer(model_dir: impl AsRef<Path>) -> Box<dyn Tokenize + Send + Sync> {
    use std::io::ErrorKind::NotFound;

    let mmap = |name: &str| {
        File::open(model_dir.as_ref().join(name)).and_then(|f| unsafe { memmap2::Mmap::map(&f) })
    };

    match mmap("tokenizer.model") {
        Ok(f) => return Box::new(Tokeneer::new(Bpe::from_tokenizer_model(&f))),
        Err(e) if e.kind() == NotFound => {}
        Err(e) => panic!("{e:?}"),
    }
    match mmap("vocabs.txt") {
        Ok(f) => return Box::new(Tokeneer::new(Lpe::from_vocabs_txt(&f))),
        Err(e) if e.kind() == NotFound => {}
        Err(e) => panic!("{e:?}"),
    }
    panic!("Tokenizer file not found");
}
