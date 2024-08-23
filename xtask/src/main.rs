mod chat;
mod deploy;
mod generate;
mod list_turbo;
mod service;

use causal_lm::{CausalLM, SampleArgs};
use clap::Parser;
use deploy::DeployArgs;
use service::ServiceArgs;
use std::{ffi::c_int, fmt, num::ParseIntError, str::FromStr};
use time::UtcOffset;

#[macro_use]
extern crate clap;

fn main() {
    use Commands::*;
    match Cli::parse().command {
        ListTurbo => list_turbo::list_turbo(),
        Deploy(deploy) => deploy.deploy(),
        Cast(cast) => cast.invoke(),
        Generate(args) => args.run(),
        Chat(chat) => chat.run(),
        Service(service) => service.run(),
    }
}

#[derive(Parser)]
#[clap(name = "transformer-utils")]
#[clap(version, about, long_about = None)]
struct Cli {
    #[clap(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// List available turbo hardware
    ListTurbo,
    /// Deploy binary
    Deploy(DeployArgs),
    /// Cast model
    Cast(cast::CastArgs),
    /// Generate following text
    Generate(generate::GenerateArgs),
    /// Chat locally
    Chat(chat::ChatArgs),
    /// Start the service
    Service(ServiceArgs),
}

#[derive(Args, Default)]
struct InferenceArgs {
    /// Model directory.
    #[clap(short, long)]
    model: String,
    /// Model type, maybe "llama", "mixtral", "llama" by default.
    #[clap(long)]
    model_type: Option<String>,

    /// Log level, may be "off", "trace", "debug", "info" or "error".
    #[clap(long)]
    log: Option<String>,

    /// Random sample temperature.
    #[clap(long)]
    temperature: Option<f32>,
    /// Random sample top-k.
    #[clap(long)]
    top_k: Option<usize>,
    /// Random sample top-p.
    #[clap(long)]
    top_p: Option<f32>,

    /// Select turbo hardware, the format is "ty:detail".
    #[clap(long)]
    turbo: Option<String>,
}

/// TODO 应该根据参数自动识别模型
#[derive(PartialEq)]
enum ModelType {
    Llama,
    Mixtral,
}

impl InferenceArgs {
    fn init_log(&self) {
        use log::LevelFilter;
        use simple_logger::SimpleLogger;

        let log = self
            .log
            .as_ref()
            .and_then(|log| match log.to_lowercase().as_str() {
                "off" | "none" => Some(LevelFilter::Off),
                "all" | "trace" => Some(LevelFilter::Trace),
                "debug" => Some(LevelFilter::Debug),
                "info" => Some(LevelFilter::Info),
                "error" => Some(LevelFilter::Error),
                _ => None,
            })
            .unwrap_or(LevelFilter::Warn);

        const EAST8: UtcOffset = match UtcOffset::from_hms(8, 0, 0) {
            Ok(it) => it,
            Err(_) => unreachable!(),
        };
        SimpleLogger::new()
            .with_level(log)
            .with_utc_offset(UtcOffset::current_local_offset().unwrap_or(EAST8))
            .init()
            .unwrap();
    }

    fn turbo(&self) -> (&str, &str) {
        if let Some(turbo) = self.turbo.as_ref() {
            if let Some((ty, detail)) = turbo.split_once(':') {
                (ty.trim(), detail.trim())
            } else {
                (turbo.trim(), "")
            }
        } else {
            ("", "")
        }
    }

    #[inline]
    fn model_type(&self) -> ModelType {
        if let Some(model_type) = self.model_type.as_ref() {
            match model_type.to_lowercase().as_str() {
                "llama" => ModelType::Llama,
                "mixtral" => ModelType::Mixtral,
                _ => panic!("Unsupported model type: {model_type}"),
            }
        } else {
            ModelType::Llama
        }
    }

    #[inline]
    fn sample_args(&self) -> SampleArgs {
        SampleArgs {
            temperature: self.temperature.unwrap_or(0.),
            top_k: self.top_k.unwrap_or(usize::MAX),
            top_p: self.top_p.unwrap_or(1.),
        }
    }
}

/// 模型相关的推理任务。
trait Task: Sized {
    /// 解析推理参数。
    fn inference(&self) -> &InferenceArgs;

    /// 在指定类型的模型上调用推理任务。
    ///
    /// 特性约束继承自 [`Service`](::service::Service)。
    async fn typed<M>(self, meta: M::Meta)
    where
        M: CausalLM + Send + Sync + 'static,
        M::Storage: Send,
        M::Error: fmt::Debug;

    fn run(self) {
        // 初始化日志器
        self.inference().init_log();
        // 启动 tokio 运行时
        let runtime = tokio::runtime::Runtime::new().unwrap();

        let (turbo, _detail) = self.inference().turbo();
        match self.inference().model_type() {
            ModelType::Llama => match turbo.to_ascii_lowercase().as_str() {
                "" => {
                    use llama_cpu::Transformer as M;
                    runtime.block_on(self.typed::<M>(()));
                }
                #[cfg(detected_cuda)]
                "nv" | "nvidia" if llama_nv::cuda::init().is_ok() => {
                    match &*_detail
                        .parse::<VecOrRange>()
                        .unwrap()
                        .into_vec(llama_nv::cuda::Device::count)
                    {
                        [] => {
                            use llama_nv::{ModelLoadMeta, Transformer as M};
                            let meta = ModelLoadMeta::load_all_to(0);
                            runtime.block_on(self.typed::<M>(meta));
                        }
                        &[n] => {
                            use llama_nv::{ModelLoadMeta, Transformer as M};
                            let meta = ModelLoadMeta::load_all_to(n);
                            runtime.block_on(self.typed::<M>(meta));
                        }
                        #[cfg(detected_nccl)]
                        list => {
                            use llama_nv_distributed::{cuda::Device, Transformer as M};
                            let meta = list.iter().copied().map(Device::new).collect();
                            runtime.block_on(self.typed::<M>(meta));
                        }
                        #[cfg(not(detected_nccl))]
                        _ => panic!("NCCL not detected"),
                    }
                    llama_nv::synchronize();
                }
                #[cfg(detected_neuware)]
                "cn" | "cambricon" => {
                    llama_cn::cndrv::init();
                    match &*_detail
                        .parse::<VecOrRange>()
                        .unwrap()
                        .into_vec(llama_cn::cndrv::Device::count)
                    {
                        [] => todo!(),
                        &[_n] => todo!(),
                        _list => todo!(),
                    }
                    llama_cn::synchronize();
                }
                _ => panic!("Turbo environment not detected"),
            },
            ModelType::Mixtral => {
                use mixtral_cpu::MixtralCPU as M;
                runtime.block_on(self.typed::<M>(()));
            }
        }
        // 关闭 tokio 运行时
        runtime.shutdown_background();
    }
}

#[allow(dead_code)]
enum VecOrRange {
    Vec(Vec<c_int>),
    Range(c_int, Option<c_int>),
}

impl FromStr for VecOrRange {
    type Err = ParseIntError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if s.is_empty() {
            Ok(VecOrRange::Vec(Vec::new()))
        } else if let Some((start, end)) = s.split_once("..") {
            Ok(VecOrRange::Range(
                match start.trim_end() {
                    "" => 0,
                    num => num.parse::<c_int>()?,
                },
                match end.trim_start() {
                    "" => None,
                    num => Some(num.parse::<c_int>()?),
                },
            ))
        } else {
            let mut list = Vec::new();
            for s in s.split(',') {
                let s = s.trim();
                if !s.is_empty() {
                    list.push(s.parse::<c_int>()?);
                }
            }
            Ok(Self::Vec(list))
        }
    }
}

impl VecOrRange {
    #[allow(dead_code)]
    fn into_vec(self, len: impl FnOnce() -> usize) -> Vec<c_int> {
        match self {
            Self::Vec(vec) => vec,
            Self::Range(start, end) => (start..end.unwrap_or_else(|| len() as _)).collect(),
        }
    }
}

#[macro_export]
macro_rules! print_now {
    ($($arg:tt)*) => {{
        use std::io::Write;

        print!($($arg)*);
        std::io::stdout().flush().unwrap();
    }};
}
