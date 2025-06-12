use crate::BaseArgs;
use llama_cu::{Message, Received, Service, Session, SessionId};
use log::info;
use std::time::{Duration, Instant};

#[derive(Args)]
pub struct BenchArgs {
    #[clap(flatten)]
    base: BaseArgs,
    #[clap(short, long)]
    prompt: Option<String>,
    #[clap(short = 't', long)]
    use_template: bool,
    #[clap(short, long)]
    batch: Option<usize>,
}

impl BenchArgs {
    pub fn bench(self) {
        let Self {
            base,
            prompt,
            use_template,
            batch,
        } = self;
        let gpus = base.gpus();
        let max_steps = base.max_steps();
        let mut prompt = prompt.unwrap_or("Once upon a time,".into());
        let batch = batch.unwrap_or(1);

        let mut service = Service::new(base.model, &gpus, !base.no_cuda_graph);
        if use_template {
            prompt = service.terminal().render(&[Message::user(&prompt)])
        }

        for i in 0..batch {
            let session = Session {
                id: SessionId(i),
                sample_args: Default::default(),
                cache: service.terminal().new_cache(),
            };
            service
                .terminal()
                .start(session, &service.terminal().tokenize(&prompt), max_steps);
        }

        let mut prefill = Duration::ZERO;
        let mut decode = Duration::ZERO;
        let mut n_toks = 0;
        let mut remain = batch;
        let mut steps = 0;
        loop {
            let time = Instant::now();
            let Received { sessions, outputs } = service.recv(Duration::from_millis(100));
            let time = time.elapsed();
            println!("{steps:03}. time = {time:?}");
            steps += 1;
            if prefill.is_zero() {
                prefill = time
            } else {
                decode += time
            }
            n_toks += outputs.values().map(Vec::len).sum::<usize>();
            remain -= sessions.len();
            if remain == 0 {
                break;
            }
        }
        println!();
        info!("prefill = {prefill:?}, decode = {decode:?}");
        let time = decode / n_toks as _;
        info!(
            "Throughput: n toks = {n_toks}, perf: {time:?}/tok, {}tok/s",
            Duration::from_secs(1).div_duration_f32(time),
        );
        info!(
            "QOS: steps = {steps}, perf: {:?}/step, {}step/s",
            decode / (steps - 1) as _,
            (steps - 1) as f64 / decode.as_secs_f64()
        )
    }
}
