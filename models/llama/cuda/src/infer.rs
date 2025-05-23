use crate::{Operators, RandomSample, Weights};
use common::Distribution;
use gguf::GGufModel;
use llama::{
    ext::ggml_quants::f16, LlamaArgs, LlamaMeta, LlamaRequest, LlamaStorage, LlamaWorker, Tensor,
};
use operators::{
    cuda::{self, memcpy_d2h, Config, Device, Gpu, MemPoolBlob, NoDevice, StreamMemPool},
    random_sample::{KVPair, SampleArgs},
    Alloc, QueueAlloc,
};
use std::{slice::from_raw_parts_mut, time::Instant};
use test_utils::{Inference, TokenizerAndPrompt};

type Worker<'w> = LlamaWorker<Operators, Weights<'w>>;

#[test]
fn test_infer() {
    let Some(Inference {
        model,
        devices,
        prompt,
        as_user,
        temperature,
        top_p,
        top_k,
        max_steps,
        ..
    }) = Inference::load()
    else {
        return;
    };
    let gguf = GGufModel::read(model.iter().map(|s| &**s));

    let TokenizerAndPrompt {
        eos,
        tokenizer,
        prompt,
    } = TokenizerAndPrompt::new(&gguf, prompt, as_user);

    let model = LlamaStorage::from_gguf(&gguf);
    println!("{:?}", model.meta);

    let sample_args = SampleArgs::new(temperature, top_p, top_k).expect("invalid sample args");
    println!("{sample_args:?}");

    let device = devices.map_or(0, |devices| devices.parse().unwrap());
    println!("using gpu{device}");

    let gpu = match cuda::init() {
        Ok(()) => Device::new(device),
        Err(NoDevice) => return,
    };
    let gpu = Gpu::new(gpu.context(), Config::default());
    let gpu = &gpu;

    let meta = &model.meta;
    let &LlamaMeta {
        dt_embd,
        nctx,
        nvoc,
        dh,
        ..
    } = meta;

    gpu.apply(|ctx| {
        let stream = ctx.stream();

        let time = Instant::now();
        let mut host_ = ctx.malloc_host::<u8>(model.token_embd.len());
        host_.copy_from_slice(model.token_embd);
        let token_embd = stream.ctx().from_host(&host_);
        let weights = Weights::new(&model, Distribution::MONO, ctx);
        println!("load weights: {:?}", time.elapsed());

        let (free, _) = ctx.mem_info();
        let mut cache = meta
            // 用剩余空闲空间的一半存储 kv cache
            .kv_cache_in_size(nctx, free.0 / 2)
            .map(|len| ctx.malloc::<u8>(len));
        println!("cache len = {}", cache.shape()[0]);

        let queue_alloc = StreamMemPool::new(stream);
        let alloc = |size| -> MemPoolBlob { queue_alloc.alloc(size) };

        let (free, _) = ctx.mem_info();
        // 去除 1GiB 以下的零头
        queue_alloc.put(free.0 & !((1 << 30) - 1));

        let mut worker = Worker::new(0, gpu, meta.clone(), weights);
        let sin_cos =
            <Operators as llama::Operators>::build_sin_cos(dt_embd, nctx, dh, &queue_alloc);
        let indices = RandomSample::build_indices(nvoc, &queue_alloc);
        let sample = RandomSample::new(gpu);

        test_utils::test_infer(eos, tokenizer, &prompt, max_steps, |input, pos| {
            let mut embd = meta.embd(input.len()).map(alloc);
            let mut logits = meta.logits(1).map(alloc);

            let d = embd.get().len() / input.len();
            for (i, &tok) in input.iter().enumerate() {
                queue_alloc.queue().memcpy_d2d(
                    &mut embd.get_mut()[i * d..][..d],
                    &token_embd[tok as usize * d..][..d],
                )
            }

            worker
                .launch(
                    LlamaArgs {
                        embd: embd.map_slice_mut(),
                        logits: logits.map_slice_mut(),
                        sin_cos: sin_cos.map_slice(),
                        requests: vec![LlamaRequest {
                            cache: cache.map_slice_mut(),
                            seq_len: input.len(),
                            out_len: 1,
                            pos,
                        }],
                        num_tokens: input.len(),
                        max_seq_len: input.len(),
                        max_att_len: pos + input.len(),
                    },
                    &mut [],
                    &queue_alloc,
                )
                .unwrap();

            let mut pairs = Tensor::kv_pair_vec(1, alloc);

            sample
                .launch(
                    &mut pairs,
                    &logits,
                    &indices,
                    sample_args,
                    &mut [],
                    &queue_alloc,
                )
                .unwrap();

            let mut pair = KVPair::new(0, f16::ZERO);
            memcpy_d2h(
                unsafe { from_raw_parts_mut(&mut pair as *mut _ as *mut u8, size_of_val(&pair)) },
                pairs.get(),
            );

            pair.idx() as _
        });
    });
}
