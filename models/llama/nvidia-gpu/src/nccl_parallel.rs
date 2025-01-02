use crate::{Operators, RandomSample, Weights};
use gguf::GGufModel;
use llama::{ext::ggml_quants::f16, LlamaRequest, LlamaStorage, LlamaWorker, Tensor};
use log::info;
use operators::{
    all_reduce::nccl::Operator as AllReduce,
    cuda::{self, memcpy_d2h, NcclNode, NoDevice},
    nccl::CommunicatorGroup,
    random_sample::{KVPair, SampleArgs},
    TopoNode,
};
use regex::Regex;
use std::{
    iter::zip,
    slice::{from_raw_parts, from_raw_parts_mut},
    thread, u64,
};
use test_utils::{test_infer_paralle, Inference, Task, TokenizerAndPrompt, WorkerSeed};

type Worker<'w> = LlamaWorker<Operators<NcclNode, AllReduce>, Weights<'w>>;

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

    let devices = devices
        .map(|devices| {
            Regex::new(r"\d+")
                .unwrap()
                .find_iter(&devices)
                .map(|c| c.as_str().parse().unwrap())
                .collect()
        })
        .unwrap_or_else(|| vec![0]);
    let lens = vec![1; devices.len()];
    let count = devices.len();
    println!("distribution: {devices:?}");

    let (seeds, senders) = match cuda::init() {
        Ok(()) => WorkerSeed::new(
            CommunicatorGroup::new(&devices)
                .into_vec()
                .into_iter()
                .map(|comm| NcclNode::new(comm, Default::default()))
                .collect(),
        ),
        Err(NoDevice) => return,
    };
    thread::scope(|s| {
        let _workers = zip(lens, seeds)
            .enumerate()
            .scan(0, |start, (id, (len, seed))| {
                let range = *start..*start + len;
                *start = range.end;

                let mut meta = model.meta.clone();
                meta.distribute(range.clone(), count);

                let model = &model;
                Some(s.spawn(move || {
                    info!("worker[{id}] started");
                    let WorkerSeed { node, tasks } = seed;
                    node.processor().apply(|ctx| {
                        let stream = ctx.stream();
                        let (free, _) = ctx.mem_info();

                        ctx.dev().set_mempool_threshold(u64::MAX);
                        let _ = stream.malloc::<u8>((free.0 >> 30).saturating_sub(1) << 30);

                        info!("worker[{id}] loading weights...");
                        let weights = Weights::new(model, range, count, usize::MAX, ctx);
                        let mut worker = Worker::new(id, &node, meta.clone(), weights);
                        info!("worker[{id}] created");
                        let mut cache = meta
                            .kv_cache(meta.nctx)
                            .map(|size| stream.malloc::<u8>(size));
                        let sin_cos = <Operators as llama::Operators>::build_sin_cos(
                            meta.dt_embd,
                            meta.nctx,
                            meta.dh,
                            &stream,
                        );

                        let sample = RandomSample::new(&node);
                        let indices = RandomSample::build_indices(model.meta.nvoc, &stream);
                        let mut pair = KVPair::new(0, f16::ZERO);
                        let mut pairs = Tensor::kv_pair_vec(1, |size| stream.malloc::<u8>(size));

                        for task in tasks {
                            let Task {
                                nt,
                                pos,
                                embd,
                                next,
                            } = task;
                            let mut embd = meta
                                .embd(nt)
                                // NOTICE NCCL 无法与异步分配存储协同工作，所以 NCCL 会用到的存储只能使用 ctx 同步分配
                                .map(|size| ctx.from_host(unsafe { from_raw_parts(embd, size) }));
                            let mut logits = meta
                                .logits(if id == 0 { 1 } else { 0 })
                                .map(|size| stream.malloc::<u8>(size));
                            worker
                                .launch(
                                    llama::LlamaArgs {
                                        embd: embd.map_slice_mut(),
                                        logits: logits.map_slice_mut(),
                                        sin_cos: sin_cos.map_slice(),
                                        requests: vec![LlamaRequest {
                                            cache: cache.map_slice_mut(),
                                            seq_len: nt,
                                            out_len: if id == 0 { 1 } else { 0 },
                                            pos,
                                        }],
                                        num_tokens: nt,
                                        max_seq_len: nt,
                                        max_att_len: nt + pos,
                                    },
                                    &mut [],
                                    &stream,
                                )
                                .unwrap();
                            if id == 0 {
                                sample
                                    .launch(
                                        &mut pairs,
                                        &logits,
                                        &indices,
                                        sample_args,
                                        &mut [],
                                        &stream,
                                    )
                                    .unwrap();

                                stream.synchronize();
                                memcpy_d2h(
                                    unsafe {
                                        from_raw_parts_mut(
                                            &mut pair as *mut _ as *mut u8,
                                            pairs.get().len(),
                                        )
                                    },
                                    pairs.get(),
                                );

                                next.send(pair.idx() as _).unwrap()
                            }
                        }
                    });
                }))
            })
            .collect::<Vec<_>>();

        let senders = senders.into_boxed_slice();
        test_infer_paralle(&model, senders, eos, tokenizer, &prompt, max_steps)
    })
}
