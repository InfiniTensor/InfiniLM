﻿use crate::{Operators, RandomSample, Weights};
use gguf::GGufModel;
use llama::{
    ext::ggml_quants::f16, LlamaArgs, LlamaMeta, LlamaRequest, LlamaStorage, LlamaWorker, Tensor,
};
use operators::{
    ascend::Npu,
    ascendcl::{self, memcpy_d2h, Device},
    random_sample::{KVPair, SampleArgs},
};
use std::{slice::from_raw_parts_mut, thread, usize};
use test_utils::{Inference, TokenizerAndPrompt};

type Worker<'w> = LlamaWorker<Operators, Weights<'w>>;

#[test]
fn test_infer() {
    let Some(Inference {
        model,
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

    ascendcl::init();
    let npu = Device::new(0);
    let npu = Npu::new(npu.context());
    let npu = &npu;

    let meta = &model.meta;
    let &LlamaMeta {
        dt_embd,
        nctx,
        nvoc,
        dh,
        ..
    } = meta;

    thread::scope(|s| {
        let sample = s.spawn(move || {
            let mut sample = RandomSample::new(npu);
            sample.scheme(dt_embd, nvoc).unwrap();
            sample
        });
        npu.apply(|ctx| {
            let stream = ctx.stream();

            let token_embd = ctx.from_host(model.token_embd);
            let weights = Weights::new(&model, .., 1, usize::MAX, &stream);
            let mut worker = Worker::new(&npu, meta.clone(), weights, true);
            let mut cache = meta.kv_cache(nctx).map(|size| ctx.malloc::<u8>(size));
            let sin_cos =
                <Operators as llama::Operators>::build_sin_cos(dt_embd, nctx, dh, &stream);
            let indices = RandomSample::build_indices(nvoc, &stream);

            let sample = sample.join().unwrap();
            test_utils::test_infer(eos, tokenizer, &prompt, max_steps, |input, pos| {
                let mut embd = meta.embd(input.len()).map(|len| ctx.malloc::<u8>(len));
                let mut logits = meta.logits(1).map(|len| ctx.malloc::<u8>(len));

                let d = embd.get().len() / input.len();
                for (i, &tok) in input.iter().enumerate() {
                    stream.memcpy_d2d(
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
                        &stream,
                    )
                    .unwrap();

                let mut pairs = Tensor::kv_pair_vec(1, |size| ctx.malloc::<u8>(size));

                sample
                    .launch(&mut pairs, &logits, &indices, sample_args, &mut [], &stream)
                    .unwrap();

                let mut pair = KVPair::new(0, f16::ZERO);
                memcpy_d2h(
                    unsafe {
                        from_raw_parts_mut(&mut pair as *mut _ as *mut u8, size_of_val(&pair))
                    },
                    pairs.get(),
                );

                pair.idx() as _
            });
        });
    });
}