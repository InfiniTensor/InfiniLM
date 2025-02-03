use crate::{Operators, RandomSample, Weights};
use gguf::GGufModel;
use gpt2::{ext::ggml_quants::f16, GPT2Storage, Gpt2Meta, Gpt2Worker, Tensor};
use operators::{
    common_cpu::{Cpu, ThisThread},
    random_sample::{KVPair, SampleArgs},
    Blob,
};
use std::slice::from_raw_parts_mut;
use test_utils::{Inference, TokenizerAndPrompt};

type Worker<'w> = Gpt2Worker<Operators, Weights<'w>>;

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

    let model = GPT2Storage::from_gguf(&gguf);
    println!("{:?}", model.meta);

    let sample_args = SampleArgs::new(temperature, top_p, top_k).expect("invalid sample args");
    println!("{sample_args:?}");

    let &Gpt2Meta {
        dt_embd,
        nctx,
        nvoc,
        d,
        ..
    } = &model.meta;
    let weights = Weights::new(&model);
    let mut worker = Worker::new(0, &Cpu, model.meta.clone(), weights);
    let mut cache = model.meta.kv_cache(nctx).map(Blob::new);
    let indices = RandomSample::build_indices(nvoc, &ThisThread);
    let sample = RandomSample::new(&Cpu);

    test_utils::test_infer(eos, tokenizer, &prompt, max_steps, |input, pos| {
        // 词汇编码缓存
        let mut embd = Tensor::new(dt_embd, &[input.len(), d]).map(Blob::new);
        // 词汇位置缓存
        let mut logits = model.meta.logits(1).map(Blob::new);
        let l = embd.get().len() / input.len();
        for (i, &tok) in input.iter().enumerate() {
            embd.get_mut()[i * l..][..l]
                .copy_from_slice(&model.token_embd[tok as usize * l..][..l]);
        }
        worker
            .launch(
                gpt2::args::Args {
                    embd: embd.map_slice_mut(),
                    logits: logits.map_slice_mut(),
                    idx: postion(input.len(), pos).map_slice(),
                    requests: vec![gpt2::args::Request {
                        cache: cache.map_slice_mut(),
                        seq_len: input.len(),
                        out_len: 1,
                        pos,
                    }],
                    max_seq_len: input.len(),
                    max_att_len: pos + input.len(),
                },
                &mut [],
                &ThisThread,
            )
            .unwrap();

        let mut pair = KVPair::new(0, f16::ZERO);
        let mut pairs = Tensor::kv_pair_vec(1, |_| unsafe {
            from_raw_parts_mut(&mut pair as *mut _ as _, size_of_val(&pair))
        });

        sample
            .launch(
                &mut pairs,
                &logits,
                &indices,
                sample_args,
                &mut [],
                &ThisThread,
            )
            .unwrap();

        pair.idx() as _
    });
}

fn postion(l: usize, pos: usize) -> Tensor<Blob> {
    use gguf::ggml_quants::digit_layout::types as ty;
    let mut ans = Tensor::new(ty::U32, &[1, l]).map(Blob::new);
    let (&mut [], data, &mut []) = (unsafe { ans.get_mut().align_to_mut::<u32>() }) else {
        panic!()
    };
    for i in 0..l {
        data[i] = (pos + i) as u32;
    }
    ans
}
