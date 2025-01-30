use crate::{Operators, RandomSample, Weights};
use common::Distribution;
use gguf::GGufModel;
use llama::{
    ext::ggml_quants::f16, LlamaArgs, LlamaMeta, LlamaRequest, LlamaStorage, LlamaWorker, Tensor,
};
use operators::{
    clrt::Platform,
    opencl::ClDevice,
    random_sample::{KVPair, SampleArgs},
};
use test_utils::{Inference, TokenizerAndPrompt};

type Worker = LlamaWorker<Operators, Weights>;

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

    let meta = &model.meta;
    let &LlamaMeta {
        dt_embd,
        nctx,
        nvoc,
        dh,
        ..
    } = meta;

    let Some(context) = Platform::all()
        .into_iter()
        .flat_map(|platform| platform.devices())
        .find(|d| {
            devices
                .as_ref()
                .is_none_or(|filter| d.name().contains(&*filter))
        })
        .map(|d| d.context())
    else {
        return;
    };
    let cl_dev = ClDevice::new(context.clone(), Default::default());
    let queue = context.queue();

    let weights = Weights::new(&model, Distribution::MONO, &context);
    let mut worker = Worker::new(0, &cl_dev, model.meta.clone(), weights).use_u32_pos();
    let mut cache = model
        .meta
        .kv_cache(nctx)
        .map(|size| context.malloc::<u8>(size));
    let sin_cos = <Operators as llama::Operators>::build_sin_cos(dt_embd, nctx, dh, &queue);
    let indices = RandomSample::build_indices(nvoc, &queue);

    let sample = RandomSample::new(&cl_dev);

    test_utils::test_infer(eos, tokenizer, &prompt, max_steps, |input, pos| {
        let mut embd = model
            .meta
            .embd(input.len())
            .map(|size| context.malloc::<u8>(size));
        let mut logits = model.meta.logits(1).map(|size| context.malloc::<u8>(size));

        let d = embd.get().len() / input.len();
        for (i, &tok) in input.iter().enumerate() {
            queue.memcpy_from_host(
                &mut embd.get_mut()[i * d..][..d],
                &model.token_embd[tok as usize * d..][..d],
                None,
            );
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
                &queue,
            )
            .unwrap();

        let mut pairs = Tensor::kv_pair_vec(1, |size| context.malloc::<u8>(size));

        sample
            .launch(&mut pairs, &logits, &indices, sample_args, &mut [], &queue)
            .unwrap();

        let mapped = queue.map_blob(pairs.take());
        let mapped = &mapped[..];

        let mut pair = KVPair::new(0, f16::ZERO);
        unsafe {
            std::ptr::copy_nonoverlapping(
                mapped.as_ptr(),
                &mut pair as *mut _ as _,
                size_of_val(&pair),
            )
        }

        pair.idx() as _
    })
}
