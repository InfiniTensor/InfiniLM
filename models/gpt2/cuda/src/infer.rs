use crate::{Operators, RandomSample, Weights};
use common::Distribution;
use gguf::{ggml_quants::digit_layout::types, GGufModel};
use gpt2::{ext::ggml_quants::f16, GPT2Storage, Gpt2Meta, Gpt2Worker, Tensor};
use operators::{
    cuda::{self, Config, Device, Gpu, NoDevice, StreamMemPool},
    random_sample::{KVPair, SampleArgs},
    Blob, QueueAlloc,
};
use std::slice::from_raw_parts_mut;
use test_utils::{Inference, TokenizerAndPrompt};

type Worker<'w> = Gpt2Worker<Operators, Weights<'w>>;

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

    let model = GPT2Storage::from_gguf(&gguf);
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
    let meta = &model.meta;
    let gpu = &gpu;
    let &Gpt2Meta { nctx, nvoc, .. } = &model.meta;

    gpu.apply(|ctx| {
        let stream = ctx.stream();
        let token_embd = ctx.from_host(model.token_embd);
        let (free, _) = ctx.mem_info();
        let queue_alloc = StreamMemPool::new(stream);
        queue_alloc.put((free.0 >> 30) << 30);
        let weights = Weights::new(&model, Distribution::MONO, ctx);

        let mut worker = Worker::new(0, &gpu, meta.clone(), weights);
        let mut cache = meta.kv_cache(nctx).map(|size| ctx.malloc::<u8>(size));
        let indices = RandomSample::build_indices(nvoc, &queue_alloc);
        let sample = RandomSample::new(gpu);

        test_utils::test_infer(eos, tokenizer, &prompt, max_steps, |input, pos| {
            let mut embd = meta.embd(input.len()).map(|len| ctx.malloc::<u8>(len));
            let mut logits = meta.logits(1).map(|len| ctx.malloc::<u8>(len));
            let d = embd.get().len() / input.len();
            for (i, &tok) in input.iter().enumerate() {
                queue_alloc.queue().memcpy_d2d(
                    &mut embd.get_mut()[i * d..][..d],
                    &token_embd[tok as usize * d..][..d],
                )
            }
            let mut idx =
                Tensor::new(types::U32, &[1, input.len()]).map(|len| ctx.malloc::<u8>(len));
            queue_alloc
                .queue()
                .memcpy_h2d(&mut idx.get_mut(), postion(input.len(), pos).get());
            worker
                .launch(
                    gpt2::args::Args {
                        embd: embd.map_slice_mut(),
                        logits: logits.map_slice_mut(),
                        idx: idx.map_slice(),
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
                    &queue_alloc,
                )
                .unwrap();

            let mut pairs = Tensor::kv_pair_vec(1, |size| ctx.malloc::<u8>(size));

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
            cuda::memcpy_d2h(
                unsafe { from_raw_parts_mut(&mut pair as *mut _ as *mut u8, size_of_val(&pair)) },
                pairs.get(),
            );

            pair.idx() as _
        });
    });
}

fn postion(d: usize, pos: usize) -> Tensor<Blob> {
    use gguf::ggml_quants::digit_layout::types as ty;
    let mut ans = Tensor::new(ty::U32, &[1, d]).map(Blob::new);
    let (&mut [], data, &mut []) = (unsafe { ans.get_mut().align_to_mut::<u32>() }) else {
        panic!()
    };
    for i in 0..d {
        data[i] = (pos + i) as u32;
    }
    ans
}
