use llama::{
    ext::{f16, primitive, Mmap},
    BlkWeight, LlamaArgs, LlamaBlkStorage, LlamaMeta, LlamaRequest, LlamaStorage, LlamaWorker,
    RandomSample, Tensor, WeightLoader,
};
use operators::{
    all_reduce::NonAllReduce,
    common_cpu::{Cpu, ThisThread},
    random_sample::{common_cpu::Operator as CpuOp, KVPair, SampleArgs},
    ByteOf, QueueOf,
};
use std::{ops::Deref, slice::from_raw_parts_mut};

pub struct Llama {
    _storage: Box<[Mmap]>,
    token_embed: &'static [u8],
    single: LlamaWorker<Operators, Weights>,
    sample: RandomSample<Cpu, CpuOp>,
}

impl Llama {
    pub fn new(_storage: Box<[Mmap]>, model: LlamaStorage<&'static [u8]>) -> Self {
        let LlamaStorage {
            meta,
            token_embed,
            output_norm,
            output,
            blocks,
        } = model;
        assert_eq!(meta.distribute, 1);
        Self {
            _storage,
            token_embed,
            single: LlamaWorker::new(
                &Cpu,
                meta,
                Weights {
                    blks: blocks,
                    output_norm,
                    output,
                },
            ),
            sample: RandomSample::new(&Cpu),
        }
    }

    pub fn infer(&mut self, input: &[u32], cache: &mut [u8], pos: usize) -> u32 {
        const EMPTY: &[u8] = &[];

        let meta = self.single.meta();
        let &LlamaMeta {
            dt_mat: element,
            dctx,
            dh,
            ..
        } = meta;
        let cache = meta.kv_cache(dctx).map(|_| cache);
        let mut embd = meta.embd(input.len()).map(|size| vec![0u8; size]);
        let mut logits = meta.logits(1).map(|size| vec![0u8; size]);

        let d = embd.get().len() / input.len();
        for (i, &tok) in input.iter().enumerate() {
            embd.get_mut()[i * d..][..d]
                .copy_from_slice(&self.token_embed[tok as usize * d..][..d]);
        }

        self.single
            .launch(
                LlamaArgs {
                    embd: embd.map_slice_mut(),
                    logits: logits.map_slice_mut(),
                    sin: Tensor::new(element, &[0, dh]).map(|_| EMPTY),
                    cos: Tensor::new(element, &[0, dh]).map(|_| EMPTY),
                    requests: vec![LlamaRequest {
                        cache,
                        seq_len: input.len(),
                        out_len: 1,
                        pos,
                    }],
                    num_tokens: input.len(),
                    max_seq_len: input.len(),
                    max_att_len: pos + input.len(),
                    mlp_alpha: 1.,
                },
                &mut [],
                &ThisThread,
            )
            .unwrap();

        let mut pair = KVPair::new(0, f16::ZERO);
        let mut pairs = Tensor::kv_pair_vec(1, |_| unsafe {
            from_raw_parts_mut(&mut pair as *mut _ as *mut u8, size_of_val(&pair))
        });

        self.sample
            .launch(
                &mut pairs,
                &logits,
                &Tensor::new(primitive::U32, &[0]).map(|_| EMPTY),
                SampleArgs::ARG_MAX,
                &mut [],
                &ThisThread,
            )
            .unwrap();

        pair.idx() as u32
    }
}

struct Operators;

macro_rules! op {
    ($name:ident) => {
        operators::$name::common_cpu::Operator
    };
}

impl llama::Operators for Operators {
    type Hardware = Cpu;
    type TopoNode = Cpu;
    type RmsNorm = op!(rms_norm);
    type MatMul = op!(mat_mul);
    type Rope = op!(rope);
    type AttnKVCached = op!(attention_kv_cached);
    type Mlp = op!(mlp);
    type Rearrange = op!(rearrange);
    type AllReduce = NonAllReduce<Cpu>;

    fn debug<T>(tensor: &Tensor<T>)
    where
        T: Deref<Target = [ByteOf<Self::Hardware>]>,
    {
        println!("{tensor}");
    }
}

struct Weights {
    blks: Box<[LlamaBlkStorage<&'static [u8]>]>,
    output_norm: &'static [u8],
    output: &'static [u8],
}

impl WeightLoader for Weights {
    type Hardware = Cpu;
    type Memory<'s> = &'s [u8];

    #[inline]
    fn load_blk(
        &self,
        which: BlkWeight,
        iblk: usize,
        _queue: &QueueOf<Self::Hardware>,
    ) -> Self::Memory<'_> {
        let blk = &self.blks[iblk];
        match which {
            BlkWeight::AttnNorm => blk.attn_norm,
            BlkWeight::AttnQKV => blk.attn_qkv,
            BlkWeight::AttnO => blk.attn_o,
            BlkWeight::FfnNorm => blk.ffn_norm,
            BlkWeight::FfnGateUp => blk.ffn_gate_up,
            BlkWeight::FfnDown => blk.ffn_down,
        }
    }

    #[inline]
    fn output_norm(&self, _queue: &QueueOf<Self::Hardware>) -> Self::Memory<'_> {
        self.output_norm
    }

    #[inline]
    fn output(&self, _queue: &QueueOf<Self::Hardware>) -> Self::Memory<'_> {
        self.output
    }
}

#[test]
fn test_infer() {
    use gguf::{GGufMetaMapExt, GGufModel};
    use std::{
        io::Write,
        slice::from_raw_parts,
        time::{Duration, Instant},
    };

    let Some(shards) = test_utils::map_gguf_files() else {
        return;
    };
    let gguf = GGufModel::read(shards.iter().map(|s| &**s));
    let eos = gguf.tokenizer_ggml_eos_token_id().unwrap();
    let tokenizer = gguf.tokenizer();
    let llama =
        LlamaStorage::from_gguf(&gguf).map(&mut |s| unsafe { from_raw_parts(s.as_ptr(), s.len()) });
    let mut llama = Llama::new(shards, llama);

    let meta = llama.single.meta();
    println!("{meta:?}");

    let cache = meta.kv_cache(meta.dctx);
    let mut cache_buf = vec![0u8; cache.shape().iter().product::<usize>() * size_of::<f16>()];

    let mut prompt = "Once upon a time,".to_string();

    print!("{prompt}");
    std::io::stdout().flush().unwrap();

    let mut tokens = tokenizer.encode(&prompt);
    let num_prompt_tokens = tokens.len();

    let mut prefill = Duration::ZERO;
    let mut decode = Duration::ZERO;

    let mut pos = 0;
    loop {
        let time = Instant::now();
        let next = llama.infer(&tokens, &mut cache_buf, pos);
        let time = time.elapsed();

        if prefill.is_zero() {
            prefill = time;
        } else {
            decode += time;
        }

        pos += tokens.len();
        if next == eos {
            break;
        }

        let piece = tokenizer.decode(next);
        print!("{piece}");
        std::io::stdout().flush().unwrap();
        prompt.push_str(&piece);
        tokens = vec![next];
    }

    println!();
    println!();
    print_time("total", prefill + decode, pos);
    print_time("prefill", prefill, num_prompt_tokens);
    print_time("decode", decode, pos - num_prompt_tokens);

    fn print_time(name: &str, time: Duration, n: usize) {
        println!(
            "{name}: {time:?} for {n} tokens, avg: {:?} per token",
            time.div_f64(n as _)
        )
    }
}
