use causal_lm::{
    CausalLM, ChatTemplate, DecodingMeta, FromGGuf, Model, QueryContext, SampleMeta, Tokenizer,
};
use common::{map_files, upos, utok, Blob, GGufModel};
use common_cpu::{
    tensor::{reslice, slice, udim, Tensor},
    CpuKernels, Kernels, KernelsA, KernelsB, ThisThread,
};
use half::f16;
use llama::{
    duplicate_cache, ComputeStream, Handle, LlamaBlk, LlamaMeta, LlamaModel, QueueOf, SliceOn,
};
use memmap2::Mmap;
use std::{iter::repeat, ops::Deref, path::Path, slice::from_raw_parts};

pub struct Transformer {
    _files: Box<[Mmap]>,
    meta: LlamaMeta,
    token_embed: &'static [u8],
    output_norm: &'static [u8],
    output: &'static [u8],
    blocks: Box<[LlamaBlk<&'static [u8]>]>,
    kernels: CpuKernels,
}

impl Model for Transformer {
    type Config = ();
    type Error = ();

    #[inline]
    fn load(gguf: impl AsRef<Path>, _meta: Self::Config) -> Result<FromGGuf<Self>, Self::Error> {
        let _files = map_files(gguf);
        let gguf = GGufModel::read(_files.iter().map(|f| &**f));

        let tokenizer = Tokenizer::from_gguf(&gguf);
        let chat_template = ChatTemplate::from_gguf(&gguf, &tokenizer);
        let llama = LlamaModel::from_gguf(&gguf);

        #[inline(always)]
        const fn keep_lifetime(data: &[u8]) -> &'static [u8] {
            unsafe { std::mem::transmute(data) }
        }

        let model = Self {
            meta: llama.meta.clone(),
            token_embed: keep_lifetime(llama.token_embed),
            output_norm: keep_lifetime(llama.output_norm),
            output: keep_lifetime(llama.output),
            blocks: llama
                .blocks
                .iter()
                .map(|blk| blk.as_ref().map(|s| keep_lifetime(s)))
                .collect(),
            kernels: Default::default(),
            _files,
        };

        Ok(FromGGuf {
            model,
            tokenizer,
            chat_template,
        })
    }
}

impl ComputeStream for Transformer {
    type Handle = common_cpu::Cpu;
    type Storage = Blob;
    type Buf<'m> = Blob;
    type Pos<'m> = &'m [u8];

    #[inline]
    fn malloc(&self, len: usize) -> Self::Buf<'_> {
        Blob::new(len)
    }
    #[inline]
    fn map_pos<'p>(&self, pos: &'p [u32]) -> Self::Pos<'p>
    where
        Self: 'p,
    {
        reslice(pos)
    }
    #[inline]
    fn map_storage<'a>(&'a self, storage: &'a mut Self::Storage) -> &'a mut SliceOn<Self::Handle> {
        storage
    }
    #[inline]
    fn meta(&self) -> &LlamaMeta {
        &self.meta
    }
    #[inline]
    fn kernels(&self) -> &impl Kernels<Self::Handle> {
        &self.kernels
    }
    #[inline]
    fn queue(&self) -> &QueueOf<Self::Handle> {
        &ThisThread
    }

    fn debug<T>(tensor: &Tensor<T>)
    where
        T: Deref<Target = SliceOn<Self::Handle>>,
    {
        println!("{tensor}");
    }

    #[inline]
    fn layers(
        &self,
    ) -> impl Iterator<Item = impl llama::LLamaLayer<Byte = <Self::Handle as Handle>::Byte>> {
        (0..self.meta.nblk).map(|i| LlamaLayer(&self.meta, &self.blocks[i]))
    }
}

struct LlamaLayer<'a>(&'a LlamaMeta, &'a LlamaBlk<&'static [u8]>);

impl<'a> llama::LLamaLayer for LlamaLayer<'a> {
    type Byte = u8;
    type Storage<'m> = &'m [u8] where Self: 'm;

    #[inline]
    fn att_layernorm(&self) -> Tensor<Self::Storage<'_>> {
        let &LlamaMeta {
            dt_norm, nh, dh, ..
        } = self.0;
        Tensor::new(dt_norm, &[(nh * dh) as _], self.1.attn_norm)
    }
    #[inline]
    fn att_qkv(&self) -> Tensor<Self::Storage<'_>> {
        let &LlamaMeta {
            dt_mat,
            nh,
            nkvh,
            dh,
            ..
        } = self.0;
        Tensor::new(
            dt_mat,
            &[((nh + nkvh + nkvh) * dh) as _, (nh * dh) as _],
            self.1.attn_qkv,
        )
        .transpose(&[1, 0])
    }
    #[inline]
    fn att_o(&self) -> Tensor<Self::Storage<'_>> {
        let &LlamaMeta { dt_mat, nh, dh, .. } = self.0;
        Tensor::new(dt_mat, &[(nh * dh) as _, (nh * dh) as _], self.1.attn_o).transpose(&[1, 0])
    }
    #[inline]
    fn mlp_layernorm(&self) -> Tensor<Self::Storage<'_>> {
        let &LlamaMeta {
            dt_norm, nh, dh, ..
        } = self.0;
        Tensor::new(dt_norm, &[(nh * dh) as _], self.1.ffn_norm)
    }
    #[inline]
    fn mlp_gate_up(&self) -> Tensor<Self::Storage<'_>> {
        let &LlamaMeta {
            dt_mat, nh, dh, di, ..
        } = self.0;
        Tensor::new(
            dt_mat,
            &[(di + di) as _, (nh * dh) as _],
            self.1.ffn_gate_up,
        )
        .transpose(&[1, 0])
    }
    #[inline]
    fn mlp_down(&self) -> Tensor<Self::Storage<'_>> {
        let &LlamaMeta {
            dt_mat, nh, dh, di, ..
        } = self.0;
        Tensor::new(dt_mat, &[(nh * dh) as _, di as _], self.1.ffn_down).transpose(&[1, 0])
    }
}

impl CausalLM for Transformer {
    type Storage = Blob;

    #[inline]
    fn max_seq_len(&self) -> upos {
        self.meta.dctx as _
    }
    #[inline]
    fn bos_token(&self) -> utok {
        1
    }
    #[inline]
    fn eos_token(&self) -> utok {
        2
    }
    #[inline]
    fn new_cache(&self) -> Tensor<Self::Storage> {
        self.meta.new_cache(Blob::new)
    }
    #[inline]
    fn duplicate_cache(&self, cache: &Tensor<Self::Storage>, pos: upos) -> Tensor<Self::Storage> {
        duplicate_cache(cache, pos, Blob::new, |dst, src| {
            src.map_physical(|u| &**u)
                .reform_to(&mut dst.map_physical(|u| &mut **u))
        })
    }

    fn token_embed(&self, queries: impl IntoIterator<Item = utok>) -> Tensor<Self::Storage> {
        let LlamaMeta {
            dt_mat,
            nh,
            dh,
            dvoc,
            ..
        } = self.meta;
        let d = (nh * dh) as udim;

        let tokens = queries.into_iter().collect::<Vec<_>>();
        let nt = tokens.len() as udim;

        let mut x = Tensor::alloc(dt_mat, &[nt, d], Blob::new);
        let token_embed = Tensor::new(dt_mat, &[dvoc as _, d], self.token_embed);
        self.kernels
            .gather(&mut x, &token_embed, tokens, &ThisThread);
        x
    }

    fn forward<'a>(
        &self,
        queries: impl IntoIterator<Item = QueryContext<'a, Self::Storage>>,
        token_embedded: Tensor<Self::Storage>,
    ) -> Tensor<Self::Storage> {
        <Self as ComputeStream>::forward(self, queries, token_embedded)
    }

    fn decode(
        &self,
        decoding: impl IntoIterator<Item = DecodingMeta>,
        hidden_state: Tensor<Self::Storage>,
    ) -> Tensor<Self::Storage> {
        let LlamaMeta {
            dt_norm,
            dt_mat,
            nh,
            dh,
            dvoc,
            epsilon,
            ..
        } = self.meta;
        let d = (nh * dh) as udim;

        let mut x = hidden_state;
        let range = DecodingMeta::select(&mut x, decoding, |dst, src| dst.copy_from_slice(src));

        if range.is_empty() {
            return Tensor::alloc(dt_mat, &[0, d as _], Blob::new);
        }

        let lm_layernorm = Tensor::new(dt_norm, &[d], self.output_norm);
        let lm_head = Tensor::new(dt_mat, &[dvoc as _, d], self.output).transpose(&[1, 0]);
        let mut x = x.slice(&[slice![range.start => range.end], slice![=>]]);
        let mut logits = Tensor::alloc(dt_mat, &[x.shape()[0], lm_head.shape()[1]], Blob::new);

        // 复制一个 x 以实现原地归一化
        let x_ = x
            .as_ref()
            .map_physical(|u| unsafe { from_raw_parts(u.as_ptr(), u.len()) });
        self.kernels()
            .rms_norm(&mut x, &x_, &lm_layernorm, epsilon, self.queue());
        self.kernels()
            .mat_mul(&mut logits, 0., &x, &lm_head, 1., self.queue());

        logits
    }

    fn sample(
        &self,
        args: impl IntoIterator<Item = SampleMeta>,
        logits: Tensor<Self::Storage>,
    ) -> Vec<utok> {
        let &[_, voc] = logits.shape() else { panic!() };
        let logits: &[f16] = reslice(logits.as_slice());
        args.into_iter()
            .flat_map(|meta| repeat(meta.args).take(meta.num_decode))
            .enumerate()
            .map(|(i, args)| {
                self.kernels.sample(
                    args.temperature,
                    args.top_p,
                    args.top_k,
                    &common_cpu::slice!(logits; voc; [i]),
                )
            })
            .collect()
    }
}

#[test]
fn test_infer() {
    causal_lm::test_impl::<Transformer>((), 100, "Once upon a time,");
}
