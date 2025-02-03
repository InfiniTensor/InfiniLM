use std::ops::DerefMut;

use crate::{GPT2BlkWeight, Gpt2Meta};
use common::{borrow, own, Contiguous, Distribution};
use gguf::{meta, tensor, GGufMetaMapExt, GGufModel};
use tensor::{rearrange, split, Tensor};

#[derive(Clone)]
pub struct Storage<T> {
    pub meta: Gpt2Meta,
    pub token_embd: T,
    pub pos_embd: T,
    pub blocks: Box<[BlkStorage<T>]>,
    pub output_norm_b: T,
    pub output_norm_w: T,
    pub output: T,
}

#[derive(Clone, Copy)]
pub struct BlkStorage<T> {
    pub attn_qkv_b: T,
    pub attn_qkv_w: T,
    pub attn_o_b: T,
    pub attn_o_w: T,
    pub attn_norm_b: T,
    pub attn_norm_w: T,

    pub ffn_up_b: T,
    pub ffn_up_w: T,
    pub ffn_down_b: T,
    pub ffn_down_w: T,
    pub ffn_norm_b: T,
    pub ffn_norm_w: T,
}

impl<'a> Storage<&'a [u8]> {
    pub fn from_gguf(gguf: &GGufModel<'a>) -> Self {
        let token_embd = &gguf.tensors["token_embd.weight"];
        let position_embd = &gguf.tensors["position_embd.weight"];
        let output_norm_b = &gguf.tensors["output_norm.bias"];
        let output_norm_w = &gguf.tensors["output_norm.weight"];
        let output = &gguf.tensors["output.weight"];
        let qkv0 = &gguf.tensors["blk.0.attn_qkv.weight"];

        let d = meta![gguf => llm_embedding_length];
        let nh = meta![gguf => llm_attention_head_count];

        #[rustfmt::skip]
        let meta = Gpt2Meta {
            dt_embd:  token_embd.ty,
            dt_token_embd:  token_embd.ty,
            dt_postion_embd: position_embd.ty,
            dt_norm: output_norm_w.ty,
            dt_mat :               qkv0.ty,

            nctx: meta![gguf => llm_context_length   ],
            nvoc: meta![gguf => tokenizer_ggml_tokens].len(),

            d, nh,
            nblk: meta![gguf => llm_block_count                 ],
            nkvh: meta![gguf => llm_attention_head_count_kv;  nh],
            dh  : meta![gguf => llm_rope_dimension_count; d / nh],
            di  : meta![gguf => llm_feed_forward_length         ],

            epsilon: meta![gguf => llm_attention_layer_norm_rms_epsilon; 1e-5],
        };
        #[rustfmt::skip]
        let blocks = (0..meta.nblk)
            .map(|i| BlkStorage {
                attn_norm_w: tensor![gguf => format!("blk.{i}.attn_norm.weight"  )].data,
                attn_norm_b: tensor![gguf => format!("blk.{i}.attn_norm.bias"    )].data,
                attn_qkv_w:  tensor![gguf => format!("blk.{i}.attn_qkv.weight"   )].data,
                attn_qkv_b:  tensor![gguf => format!("blk.{i}.attn_qkv.bias"     )].data,
                attn_o_w:    tensor![gguf => format!("blk.{i}.attn_output.weight")].data,
                attn_o_b:    tensor![gguf => format!("blk.{i}.attn_output.bias"  )].data,

                ffn_norm_w:  tensor![gguf => format!("blk.{i}.ffn_norm.weight"   )].data,
                ffn_norm_b:  tensor![gguf => format!("blk.{i}.ffn_norm.bias"     )].data,
                ffn_up_w:    tensor![gguf => format!("blk.{i}.ffn_up.weight"     )].data,
                ffn_up_b:    tensor![gguf => format!("blk.{i}.ffn_up.bias"       )].data,
                ffn_down_w:  tensor![gguf => format!("blk.{i}.ffn_down.weight"   )].data,
                ffn_down_b:  tensor![gguf => format!("blk.{i}.ffn_down.bias"     )].data,
            })
            .collect();

        Self {
            meta,
            token_embd: token_embd.data,
            pos_embd: position_embd.data,
            blocks,
            output_norm_b: output_norm_b.data,
            output_norm_w: output_norm_w.data,
            output: output.data,
        }
    }
}

impl<T> BlkStorage<T> {
    pub fn map<U>(self, mut f: impl FnMut(T) -> U) -> BlkStorage<U> {
        BlkStorage {
            attn_norm_b: f(self.attn_norm_b),
            attn_norm_w: f(self.attn_norm_w),
            attn_qkv_b: f(self.attn_qkv_b),
            attn_qkv_w: f(self.attn_qkv_w),
            attn_o_b: f(self.attn_o_b),
            attn_o_w: f(self.attn_o_w),

            ffn_up_b: f(self.ffn_up_b),
            ffn_up_w: f(self.ffn_up_w),
            ffn_down_b: f(self.ffn_down_b),
            ffn_down_w: f(self.ffn_down_w),
            ffn_norm_b: f(self.ffn_norm_b),
            ffn_norm_w: f(self.ffn_norm_w),
        }
    }

    pub fn as_ref(&self) -> BlkStorage<&T> {
        BlkStorage {
            attn_norm_b: &self.attn_norm_b,
            attn_norm_w: &self.attn_norm_w,
            attn_qkv_b: &self.attn_qkv_b,
            attn_qkv_w: &self.attn_qkv_w,
            attn_o_b: &self.attn_o_b,
            attn_o_w: &self.attn_o_w,

            ffn_up_b: &self.ffn_up_b,
            ffn_up_w: &self.ffn_up_w,
            ffn_down_b: &self.ffn_down_b,
            ffn_down_w: &self.ffn_down_w,
            ffn_norm_b: &self.ffn_norm_b,
            ffn_norm_w: &self.ffn_norm_w,
        }
    }

        #[rustfmt::skip]
    pub fn into_vec(self) -> Vec<(GPT2BlkWeight, T)> {
            use GPT2BlkWeight as W;
            vec![
                (W::AttnQkvB    , self.attn_qkv_b   ),
                (W::AttnQkvW    , self.attn_qkv_w   ),
                (W::AttnOB      , self.attn_o_b     ),
                (W::AttnOW      , self.attn_o_w     ),
                (W::AttnNormB   , self.attn_norm_b  ),
                (W::AttnNormW   , self.attn_norm_w  ),
                (W::FfnUpB      , self.ffn_up_b     ),
                (W::FfnUpW      , self.ffn_up_w     ),
                (W::FfnDownB    , self.ffn_down_b   ),
                (W::FfnDownW    , self.ffn_down_w   ),
                (W::FfnNormB    , self.ffn_norm_b   ),
                (W::FfnNormW    , self.ffn_norm_w   ),
            ]
        }
}

impl<T> FromIterator<(GPT2BlkWeight, T)> for BlkStorage<T> {
    #[rustfmt::skip]
    fn from_iter<U>(iter: U) -> Self
    where
        U: IntoIterator<Item = (GPT2BlkWeight, T)>,
    {
        let mut collector: BlkStorage<Option<T>> = BlkStorage {
            attn_qkv_b: None,
            attn_qkv_w: None,
            attn_o_b: None,
            attn_o_w: None,
            attn_norm_b: None,
            attn_norm_w: None,
            ffn_up_b: None,
            ffn_up_w: None,
            ffn_down_b: None,
            ffn_down_w: None,
            ffn_norm_b: None,
            ffn_norm_w: None,
        };
        for (which, data) in iter {
            use GPT2BlkWeight as W;
            match which {
                W::AttnQkvB =>  collector.attn_qkv_b=Some(data),
                W::AttnQkvW => collector.attn_qkv_w =Some(data),
                W::AttnOB => collector.attn_o_b =Some(data),
                W::AttnOW => collector.attn_o_w=Some(data),
                W::AttnNormB => collector.attn_norm_b=Some(data),
                W::AttnNormW => collector.attn_norm_w=Some(data),
                W::FfnUpB => collector.ffn_up_b=Some(data),
                W::FfnUpW => collector.ffn_up_w=Some(data),
                W::FfnDownB => collector.ffn_down_b=Some(data),
                W::FfnDownW => collector.ffn_down_w=Some(data),
                W::FfnNormB => collector.ffn_norm_b=Some(data),
                W::FfnNormW => collector.ffn_norm_w=Some(data),
            };
        }

        BlkStorage  {
           attn_qkv_b:collector.attn_qkv_b.unwrap(),
           attn_qkv_w:collector.attn_qkv_w.unwrap(),
           attn_o_b:collector.attn_o_b.unwrap(),
           attn_o_w:collector.attn_o_w.unwrap(),
           attn_norm_b:collector.attn_norm_b.unwrap(),
           attn_norm_w:collector.attn_norm_w.unwrap(),
           ffn_up_b:collector.ffn_up_b.unwrap(),
           ffn_up_w:collector.ffn_up_w.unwrap(),
           ffn_down_b:collector.ffn_down_b.unwrap(),
           ffn_down_w:collector.ffn_down_w.unwrap(),
           ffn_norm_b:collector.ffn_norm_b.unwrap(),
           ffn_norm_w:collector.ffn_norm_w.unwrap(),
        }
    }
}

impl Gpt2Meta {
    /// TODO 分布式未测试
    pub fn distribute_data<'w, U>(
        &self,
        which: GPT2BlkWeight,
        data: &'w [u8],
        dist: Distribution,
        mut f: impl FnMut(usize) -> U,
    ) -> Contiguous<'w, U>
    where
        U: DerefMut<Target = [u8]>,
    {
        use crate::TensorUsage::Storage as TensorMem;
        use GPT2BlkWeight as W;
        match which {
            W::AttnNormB | W::AttnNormW | W::FfnNormB | W::FfnNormW => borrow(data),
            _ if dist.is_mono() || data.is_empty() => borrow(data),
            W::AttnQkvB => {
                let meta = self.distribute(dist);
                self.distribute_qkv(
                    dist,
                    meta.attn_qkv_b(TensorMem).map(&mut f),
                    self.attn_qkv_b(TensorMem).map(|_| data),
                )
            }
            W::AttnQkvW => {
                let meta = self.distribute(dist);
                self.distribute_qkv(
                    dist,
                    meta.attn_qkv_w(TensorMem).map(&mut f),
                    self.attn_qkv_w(TensorMem).map(|_| data),
                )
            }
            W::AttnOB => {
                let [start, len, total] = dist.info();
                let o = self.attn_o_b(TensorMem).map(|_| data);

                let d = o.shape()[1] / total;
                let o = o.slice(1, d * start, 1, d * len);

                let mut o_ = Tensor::new(o.dt(), o.shape()).map(&mut f);
                rearrange(&mut o_, &o);
                own(o_.take())
            }
            W::AttnOW => {
                let [start, len, total] = dist.info();
                let o = self.attn_o_w(TensorMem).map(|_| data);

                let d = o.shape()[1] / total;
                let o = o.slice(1, d * start, 1, d * len);

                let mut o_ = Tensor::new(o.dt(), o.shape()).map(&mut f);
                rearrange(&mut o_, &o);
                own(o_.take())
            }
            W::FfnUpB => {
                let &Gpt2Meta { di, .. } = self;
                let [start, len, total] = dist.info();
                let dist = self.distribute(dist);

                let gu = self.ffn_up_b(TensorMem).map(|_| data);
                split!(gu => g, u; [di, di] @ 1);

                let di = di / total;

                let g = g.slice(1, di * start, 1, di * len);
                let u = u.slice(1, di * start, 1, di * len);

                let mut ans = dist.ffn_up_b(TensorMem).map(&mut f);
                {
                    let ans = ans.map_slice_mut();
                    split!(ans => g_, u_; [di * len , di * len] @ 1);
                    let mut g_ = g_;
                    let mut u_ = u_;
                    rearrange(&mut g_, &g);
                    rearrange(&mut u_, &u);
                }
                own(ans.take())
            }
            W::FfnUpW => {
                let &Gpt2Meta { di, .. } = self;
                let [start, len, total] = dist.info();
                let dist = self.distribute(dist);

                let gu = self.ffn_up_w(TensorMem).map(|_| data);
                split!(gu => g, u; [di, di] @ 1);

                let di = di / total;

                let g = g.slice(1, di * start, 1, di * len);
                let u = u.slice(1, di * start, 1, di * len);

                let mut ans = dist.ffn_up_w(TensorMem).map(&mut f);
                {
                    let ans = ans.map_slice_mut();
                    split!(ans => g_, u_; [di * len , di * len] @ 1);
                    let mut g_ = g_;
                    let mut u_ = u_;
                    rearrange(&mut g_, &g);
                    rearrange(&mut u_, &u);
                }
                own(ans.take())
            }
            W::FfnDownB => {
                let [start, len, total] = dist.info();
                let down = self.ffn_down_b(TensorMem).map(|_| data);

                let d = down.shape()[2] / total;
                let down = down.slice(2, d * start, 1, d * len);

                let mut down_ = Tensor::new(down.dt(), down.shape()).map(&mut f);
                rearrange(&mut down_, &down);
                own(down_.take())
            }
            W::FfnDownW => {
                let [start, len, total] = dist.info();
                let down = self.ffn_down_w(TensorMem).map(|_| data);

                let d = down.shape()[2] / total;
                let down = down.slice(2, d * start, 1, d * len);

                let mut down_ = Tensor::new(down.dt(), down.shape()).map(&mut f);
                rearrange(&mut down_, &down);
                own(down_.take())
            }
        }
    }

    pub fn distribute_qkv<'w, U>(
        &self,
        dist: Distribution,
        dst: Tensor<U>,
        src: Tensor<&'w [u8]>,
    ) -> Contiguous<'w, U>
    where
        U: DerefMut<Target = [u8]>,
    {
        let &Gpt2Meta { nh, nkvh, dh, .. } = self;
        let [start, len, total] = dist.info();

        let dq = nh * dh;
        let dkv = nkvh * dh;

        let qkv = src;
        split!(qkv => q, k, v; [dq, dkv, dkv] @ 0);

        let dq = dq / total;
        let dkv = dkv / total;

        let q = q.slice(0, dq * start, 1, dq * len);
        let k = k.slice(0, dkv * start, 1, dkv * len);
        let v = v.slice(0, dkv * start, 1, dkv * len);
        debug_assert!(q.is_contiguous() && k.is_contiguous() && v.is_contiguous());

        let mut ans = dst;
        {
            let ans = ans.map_slice_mut();
            split!(ans => q_, k_, v_; [dq * len , dkv * len, dkv * len] @ 0);
            let mut q_ = q_;
            let mut k_ = k_;
            let mut v_ = v_;
            rearrange(&mut q_, &q);
            rearrange(&mut k_, &k);
            rearrange(&mut v_, &v);
        }
        own(ans.take())
    }
}
