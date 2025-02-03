pub mod args;
pub mod compute;
pub mod storage;

pub use args::{Args as GPT2Args, Request as GPT2Request};
pub use common::Contiguous;
use common::Distribution;
pub use compute::{BlkWeight, Gpt2Worker, Operators, WeightLoader};
use gguf::ggml_quants::digit_layout::DigitLayout;
pub use storage::{BlkStorage as GPT2BlkStorage, Storage as GPT2Storage};
pub use tensor::{RandomSample, Tensor};
pub mod ext {
    pub use gguf::{
        ext::{utok, Mmap},
        ggml_quants,
    };
}
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum GPT2BlkWeight {
    AttnQkvB,
    AttnQkvW,
    AttnOB,
    AttnOW,
    AttnNormB,
    AttnNormW,
    FfnUpB,
    FfnUpW,
    FfnDownB,
    FfnDownW,
    FfnNormB,
    FfnNormW,
}
#[derive(Clone, Debug)]
pub struct Gpt2Meta {
    pub dt_embd: DigitLayout,
    pub dt_token_embd: DigitLayout,   // 词汇编码布局
    pub dt_postion_embd: DigitLayout, // 位置编码布局
    pub dt_norm: DigitLayout,
    pub dt_mat: DigitLayout,

    pub nctx: usize,
    pub nvoc: usize,

    pub nblk: usize,
    pub nh: usize,
    pub nkvh: usize,
    pub d: usize,
    pub dh: usize,
    pub di: usize,

    pub epsilon: f32,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum TensorUsage {
    Storage,
    Computation,
}

impl Gpt2Meta {
    /// TODO 分布式未测试
    pub fn distribute(&self, dist: Distribution) -> Self {
        let [_, len, total] = dist.info();
        assert_eq!(self.nkvh % total, 0);
        assert_eq!(self.di % total, 0);

        Self {
            nh: self.nh / total * len,
            nkvh: self.nkvh / total * len,
            di: self.di / total * len,
            ..self.clone()
        }
    }
    pub fn blk(&self) -> GPT2BlkStorage<usize> {
        use TensorUsage::Storage as TensorMem;
        GPT2BlkStorage {
            attn_qkv_b: self.attn_qkv_b(TensorMem).take(),
            attn_qkv_w: self.attn_qkv_w(TensorMem).take(),
            attn_o_b: self.attn_o_b(TensorMem).take(),
            attn_o_w: self.attn_o_w(TensorMem).take(),
            attn_norm_b: self.norm().take(),
            attn_norm_w: self.norm().take(),
            ffn_up_b: self.ffn_up_b(TensorMem).take(),
            ffn_up_w: self.ffn_up_w(TensorMem).take(),
            ffn_down_b: self.ffn_down_b(TensorMem).take(),
            ffn_down_w: self.ffn_down_w(TensorMem).take(),
            ffn_norm_b: self.norm().take(),
            ffn_norm_w: self.norm().take(),
        }
    }

    pub fn kv_cache(&self, buf: usize) -> Tensor<usize> {
        let &Self {
            dt_embd,
            nblk,
            nkvh,
            ..
        } = self;
        Tensor::new(dt_embd, &[buf, nblk, 2, nkvh, 64])
    }

    pub fn embd(&self, nt: usize) -> Tensor<usize> {
        let &Self { dt_embd, d, .. } = self;
        Tensor::new(dt_embd, &[nt, d])
    }

    pub fn logits(&self, nt: usize) -> Tensor<usize> {
        let &Self { dt_embd, nvoc, .. } = self;
        Tensor::new(dt_embd, &[nt, nvoc])
    }

    pub fn pos_embd(&self) -> Tensor<usize> {
        let &Self {
            dt_embd, nvoc, d, ..
        } = self;
        Tensor::new(dt_embd, &[nvoc, d])
    }

    pub fn norm(&self) -> Tensor<usize> {
        let &Self { dt_norm, d, .. } = self;
        Tensor::new(dt_norm, &[d])
    }

    pub fn attn_qkv_w(&self, usage: TensorUsage) -> Tensor<usize> {
        let &Self { d, .. } = self;
        self.mat(3 * d, d, usage)
    }

    pub fn attn_qkv_b(&self, usage: TensorUsage) -> Tensor<usize> {
        let &Self { d, .. } = self;
        self.mat(3 * d, 1, usage)
    }

    pub fn attn_o_w(&self, usage: TensorUsage) -> Tensor<usize> {
        let &Self { d, .. } = self;
        self.mat(d, d, usage)
    }

    pub fn attn_o_b(&self, usage: TensorUsage) -> Tensor<usize> {
        let &Self { d, .. } = self;
        self.mat(d, 1, usage)
    }

    pub fn ffn_up_w(&self, usage: TensorUsage) -> Tensor<usize> {
        let &Self { d, di, .. } = self;
        self.mat(di, d, usage)
    }

    pub fn ffn_up_b(&self, usage: TensorUsage) -> Tensor<usize> {
        let &Self { di, .. } = self;
        self.mat(di, 1, usage)
    }

    pub fn ffn_down_w(&self, usage: TensorUsage) -> Tensor<usize> {
        let &Self { d, di, .. } = self;
        self.mat(d, di, usage)
    }

    pub fn ffn_down_b(&self, usage: TensorUsage) -> Tensor<usize> {
        let &Self { d, .. } = self;
        self.mat(d, 1, usage)
    }

    pub fn output_weight(&self) -> Tensor<usize> {
        Tensor::new(self.dt_embd, &[self.nvoc, self.d])
    }

    fn mat(&self, row: usize, col: usize, usage: TensorUsage) -> Tensor<usize> {
        // NOTICE: 权重矩阵以 mat 类型存储但以 embd 类型参与计算
        match usage {
            TensorUsage::Storage => {
                Tensor::new(self.dt_mat, &[row, col / self.dt_mat.group_size()])
            }
            TensorUsage::Computation => {
                assert_eq!(self.dt_embd.group_size(), 1);
                Tensor::new(self.dt_embd, &[row, col]).transpose(&[1, 0])
            }
        }
    }
}
