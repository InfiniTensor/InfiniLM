#[macro_export]
macro_rules! slice {
    ($blob:expr; $width:expr; [$line:expr]) => {
        $blob[$line as usize * $width as usize..][..$width as usize]
    };
}

mod gather;

use common::utok;
use common_devices::{Operators, SliceOn};
use digit_layout::types::F16;
use half::f16;
use operators::{
    fuesd_softmax::common_cpu as softmax,
    mat_mul::common_cpu as mat_mul,
    mlp::common_cpu as mlp,
    random_sample::{common_cpu as random_sample, Args, KVPair, SampleArgs},
    reform::common_cpu as reform,
    rms_norm::common_cpu as rms_norm,
    rope::common_cpu as rope,
    Operator, QueueOf,
};
use std::ops::{Deref, DerefMut};
use tensor::Tensor;

pub extern crate tensor;

pub use common_devices::{Kernels, KernelsA, KernelsB};
pub use operators::common_cpu::{Handle as Cpu, ThisThread};

pub struct CpuKernels {
    reform: reform::Operator,
    mat_mul: mat_mul::Operator,
    rms_norm: rms_norm::Operator,
    rope: rope::Operator,
    softmax: softmax::Operator,
    mlp: mlp::Operator,
    sample: random_sample::Operator,
}

impl CpuKernels {
    pub fn sample(&self, temperature: f32, top_p: f32, top_k: usize, logits: &[f16]) -> utok {
        let mut kv_pair = KVPair::new(0, f16::ZERO);
        let mut args = Args::<Cpu>::new(F16, logits.len());
        args.kv_pair_base = &mut kv_pair as *mut _ as _;
        args.data_base = logits.as_ptr() as _;
        args.detail = SampleArgs {
            temperature,
            top_p,
            top_k,
        };
        self.sample.launch(&args, &ThisThread).unwrap();
        kv_pair.idx() as _
    }
}

impl Default for CpuKernels {
    fn default() -> Self {
        Self {
            reform: reform::Operator::new(&Cpu),
            mat_mul: mat_mul::Operator::new(&Cpu),
            rms_norm: rms_norm::Operator::new(&Cpu),
            rope: rope::Operator::new(&Cpu),
            softmax: softmax::Operator::new(&Cpu),
            mlp: mlp::Operator::new(&Cpu),
            sample: random_sample::Operator::new(&Cpu),
        }
    }
}

impl Kernels<Cpu> for CpuKernels {}

impl Operators for CpuKernels {
    type Handle = Cpu;

    fn reform_op(
        &self,
        _: &QueueOf<Self::Handle>,
    ) -> &impl operators::reform::Reform<Self::Handle> {
        &self.reform
    }
    fn rms_norm_op(
        &self,
        _: &QueueOf<Self::Handle>,
    ) -> &impl operators::rms_norm::RmsNorm<Self::Handle> {
        &self.rms_norm
    }
    fn mat_mul_op(
        &self,
        _: &QueueOf<Self::Handle>,
    ) -> &impl operators::mat_mul::MatMul<Self::Handle> {
        &self.mat_mul
    }
    fn rope_op(&self, _: &QueueOf<Self::Handle>) -> &impl operators::rope::Rope<Self::Handle> {
        &self.rope
    }
    fn softmax_op(
        &self,
        _: &QueueOf<Self::Handle>,
    ) -> &impl operators::fuesd_softmax::FusedSoftmax<Self::Handle> {
        &self.softmax
    }
    fn mlp_op(&self, _: &QueueOf<Self::Handle>) -> &impl operators::mlp::Mlp<Self::Handle> {
        &self.mlp
    }
}

impl KernelsB for CpuKernels {
    type Handle = Cpu;

    fn gather<T, U, I>(
        &self,
        x: &mut Tensor<T>,
        table: &Tensor<U>,
        tokens: I,
        _queue: &QueueOf<Self::Handle>,
    ) where
        T: DerefMut<Target = SliceOn<Self::Handle>>,
        U: Deref<Target = [u8]>,
        I: IntoIterator<Item = utok>,
    {
        gather::gather(x, table, tokens);
    }
}
