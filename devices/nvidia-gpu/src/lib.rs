#![cfg(detected_cuda)]

mod gather;

use common::utok;
use common_devices::{Operators, SliceOn};
use cuda::{AsRaw, Device};
use digit_layout::{
    types::{F16, U32},
    DigitLayout,
};
use half::f16;
use operators::{
    cuda::{memcpy_d2h, DevByte, DevMem, Stream},
    dyn_,
    fuesd_softmax::nvidia_gpu as softmax,
    mat_mul::nvidia_gpu as mat_mul,
    mlp::nvidia_gpu as mlp,
    random_sample::{nvidia_gpu as random_sample, KVPair, RandomSample, SampleArgs},
    reform::nvidia_gpu as reform,
    rms_norm::nvidia_gpu as rms_norm,
    rope::nvidia_gpu as rope,
    Operator, QueueOf, TensorLayout, Workspace,
};
use std::{
    collections::HashMap,
    ops::{Deref, DerefMut},
    ptr::{null, null_mut},
};

pub use common_devices::{Kernels, KernelsA, KernelsB};
pub use operators::{cuda, nvidia_gpu::Handle as Gpu};
pub use tensor::{reslice, reslice_mut, slice, split, udim, LocalSplitable, Tensor};

#[cfg(detected_nccl)]
pub use operators::nccl;

pub struct NvidiaKernels(HashMap<i32, Internal>);

struct Internal {
    mat_mul: mat_mul::Operator,
    rms_norm: rms_norm::Operator,
    rope: rope::Operator,
    reform: reform::Operator,
    softmax: softmax::Operator,
    mlp: mlp::Operator,
    random_sample: random_sample::Operator,
}

impl Internal {
    pub fn new(
        handle: &Gpu,
        dt_norm: DigitLayout,
        dt_mat: DigitLayout,
        d: usize,
        voc: usize,
    ) -> Self {
        let d = d as u64;
        let hidden_layout = TensorLayout::init(dt_mat, [dyn_(), d.into()], [dyn_(); 2]);
        let mat_mul = mat_mul::Operator::new(handle);

        let mut rms_norm = rms_norm::Operator::new(handle);
        rms_norm
            .scheme(&operators::rms_norm::Args {
                y_layout: hidden_layout.clone(),
                y_base: null_mut(),
                x_layout: hidden_layout.clone(),
                x_base: null(),
                w_layout: TensorLayout::init(dt_norm, [d.into()], [dyn_()]),
                w_base: null(),
                epsilon: 0.,
            })
            .unwrap();

        let mut rope = rope::Operator::new(handle);
        rope.scheme(&operators::rope::Args {
            t_layout: TensorLayout::dyn_(dt_mat, 3),
            t_base: null_mut(),
            p_layout: TensorLayout::dyn_(U32, 1),
            p_base: null(),
            theta: 0.,
        })
        .unwrap();

        let mut reform = reform::Operator::new(handle);
        reform
            .scheme(&operators::reform::Args {
                dst_layout: TensorLayout::dyn_(dt_mat, 2),
                dst_base: null_mut(),
                src_layout: TensorLayout::dyn_(dt_mat, 2),
                src_base: null(),
            })
            .unwrap();

        let mut softmax = softmax::Operator::new(handle);
        softmax
            .scheme(&operators::fuesd_softmax::Args {
                att_layout: TensorLayout::dyn_(dt_mat, 3),
                att_base: null_mut(),
            })
            .unwrap();

        let mut mlp = mlp::Operator::new(handle);
        mlp.scheme(&operators::mlp::Args {
            y_layout: hidden_layout.clone(),
            y_base: null_mut(),
            x_layout: hidden_layout.clone(),
            x_base: null(),
            gate_up_layout: TensorLayout::dyn_(dt_mat, 2),
            gate_up_base: null_mut(),
            w_gate_up_layout: TensorLayout::dyn_(dt_mat, 2),
            w_gate_up_base: null(),
            w_down_layout: TensorLayout::dyn_(dt_mat, 2),
            w_down_base: null(),
            down_alpha: 1.,
            down_bias: true,
        })
        .unwrap();

        let mut random_sample = random_sample::Operator::new(handle);
        random_sample
            .scheme(&operators::random_sample::Args::new(dt_mat, voc))
            .unwrap();

        Self {
            mat_mul,
            rms_norm,
            rope,
            reform,
            softmax,
            mlp,
            random_sample,
        }
    }
}

impl NvidiaKernels {
    pub fn new(
        devices: &[Device],
        dt_norm: DigitLayout,
        dt_mat: DigitLayout,
        rms_norm_size: usize,
        voc_size: usize,
    ) -> Self {
        Self(
            devices
                .iter()
                .map(|d| {
                    (
                        unsafe { d.as_raw() },
                        Internal::new(
                            &Gpu::new(d.retain_primary()),
                            dt_norm,
                            dt_mat,
                            rms_norm_size,
                            voc_size,
                        ),
                    )
                })
                .collect(),
        )
    }

    fn get(&self, queue: &QueueOf<Gpu>) -> &Internal {
        self.0.get(&unsafe { queue.ctx().dev().as_raw() }).unwrap()
    }

    pub fn sample_workspace<'ctx>(&self, queue: &QueueOf<'ctx, Gpu>) -> DevMem<'ctx> {
        self.get(queue).random_sample.workspace(queue)
    }

    pub fn sample(
        &self,
        voc_size: usize,
        args: impl IntoIterator<Item = SampleArgs>,
        logits: &[DevByte],
        workspace: &mut [DevByte],
        stream: &Stream,
    ) -> Vec<utok> {
        let random_sample = &self.get(stream).random_sample;
        let logits = logits.as_ptr();

        let details = args.into_iter().collect::<Vec<_>>();
        let kv_pair_size = KVPair::<()>::LAYOUT.nbytes();
        let mut kv_pairs = stream.malloc::<u8>(details.len() * kv_pair_size);

        let mut args = operators::random_sample::Args::<Gpu>::new(F16, voc_size);
        args.workspace = Workspace {
            ptr: workspace.as_mut_ptr(),
            len: workspace.len(),
        };
        for (i, detail) in details.iter().enumerate() {
            args.kv_pair_base = unsafe { kv_pairs.as_mut_ptr().add(i * kv_pair_size) };
            args.data_base = unsafe { logits.add(i * voc_size * F16.nbytes()) };
            args.detail = *detail;
            random_sample.launch(&args, stream).unwrap();
        }

        let mut host = vec![KVPair::new(0, f16::ZERO); details.len()];
        stream.synchronize();
        memcpy_d2h(&mut host, &kv_pairs);

        host.into_iter().map(|kv| kv.idx() as _).collect()
    }
}

impl Kernels<Gpu> for NvidiaKernels {}

impl Operators for NvidiaKernels {
    type Handle = Gpu;

    fn reform_op(
        &self,
        queue: &QueueOf<Self::Handle>,
    ) -> &impl operators::reform::Reform<Self::Handle> {
        &self.get(queue).reform
    }

    fn rms_norm_op(
        &self,
        queue: &QueueOf<Self::Handle>,
    ) -> &impl operators::rms_norm::RmsNorm<Self::Handle> {
        &self.get(queue).rms_norm
    }

    fn mat_mul_op(
        &self,
        queue: &QueueOf<Self::Handle>,
    ) -> &impl operators::mat_mul::MatMul<Self::Handle> {
        &self.get(queue).mat_mul
    }

    fn rope_op(&self, queue: &QueueOf<Self::Handle>) -> &impl operators::rope::Rope<Self::Handle> {
        &self.get(queue).rope
    }

    fn softmax_op(
        &self,
        queue: &QueueOf<Self::Handle>,
    ) -> &impl operators::fuesd_softmax::FusedSoftmax<Self::Handle> {
        &self.get(queue).softmax
    }

    fn mlp_op(&self, queue: &QueueOf<Self::Handle>) -> &impl operators::mlp::Mlp<Self::Handle> {
        &self.get(queue).mlp
    }
}

impl KernelsB for NvidiaKernels {
    type Handle = Gpu;

    fn gather<T, U, I>(
        &self,
        x: &mut Tensor<T>,
        table: &Tensor<U>,
        tokens: I,
        queue: &QueueOf<Self::Handle>,
    ) where
        T: DerefMut<Target = SliceOn<Self::Handle>>,
        U: Deref<Target = [u8]>,
        I: IntoIterator<Item = utok>,
    {
        gather::gather(x, table, tokens, queue);
    }
}

pub fn synchronize() {
    if let Err(cuda::NoDevice) = cuda::init() {
        return;
    }
    for i in 0..cuda::Device::count() {
        cuda::Device::new(i as _)
            .retain_primary()
            .apply(|ctx| ctx.synchronize());
    }
}
