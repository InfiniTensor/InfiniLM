#[allow(warnings)]
mod bindings {
    include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

    #[macro_export]
    macro_rules! check {
        ($f:expr) => {{
            #[allow(unused_imports)]
            use $crate::op::random_sample::bindings::*;
            #[allow(unused_unsafe, clippy::macro_metavars_in_unsafe)]
            let err = unsafe { $f };
            assert_eq!(err, cudaError_t::hcSuccess);
        }};
    }
}

use crate::{
    check,
    utils::{dims, offset_ptr, strides},
};
use ggus::ggml_quants::f16;
use nn::{
    Tensor,
    digit_layout::{layout, types as ty},
};
use operators::cuda::{AsRaw, CurrentCtx, DevMem, Stream, VirByte};
use std::ffi::c_uint;

#[derive(Clone, Copy)]
#[repr(C)]
pub struct KVPair {
    pub idx: c_uint,
    pub val: f16,
}

impl KVPair {
    pub const ZERO: Self = Self {
        idx: 0,
        val: f16::ZERO,
    };
}

layout!(KV_PAIR = "kvpair"; [1] in 8);

#[derive(Clone, Copy, Debug)]
pub struct SampleArgs {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SampleArgsError {
    NegativeTemperature,
    NonPositiveTop,
}

impl Default for SampleArgs {
    #[inline]
    fn default() -> Self {
        Self::ARG_MAX
    }
}

impl SampleArgs {
    pub const ARG_MAX: Self = Self {
        temperature: 0.,
        top_p: 1.,
        top_k: usize::MAX,
    };

    pub fn new(temperature: f32, top_p: f32, top_k: usize) -> Result<Self, SampleArgsError> {
        if temperature < 0. {
            return Err(SampleArgsError::NegativeTemperature);
        }
        if top_k == 0 || top_p <= 0. {
            return Err(SampleArgsError::NonPositiveTop);
        }
        Ok(Self {
            temperature,
            top_p: f32::min(top_p, 1.),
            top_k,
        })
    }

    #[inline]
    pub fn is_argmax(&self) -> bool {
        self.temperature == 0. || self.top_k == 1
    }
}

pub struct RandomSample<'ctx> {
    workspace: DevMem<'ctx>,
    indices: DevMem<'ctx>,
}

impl<'ctx> RandomSample<'ctx> {
    pub fn new(n: usize, ctx: &'ctx CurrentCtx) -> Self {
        let mut argmax_size = 0;
        let mut sample_size = 0;
        check!(calculate_workspace_size_half(
            &mut argmax_size,
            &mut sample_size,
            n,
        ));
        Self {
            workspace: ctx.malloc::<u8>(argmax_size.max(sample_size)),
            indices: ctx.from_host(&(0..n as c_uint).collect::<Vec<_>>()),
        }
    }
}

impl RandomSample<'_> {
    pub fn argmax<const N: usize>(
        &mut self,
        kv_pair: Tensor<*const VirByte, N>,
        logits: Tensor<*const VirByte, N>,
        stream: &Stream,
    ) {
        assert_eq!(kv_pair.dt(), KV_PAIR);
        assert_eq!(logits.dt(), ty::F16);

        dims!([] = kv_pair);
        dims!([n] = logits);

        strides!([sl] = logits);
        assert_eq!(sl, logits.dt().nbytes() as _);

        check!(argmax_half(
            offset_ptr(&kv_pair).cast_mut().cast(),
            offset_ptr(&logits).cast(),
            n,
            self.workspace.as_mut_ptr().cast(),
            self.workspace.len(),
            stream.as_raw().cast(),
        ))
    }

    pub fn sample<const N: usize>(
        &mut self,
        kv_pair: Tensor<*const VirByte, N>,
        logits: Tensor<*const VirByte, N>,
        args: SampleArgs,
        seed: f32,
        stream: &Stream,
    ) {
        assert_eq!(kv_pair.dt(), KV_PAIR);
        assert_eq!(logits.dt(), ty::F16);

        dims!([] = kv_pair);
        dims!([n] = logits);

        strides!([sl] = logits);
        assert_eq!(sl, logits.dt().nbytes() as isize);

        check!(sample_half(
            offset_ptr(&kv_pair).cast_mut().cast(),
            offset_ptr(&logits).cast(),
            self.indices.as_ptr().cast(),
            n,
            seed,
            args.temperature,
            args.top_p,
            args.top_k,
            self.workspace.as_mut_ptr().cast(),
            self.workspace.len(),
            stream.as_raw().cast(),
        ));
    }
}
