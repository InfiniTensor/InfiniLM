use llama::{ext::Mmap, LlamaStorage, Tensor, WeightLoader};
use operators::{
    all_reduce::NonAllReduce,
    cuda::{memcpy_d2h, DevByte},
    nvidia_gpu::Gpu,
    ByteOf,
};
use std::ops::Deref;

pub struct Llama {}

impl Llama {
    pub fn new(_storage: Box<[Mmap]>, _model: LlamaStorage<&'static [u8]>) -> Self {
        Self {}
    }

    pub fn infer(&mut self, input: &[u32], cache: &mut [u8], pos: usize) -> u32 {
        todo!()
    }
}

struct Operators;

macro_rules! op {
    ($name:ident) => {
        operators::$name::nvidia_gpu::Operator
    };
}

impl llama::Operators for Operators {
    type Hardware = Gpu;
    type TopoNode = Gpu;
    type RmsNorm = op!(rms_norm);
    type MatMul = op!(mat_mul);
    type Rope = op!(rope);
    type AttnKVCached = op!(attention_kv_cached);
    type Mlp = op!(mlp);
    type Rearrange = op!(rearrange);
    type AllReduce = NonAllReduce<Gpu>;

    fn debug<T>(tensor: &Tensor<T>)
    where
        T: Deref<Target = [ByteOf<Self::Hardware>]>,
    {
        let tensor = tensor.as_ref().map(|mem| {
            let mut buf = vec![0u8; mem.len()];
            memcpy_d2h(&mut buf, mem);
            buf
        });
        println!("{tensor}");
    }
}

struct Weights {}

impl WeightLoader for Weights {
    type Hardware = Gpu;
    type Memory<'s> = &'s [DevByte];

    fn load_blk(
        &self,
        which: llama::BlkWeight,
        iblk: usize,
        queue: &operators::QueueOf<Self::Hardware>,
    ) -> Self::Memory<'_> {
        todo!()
    }

    fn output_norm(&self, queue: &operators::QueueOf<Self::Hardware>) -> Self::Memory<'_> {
        todo!()
    }

    fn output(&self, queue: &operators::QueueOf<Self::Hardware>) -> Self::Memory<'_> {
        todo!()
    }
}
