use crate::Tensor;
use ggus::ggml_quants::digit_layout::types as ty;
use operators::{
    random_sample::{self, Indices, RandomSample as Trait, SampleArgs},
    Hardware, LaunchError, QueueAlloc, TopoNode,
};
use std::{
    marker::PhantomData,
    ops::{Deref, DerefMut},
    ptr::{null, null_mut},
};

pub struct RandomSample<H, Op>(Op, PhantomData<H>);

impl<H, Op> RandomSample<H, Op>
where
    H: Hardware,
    Op: Trait<H>,
{
    pub fn build_indices<QA>(n: usize, queue_alloc: &QA) -> Tensor<QA::DevMem>
    where
        QA: QueueAlloc<Hardware = H>,
    {
        let Indices { n, mem } = Op::build_indices(n, queue_alloc);
        Tensor::new(ty::U32, &[n]).map(|_| mem)
    }

    pub fn new(node: &impl TopoNode<H>) -> Self {
        Self(Op::new(node.processor()), PhantomData)
    }

    pub fn launch<Pair, L, I, QA>(
        &self,
        pairs: &mut Tensor<Pair>,
        logits: &Tensor<L>,
        indices: &Tensor<I>,
        config: SampleArgs,
        workspace: &mut [H::Byte],
        queue_alloc: &QA,
    ) -> Result<(), LaunchError>
    where
        Pair: DerefMut<Target = [H::Byte]>,
        L: Deref<Target = [H::Byte]>,
        I: Deref<Target = [H::Byte]>,
        QA: QueueAlloc<Hardware = H>,
    {
        let layout = indices.layout();
        let mut args = random_sample::Args {
            kv_pair: layout.clone(),
            kv_pair_base: null_mut(),
            logits: layout.clone(),
            logits_base: null(),
            indices: layout.clone(),
            indices_base: indices.base(),
            config,
            seed: 0.,
        };

        for i in 0..logits.shape()[0] {
            let mut pair = pairs.map_slice_mut().index(0, i);
            let logits = logits.map_slice().index(0, i);

            args.kv_pair = pair.layout();
            args.kv_pair_base = pair.base_mut();
            args.logits = logits.layout();
            args.logits_base = logits.base();
            args.seed = rand::random();

            self.0.launch(&args, workspace, queue_alloc)?;
        }

        Ok(())
    }
}
