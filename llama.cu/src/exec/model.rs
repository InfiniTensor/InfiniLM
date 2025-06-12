use super::step::Step;
use crate::{
    batch::Req,
    handle::Handle,
    memory::MemPages,
    utils::{self, destruct},
};
use bytesize::ByteSize;
use log::trace;
use nn::{NNGraph, Tensor};
use operators::{
    attention_kv_cached::cuda::Operator as Attn,
    cuda::{DevByte, Stream, VirByte, VirMem},
};
use std::time::Instant;

pub(super) struct ModelExec<'ctx> {
    execs: Box<[Step<'ctx>]>,
    workspace: VirMem,
    inputs: Box<[Tensor<*const VirByte, 2>]>,
    outputs: Box<[Tensor<*const VirByte, 2>]>,
}

impl<'ctx> ModelExec<'ctx> {
    pub fn new(
        graph: NNGraph<Tensor<*const VirByte, 2>>,
        n_tok: usize,
        handle: &mut Handle<'ctx>,
        pages: &mut MemPages,
        use_cuda_graph: bool,
    ) -> Self {
        let graph = graph.lower(&[("n_tok", n_tok)].into(), |t| t);

        let mem_range_map = graph.mem_range_map(8 << 30, 512);

        let mut workspace = pages.reserve_vir(mem_range_map.range.len());
        let ptr = workspace.as_ptr();
        let graph = graph.lower(
            |key| unsafe { ptr.byte_add(mem_range_map.map[&key].start) },
            |&data| data,
        );
        let inputs: Box<[Tensor<*const VirByte, 2>]> = graph
            .0
            .topo
            .global_inputs()
            .map(|i| graph.0.edges[i].clone())
            .collect::<Box<_>>();
        let outputs = graph
            .0
            .topo
            .global_outputs()
            .iter()
            .map(|&i| graph.0.edges[i].clone())
            .collect::<Box<_>>();
        let exec = graph.into_exec();

        // memcpy node 要求当时虚地址有对应的物理页
        pages.map(&mut workspace, ..);

        // 构造 cuda graph
        let time = Instant::now();
        let execs = handle.build_steps(exec, use_cuda_graph);
        trace!(
            "model compiled @{} in {:.2?}, seq len = {n_tok}, workspace = {}",
            handle.ctx.dev().index(),
            time.elapsed(),
            ByteSize::b(workspace.len() as _).display(),
        );

        // 解除映射回收物理页
        pages.unmap(&mut workspace, ..);

        Self {
            execs,
            workspace,
            inputs,
            outputs,
        }
    }
}

impl ModelExec<'_> {
    /// 映射虚页
    pub fn map(&mut self, pages: &mut MemPages) {
        pages.map(&mut self.workspace, ..)
    }

    /// 解映射虚页
    pub fn unmap(&mut self, pages: &mut MemPages) {
        pages.unmap(&mut self.workspace, ..)
    }

    pub fn tok_buf(&mut self) -> &mut [DevByte] {
        as_mapped(&self.inputs[0])
    }

    pub fn pos_buf(&mut self) -> &mut [DevByte] {
        as_mapped(&self.inputs[1])
    }

    pub fn launch(
        &mut self,
        attn: &Attn,
        handle: &mut Handle,
        reqs: &[Req<Tensor<*const VirByte, 2>>],
        stream: &Stream,
    ) -> Tensor<*const VirByte, 2> {
        // 执行
        for exec in &self.execs {
            match exec {
                Step::Graph(graph, stub) => {
                    stream.launch_graph(graph);
                    if !stub.is_empty() {
                        for t in stub {
                            utils::fmt(t, stream.ctx())
                        }
                        std::process::exit(0);
                    }
                }
                Step::Attention(box_) => handle.launch_attn(attn, box_, reqs, stream),
                Step::Exec(exec) => handle.launch_nn_exec(exec, stream),
            }
        }
        destruct!([x] = self.outputs.clone());
        x
    }
}

#[allow(clippy::mut_from_ref)]
fn as_mapped(input: &Tensor<*const VirByte, 2>) -> &mut [DevByte] {
    let ptr = input.get().cast_mut();
    let len = Tensor::use_info(input).take();
    unsafe { std::slice::from_raw_parts_mut(ptr.cast(), len) }
}
