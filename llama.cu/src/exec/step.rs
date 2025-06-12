use crate::{
    batch::Req,
    handle::Handle,
    op::{self, Operator as _},
    utils::{destruct, layout, offset_ptr},
};
use nn::{Arg, Named, Tensor};
use operators::{
    Operator as _,
    attention_kv_cached::{Args as AttnArgs, cuda::Operator as Attn},
    cuda::{CaptureStream, GraphExec, Stream, VirByte},
};
use regex::Regex;
use std::{fmt, sync::LazyLock};

pub(super) enum Step<'ctx> {
    Graph(GraphExec<'ctx>, Box<[Tensor<*const VirByte, 2>]>),
    Attention(Box<Attention>),
    Exec(nn::Exec<*const VirByte>),
}

pub(super) struct Attention {
    pub iblk: usize,
    pub q: Tensor<*const VirByte, 2>,
    pub k: Tensor<*const VirByte, 2>,
    pub v: Tensor<*const VirByte, 2>,
    pub o: Tensor<*const VirByte, 2>,
}

impl<'ctx> Handle<'ctx> {
    pub(super) fn build_steps(
        &mut self,
        exec: impl IntoIterator<Item = nn::Exec<*const VirByte>>,
        use_cuda_graph: bool,
    ) -> Box<[Step<'ctx>]> {
        let mut stream: Option<CaptureStream<'_>> = None;
        let mut exec_ = Vec::new();
        for exec in exec {
            if exec.node.value.name == "attention" {
                static REGEX: LazyLock<Regex> =
                    LazyLock::new(|| Regex::new(r"^Ω\.blk(\d+)\.attn:attention$").unwrap());

                if let Some(stream) = stream.take() {
                    exec_.push(Step::Graph(
                        self.ctx.instantiate(&stream.end()),
                        Default::default(),
                    ))
                }

                let nn::Exec {
                    node: Named { name, value: op },
                    inputs,
                    outputs,
                } = exec;

                destruct!([q, k, v] = inputs);
                destruct!([o] = outputs);
                let Some(nn::Arg::Int(dh)) = op.arg else {
                    panic!()
                };
                let dh = dh as usize;
                // [n, nh * dh] -> [n, nh, dh] -> [nh, n, dh]
                let transform = |t: Tensor<*const VirByte, 2>| {
                    t.transform(|layout| {
                        layout
                            .tile_be(1, &[layout.shape()[1] / dh, dh])
                            .transpose(&[1, 0])
                    })
                };
                let q = transform(q);
                let k = transform(k);
                let v = transform(v);
                let o = transform(o);

                let iblk = {
                    let (_, [iblk]) = REGEX.captures(&name).unwrap().extract();
                    iblk.parse().unwrap()
                };
                exec_.push(Step::Attention(Box::new(Attention { iblk, q, k, v, o })));
                continue;
            }
            if use_cuda_graph {
                self.launch_nn_exec(
                    &exec,
                    stream.get_or_insert_with(|| self.ctx.stream().capture()),
                )
            } else {
                exec_.push(Step::Exec(exec))
            }
        }
        if let Some(stream) = stream.take() {
            exec_.push(Step::Graph(
                self.ctx.instantiate(&stream.end()),
                Default::default(),
            ))
        }
        exec_.into()
    }

    pub(super) fn launch_nn_exec(&mut self, exec: &nn::Exec<*const VirByte>, stream: &Stream) {
        let nn::Exec {
            node,
            inputs,
            outputs,
        } = exec;
        let op = &node.value;
        macro_rules! launch {
            ($op:ident) => {
                op::$op::launch(
                    self,
                    op.arg.clone(),
                    inputs.clone(),
                    outputs.clone(),
                    &stream,
                )
            };
        }
        match &*op.name {
            "embedding" => launch!(Embedding),
            "rms-norm" => launch!(RmsNorm),
            "linear" => launch!(Linear),
            "rope" => launch!(Rope),
            "swiglu" => launch!(Swiglu),
            #[cfg(nccl)]
            "all-reduce" => launch!(AllReduce),
            "empty" => {}
            _ => panic!(
                "{}",
                ErrorFmt {
                    name: &node.name,
                    ty: &op.name,
                    arg: &op.arg,
                    inputs,
                    outputs,
                }
            ),
        }
    }

    pub(super) fn launch_attn(
        &mut self,
        op: &Attn,
        attn: &Attention,
        reqs: &[Req<Tensor<*const VirByte, 2>>],
        stream: &Stream,
    ) {
        let Attention { iblk, q, k, v, o } = attn;
        let mut start = 0;
        for req in reqs {
            // [nkvh, 2, nctx, dh]
            let cache = req.cache.clone();
            let cache = cache.transform(|layout| layout.index(1, *iblk));
            let k_cache = cache.clone().transform(|layout| layout.index(1, 0));
            let v_cache = cache.clone().transform(|layout| layout.index(1, 1));
            // [nh, n, dh]
            let len = req.seq;
            let q = q.clone().transform(|layout| layout.slice(1, start, 1, len));
            let k = k.clone().transform(|layout| layout.slice(1, start, 1, len));
            let v = v.clone().transform(|layout| layout.slice(1, start, 1, len));
            let o = o.clone().transform(|layout| layout.slice(1, start, 1, len));
            start += len;
            op.launch(
                &AttnArgs {
                    q_layout: layout(&q),
                    q_base: offset_ptr(&q).cast_mut().cast(),
                    k_layout: layout(&k),
                    k_base: offset_ptr(&k).cast(),
                    v_layout: layout(&v),
                    v_base: offset_ptr(&v).cast(),
                    o_layout: layout(&o),
                    o_base: offset_ptr(&o).cast_mut().cast(),
                    k_cache_layout: layout(&k_cache),
                    k_cache_base: offset_ptr(&k_cache).cast_mut().cast(),
                    v_cache_layout: layout(&v_cache),
                    v_cache_base: offset_ptr(&v_cache).cast_mut().cast(),
                    mask: operators::fuesd_softmax::AttnMask::Causal,
                    pos: req.pos as _,
                },
                &mut [],
                stream,
            )
            .unwrap()
        }
    }
}

struct ErrorFmt<'a> {
    name: &'a str,
    ty: &'a str,
    arg: &'a Option<Arg>,
    inputs: &'a [Tensor<*const VirByte, 2>],
    outputs: &'a [Tensor<*const VirByte, 2>],
}

impl fmt::Display for ErrorFmt<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let &Self {
            name,
            ty,
            arg,
            inputs,
            outputs,
        } = self;
        write!(f, "todo! [{ty}] {name} ({arg:?})")?;
        for t in inputs {
            write!(f, " {}{:?}", t.dt(), t.shape())?
        }
        write!(f, " ->")?;
        for t in outputs {
            write!(f, " {}{:?}", t.dt(), t.shape())?
        }
        writeln!(f)
    }
}
