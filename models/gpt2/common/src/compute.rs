use super::{args::Args, Gpt2Meta};
use itertools::izip;
use operators::{
    add::{self, Add},
    add_rows::{self, AddRows},
    all_reduce::{self, AllReduce, ReduceOp},
    attention_kv_cached::{self, AttnKVCached},
    gelu::{self, Gelu},
    layer_norm::{self, LayerNorm},
    mat_mul::{self, MatMul},
    rearrange::{self, Rearrange},
    ByteOf, Hardware, LaunchError, Operator, QueueAlloc, QueueOf, TopoNode, Workspace,
};
use std::ops::{Deref, DerefMut};
use tensor::{split, Tensor};

pub trait Operators {
    type Hardware: Hardware;
    type TopoNode: TopoNode<Self::Hardware>;
    type AddRows: AddRows<Self::Hardware>;
    type LayerNorm: LayerNorm<Self::Hardware>;
    type MatMul: MatMul<Self::Hardware>;
    type AttnKVCached: AttnKVCached<Self::Hardware>;
    type Gelu: Gelu<Self::Hardware>;
    type Add: Add<Self::Hardware>;
    type Rearrange: Rearrange<Self::Hardware>;
    type AllReduce: AllReduce<Self::Hardware, Self::TopoNode>;

    fn debug<T>(tensor: &Tensor<T>)
    where
        T: Deref<Target = [ByteOf<Self::Hardware>]>;
    fn memcpy_d2h<T: Copy>(
        dst: &mut [T],
        src: &[ByteOf<Self::Hardware>],
        queue: &QueueOf<Self::Hardware>,
    );
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum BlkWeight {
    AttnNorm,
    AttnQKV,
    AttnO,
    FfnNorm,
    FfnUp,
    FfnDown,
}

pub trait WeightLoader {
    type Hardware: Hardware;
    type Memory<'s>: Deref<Target = [ByteOf<Self::Hardware>]> + 's
    where
        Self: 's;

    fn load_blk(
        &self,
        which: BlkWeight,
        iblk: usize,
        queue: &QueueOf<Self::Hardware>,
    ) -> [Self::Memory<'_>; 2];

    fn output_norm(&self, queue: &QueueOf<Self::Hardware>) -> [Self::Memory<'_>; 2];
    fn output(&self, queue: &QueueOf<Self::Hardware>) -> Self::Memory<'_>;
    fn pos_embd<'a>(&'a self, queue: &'a QueueOf<Self::Hardware>) -> Self::Memory<'a>;
}

pub struct Gpt2Worker<Ops: Operators, W> {
    #[allow(dead_code)]
    id: usize,
    meta: Gpt2Meta,
    weights: WeightDecorator<W>,
    add_rows: Ops::AddRows,
    layer_norm: Ops::LayerNorm,
    mat_mul: Ops::MatMul,
    attn_kv_cached: Ops::AttnKVCached,
    gelu: Ops::Gelu,
    add: Ops::Add,
    rearrange: Ops::Rearrange,
    all_reduce: Ops::AllReduce,
    pub debug: bool,
}

impl<Ops: Operators, W> Gpt2Worker<Ops, W> {
    pub fn new(id: usize, node: &Ops::TopoNode, meta: Gpt2Meta, weights: W) -> Self {
        let processor = node.processor();
        Self {
            id,
            weights: meta.decorator(weights), // meta.decorator
            meta,
            add_rows: Ops::AddRows::new(processor),
            layer_norm: Ops::LayerNorm::new(processor),
            mat_mul: Ops::MatMul::new(processor),
            attn_kv_cached: Ops::AttnKVCached::new(processor),
            gelu: Ops::Gelu::new(processor),
            add: Ops::Add::new(processor),
            rearrange: Ops::Rearrange::new(processor),
            all_reduce: Ops::AllReduce::new(node),
            debug: true,
        }
    }

    #[inline]
    pub const fn meta(&self) -> &Gpt2Meta {
        &self.meta
    }

    pub fn workspace_size(&self, nt: usize, max_seq_len: usize, max_att_len: usize) -> usize {
        let Gpt2Meta {
            nh, nkvh, dh, di, ..
        } = self.meta;

        let embd = self.meta.embd(nt);
        let dt = embd.dt();
        let embd = embd.take();

        let qkv = Tensor::new(dt, &[nt * (nh + nkvh + nkvh), dh]).take();
        let q = Tensor::new(dt, &[max_seq_len, nh, dh]).take();
        let att = Tensor::new(dt, &[nh, max_seq_len, max_att_len]).take();

        let up = Tensor::new(dt, &[nt, di]).take();
        embd + (qkv + q + att).max(up)
    }
}

impl<Ops, W> Gpt2Worker<Ops, W>
where
    Ops: Operators,
    W: WeightLoader<Hardware = Ops::Hardware>,
    ByteOf<Ops::Hardware>: 'static,
{
    pub fn launch<QA>(
        &mut self,
        args: Args<Ops::Hardware>,
        workspace: &mut [ByteOf<Ops::Hardware>],
        queue_alloc: &QA,
    ) -> Result<(), LaunchError>
    where
        QA: QueueAlloc<Hardware = Ops::Hardware>,
    {
        let Args {
            mut embd,
            mut logits,
            mut requests,
            max_seq_len,
            max_att_len,
            idx,
        } = args;
        let Gpt2Meta {
            nblk,
            nh,
            nkvh,
            dh,
            di,
            ..
        } = self.meta;

        let queue = queue_alloc.queue();
        {
            let pos_embd = self.weights.pos_embd(queue);
            self.add_rows(&mut embd, &pos_embd, &idx, workspace, queue_alloc)?
        }

        let nt = embd.shape()[0];
        let mut x = embd;
        let x1 = Tensor::new(x.dt(), x.shape());
        let qkv = Tensor::new(x.dt(), &[nt, (nh + nkvh + nkvh) * dh]);
        let up = Tensor::new(x.dt(), &[nt, di]);

        let workspace_size = self.workspace_size(nt, max_seq_len, max_att_len);
        let mut workspace = Workspace::new(queue_alloc, workspace, workspace_size);
        let (buf, workspace) = workspace.split_at_mut(*x1.get());
        let mut x1 = x1.map(|_| buf);

        let req_split = requests.iter().map(|req| req.seq_len).collect::<Vec<_>>();

        for iblk in 0..nblk {
            {
                let wb = self.weights.attn_norm(iblk, queue);
                self.layer_norm(&mut x1, &x, wb, workspace, queue_alloc)?;

                let (buf, workspace) = workspace.split_at_mut(*qkv.get());
                let mut qkv = qkv.clone().map(|_| buf);

                let [w, b] = self.weights.attn_qkv(iblk, queue);
                self.mat_mul(&mut qkv, &x1, (w, Some(b)), workspace, queue_alloc)?;

                let qkv = qkv.tile(1, &[nh + nkvh + nkvh, dh]);
                split!(qkv => q, k, v; [nh, nkvh, nkvh] @ 1);
                let mut q = q;
                let k = k;
                let v = v;
                {
                    let q = q.map_slice_mut().transpose(&[1, 0]);
                    let k = k.map_slice().transpose(&[1, 0]);
                    let v = v.map_slice().transpose(&[1, 0]);
                    let q = q.split(1, &req_split);
                    let k = k.split(1, &req_split);
                    let v = v.split(1, &req_split);

                    for (mut q, k, v, req) in izip!(q, k, v, &mut requests) {
                        let cache = req
                            .cache
                            .as_mut() // [buf, nblk, 2, nkvh, dh]
                            .index(1, iblk) // [buf, 2, nkvh, dh]
                            .transpose(&[2, 0]) // [nkvh, 2, buf, dh]
                            .map(|t| &mut t[..]);

                        split!(cache => kc, vc; [1, 1] @ 1);
                        let mut o = unsafe { q.map_slice_static_mut() };
                        self.attn_kv_cached(
                            &mut q,
                            &k,
                            &v,
                            &mut o,
                            &mut kc.index(1, 0),
                            &mut vc.index(1, 0),
                            req.pos,
                            workspace,
                            queue_alloc,
                        )?
                    }
                }
                let o = q.map_slice().merge(1..3).unwrap();
                let [w, b] = self.weights.attn_o(iblk, queue);
                self.mat_mul(&mut x1, &o, (w, Some(b)), workspace, queue_alloc)?
            }
            let inplace = unsafe { x.map_slice_static() };
            self.add(&mut x, &inplace, &x1, workspace, queue_alloc)?;
            self.all_reduce(&mut x, workspace, queue_alloc)?;

            let wb = self.weights.ffn_norm(iblk, queue);
            self.layer_norm(&mut x1, &x, wb, workspace, queue_alloc)?;
            {
                let (buf, workspace) = workspace.split_at_mut(*up.get());
                let mut up = up.clone().map(|_| buf);

                let [w, b] = self.weights.ffn_up(iblk, queue);
                self.mat_mul(&mut up, &x1, (w, Some(b)), workspace, queue_alloc)?;

                self.gelu(&mut up, workspace, queue_alloc)?;

                let [w, b] = self.weights.ffn_down(iblk, queue);
                self.mat_mul(&mut x1, &up, (w, Some(b)), workspace, queue_alloc)?
            }
            let inplace = unsafe { x.map_slice_static() };
            self.add(&mut x, &inplace, &x1, workspace, queue_alloc)?;
            self.all_reduce(&mut x, workspace, queue_alloc)?
        }
        if logits.shape()[0] == 0 {
            return Ok(());
        }

        // 集中要采样的 token
        // NOTICE: 输入之前将请求按 seq len 升序排列可降低移动开销
        let mut dst = 0;
        let mut src = 0;
        for req in &requests {
            src += req.seq_len;
            for src in src - req.out_len..src {
                if src != dst {
                    let src = unsafe { x.map_slice_static() }.index(0, src);
                    let mut dst = x.map_slice_mut().index(0, dst);
                    self.rearrange(&mut dst, &src, workspace, queue_alloc)?;
                }
                dst += 1;
            }
        }
        assert_eq!(dst, logits.shape()[0]);

        let mut x = x.map_slice_mut().slice(0, 0, 1, dst);

        let inplace = unsafe { x.map_slice_static() };
        let wb = self.weights.output_norm(queue);
        self.layer_norm(&mut x, &inplace, wb, workspace, queue_alloc)?;

        let w = self.weights.output_weight(queue).transpose(&[1, 0]);
        self.mat_mul(&mut logits, &x, (w, None), workspace, queue_alloc)
    }
}

#[allow(clippy::too_many_arguments)]
impl<Ops, W> Gpt2Worker<Ops, W>
where
    Ops: Operators,
    W: WeightLoader<Hardware = Ops::Hardware>,
{
    fn add_rows<Dst, Src, Idx, QA>(
        &self,
        dst: &mut Tensor<Dst>,
        src: &Tensor<Src>,
        idx: &Tensor<Idx>,
        workspace: &mut [ByteOf<Ops::Hardware>],
        queue_alloc: &QA,
    ) -> Result<(), LaunchError>
    where
        Dst: DerefMut<Target = [ByteOf<Ops::Hardware>]>,
        Src: Deref<Target = [ByteOf<Ops::Hardware>]>,
        Idx: Deref<Target = [ByteOf<Ops::Hardware>]>,
        QA: QueueAlloc<Hardware = Ops::Hardware>,
    {
        let n = dst.shape()[0];
        let mut dst = dst.map_slice_mut().tile(0, &[1, n]);
        self.add_rows.launch(
            &add_rows::Args {
                dst_layout: dst.layout(),
                dst_base: dst.base_mut(),
                src_layout: src.layout(),
                src_base: src.base(),
                idx_layout: idx.layout(),
                idx_base: idx.base(),
            },
            workspace,
            queue_alloc,
        )
    }

    fn layer_norm<Y, X, WB, QA>(
        &self,
        y: &mut Tensor<Y>,
        x: &Tensor<X>,
        [w, b]: [Tensor<WB>; 2],
        workspace: &mut [ByteOf<Ops::Hardware>],
        queue_alloc: &QA,
    ) -> Result<(), LaunchError>
    where
        Y: DerefMut<Target = [ByteOf<Ops::Hardware>]>,
        X: Deref<Target = [ByteOf<Ops::Hardware>]>,
        WB: Deref<Target = [ByteOf<Ops::Hardware>]>,
        QA: QueueAlloc<Hardware = Ops::Hardware>,
    {
        self.layer_norm.launch(
            &layer_norm::Args {
                y_layout: y.layout(),
                y_base: y.base_mut(),
                x_layout: x.layout(),
                x_base: x.base(),
                scale_layout: w.layout(),
                scale_base: w.base(),
                bias_layout: b.layout(),
                bias_base: b.base(),
                epsilon: self.meta.epsilon,
            },
            workspace,
            queue_alloc,
        )
    }

    fn mat_mul<C, A, WB, QA>(
        &self,
        c: &mut Tensor<C>,
        a: &Tensor<A>,
        (w, b): (Tensor<WB>, Option<Tensor<WB>>),
        workspace: &mut [ByteOf<Ops::Hardware>],
        queue_alloc: &QA,
    ) -> Result<(), LaunchError>
    where
        C: DerefMut<Target = [ByteOf<Ops::Hardware>]>,
        A: Deref<Target = [ByteOf<Ops::Hardware>]>,
        WB: Deref<Target = [ByteOf<Ops::Hardware>]>,
        QA: QueueAlloc<Hardware = Ops::Hardware>,
    {
        let beta = if let Some(b) = b {
            let n = c.shape()[0];
            let b = b.broadcast(0, n);
            self.rearrange(c, &b, workspace, queue_alloc)?;
            1.
        } else {
            0.
        };
        self.mat_mul.launch(
            &mat_mul::Args {
                c_layout: c.layout(),
                c_base: c.base_mut(),
                beta,
                a_layout: a.layout(),
                a_base: a.base(),
                b_layout: w.layout(),
                b_base: w.base(),
                alpha: 1.,
            },
            workspace,
            queue_alloc,
        )
    }

    fn attn_kv_cached<Q, K, V, O, KC, VC, QA>(
        &self,
        q: &mut Tensor<Q>,
        k: &Tensor<K>,
        v: &Tensor<V>,
        o: &mut Tensor<O>,
        kc: &mut Tensor<KC>,
        vc: &mut Tensor<VC>,
        pos: usize,
        workspace: &mut [ByteOf<Ops::Hardware>],
        queue_alloc: &QA,
    ) -> Result<(), LaunchError>
    where
        Q: DerefMut<Target = [ByteOf<Ops::Hardware>]>,
        K: Deref<Target = [ByteOf<Ops::Hardware>]>,
        V: Deref<Target = [ByteOf<Ops::Hardware>]>,
        O: DerefMut<Target = [ByteOf<Ops::Hardware>]>,
        KC: DerefMut<Target = [ByteOf<Ops::Hardware>]>,
        VC: DerefMut<Target = [ByteOf<Ops::Hardware>]>,
        QA: QueueAlloc<Hardware = Ops::Hardware>,
    {
        self.attn_kv_cached.launch(
            &attention_kv_cached::Args {
                q_layout: q.layout(),
                q_base: q.base_mut(),
                k_layout: k.layout(),
                k_base: k.base(),
                v_layout: v.layout(),
                v_base: v.base(),
                o_layout: o.layout(),
                o_base: o.base_mut(),
                k_cache_layout: kc.layout(),
                k_cache_base: kc.base_mut(),
                v_cache_layout: vc.layout(),
                v_cache_base: vc.base_mut(),
                pos: pos.into(),
            },
            workspace,
            queue_alloc,
        )
    }

    fn gelu<X, QA>(
        &self,
        x: &mut Tensor<X>,
        workspace: &mut [ByteOf<Ops::Hardware>],
        queue_alloc: &QA,
    ) -> Result<(), LaunchError>
    where
        X: DerefMut<Target = [ByteOf<Ops::Hardware>]>,
        QA: QueueAlloc<Hardware = Ops::Hardware>,
    {
        self.gelu.launch(
            &gelu::Args {
                layout: x.layout(),
                base: x.base_mut(),
            },
            workspace,
            queue_alloc,
        )
    }

    fn add<C, A, B, QA>(
        &self,
        c: &mut Tensor<C>,
        a: &Tensor<A>,
        b: &Tensor<B>,
        workspace: &mut [ByteOf<Ops::Hardware>],
        queue_alloc: &QA,
    ) -> Result<(), LaunchError>
    where
        C: DerefMut<Target = [ByteOf<Ops::Hardware>]>,
        A: Deref<Target = [ByteOf<Ops::Hardware>]>,
        B: Deref<Target = [ByteOf<Ops::Hardware>]>,
        QA: QueueAlloc<Hardware = Ops::Hardware>,
    {
        self.add.launch(
            &add::Args {
                c_layout: c.layout(),
                c_base: c.base_mut(),
                a_layout: a.layout(),
                a_base: a.base(),
                b_layout: b.layout(),
                b_base: b.base(),
            },
            workspace,
            queue_alloc,
        )
    }

    fn rearrange<Y, X, QA>(
        &self,
        dst: &mut Tensor<Y>,
        src: &Tensor<X>,
        workspace: &mut [ByteOf<Ops::Hardware>],
        queue_alloc: &QA,
    ) -> Result<(), LaunchError>
    where
        Y: DerefMut<Target = [ByteOf<Ops::Hardware>]>,
        X: Deref<Target = [ByteOf<Ops::Hardware>]>,
        QA: QueueAlloc<Hardware = Ops::Hardware>,
    {
        self.rearrange.launch(
            &rearrange::Args {
                dst_layout: dst.layout(),
                dst_base: dst.base_mut(),
                src_layout: src.layout(),
                src_base: src.base(),
            },
            workspace,
            queue_alloc,
        )
    }

    fn all_reduce<X, QA>(
        &self,
        x: &mut Tensor<X>,
        workspace: &mut [ByteOf<Ops::Hardware>],
        queue_alloc: &QA,
    ) -> Result<(), LaunchError>
    where
        X: DerefMut<Target = [ByteOf<Ops::Hardware>]>,
        QA: QueueAlloc<Hardware = Ops::Hardware>,
    {
        self.all_reduce.launch(
            &all_reduce::Args {
                pair: rearrange::Args {
                    dst_layout: x.layout(),
                    dst_base: x.base_mut(),
                    src_layout: x.layout(),
                    src_base: x.base(),
                },
                op: ReduceOp::Sum,
            },
            workspace,
            queue_alloc,
        )
    }
}

struct WeightDecorator<W> {
    pos_embd: Tensor<usize>,
    output_weight: Tensor<usize>,
    norm: Tensor<usize>,

    attn_qkv_w: Tensor<usize>,
    attn_qkv_b: Tensor<usize>,
    attn_o_w: Tensor<usize>,
    attn_o_b: Tensor<usize>,

    ffn_up_w: Tensor<usize>,
    ffn_up_b: Tensor<usize>,
    ffn_down_w: Tensor<usize>,
    ffn_down_b: Tensor<usize>,

    weights: W,
}

impl Gpt2Meta {
    fn decorator<W>(&self, weights: W) -> WeightDecorator<W> {
        use crate::TensorUsage::Computation;
        WeightDecorator {
            pos_embd: self.pos_embd(),
            output_weight: self.output_weight(),
            norm: self.norm(),

            attn_qkv_w: self.attn_qkv_w(Computation),
            attn_qkv_b: self.attn_qkv_b(Computation),
            attn_o_w: self.attn_o_w(Computation),
            attn_o_b: self.attn_o_b(Computation),

            ffn_up_w: self.ffn_up_w(Computation),
            ffn_up_b: self.ffn_up_b(Computation),
            ffn_down_w: self.ffn_down_w(Computation),
            ffn_down_b: self.ffn_down_b(Computation),

            weights,
        }
    }
}

impl<W: WeightLoader> WeightDecorator<W> {
    pub fn attn_norm(
        &self,
        iblk: usize,
        queue: &QueueOf<W::Hardware>,
    ) -> [Tensor<W::Memory<'_>>; 2] {
        let [w, b] = self.weights.load_blk(BlkWeight::AttnNorm, iblk, queue);
        [self.norm.clone().map(|_| w), self.norm.clone().map(|_| b)]
    }

    pub fn attn_qkv(
        &self,
        iblk: usize,
        queue: &QueueOf<W::Hardware>,
    ) -> [Tensor<W::Memory<'_>>; 2] {
        let [w, b] = self.weights.load_blk(BlkWeight::AttnQKV, iblk, queue);
        [
            self.attn_qkv_w.clone().map(|_| w),
            self.attn_qkv_b.clone().map(|_| b),
        ]
    }

    pub fn attn_o(&self, iblk: usize, queue: &QueueOf<W::Hardware>) -> [Tensor<W::Memory<'_>>; 2] {
        let [w, b] = self.weights.load_blk(BlkWeight::AttnO, iblk, queue);
        [
            self.attn_o_w.clone().map(|_| w),
            self.attn_o_b.clone().map(|_| b),
        ]
    }

    pub fn ffn_norm(
        &self,
        iblk: usize,
        queue: &QueueOf<W::Hardware>,
    ) -> [Tensor<W::Memory<'_>>; 2] {
        let [w, b] = self.weights.load_blk(BlkWeight::FfnNorm, iblk, queue);
        [self.norm.clone().map(|_| w), self.norm.clone().map(|_| b)]
    }

    pub fn ffn_up(&self, iblk: usize, queue: &QueueOf<W::Hardware>) -> [Tensor<W::Memory<'_>>; 2] {
        let [w, b] = self.weights.load_blk(BlkWeight::FfnUp, iblk, queue);
        [
            self.ffn_up_w.clone().map(|_| w),
            self.ffn_up_b.clone().map(|_| b),
        ]
    }

    pub fn ffn_down(
        &self,
        iblk: usize,
        queue: &QueueOf<W::Hardware>,
    ) -> [Tensor<W::Memory<'_>>; 2] {
        let [w, b] = self.weights.load_blk(BlkWeight::FfnDown, iblk, queue);
        [
            self.ffn_down_w.clone().map(|_| w),
            self.ffn_down_b.clone().map(|_| b),
        ]
    }

    pub fn output_norm(&self, queue: &QueueOf<W::Hardware>) -> [Tensor<W::Memory<'_>>; 2] {
        let [w, b] = self.weights.output_norm(queue);
        [self.norm.clone().map(|_| w), self.norm.clone().map(|_| b)]
    }

    pub fn output_weight(&self, queue: &QueueOf<W::Hardware>) -> Tensor<W::Memory<'_>> {
        self.output_weight
            .clone()
            .map(|_| self.weights.output(queue))
    }

    pub fn pos_embd<'a>(&'a self, queue: &'a QueueOf<W::Hardware>) -> Tensor<W::Memory<'a>> {
        let pos_embd = self.weights.pos_embd(queue);
        self.pos_embd.clone().map(|_| pos_embd)
    }
}
