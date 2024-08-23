use crate::LlamaMeta;
use causal_lm::QueryContext;
use common_devices::{Kernels, KernelsA, SliceOn};
use itertools::izip;
use operators::{Handle, QueueOf};
use std::ops::{Deref, DerefMut};
use tensor::{slice, split, LocalSplitable, Tensor};

pub trait ComputeStream {
    type Handle: Handle;
    type Storage;
    type Buf<'m>: DerefMut<Target = SliceOn<Self::Handle>>;
    type Pos<'m>: Deref<Target = SliceOn<Self::Handle>>;

    fn malloc(&self, len: usize) -> Self::Buf<'_>;
    fn free(&self, _mem: Self::Buf<'_>) {}
    fn map_pos<'p>(&self, pos: &'p [u32]) -> Self::Pos<'p>
    where
        Self: 'p;
    fn free_pos(&self, _mem: Self::Pos<'_>) {}
    fn map_storage<'a>(&'a self, storage: &'a mut Self::Storage) -> &'a mut SliceOn<Self::Handle>;

    fn meta(&self) -> &LlamaMeta;
    fn kernels(&self) -> &impl Kernels<Self::Handle>;
    fn queue(&self) -> &QueueOf<Self::Handle>;

    fn debug<T>(tensor: &Tensor<T>)
    where
        T: Deref<Target = SliceOn<Self::Handle>>;

    fn layers(
        &self,
    ) -> impl Iterator<Item = impl LLamaLayer<Byte = <Self::Handle as Handle>::Byte>>;

    fn forward<'q>(
        &self,
        queries: impl IntoIterator<Item = QueryContext<'q, Self::Storage>>,
        mut token_embedded: Tensor<Self::Storage>,
    ) -> Tensor<Self::Storage>
    where
        Self::Storage: 'q,
    {
        let mut queries = queries.into_iter().collect::<Vec<_>>();
        let mut nt = 0;
        let mut max_seq_len = 0;
        let mut max_att_len = 0;
        let seq_len = queries
            .iter()
            .map(|q| {
                let seq = q.seq_len();
                let att = q.att_len();
                nt += seq;
                max_seq_len = max_seq_len.max(seq);
                max_att_len = max_att_len.max(att);
                seq
            })
            .collect::<Vec<_>>();

        let &LlamaMeta {
            nh,
            nkvh,
            dh,
            di,
            epsilon,
            theta,
            ..
        } = self.meta();
        let dt = token_embedded.data_layout();
        let d = nh * dh;
        let dkv = nkvh * dh;
        let head_group = nh / nkvh;
        let head_div = (dh as f32).sqrt().recip();
        let queue = self.queue();

        let mut x = token_embedded
            .as_mut()
            .map_physical(|u| self.map_storage(u));
        let reusing = (d + dkv + dkv).max(di + di);
        let mut state_buf = Tensor::alloc(dt, &[nt, (d + reusing) as _], |len| self.malloc(len));

        let mut q_buf = self.malloc(nh * max_seq_len as usize * dh * dt.nbytes());
        let mut att_buf =
            self.malloc(nh * max_seq_len as usize * max_att_len as usize * dt.nbytes());
        let pos = causal_lm::pos(&queries, nt);
        let pos = pos.as_ref().map_physical(|u| self.map_pos(u));

        for (layer, params) in self.layers().enumerate() {
            let (mut x1, qkv) = split!(state_buf.as_mut().map_physical(|u| LocalSplitable::from(&mut **u)); [1]: d, reusing);
            let mut qkv = qkv.slice(&[slice![=>], slice![=> d + dkv + dkv]]);

            self.kernels()
                .rms_norm(&mut x1, &x, &params.att_layernorm(), epsilon, queue);
            self.kernels()
                .mat_mul(&mut qkv, 0., &x1, &params.att_qkv(), 1., queue);

            let (q, k, v) = split!(qkv; [1]: d, dkv, dkv);
            let mut q = q.reshape(&[nt, nh as _, dh as _]);
            let mut k = k.reshape(&[nt, nkvh as _, dh as _]);
            let v = v.reshape(&[nt, nkvh as _, dh as _]);
            let o = x1.reshape(&[nt, nh as _, dh as _]);

            self.kernels().rope(&mut q, &pos, theta, queue);
            self.kernels().rope(&mut k, &pos, theta, queue);

            let q = q.transpose(&[1, 0, 2]).split(1, &seq_len);
            let k = k.transpose(&[1, 0, 2]).split(1, &seq_len);
            let v = v.transpose(&[1, 0, 2]).split(1, &seq_len);
            let o = o.transpose(&[1, 0, 2]).split(1, &seq_len);

            for (query, q, k, v, mut o) in izip!(&mut queries, q, k, v, o) {
                let pos = query.pos();
                let seq_len = query.seq_len();
                let att_len = query.att_len();
                let mut cache = query
                    .cache
                    .as_mut()
                    .map(|t| t.as_mut().map_physical(|u| self.map_storage(u)));
                let mut query = QueryContext {
                    cache: cache.as_mut(),
                    range: query.range.clone(),
                };
                let Some((mut k_cache, mut v_cache)) = query.cache(layer as _) else {
                    continue;
                };

                let slice_cat = &[slice![=>], slice![pos =>=> seq_len], slice![=>]];
                let slice_att = &[slice![=>], slice![      => att_len], slice![=>]];
                let shape_q0 = &[(nkvh * head_group) as u32, seq_len, dh as u32];
                let shape_q1 = &[nkvh as u32, head_group as u32 * seq_len, dh as u32];
                let shape_att0 = &[nkvh as u32, head_group as u32 * seq_len, att_len];
                let shape_att1 = &[(nkvh * head_group) as u32, seq_len, att_len];

                let mut q_att = Tensor::new(dt, shape_q0, &mut q_buf[..]);
                let mut k_cat = k_cache.as_mut().slice(slice_cat).map_physical(|u| &mut **u);
                let mut v_cat = v_cache.as_mut().slice(slice_cat).map_physical(|u| &mut **u);
                self.kernels().reform(&mut q_att, &q, queue);
                self.kernels().reform(&mut k_cat, &k, queue);
                self.kernels().reform(&mut v_cat, &v, queue);

                let q_att = q_att.reshape(shape_q1);
                let k_att = k_cache.slice(slice_att).transpose(&[0, 2, 1]);
                let v_att = v_cache.slice(slice_att);

                let mut att = Tensor::new(dt, shape_att0, &mut att_buf[..]);
                self.kernels()
                    .mat_mul(&mut att, 0., &q_att, &k_att, head_div, queue);
                let mut att = att.reshape(shape_att1);
                self.kernels().softmax(&mut att, queue);
                let mut x2 = q_att;
                self.kernels()
                    .mat_mul(&mut x2, 0., &att.reshape(shape_att0), &v_att, 1., queue);

                self.kernels().reform(&mut o, &x2.reshape(shape_q0), queue);
            }

            let (mut x1, gate_up) = split!(state_buf.as_mut().map_physical(|u| LocalSplitable::from(&mut **u)); [1]: d, reusing);
            let mut gate_up = gate_up.slice(&[slice![=>], slice![=> di + di]]);

            self.kernels()
                .mat_mul(&mut x, 1., &x1, &params.att_o(), 1., queue);
            self.kernels()
                .rms_norm(&mut x1, &x, &params.mlp_layernorm(), epsilon, queue);
            self.kernels().mlp(
                &mut x,
                &x1,
                &mut gate_up,
                &params.mlp_gate_up(),
                &params.mlp_down(),
                1.,
                true,
                queue,
            );
        }
        self.free_pos(pos.take_physical());
        self.free(state_buf.take_physical());
        self.free(q_buf);
        self.free(att_buf);
        drop(x);
        token_embedded
    }
}

pub trait LLamaLayer {
    type Byte;
    type Storage<'m>: Deref<Target = [Self::Byte]>
    where
        Self: 'm;

    fn att_layernorm(&self) -> Tensor<Self::Storage<'_>>;
    fn att_qkv(&self) -> Tensor<Self::Storage<'_>>;
    fn att_o(&self) -> Tensor<Self::Storage<'_>>;
    fn mlp_layernorm(&self) -> Tensor<Self::Storage<'_>>;
    fn mlp_gate_up(&self) -> Tensor<Self::Storage<'_>>;
    fn mlp_down(&self) -> Tensor<Self::Storage<'_>>;
}
