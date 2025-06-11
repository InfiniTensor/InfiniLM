mod engine;
mod engine_manager;
mod group;
mod kv_cache;
mod model;
mod output_head;
mod step;

use crate::{
    memory::MemPages,
    op::random_sample::{KVPair, SampleArgs},
};
use kv_cache::KVCache;
use nn::Tensor;
use operators::cuda::{ContextSpore, CurrentCtx, DevMemSpore, Device, EventSpore, Stream};
use std::{
    collections::BTreeMap,
    sync::{Arc, Mutex},
};
use tokeneer::utok;

#[allow(non_camel_case_types)]
type upos = u32;

pub(crate) use engine::engine;

pub(crate) enum Command {
    ShutDown,
    Insert(Request),
    Remove(SessionId),
}

pub(crate) enum Output {
    Ready,
    Overflow(Box<[Session]>),
    Removed(Session),
    Complete {
        output: Box<[(SessionId, usize)]>,
        kv_pair: DevMemSpore,
        event: EventSpore,
        finished: Box<[Session]>,
    },
}

impl Output {
    pub(crate) fn drop_on(self, ctx: &CurrentCtx) {
        if let Self::Complete { kv_pair, event, .. } = self {
            drop((kv_pair.sprout(ctx), event.sprout(ctx)))
        }
    }
}

pub(crate) struct Request {
    pub session: Session,
    pub prompt: Box<[utok]>,
    pub out: usize,
    pub max_steps: usize,
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
#[repr(transparent)]
pub struct SessionId(pub usize);

pub struct Session {
    pub id: SessionId,
    pub sample_args: SampleArgs,
    pub cache: DistKVCache,
}

pub struct DistKVCache {
    parts: Arc<[Mutex<KVCache>]>,
    len: usize,
    pub pos: usize,
}

impl DistKVCache {
    pub(crate) fn new(template: &Tensor<usize, 2>, parts: &[(Device, usize)]) -> Self {
        let total = parts.iter().map(|(_, len)| len).sum::<usize>();
        let parts = parts
            .iter()
            .map(|(dev, len)| KVCache::new(template, *len, total, &MemPages::new(*dev)));
        Self {
            parts: parts.map(Mutex::new).collect(),
            len: template.shape()[0],
            pos: 0,
        }
    }
}

pub(crate) fn decode(
    output: Box<[(SessionId, usize)]>,
    kv_pair: DevMemSpore,
    event: EventSpore,
    stream: &Stream,
) -> BTreeMap<SessionId, Vec<utok>> {
    let ctx = stream.ctx();
    let kv_pair = kv_pair.sprout(ctx);
    let mut host = vec![KVPair::ZERO; kv_pair.len() / size_of::<KVPair>()];
    stream
        .wait_for(&event.sprout(ctx))
        .memcpy_d2h(&mut host, &kv_pair)
        .free(kv_pair);
    let mut offset = 0;
    output
        .into_iter()
        .map(|(id, len)| {
            let slice = &host[offset..][..len];
            offset += len;
            (id, slice.iter().map(|kv| kv.idx as _).collect())
        })
        .collect()
}
