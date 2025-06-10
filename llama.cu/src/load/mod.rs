mod loader;
mod range_collector;

use bytesize::ByteSize;
use log::{debug, trace};
use nn::{Edge, TPAction, TPTensor, Tensor};
use operators::cuda::{CurrentCtx, DevByte, DevMem, Stream, VirByte};
use range_collector::RangeCollector;
use std::{
    collections::HashSet,
    ops::Range,
    os::raw::c_int,
    time::{Duration, Instant},
};

pub(crate) use loader::WeightLoader;

type HostTPTensor<'a> = TPTensor<Tensor<&'a [u8], 2>>;
type VirTensor = Tensor<*const VirByte, 2>;

pub(crate) fn load_weight<'ctx>(
    edges: Box<[Edge<HostTPTensor>]>,
    ctx: &'ctx CurrentCtx,
) -> (DevMem<'ctx>, Box<[Edge<VirTensor>]>) {
    // 排布权重存储
    let align = Some(ctx.dev().alignment())
        .filter(|&n| n > 0)
        .unwrap_or(512);
    let mut ranges = RangeCollector::new(align);
    for nn::Edge { external, .. } in &edges {
        if let Some(nn::External { item, .. }) = external {
            let TPTensor { act, val } = item;
            let len = match act {
                Some(act) => val.get().len() / act.dist.total * act.dist.len,
                None => val.get().len(),
            };
            ranges.insert((act.clone(), val.get().as_ptr()), len)
        }
    }
    // 权重加载
    let time = Instant::now();
    let mut weight = ctx.malloc::<u8>(ranges.size());
    let mut loader = WeightLoader::new(
        ranges
            .sizes()
            .filter(|&(_, times)| times < 4)
            .map(|(size, _)| size),
    );

    let stream = ctx.stream();
    let mut copied = HashSet::new();
    let edges = edges
        .into_iter()
        .map(|nn::Edge { meta, external }| nn::Edge {
            meta,
            external: external.map(|external| {
                load_exteranl(
                    external,
                    &mut loader,
                    &ranges,
                    &mut weight,
                    &mut copied,
                    &stream,
                )
            }),
        })
        .collect::<Box<_>>();
    fmt_log(ctx.dev().index(), edges.len(), weight.len(), time.elapsed());
    (weight, edges)
}

fn load_exteranl<'ctx>(
    external: nn::External<TPTensor<Tensor<&[u8], 2>>>,
    loader: &mut WeightLoader<'ctx>,
    ranges: &RangeCollector<(Option<TPAction>, *const u8)>,
    mapped: &mut [DevByte],
    copied: &mut HashSet<Range<usize>>,
    stream: &Stream<'ctx>,
) -> nn::External<Tensor<*const VirByte, 2>> {
    let nn::External { name, item } = external;
    let size = ByteSize::b(item.val.get().len() as _).display();
    trace!(
        "loading weight {:>9} @{} {name}",
        size.to_string(),
        stream.ctx().dev().index(),
    );

    let TPTensor { act, val } = item;
    let range = &ranges[&(act.clone(), val.get().as_ptr())];
    let dev = &mut mapped[range.clone()];
    let ptr = dev.as_ptr().cast();
    nn::External {
        name,
        item: match act.clone() {
            Some(TPAction { wt, dist }) => {
                if copied.insert(range.clone()) {
                    loader.load(dev, stream, |dst| wt.move_data(dist, dst, &val))
                }
                let shape = wt.split_shape(dist, val.shape());
                Tensor::from_dim_slice(val.dt(), &shape).map(|_| ptr)
            }
            None => {
                if copied.insert(range.clone()) {
                    loader.load(dev, stream, |dst| dst.copy_from_slice(val.get()))
                }
                val.map(|_| ptr)
            }
        },
    }
}

fn fmt_log(dev: c_int, num: usize, size: usize, time: Duration) {
    let speed = size as f64 / time.as_secs_f64();
    debug!(
        "weight loaded @{dev} in {time:.2?}, {} for {num} tensors, {}/s",
        ByteSize::b(size as _).display(),
        ByteSize::b(speed as _).display(),
    );
}
