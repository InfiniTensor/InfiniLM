import functools
import ninetoothed
from ninetoothed import Tensor


def arrangement(
    k_cache,
    v_cache,
    k,
    v,
    past_lengths,
    block_size_m=ninetoothed.block_size(),
    block_size_n=ninetoothed.block_size(),
):
    k_cache_arranged = k_cache.tile((1, block_size_m, 1, -1)).tile((1, 1, -1, 1))
    v_cache_arranged = v_cache.tile((1, block_size_m, 1, -1)).tile((1, 1, -1, 1))

    k_arranged = k.tile((1, block_size_m, 1, -1)).tile((1, 1, -1, 1))
    v_arranged = v.tile((1, block_size_m, 1, -1)).tile((1, 1, -1, 1))

    past_lengths_arranged = (
        past_lengths.tile((1,))
        .unsqueeze(1)
        .unsqueeze(2)
        .unsqueeze(3)
        .unsqueeze(4)
        .expand((-1, *k_arranged.shape))
    )

    return (
        k_cache_arranged,
        v_cache_arranged,
        k_arranged,
        v_arranged,
        past_lengths_arranged,
    )


def application(k_cache, v_cache, k, v, past_lengths):
    pos = past_lengths

    for i in range(k.shape[-2]):
        k_cache[0, 0, pos + i, 0] = k[0, 0, i, 0]
        v_cache[0, 0, pos + i, 0] = v[0, 0, i, 0]


def premake(emb_dim=None, dtype=None, block_size_m=None, block_size_n=None):
    arrangement_ = functools.partial(
        arrangement, block_size_m=block_size_m, block_size_n=block_size_n
    )

    shape_options = (None, None, None, {"constexpr": True, "upper_bound": 256})

    tensors = (
        Tensor(4, dtype=dtype, shape_options=shape_options),
        Tensor(4, dtype=dtype, shape_options=shape_options),
        Tensor(4, dtype=dtype, shape_options=shape_options),
        Tensor(4, dtype=dtype, shape_options=shape_options),
        Tensor(1, dtype=ninetoothed.int64),
    )

    if emb_dim is not None:
        for tensor in tensors:
            tensor.shape = tensor.shape[:-1] + (emb_dim,)

    return arrangement_, application, tensors
