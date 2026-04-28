import enum
import functools

import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor

BLOCK_SIZE_M = ninetoothed.block_size()
BLOCK_SIZE_N = ninetoothed.block_size()


class CausalVariant(enum.IntEnum):
    """Please refer to `<https://docs.pytorch.org/docs/stable/generated/torch.nn.attention.bias.CausalVariant.html>`_."""

    UPPER_LEFT = enum.auto()

    LOWER_RIGHT = enum.auto()


def arrangement(
    query,
    key,
    value,
    total_kv_len,
    present_key,
    present_value,
    present_key_slot,
    present_value_slot,
    attn_mask,
    is_causal,
    scale,
    output,
    with_attn_mask,
    causal_variant,
    with_kv_cache,
    block_size_m=None,
    block_size_n=None,
):
    def arrange_query_or_output(input):
        arranged = input.tile((1, 1, block_size_m, -1)).tile(
            (1, query.shape[-3] // key.shape[-3], 1, 1)
        )
        arranged.dtype = arranged.dtype.squeeze((0, 2, 3))
        arranged.dtype.dtype = arranged.dtype.dtype.squeeze((0, 1))

        return arranged

    def arrange_key_or_value(input):
        arranged = (
            input.tile((1, 1, block_size_n, -1))
            .tile((1, 1, -1, -1))
            .expand((-1, -1, query_arranged.shape[-2], -1))
        )
        arranged.dtype = arranged.dtype.squeeze((0, 1, 3))
        arranged.dtype.dtype = arranged.dtype.dtype.squeeze((0, 1))

        return arranged

    def arrange_total_kv_len(input, shape):
        arranged = input.tile((1,))
        arranged = arranged.unsqueeze(1).unsqueeze(2).unsqueeze(3).expand(shape)
        return arranged

    def arrange_present_key_or_present_value(input):
        arranged = input.tile((1, 1, block_size_m, block_size_n))
        arranged.dtype = arranged.dtype.squeeze((0, 1))

        return arranged

    def arrange_attn_mask(input):
        arranged = input.tile((1, 1, block_size_m, block_size_n)).tile((1, 1, 1, -1))
        arranged.dtype = arranged.dtype.squeeze((0, 1, 2))
        arranged.dtype.dtype = arranged.dtype.dtype.squeeze((0, 1))

        return arranged

    if block_size_m is None:
        block_size_m = BLOCK_SIZE_M

    if block_size_n is None:
        block_size_n = BLOCK_SIZE_N

    query_arranged = arrange_query_or_output(query)
    key_arranged = arrange_key_or_value(key)
    value_arranged = arrange_key_or_value(value)
    total_kv_len_arranged = arrange_total_kv_len(total_kv_len, query_arranged.shape)
    present_key_arranged = arrange_present_key_or_present_value(present_key)
    present_value_arranged = arrange_present_key_or_present_value(present_value)
    present_key_slot_arranged = arrange_present_key_or_present_value(present_key_slot)
    present_value_slot_arranged = arrange_present_key_or_present_value(
        present_value_slot
    )
    attn_mask_arranged = arrange_attn_mask(attn_mask)
    is_causal_arranged = is_causal
    scale_arranged = scale
    output_arranged = arrange_query_or_output(output)
    with_attn_mask_arranged = with_attn_mask
    causal_variant_arranged = causal_variant

    if with_kv_cache:
        return (
            query_arranged,
            key_arranged,
            value_arranged,
            total_kv_len_arranged,
            present_key_arranged,
            present_value_arranged,
            present_key_slot_arranged,
            present_value_slot_arranged,
            attn_mask_arranged,
            is_causal_arranged,
            scale_arranged,
            output_arranged,
            with_attn_mask_arranged,
            causal_variant_arranged,
        )

    return (
        query_arranged,
        key_arranged,
        value_arranged,
        total_kv_len_arranged,
        attn_mask_arranged,
        is_causal_arranged,
        scale_arranged,
        output_arranged,
        with_attn_mask_arranged,
        causal_variant_arranged,
    )


def application_with_kv_cache(
    query,
    key,
    value,
    total_kv_len,
    present_key,
    present_value,
    present_key_slot,
    present_value_slot,
    attn_mask,
    is_causal,
    scale,
    output,
    with_attn_mask,
    causal_variant,
):
    present_key_slot = present_key  # noqa: F841
    present_value_slot = present_value  # noqa: F841

    application_without_kv_cache(
        query,
        key,
        value,
        total_kv_len,
        attn_mask,
        is_causal,
        scale,
        output,
        with_attn_mask,
        causal_variant,
    )


def application_without_kv_cache(
    query,
    key,
    value,
    total_kv_len,
    attn_mask,
    is_causal,
    scale,
    output,
    with_attn_mask,
    causal_variant,
):
    actual_kv_len = total_kv_len[0]

    for i in range(query.shape[0]):
        query_i = (1.4426950408889634 * scale * query[i]).to(query[i].dtype)

        acc = ntl.zeros((query_i.shape[-2], query_i.shape[-1]), dtype=ntl.float32)
        lse = ntl.full((query_i.shape[-2],), 1, dtype=ntl.float32)
        max = ntl.full((query_i.shape[-2],), float("-inf"), dtype=ntl.float32)

        for j in range(-(-actual_kv_len // key.dtype.shape[0])):

            qk = ntl.dot(query_i, ntl.trans(key[j]))

            key_pos = key[j].offsets(-2)
            qk = ntl.where(key_pos < actual_kv_len, qk, float("-inf"))

            if with_attn_mask:
                qk += attn_mask[j]

            if is_causal:
                query_pos = query[i].offsets(-2)

                if causal_variant == 2:  # CausalVariant.LOWER_RIGHT:
                    mask = (
                        query_pos[:, None] + actual_kv_len - query.source.shape[-2]
                        >= key_pos[None, :]
                    )
                else:
                    mask = query_pos[:, None] >= key_pos[None, :]

                qk = ntl.where(mask, qk, float("-inf"))

            next_max = ntl.maximum(max, ntl.max(qk, 1))
            stable_qk = ntl.exp2(qk - next_max[:, None])

            alpha = ntl.exp2(max - next_max)
            acc = acc * alpha[:, None] + ntl.dot(stable_qk.to(value[i].dtype), value[j])
            max = next_max
            lse = lse * alpha + ntl.sum(stable_qk, 1)

        acc /= lse[:, None]
        output[i] = acc  # noqa: F841


def premake(
    with_kv_cache,
    emb_dim=None,
    is_causal=None,
    with_attn_mask=None,
    causal_variant=None,
    dtype=None,
    block_size_m=None,
    block_size_n=None,
):
    arrangement_ = functools.partial(
        arrangement,
        with_kv_cache=with_kv_cache,
        block_size_m=block_size_m,
        block_size_n=block_size_n,
    )

    query, key, value, attn_mask, output = (
        Tensor(
            4,
            dtype=dtype,
            shape_options=(None, None, None, {"constexpr": True, "upper_bound": 128}),
        )
        for _ in range(5)
    )
    total_kv_len = Tensor(1, dtype=ninetoothed.int32)
    present_key, present_value, present_key_slot, present_value_slot = (
        Tensor(4, dtype=dtype) for _ in range(4)
    )
    scale = Tensor(0, dtype=ninetoothed.float64)
    is_causal = Tensor(0, constexpr=True, value=is_causal)
    with_attn_mask = Tensor(0, constexpr=True, value=with_attn_mask)
    causal_variant = Tensor(0, constexpr=True, value=causal_variant)

    if emb_dim is not None:
        for tensor in (query, key, value, attn_mask, output):
            tensor.shape = tensor.shape[:-1] + (emb_dim,)

    if with_kv_cache:
        application = application_with_kv_cache
    else:
        application = application_without_kv_cache

    tensors = (
        query,
        key,
        value,
        total_kv_len,
        present_key,
        present_value,
        present_key_slot,
        present_value_slot,
        attn_mask,
        is_causal,
        scale,
        output,
        with_attn_mask,
        causal_variant,
    )

    return arrangement_, application, tensors
