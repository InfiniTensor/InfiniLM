import math

from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def flash_attention(
    query,
    key,
    value,
    total_kv_len,
    attn_mask=None,
    dropout_p=0,
    is_causal=False,
    scale=None,
    enable_gqa=False,
):
    assert attn_mask is None and dropout_p == 0 and not enable_gqa

    emb_dim = query.shape[-1]

    if scale is None:
        scale = 1 / math.sqrt(emb_dim)

    return Tensor(
        _infinicore.flash_attention(
            query._underlying,
            key._underlying,
            value._underlying,
            total_kv_len._underlying,
            scale,
            is_causal,
        )
    )
