from typing import Optional

from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def mha_kvcache(
    q: Tensor,
    k_cache: Tensor,
    v_cache: Tensor,
    seqlens_k: Tensor,
    block_table: Tensor,
    alibi_slopes: Optional[Tensor] = None,
    scale: float = 1.0,
    *,
    out: Optional[Tensor] = None,
) -> Tensor:
    """
    Flash attention KV-cache decode for single-step attention over a paged KV cache.

    This function performs attention decoding using a paged KV cache layout,
    which is efficient for inference with large sequence lengths.

    Args:
        q: Query tensor of shape [batch_size, seqlen_q, num_heads, head_size]
        k_cache: Key cache tensor of shape [num_blocks, block_size, num_heads_k, head_size] (paged layout)
        v_cache: Value cache tensor of shape [num_blocks, block_size, num_heads_k, head_size] (paged layout)
        seqlens_k: Total KV length per request of shape [batch_size] (int32)
        block_table: Block mapping table of shape [batch_size, max_num_blocks_per_seq] (int32)
        alibi_slopes: Optional ALiBi slopes tensor, if None then ALiBi is disabled
        scale: Scaling factor for attention scores (typically 1.0/sqrt(head_size))
        out: Optional output tensor. If provided, the operation will be performed in-place.

    Returns:
        Output tensor of shape [batch_size, seqlen_q, num_heads, head_size]

    Note:
        The KV cache uses a paged layout where:
        - k_cache and v_cache are organized into fixed-size blocks
        - block_table maps logical positions to physical blocks for each sequence
        - seqlens_k indicates the current total length of each sequence in the cache
    """
    if out is None:
        return Tensor(
            _infinicore.mha_kvcache(
                q._underlying,
                k_cache._underlying,
                v_cache._underlying,
                seqlens_k._underlying,
                block_table._underlying,
                alibi_slopes._underlying if alibi_slopes is not None else None,
                scale,
            )
        )

    _infinicore.mha_kvcache_(
        out._underlying,
        q._underlying,
        k_cache._underlying,
        v_cache._underlying,
        seqlens_k._underlying,
        block_table._underlying,
        alibi_slopes._underlying if alibi_slopes is not None else None,
        scale,
    )

    return out
