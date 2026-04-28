from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def mha_varlen(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    cum_seqlens_q: Tensor,
    cum_seqlens_k: Tensor,
    block_table: Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    alibi_slopes: Tensor | None = None,
    scale: float = 1.0,
    *,
    out: Tensor | None = None,
):
    if out is None:
        return Tensor(
            _infinicore.mha_varlen(
                q._underlying,
                k._underlying,
                v._underlying,
                cum_seqlens_q._underlying,
                cum_seqlens_k._underlying,
                block_table._underlying,
                max_seqlen_q,
                max_seqlen_k,
                alibi_slopes._underlying if alibi_slopes is not None else None,
                scale,
            )
        )

    _infinicore.mha_varlen_(
        out._underlying,
        q._underlying,
        k._underlying,
        v._underlying,
        cum_seqlens_q._underlying,
        cum_seqlens_k._underlying,
        block_table._underlying,
        max_seqlen_q,
        max_seqlen_k,
        alibi_slopes._underlying if alibi_slopes is not None else None,
        scale,
    )

    return out
