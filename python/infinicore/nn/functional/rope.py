from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


class RopeAlgo:
    r"""Different types of RoPE algorithms."""

    GPT_J = _infinicore.RoPEAlgo.GPT_J
    GPT_NEOX = _infinicore.RoPEAlgo.GPT_NEOX


def rope(
    x: Tensor,
    pos_ids: Tensor,
    sin_table: Tensor,
    cos_table: Tensor,
    algo: RopeAlgo = RopeAlgo.GPT_NEOX,
    *,
    out=None,
) -> Tensor:
    r"""Rotary Position Embedding(RoPE)."""

    if out is None:
        return Tensor(
            _infinicore.rope(
                x._underlying,
                pos_ids._underlying,
                sin_table._underlying,
                cos_table._underlying,
                algo,
            )
        )

    _infinicore.rope_(
        out._underlying,
        x._underlying,
        pos_ids._underlying,
        sin_table._underlying,
        cos_table._underlying,
        algo,
    )
    return out
