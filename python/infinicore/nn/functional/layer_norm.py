from typing import List

from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def layer_norm(
    input: Tensor,
    normalized_shape: List[int],
    weight: Tensor,
    bias: Tensor,
    eps: float = 1e-5,
    *,
    out=None,
) -> Tensor:
    r"""Apply Layer Normalization."""

    assert normalized_shape == weight.shape, (
        "normalized_shape does not match weight.shape."
    )

    if out is None:
        return Tensor(
            _infinicore.layer_norm(
                input._underlying, weight._underlying, bias._underlying, eps
            )
        )

    _infinicore.layer_norm_(
        out._underlying, input._underlying, weight._underlying, bias._underlying, eps
    )

    return out
