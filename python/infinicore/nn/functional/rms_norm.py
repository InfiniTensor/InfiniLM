from typing import List

from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def rms_norm(
    input: Tensor,
    normalized_shape: List[int],
    weight: Tensor,
    eps: float = 1e-5,
    *,
    out=None,
) -> Tensor:
    r"""Apply Root Mean Square Layer Normalization."""

    assert normalized_shape == weight.shape, (
        "normalized_shape does not match weight.shape."
    )

    if out is None:
        return Tensor(_infinicore.rms_norm(input._underlying, weight._underlying, eps))

    _infinicore.rms_norm_(out._underlying, input._underlying, weight._underlying, eps)

    return out
