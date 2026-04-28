from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def embedding(
    input: Tensor,
    weight: Tensor,
    padding_idx=None,
    max_norm=None,
    norm_type=2.0,
    scale_grad_by_freq=False,
    sparse=False,
    *,
    out=None,
) -> Tensor:
    r"""Generate a simple lookup table that looks up embeddings in a fixed dictionary and size."""

    assert (
        (padding_idx is None)
        and (max_norm is None)
        and (scale_grad_by_freq is False)
        and (sparse is False)
    ), "Unsupported parameters."

    # Note: embedding now supports device-side input for graph recording
    # The C++ implementation handles both CPU and device-side inputs

    if out is None:
        return Tensor(_infinicore.embedding(input._underlying, weight._underlying))

    _infinicore.embedding_(out._underlying, input._underlying, weight._underlying)
    return out
