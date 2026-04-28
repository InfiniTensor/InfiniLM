from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def linear_w8a8i8(
    input: Tensor,
    weight_packed: Tensor,
    weight_scale: Tensor,
    bias=None,
    out=None,
) -> Tensor:
    r"""Linear layer with weight quantized to int8 and input quantized to int8 with per-tensor scale."""

    if out is None:
        return Tensor(
            _infinicore.linear_w8a8i8(
                input._underlying,
                weight_packed._underlying,
                weight_scale._underlying,
                None if bias is None else bias._underlying,
            )
        )

    _infinicore.linear_w8a8i8_(
        out._underlying,
        input._underlying,
        weight_packed._underlying,
        weight_scale._underlying,
        None if bias is None else bias._underlying,
    )
    return out
