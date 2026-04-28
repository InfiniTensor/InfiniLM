from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def bitwise_right_shift(
    input: Tensor, other: Tensor, *, out: Tensor | None = None
) -> Tensor:
    if out is None:
        return Tensor(
            _infinicore.bitwise_right_shift(input._underlying, other._underlying)
        )

    _infinicore.bitwise_right_shift_(
        out._underlying, input._underlying, other._underlying
    )
    return out
