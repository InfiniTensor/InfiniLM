from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def avg_pool1d(
    input: Tensor,
    kernel_size: int,
    stride: int | None = None,
    padding: int = 0,
    *,
    out=None,
) -> Tensor:
    if stride is None:
        stride = 0

    if out is None:
        return Tensor(
            _infinicore.avg_pool1d(input._underlying, kernel_size, stride, padding)
        )

    _infinicore.avg_pool1d_(
        out._underlying, input._underlying, kernel_size, stride, padding
    )
    return out
