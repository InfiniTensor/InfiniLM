import infinicore
from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def logical_and(input: Tensor, other: Tensor, *, out=None) -> Tensor:
    r"""Computes the element-wise logical AND of the given input tensors."""
    if input.device.type not in ("cpu"):
        assert infinicore.use_ntops
        return infinicore.ntops.torch.logical_and(input, other, out=out)

    if out is None:
        return Tensor(_infinicore.logical_and(input._underlying, other._underlying))

    _infinicore.logical_and_(out._underlying, input._underlying, other._underlying)
    return out
