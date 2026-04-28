import infinicore
from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def silu(input: Tensor, inplace: bool = False, *, out=None) -> Tensor:
    r"""Apply the Sigmoid Linear Unit (SiLU) function, element-wise."""

    if infinicore.use_ntops and input.device.type in ("cuda", "musa") and out is None:
        return infinicore.ntops.torch.silu(input, inplace=inplace)

    if inplace:
        _infinicore.silu_(input._underlying, input._underlying)
        return input

    if out is None:
        return Tensor(_infinicore.silu(input._underlying))

    _infinicore.silu_(out._underlying, input._underlying)

    return out
