import infinicore
from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def hardswish(input: Tensor, inplace: bool = False, *, out=None) -> Tensor:
    r"""Apply the Hardswish activation function element-wise."""

    if (
        infinicore.use_ntops
        and input.device.type in ("cuda", "musa")
        and out is None
        and hasattr(infinicore.ntops.torch, "hardswish")
    ):
        try:
            return infinicore.ntops.torch.hardswish(input, inplace=inplace)
        except AttributeError:
            pass

    if inplace:
        _infinicore.hardswish_(input._underlying, input._underlying)
        return input

    if out is None:
        return Tensor(_infinicore.hardswish(input._underlying))

    _infinicore.hardswish_(out._underlying, input._underlying)
    return out
