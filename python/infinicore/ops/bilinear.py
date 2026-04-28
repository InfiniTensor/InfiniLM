from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def bilinear(input1, input2, weight, bias=None, *, out=None):
    if out is None:
        return Tensor(
            _infinicore.bilinear(
                input1._underlying,
                input2._underlying,
                weight._underlying,
                bias._underlying if bias is not None else None,
            )
        )
    _infinicore.bilinear_(
        out._underlying,
        input1._underlying,
        input2._underlying,
        weight._underlying,
        bias._underlying if bias is not None else None,
    )

    return out
