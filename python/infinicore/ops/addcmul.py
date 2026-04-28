from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def addcmul(input, tensor1, tensor2, value=1.0, *, out=None):
    if out is None:
        return Tensor(
            _infinicore.addcmul(
                input._underlying,
                tensor1._underlying,
                tensor2._underlying,
                float(value),
            )
        )

    _infinicore.addcmul_(
        out._underlying,
        input._underlying,
        tensor1._underlying,
        tensor2._underlying,
        float(value),
    )

    return out
