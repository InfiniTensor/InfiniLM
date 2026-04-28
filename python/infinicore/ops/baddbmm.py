from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def baddbmm(input, batch1, batch2, *, beta=1.0, alpha=1.0, out=None):
    if out is None:
        return Tensor(
            _infinicore.baddbmm(
                input._underlying,
                batch1._underlying,
                batch2._underlying,
                float(beta),
                float(alpha),
            )
        )
    _infinicore.baddbmm_(
        out._underlying,
        input._underlying,
        batch1._underlying,
        batch2._underlying,
        float(beta),
        float(alpha),
    )

    return out
