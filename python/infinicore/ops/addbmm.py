from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def addbmm(input, batch1, batch2, *, beta=1.0, alpha=1.0, out=None):
    # 1. Out-of-place 模式 (如果没有指定 out)
    if out is None:
        return Tensor(
            _infinicore.addbmm(
                input._underlying, batch1._underlying, batch2._underlying, beta, alpha
            )
        )

    # 2. In-place 模式 (指定了 out)
    _infinicore.addbmm_(
        out._underlying,
        input._underlying,
        batch1._underlying,
        batch2._underlying,
        beta,
        alpha,
    )

    return out
