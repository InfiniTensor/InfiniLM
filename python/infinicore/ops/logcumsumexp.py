from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def logcumsumexp(input, dim, exclusive=False, reverse=False, *, out=None):
    if out is None:
        # 调用 C++ 绑定的非原地操作，返回新创建的 _underlying 句柄
        return Tensor(
            _infinicore.logcumsumexp(input._underlying, dim, exclusive, reverse)
        )

    # 调用 C++ 绑定的原地/指定输出操作
    _infinicore.logcumsumexp_(
        out._underlying, input._underlying, dim, exclusive, reverse
    )

    return out
