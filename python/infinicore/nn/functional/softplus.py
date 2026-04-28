from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def softplus(input, beta=1, threshold=20, *, out=None):
    if out is None:
        # 修改：将 beta 和 threshold 传递给底层 C++
        return Tensor(_infinicore.softplus(input._underlying, beta, threshold))

    # 修改：将 beta 和 threshold 传递给底层 C++ (In-place)
    _infinicore.softplus_(out._underlying, input._underlying, beta, threshold)

    return out
