from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def argwhere(x: Tensor) -> Tensor:
    return Tensor(_infinicore.argwhere(x._underlying))
