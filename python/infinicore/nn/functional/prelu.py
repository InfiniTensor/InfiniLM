from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def prelu(input: Tensor, weight: Tensor) -> Tensor:
    return Tensor(_infinicore.prelu(input._underlying, weight._underlying))
