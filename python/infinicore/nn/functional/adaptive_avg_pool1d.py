from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def adaptive_avg_pool1d(input: Tensor, output_size: int) -> Tensor:
    r"""Apply a 1D adaptive average pooling."""
    return Tensor(_infinicore.adaptive_avg_pool1d(input._underlying, output_size))
