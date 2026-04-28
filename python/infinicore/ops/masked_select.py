from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def masked_select(input, mask):
    return Tensor(_infinicore.masked_select(input._underlying, mask._underlying))
