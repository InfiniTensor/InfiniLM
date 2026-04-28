from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def var(input, dim=None, unbiased=True, keepdim=False, out=None):
    if out is None:
        var_tensor = _infinicore.var(input._underlying, dim, unbiased, keepdim)
        return Tensor(var_tensor)
    var_output = out
    _infinicore.var_(var_output._underlying, input._underlying, dim, unbiased, keepdim)

    return out
