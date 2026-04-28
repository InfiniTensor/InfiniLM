from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def var_mean(input, dim=None, unbiased=True, keepdim=False, out=None):
    if out is None:
        var_tensor, mean_tensor = _infinicore.var_mean(
            input._underlying, dim, unbiased, keepdim
        )
        return Tensor(var_tensor), Tensor(mean_tensor)
    var_output, mean_output = out
    _infinicore.var_mean_(
        var_output._underlying,
        mean_output._underlying,
        input._underlying,
        dim,
        unbiased,
        keepdim,
    )

    return out
