from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


# 修改说明：将参数名 'shape' 改为 'size'，以匹配测试用例中的调用方式 kwargs={size=...}
def broadcast_to(input, size, *, out=None):
    if out is None:
        return Tensor(_infinicore.broadcast_to(input._underlying, size))

    _infinicore.broadcast_to_(out._underlying, input._underlying)

    return out
