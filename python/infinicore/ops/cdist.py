from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def cdist(x1, x2, p=2.0, *, out=None):
    """
    计算两组向量集合中每一对向量之间的 p-norm 距离。

    参数:
        x1 (Tensor): 形状为 (M, D) 的输入张量。
        x2 (Tensor): 形状为 (N, D) 的输入张量。
        p (float): p-norm 的阶数，默认为 2.0。
        out (Tensor, optional): 结果输出张量。

    返回:
        Tensor: 形状为 (M, N) 的距离矩阵。
    """
    if out is None:
        # 非原地操作：由底层 C++ 接口根据 x1, x2 推导形状并创建新 Tensor
        return Tensor(
            _infinicore.cdist(
                x1._underlying,
                x2._underlying,
                float(p),
            )
        )

    # 原地/指定输出操作：结果写入用户提供的 out 张量
    _infinicore.cdist_(
        out._underlying,
        x1._underlying,
        x2._underlying,
        float(p),
    )

    return out
