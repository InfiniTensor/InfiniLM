from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def affine_grid(theta: Tensor, size: list[int], align_corners: bool = False) -> Tensor:
    r"""Generates a 2D flow field (sampling grid), given a batch of affine matrices theta."""

    # 直接调用底层绑定
    # theta._underlying: 传递底层 C++ Tensor 对象
    # size: Python list[int] 自动转换为 C++ std::vector<int64_t>
    return Tensor(_infinicore.affine_grid(theta._underlying, size, align_corners))
