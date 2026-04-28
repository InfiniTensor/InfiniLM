from typing import Optional, Sequence, Union

from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def upsample_bilinear(
    input: Tensor,
    size: Optional[Union[int, Sequence[int]]] = None,
    scale_factor: Optional[Union[float, Sequence[float]]] = None,
    align_corners: bool = False,
    *,
    out: Optional[Tensor] = None,
) -> Tensor:
    r"""
    Applies bilinear interpolation upsampling to the input tensor.
    """
    # 确保输入连续
    if not input.is_contiguous():
        input = input.contiguous()

    # 参数互斥检查
    if (size is None) == (scale_factor is None):
        raise ValueError("Either size or scale_factor should be defined, but not both.")

    # 计算目标输出尺寸 (H, W)
    output_size = []

    if size is not None:
        if isinstance(size, int):
            # 如果是单个整数，应用于 H 和 W
            output_size = [size, size]
        elif isinstance(size, (list, tuple)):
            if len(size) < 2:
                raise ValueError(
                    "size sequence must contain at least 2 elements for bilinear upsampling"
                )
            output_size = [size[0], size[1]]
        else:
            raise ValueError("size must be int or sequence of int")
    else:
        # 基于 scale_factor 计算
        if isinstance(scale_factor, float):
            scale_h = scale_factor
            scale_w = scale_factor
        elif isinstance(scale_factor, (list, tuple)):
            if len(scale_factor) < 2:
                raise ValueError(
                    "scale_factor sequence must contain at least 2 elements"
                )
            scale_h = scale_factor[0]
            scale_w = scale_factor[1]
        else:
            raise ValueError("scale_factor must be float or sequence of float")

        # 假设输入是 (..., H, W)，取最后两维
        h_in = input.shape[-2]
        w_in = input.shape[-1]
        output_size = [int(h_in * scale_h), int(w_in * scale_w)]

    # 1. 显式输出 (In-place / Out parameter)
    if out is not None:
        if not out.is_contiguous():
            raise RuntimeError("out tensor must be contiguous")

        _infinicore.upsample_bilinear_(
            out._underlying, input._underlying, align_corners
        )
        return out

    # 2. 函数式调用 (Functional API)
    return Tensor(
        _infinicore.upsample_bilinear(input._underlying, output_size, align_corners)
    )
