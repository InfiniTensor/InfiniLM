from typing import Optional, Sequence, Union

from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def upsample_nearest(
    input: Tensor,
    size: Optional[Union[int, Sequence[int]]] = None,
    scale_factor: Optional[Union[float, Sequence[float]]] = None,
    *,
    out: Optional[Tensor] = None,
) -> Tensor:
    if not input.is_contiguous():
        input = input.contiguous()

    if (size is None) == (scale_factor is None):
        raise ValueError("Either size or scale_factor should be defined, but not both.")

    ndim = len(input.shape)
    output_size = []

    if size is not None:
        if isinstance(size, int):
            if ndim == 3:
                output_size = [size]
            else:
                output_size = [size, size]
        elif isinstance(size, (list, tuple)):
            output_size = [int(s) for s in size]
        else:
            raise ValueError("size must be int or sequence of int")
    else:
        if isinstance(scale_factor, (float, int)):
            scales = [float(scale_factor)]
        elif isinstance(scale_factor, (list, tuple)):
            scales = [float(s) for s in scale_factor]
        else:
            raise ValueError("scale_factor must be float or sequence of float")

        if ndim == 3:
            w_in = input.shape[-1]
            scale_w = scales[0] if len(scales) == 1 else scales[-1]
            output_size = [int(w_in * scale_w)]
        else:
            if len(scales) == 1:
                scale_h = scale_w = scales[0]
            elif len(scales) >= 2:
                scale_h, scale_w = scales[0], scales[1]
            else:
                raise ValueError("scale_factor sequence length mismatch")

            h_in = input.shape[-2]
            w_in = input.shape[-1]
            output_size = [int(h_in * scale_h), int(w_in * scale_w)]

    if out is not None:
        if not out.is_contiguous():
            raise RuntimeError("out tensor must be contiguous")

        _infinicore.upsample_nearest_(out._underlying, input._underlying)
        return out

    return Tensor(_infinicore.upsample_nearest(input._underlying, output_size))
