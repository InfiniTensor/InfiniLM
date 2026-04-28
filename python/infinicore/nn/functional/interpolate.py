from collections.abc import Iterable

from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def _to_int64_list(value) -> list[int]:
    if isinstance(value, int):
        return [int(value)]
    if isinstance(value, Iterable):
        return [int(v) for v in value]
    raise TypeError(f"Expected int or iterable of ints, got {type(value).__name__}")


def _to_double_list(value) -> list[float]:
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, Iterable):
        return [float(v) for v in value]
    raise TypeError(f"Expected float or iterable of floats, got {type(value).__name__}")


# def interpolate(
#     input: Tensor,
#     size: Optional[Union[int, Sequence[int]]] = None,
#     scale_factor: Optional[Union[float, Sequence[float]]] = None,
#     mode: str = "nearest",
#     align_corners: Optional[bool] = None,
#     recompute_scale_factor: Optional[bool] = None,
# ) -> Tensor:
#     if mode == "nearest":
#         if align_corners is not None:
#             raise ValueError(
#                 "align_corners option can only be set with the "
#                 "interpolating modes: linear | bilinear | bicubic | trilinear"
#             )
#         return upsample_nearest(input, size, scale_factor)

#     if mode == "bilinear":
#         if align_corners is None:
#             align_corners = False
#         return upsample_bilinear(input, size, scale_factor, align_corners)

#     raise NotImplementedError(
#         f"Interpolation mode '{mode}' is not currently supported."
#         )


def interpolate(
    input: Tensor,
    size=None,
    scale_factor=None,
    mode: str = "nearest",
    align_corners=None,
) -> Tensor:
    size_list: list[int] = [] if size is None else _to_int64_list(size)
    scale_list: list[float] = (
        [] if scale_factor is None else _to_double_list(scale_factor)
    )

    if bool(size_list) == bool(scale_list):
        raise ValueError("Expected exactly one of size or scale_factor")

    spatial_ndim = input.ndim - 2
    if spatial_ndim < 1:
        raise ValueError("interpolate expects input with at least 3 dimensions")

    if size_list:
        if len(size_list) == 1 and spatial_ndim > 1:
            size_list = size_list * spatial_ndim
        if len(size_list) != spatial_ndim:
            raise ValueError(
                f"Expected size to have length {spatial_ndim}, got {len(size_list)}"
            )

    if scale_list:
        if len(scale_list) == 1 and spatial_ndim > 1:
            scale_list = scale_list * spatial_ndim
        if len(scale_list) != spatial_ndim:
            raise ValueError(
                f"Expected scale_factor to have length {spatial_ndim}, got {len(scale_list)}"
            )
        if any(v != scale_list[0] for v in scale_list[1:]):
            raise ValueError(
                "Per-dimension scale_factor is not supported; pass a scalar (or equal values)."
            )

    align_i = 0 if align_corners is None else int(bool(align_corners))

    return Tensor(
        _infinicore.interpolate(
            input._underlying, str(mode), size_list, scale_list, align_i
        )
    )
