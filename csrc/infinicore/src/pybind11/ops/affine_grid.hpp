#pragma once

#include "infinicore/ops/affine_grid.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_affine_grid(py::module &m) {
    // 绑定函数接口: grid = affine_grid(theta, size, align_corners)
    m.def("affine_grid",
          &op::affine_grid,
          py::arg("theta"),
          py::arg("size"),
          py::arg("align_corners") = false, // 设置默认值
          R"doc(Generates a 2D or 3D flow field (sampling grid), given a batch of affine matrices theta.

Args:
    theta (Tensor): Input affine matrices of shape (N, 2, 3) for 2D or (N, 3, 4) for 3D.
    size (List[int]): The target output image size. Usually (N, C, H, W) for 2D or (N, C, D, H, W) for 3D.
    align_corners (bool, optional): Geometrically, we consider the pixels of the input as squares rather than points. If set to True, the extrema (-1 and 1) are considered as referring to the center points of the input's corner pixels. If set to False, they are instead considered as referring to the corner points of the input's corner pixels, making the sampling more resolution agnostic. Defaults to False.

Returns:
    Tensor: Output tensor of shape (N, H, W, 2).
)doc");
}

} // namespace infinicore::ops
