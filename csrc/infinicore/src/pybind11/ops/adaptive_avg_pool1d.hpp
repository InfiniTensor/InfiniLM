#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/adaptive_avg_pool1d.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_adaptive_avg_pool1d(py::module &m) {
    // 绑定函数接口: output = adaptive_avg_pool1d(input, output_size)
    m.def("adaptive_avg_pool1d",
          &op::adaptive_avg_pool1d,
          py::arg("input"),
          py::arg("output_size"),
          R"doc(Applies a 1D adaptive average pooling over an input signal composed of several input planes.

Args:
    input (Tensor): Input tensor of shape (C, L) or (N, C, L).
    output_size (int): The target output size.

Returns:
    Tensor: Output tensor of shape (C, output_size) or (N, C, output_size).
)doc");
}

} // namespace infinicore::ops
