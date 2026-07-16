#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/adaptive_max_pool1d.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_adaptive_max_pool1d(py::module &m) {
    m.def("adaptive_max_pool1d",
          &op::adaptive_max_pool1d,
          py::arg("x"),
          py::arg("output_size"),
          R"doc(1D Adaptive Max Pooling.

Args:
    x: Input tensor of shape (N, C, L_in) or (N, L_in)
    output_size: Target output size L_out
Returns:
    Output tensor of shape (N, C, L_out) or (N, L_out)
)doc");

    m.def("adaptive_max_pool1d_",
          &op::adaptive_max_pool1d_,
          py::arg("y"),
          py::arg("x"),
          py::arg("output_size"),
          R"doc(In-place 1D Adaptive Max Pooling.

Args:
    y: Output tensor of shape (N, C, L_out) or (N, L_out)
    x: Input tensor of shape (N, C, L_in) or (N, L_in)
    output_size: Target output size L_out
)doc");
}

} // namespace infinicore::ops
