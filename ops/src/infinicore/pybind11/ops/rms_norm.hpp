#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/rms_norm.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_rms_norm(py::module &m) {
    m.def("rms_norm",
          &op::rms_norm,
          py::arg("x"),
          py::arg("weight"),
          py::arg("epsilon") = 1e-5f,
          R"doc(Root Mean Square Normalization.

Args:
    x: Input tensor
    weight: Scale weights
    epsilon: Small constant for numerical stability, default is 1e-5

Returns:
    Normalized tensor with same shape as input
)doc");

    m.def("rms_norm_",
          &op::rms_norm_,
          py::arg("y"),
          py::arg("x"),
          py::arg("weight"),
          py::arg("epsilon") = 1e-5f,
          R"doc(In-place Root Mean Square Normalization.

Args:
    y: Output tensor
    x: Input tensor
    weight: Scale weights
    epsilon: Small constant for numerical stability, default is 1e-5
)doc");
}

} // namespace infinicore::ops
