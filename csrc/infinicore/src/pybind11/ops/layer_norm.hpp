#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/layer_norm.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_layer_norm(py::module &m) {
    m.def("layer_norm",
          &op::layer_norm,
          py::arg("x"),
          py::arg("weight"),
          py::arg("bias"),
          py::arg("epsilon") = 1e-5f,
          R"doc(Layer Normalization.

Args:
    x: Input tensor
    weight: Scale weights
    bias: Bias weights
    epsilon: Small constant for numerical stability, default is 1e-5

Returns:
    Normalized tensor with same shape as input
)doc");

    m.def("layer_norm_",
          &op::layer_norm_for_pybind,
          py::arg("y"),
          py::arg("x"),
          py::arg("weight"),
          py::arg("bias"),
          py::arg("epsilon") = 1e-5f,
          R"doc(In-place Layer Normalization.

Args:
    y: Output tensor
    x: Input tensor
    weight: Scale weights
    bias: Bias weights
    epsilon: Small constant for numerical stability, default is 1e-5
)doc");
}

} // namespace infinicore::ops
