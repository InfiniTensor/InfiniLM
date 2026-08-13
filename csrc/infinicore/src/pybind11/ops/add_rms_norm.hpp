#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/add_rms_norm.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_add_rms_norm(py::module &m) {
    m.def("add_rms_norm",
          &op::add_rms_norm,
          py::arg("a"),
          py::arg("b"),
          py::arg("weight"),
          py::arg("epsilon") = 1e-5f,
          R"doc(Fused Add and RMS Normalization.

Args:
    a: First input tensor
    b: Second input tensor
    weight: Scale weights
    epsilon: Small constant for numerical stability, default is 1e-5

Returns:
    Tuple of (normalized_result, add_result): (RMSNorm(a + b) * weight, a + b)
    The add_result can be used as residual for subsequent layers.
)doc");

    m.def("add_rms_norm_",
          &op::add_rms_norm_,
          py::arg("y"),
          py::arg("residual_out"),
          py::arg("a"),
          py::arg("b"),
          py::arg("weight"),
          py::arg("epsilon") = 1e-5f,
          R"doc(In-place Fused Add and RMS Normalization.

Args:
    y: Output tensor for normalized result
    residual_out: Output tensor for add result (a + b) before normalization
    a: First input tensor
    b: Second input tensor
    weight: Scale weights
    epsilon: Small constant for numerical stability, default is 1e-5
)doc");
}

} // namespace infinicore::ops
