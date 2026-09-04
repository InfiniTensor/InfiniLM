#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/bilinear.hpp"

namespace py = pybind11;

namespace infinicore::ops {

Tensor py_bilinear(Tensor x1, Tensor x2, Tensor weight, pybind11::object bias) {
    std::optional<Tensor> bias_tensor = std::nullopt;
    if (!bias.is_none()) {
        bias_tensor = bias.cast<Tensor>();
    }
    return op::bilinear(x1, x2, weight, bias_tensor);
}

void py_bilinear_(Tensor out, Tensor x1, Tensor x2, Tensor weight, pybind11::object bias) {
    std::optional<Tensor> bias_tensor = std::nullopt;
    if (!bias.is_none()) {
        bias_tensor = bias.cast<Tensor>();
    }
    op::bilinear_(out, x1, x2, weight, bias_tensor);
}

inline void bind_bilinear(py::module &m) {
    m.def("bilinear",
          &py_bilinear,
          py::arg("x1"),
          py::arg("x2"),
          py::arg("weight"),
          py::arg("bias"),
          R"doc(Bilinear transformation of two input tensors.
Args:
    x1: First input tensor
    x2: Second input tensor
    weight: Weight tensor
    bias: Bias tensor (optional)
Returns:
    Output tensor after bilinear transformation
)doc");

    m.def("bilinear_",
          &py_bilinear_,
          py::arg("out"),
          py::arg("x1"),
          py::arg("x2"),
          py::arg("weight"),
          py::arg("bias"),
          R"doc(In-place bilinear transformation of two input tensors.
Args:
      out: Output tensor
      x1: First input tensor
      x2: Second input tensor
      weight: Weight tensor
      bias: Bias tensor (optional)
)doc");
}

} // namespace infinicore::ops
