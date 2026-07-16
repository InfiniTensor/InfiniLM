#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/softplus.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_softplus(py::module &m) {
    // Functional interface: returns a new Tensor
    m.def("softplus",
          &op::softplus,
          py::arg("x"),
          py::arg("beta") = 1.0f,
          py::arg("threshold") = 20.0f,
          R"doc(Computes the softplus function element-wise: y = 1/beta * log(1 + exp(beta * x)).)doc");

    // In-place/Out-variant interface: writes to provided output Tensor
    m.def("softplus_",
          &op::softplus_,
          py::arg("y"),
          py::arg("x"),
          py::arg("beta") = 1.0f,
          py::arg("threshold") = 20.0f,
          R"doc(In-place softplus activation. Writes result into y.)doc");
}

} // namespace infinicore::ops
