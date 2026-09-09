#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/cross_entropy.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_cross_entropy(py::module &m) {
    m.def("cross_entropy",
          &op::cross_entropy,
          py::arg("logits"),
          py::arg("target"),
          R"doc(Token-wise cross entropy loss without reduction.)doc");

    m.def("cross_entropy_",
          &op::cross_entropy_,
          py::arg("loss"),
          py::arg("logits"),
          py::arg("target"),
          R"doc(Write cross entropy loss into a provided tensor.)doc");
}

} // namespace infinicore::ops
