#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/prelu.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_prelu(py::module &m) {
    m.def("prelu",
          &op::prelu,
          py::arg("input"),
          py::arg("weight"),
          R"doc(Parametric ReLU.)doc");

    m.def("prelu_",
          &op::prelu_,
          py::arg("out"),
          py::arg("input"),
          py::arg("weight"),
          R"doc(In-place Parametric ReLU (writes to out).)doc");
}

} // namespace infinicore::ops
