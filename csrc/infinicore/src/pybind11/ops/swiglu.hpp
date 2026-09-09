#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/swiglu.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_swiglu(py::module &m) {
    m.def("swiglu",
          &op::swiglu,
          py::arg("a"),
          py::arg("b"),
          R"doc(SwiGLU activation function.)doc");

    m.def("swiglu_",
          &op::swiglu_,
          py::arg("c"),
          py::arg("a"),
          py::arg("b"),
          R"doc(In-place SwiGLU activation function.)doc");
}

} // namespace infinicore::ops
