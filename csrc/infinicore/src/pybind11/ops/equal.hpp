#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/equal.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_equal(py::module &m) {
    m.def("equal",
          &op::equal,
          py::arg("a"),
          py::arg("b"),
          R"doc(Elementwise equality returning a bool tensor.)doc");

    m.def("equal_",
          &op::equal_,
          py::arg("out"),
          py::arg("a"),
          py::arg("b"),
          R"doc(In-place elementwise equality writing into `out`.)doc");
}

} // namespace infinicore::ops
