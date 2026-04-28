#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/diff.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_diff(py::module &m) {
    m.def("diff",
          &op::diff,
          py::arg("x"),
          py::arg("n") = 1,
          py::arg("dim") = -1,
          R"doc(Difference of adjacent elements along a dimension.)doc");

    m.def("diff_",
          &op::diff_,
          py::arg("y"),
          py::arg("x"),
          py::arg("n") = 1,
          py::arg("dim") = -1,
          R"doc(Out variant of diff.)doc");
}

} // namespace infinicore::ops
