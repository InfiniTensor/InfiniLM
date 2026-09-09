#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/dist.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_dist(py::module &m) {
    m.def("dist",
          &op::dist,
          py::arg("x1"),
          py::arg("x2"),
          py::arg("p") = 2.0,
          R"doc(p-norm distance between two tensors.)doc");

    m.def("dist_",
          &op::dist_,
          py::arg("y"),
          py::arg("x1"),
          py::arg("x2"),
          py::arg("p") = 2.0,
          R"doc(Out variant of dist.)doc");
}

} // namespace infinicore::ops
