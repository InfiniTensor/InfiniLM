#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/fmin.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_fmin(py::module &m) {
    m.def("fmin",
          &op::fmin,
          py::arg("a"),
          py::arg("b"),
          R"doc(fmin of two tensors.)doc");

    m.def("fmin_",
          &op::fmin_,
          py::arg("c"),
          py::arg("a"),
          py::arg("b"),
          R"doc(In-place tensor fmin.)doc");
}

} // namespace infinicore::ops
