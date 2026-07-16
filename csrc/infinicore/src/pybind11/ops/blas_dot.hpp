#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/blas_dot.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_blas_dot(py::module &m) {
    m.def("blas_dot",
          &op::blas_dot,
          py::arg("x"),
          py::arg("y"),
          R"doc(BLAS level-1 dot.)doc");

    m.def("blas_dot_",
          &op::blas_dot_,
          py::arg("x"),
          py::arg("y"),
          py::arg("result"),
          R"doc(In-place BLAS level-1 dot.)doc");
}

} // namespace infinicore::ops
