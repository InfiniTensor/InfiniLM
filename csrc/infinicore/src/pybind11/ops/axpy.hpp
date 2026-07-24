#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/axpy.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_axpy(py::module &m) {
    m.def("axpy_",
          &op::axpy_,
          py::arg("alpha"),
          py::arg("x"),
          py::arg("y"),
          R"doc(In-place BLAS level-1 axpy, updating y.)doc");
}

} // namespace infinicore::ops
