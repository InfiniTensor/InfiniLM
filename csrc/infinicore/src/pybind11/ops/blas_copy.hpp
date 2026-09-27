#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/blas_copy.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_blas_copy(py::module &m) {
    m.def("blas_copy_",
          &op::blas_copy_,
          py::arg("x"),
          py::arg("y"),
          R"doc(In-place BLAS level-1 copy from x to y.)doc");
}

} // namespace infinicore::ops
