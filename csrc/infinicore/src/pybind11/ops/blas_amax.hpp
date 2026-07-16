#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/blas_amax.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_blas_amax(py::module &m) {
    m.def("blas_amax",
          &op::blas_amax,
          py::arg("x"),
          R"doc(BLAS level-1 amax.)doc");

    m.def("blas_amax_",
          &op::blas_amax_,
          py::arg("x"),
          py::arg("result"),
          R"doc(In-place BLAS level-1 amax.)doc");
}

} // namespace infinicore::ops
