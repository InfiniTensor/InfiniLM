#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/blas_amin.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_blas_amin(py::module &m) {
    m.def("blas_amin",
          &op::blas_amin,
          py::arg("x"),
          R"doc(BLAS level-1 amin.)doc");

    m.def("blas_amin_",
          &op::blas_amin_,
          py::arg("x"),
          py::arg("result"),
          R"doc(In-place BLAS level-1 amin.)doc");
}

} // namespace infinicore::ops
