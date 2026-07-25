#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/asum.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_asum(py::module &m) {
    m.def("asum",
          &op::asum,
          py::arg("x"),
          R"doc(BLAS level-1 asum.)doc");

    m.def("asum_",
          &op::asum_,
          py::arg("x"),
          py::arg("result"),
          R"doc(In-place BLAS level-1 asum.)doc");
}

} // namespace infinicore::ops
