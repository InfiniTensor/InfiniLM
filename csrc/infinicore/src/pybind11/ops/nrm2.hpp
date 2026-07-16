#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/nrm2.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_nrm2(py::module &m) {
    m.def("nrm2",
          &op::nrm2,
          py::arg("x"),
          R"doc(BLAS level-1 nrm2.)doc");

    m.def("nrm2_",
          &op::nrm2_,
          py::arg("x"),
          py::arg("result"),
          R"doc(In-place BLAS level-1 nrm2.)doc");
}

} // namespace infinicore::ops
