#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/logdet.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_logdet(py::module &m) {
    m.def("logdet",
          &op::logdet,
          py::arg("x"),
          R"doc(Log determinant of a square matrix (NaN if determinant is negative).)doc");

    m.def("logdet_",
          &op::logdet_,
          py::arg("y"),
          py::arg("x"),
          R"doc(Out variant of logdet.)doc");
}

} // namespace infinicore::ops
