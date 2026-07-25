#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/logaddexp.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_logaddexp(py::module &m) {
    m.def("logaddexp",
          &op::logaddexp,
          py::arg("a"),
          py::arg("b"),
          R"doc(Logarithm of the sum of exponentiations of the inputs.)doc");
    m.def("logaddexp_",
          &op::logaddexp_,
          py::arg("c"),
          py::arg("a"),
          py::arg("b"),
          R"doc(In-place logaddexp operation. Writes results into c tensor.)doc");
}

} // namespace infinicore::ops
