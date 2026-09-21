#pragma once

#include "infinicore/ops/logcumsumexp.hpp"
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_logcumsumexp(py::module &m) {
    m.def("logcumsumexp",
          &op::logcumsumexp,
          py::arg("input"),
          py::arg("dim"),
          py::arg("exclusive") = false,
          py::arg("reverse") = false,
          R"doc(Computes the logarithm of the cumulative summation of the exponentiation of elements.)doc");

    m.def("logcumsumexp_",
          &op::logcumsumexp_,
          py::arg("out"),
          py::arg("input"),
          py::arg("dim"),
          py::arg("exclusive") = false,
          py::arg("reverse") = false,
          R"doc(In-place version of logcumsumexp.)doc");
}

} // namespace infinicore::ops
