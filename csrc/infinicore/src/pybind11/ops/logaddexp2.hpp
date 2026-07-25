#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/logaddexp2.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_logaddexp2(py::module &m) {
    m.def("logaddexp2",
          &op::logaddexp2,
          py::arg("a"),
          py::arg("b"),
          R"doc(Logarithm of the sum of exponentiations of the inputs in base-2.)doc");
    m.def("logaddexp2_",
          &op::logaddexp2_,
          py::arg("c"),
          py::arg("a"),
          py::arg("b"),
          R"doc(In-place logaddexp2 operation. Writes results into c tensor.)doc");
}

} // namespace infinicore::ops
