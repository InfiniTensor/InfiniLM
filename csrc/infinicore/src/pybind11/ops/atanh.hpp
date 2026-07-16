#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/atanh.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_atanh(py::module &m) {
    m.def("atanh",
          &op::atanh,
          py::arg("a"),
          R"doc(Inverse hyperbolic tangent of a tensor.)doc");

    m.def("atanh_",
          &op::atanh_,
          py::arg("y"),
          py::arg("a"),
          R"doc(Compute inverse hyperbolic tangent and store in the provided output tensor.)doc");
}

} // namespace infinicore::ops
