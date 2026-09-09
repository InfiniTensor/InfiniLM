#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/reciprocal.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_reciprocal(py::module &m) {
    m.def("reciprocal",
          &op::reciprocal,
          py::arg("x"),
          R"doc(Computes the reciprocal of the input tensor.)doc");

    m.def("reciprocal_",
          &op::reciprocal_,
          py::arg("y"),
          py::arg("x"),
          R"doc(Computes the reciprocal of the input tensor and stores in the output tensor.)doc");
}

} // namespace infinicore::ops
