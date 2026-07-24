#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/matmul.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_matmul(py::module &m) {
    m.def("matmul",
          &op::matmul,
          py::arg("a"),
          py::arg("b"),
          py::arg("alpha") = 1.0f,
          R"doc(Matrix multiplication of two tensors.)doc");

    m.def("matmul_",
          &op::matmul_,
          py::arg("c"),
          py::arg("a"),
          py::arg("b"),
          py::arg("alpha") = 1.0f,
          R"doc(In-place matrix multiplication.)doc");
}

} // namespace infinicore::ops
