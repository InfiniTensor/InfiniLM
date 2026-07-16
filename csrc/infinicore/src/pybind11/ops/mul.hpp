#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/mul.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_mul(py::module &m) {
    m.def("mul",
          &op::mul,
          py::arg("a"),
          py::arg("b"),
          R"doc(Element-wise multiplication of two tensors.)doc");

    m.def("mul_",
          &op::mul_,
          py::arg("c"),
          py::arg("a"),
          py::arg("b"),
          R"doc(In-place element-wise tensor multiplication.)doc");
}

} // namespace infinicore::ops
