#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/asinh.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_asinh(py::module &m) {
    m.def("asinh",
          &op::asinh,
          py::arg("x"),
          R"doc(Element-wise inverse hyperbolic sine function.)doc");

    m.def("asinh_",
          &op::asinh_,
          py::arg("y"),
          py::arg("x"),
          R"doc(In-place element-wise inverse hyperbolic sine function.)doc");
}

} // namespace infinicore::ops
