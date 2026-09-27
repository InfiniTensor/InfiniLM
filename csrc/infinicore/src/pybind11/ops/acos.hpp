#pragma once

#include "infinicore/ops/acos.hpp"
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_acos(py::module &m) {
    m.def("acos",
          &op::acos,
          py::arg("input"),
          R"doc(Computes the inverse cosine (arccosine) of each element of input.

Returns a new tensor with the arccosine of the elements of input.
The range of the result is [0, pi].)doc");

    m.def("acos_",
          &op::acos_,
          py::arg("output"),
          py::arg("input"),
          R"doc(In-place acos operation. Writes result into output tensor.)doc");
}

} // namespace infinicore::ops
