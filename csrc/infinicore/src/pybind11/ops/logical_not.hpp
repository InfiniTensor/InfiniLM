#pragma once

#include "infinicore/ops/logical_not.hpp"
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_logical_not(py::module &m) {
    // Out-of-place: output = logical_not(input)
    m.def("logical_not",
          &op::logical_not,
          py::arg("input"),
          R"doc(Logical NOT of the tensor.)doc");

    // In-place / Explicit Output: logical_not_(output, input)
    // 对应 C++: void logical_not_(Tensor output, Tensor input)
    m.def("logical_not_",
          &op::logical_not_,
          py::arg("output"),
          py::arg("input"),
          R"doc(In-place logical NOT computation.)doc");
}

} // namespace infinicore::ops
