#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/scaled_mm_i8.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_scaled_mm_i8(py::module &m) {
    m.def("scaled_mm_i8",
          &op::scaled_mm_i8,
          py::arg("a_p"),
          py::arg("a_s"),
          py::arg("b_p"),
          py::arg("b_s"),
          py::arg("bias"),
          R"doc(Scaled matrix multiplication of two tensors.)doc");

    m.def("scaled_mm_i8_",
          &op::scaled_mm_i8_,
          py::arg("a"),
          py::arg("b"),
          py::arg("a_scale"),
          py::arg("b_scale"),
          R"doc(In-place Scaled matrix multiplication of two tensors.)doc");
}

} // namespace infinicore::ops
