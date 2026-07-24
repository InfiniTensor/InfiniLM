#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/bitwise_right_shift.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_bitwise_right_shift(py::module &m) {
    m.def("bitwise_right_shift",
          &op::bitwise_right_shift,
          py::arg("input"),
          py::arg("other"),
          R"doc(Element-wise bitwise right shift.)doc");

    m.def("bitwise_right_shift_",
          &op::bitwise_right_shift_,
          py::arg("out"),
          py::arg("input"),
          py::arg("other"),
          R"doc(In-place element-wise bitwise right shift.)doc");
}

} // namespace infinicore::ops
