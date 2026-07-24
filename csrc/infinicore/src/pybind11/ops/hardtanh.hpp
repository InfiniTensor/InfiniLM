#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/hardtanh.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_hardtanh(py::module &m) {
    m.def("hardtanh",
          &op::hardtanh,
          py::arg("input"),
          py::arg("min_val") = -1.0f,
          py::arg("max_val") = 1.0f,
          R"doc(Apply the HardTanh activation.)doc");

    m.def("hardtanh_",
          &op::hardtanh_,
          py::arg("output"),
          py::arg("input"),
          py::arg("min_val") = -1.0f,
          py::arg("max_val") = 1.0f,
          R"doc(In-place HardTanh activation.)doc");
}

} // namespace infinicore::ops
