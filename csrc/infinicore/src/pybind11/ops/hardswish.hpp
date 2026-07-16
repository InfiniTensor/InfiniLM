#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/hardswish.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_hardswish(py::module &m) {
    m.def("hardswish",
          &op::hardswish,
          py::arg("input"),
          R"doc(Out-of-place Hardswish activation.)doc");

    m.def("hardswish_",
          &op::hardswish_,
          py::arg("output"),
          py::arg("input"),
          R"doc(In-place Hardswish activation.)doc");
}

} // namespace infinicore::ops
