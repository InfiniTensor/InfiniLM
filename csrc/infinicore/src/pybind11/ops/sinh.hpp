#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/sinh.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_sinh(py::module &m) {
    m.def(
        "sinh",
        &op::sinh,
        py::arg("input"),
        R"doc(Sinh activation function.)doc");

    m.def(
        "sinh_",
        &op::sinh_,
        py::arg("output"),
        py::arg("input"),
        R"doc(In-place Sinh activation function.)doc");
}

} // namespace infinicore::ops
