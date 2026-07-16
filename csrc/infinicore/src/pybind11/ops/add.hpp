#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/add.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_add(py::module &m) {
    m.def("add",
          &op::add,
          py::arg("a"),
          py::arg("b"),
          R"doc(Addition of two tensors.)doc");

    m.def("add_",
          &op::add_,
          py::arg("c"),
          py::arg("a"),
          py::arg("b"),
          R"doc(In-place tensor addition.)doc");
}

} // namespace infinicore::ops
