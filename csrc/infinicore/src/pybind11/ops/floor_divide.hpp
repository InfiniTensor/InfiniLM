#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/floor_divide.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_floor_divide(py::module &m) {
    m.def("floor_divide",
          &op::floor_divide,
          py::arg("a"),
          py::arg("b"),
          R"doc(Floor division of two tensors.)doc");

    m.def("floor_divide_",
          &op::floor_divide_,
          py::arg("c"),
          py::arg("a"),
          py::arg("b"),
          R"doc(In-place tensor floor division.)doc");
}

} // namespace infinicore::ops
