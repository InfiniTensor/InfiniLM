#pragma once

#include "infinicore/ops/hypot.hpp"
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_hypot(py::module &m) {
    m.def("hypot",
          &op::hypot,
          py::arg("input"),
          py::arg("other"),
          R"doc(Computes the hypotenuse of input and other arguments, i.e. sqrt(input^2 + other^2).)doc");

    m.def("hypot_",
          &op::hypot_,
          py::arg("output"),
          py::arg("input"),
          py::arg("other"),
          R"doc(In-place hypot operation. Writes result into output tensor.)doc");
}

} // namespace infinicore::ops
