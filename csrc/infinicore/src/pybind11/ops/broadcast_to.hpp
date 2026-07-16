#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "infinicore/ops/broadcast_to.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_broadcast_to(py::module &m) {
    m.def("broadcast_to",
          &op::broadcast_to,
          py::arg("x"),
          py::arg("shape"),
          R"doc(Broadcast tensor to target shape.)doc");

    m.def("broadcast_to_",
          &op::broadcast_to_,
          py::arg("y"),
          py::arg("x"),
          R"doc(In-place/Out broadcast tensor.)doc");
}

} // namespace infinicore::ops
