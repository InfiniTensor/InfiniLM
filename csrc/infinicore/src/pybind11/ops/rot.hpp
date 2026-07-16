#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/rot.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_rot(py::module &m) {
    m.def("rot_",
          &op::rot_,
          py::arg("x"),
          py::arg("y"),
          py::arg("c"),
          py::arg("s"),
          R"doc(In-place BLAS level-1 rot, updating x and y.)doc");
}

} // namespace infinicore::ops
