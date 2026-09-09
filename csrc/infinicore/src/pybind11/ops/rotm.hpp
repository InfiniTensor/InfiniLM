#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/rotm.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_rotm(py::module &m) {
    m.def("rotm_",
          py::overload_cast<Tensor, Tensor, Tensor>(&op::rotm_),
          py::arg("x"),
          py::arg("y"),
          py::arg("param"),
          R"doc(In-place BLAS level-1 rotm, updating x and y.)doc");
}

} // namespace infinicore::ops
