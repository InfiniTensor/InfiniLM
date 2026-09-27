#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/rotmg.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_rotmg(py::module &m) {
    m.def("rotmg_",
          py::overload_cast<Tensor, Tensor, Tensor, Tensor, Tensor>(&op::rotmg_),
          py::arg("d1"),
          py::arg("d2"),
          py::arg("x1"),
          py::arg("y1"),
          py::arg("param"),
          R"doc(In-place BLAS level-1 rotmg, updating d1, d2, x1, and param.)doc");
}

} // namespace infinicore::ops
