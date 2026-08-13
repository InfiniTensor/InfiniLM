#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/rotg.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_rotg(py::module &m) {
    m.def("rotg_",
          py::overload_cast<Tensor, Tensor, Tensor, Tensor>(&op::rotg_),
          py::arg("x"),
          py::arg("y"),
          py::arg("c"),
          py::arg("s"),
          R"doc(In-place BLAS level-1 rotg, updating x, y, c, and s.)doc");
}

} // namespace infinicore::ops
