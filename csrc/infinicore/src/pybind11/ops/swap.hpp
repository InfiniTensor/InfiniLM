#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/swap.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_swap(py::module &m) {
    m.def("swap_",
          &op::swap_,
          py::arg("x"),
          py::arg("y"),
          R"doc(In-place BLAS level-1 swap.)doc");
}

} // namespace infinicore::ops
