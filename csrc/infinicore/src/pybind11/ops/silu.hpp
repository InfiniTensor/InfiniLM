#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/silu.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_silu(py::module &m) {
    m.def("silu",
          &op::silu,
          py::arg("input"),
          R"doc(SiLU (Swish) activation function.)doc");

    m.def("silu_",
          &op::silu_,
          py::arg("output"),
          py::arg("input"),
          R"doc(In-place SiLU (Swish) activation function.)doc");
}

} // namespace infinicore::ops
