#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/tanhshrink.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_tanhshrink(py::module &m) {
    m.def("tanhshrink",
          &op::tanhshrink,
          py::arg("input"),
          R"doc(opertor: torch.tanhshrink, out-of-place mode)doc");

    m.def("tanhshrink_",
          &op::tanhshrink_,
          py::arg("output"),
          py::arg("input"),
          R"doc(opertor: torch.tanhshrink, in-place mode)doc");
}

} // namespace infinicore::ops
