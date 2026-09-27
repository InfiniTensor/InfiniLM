#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/tan.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_tan(py::module &m) {
    m.def("tan",
          &op::tan,
          py::arg("input"),
          R"doc(opertor: torch.tan, out-of-place mode)doc");

    m.def("tan_",
          &op::tan_,
          py::arg("output"),
          py::arg("input"),
          R"doc(opertor: torch.tan, in-place mode)doc");
}

} // namespace infinicore::ops
