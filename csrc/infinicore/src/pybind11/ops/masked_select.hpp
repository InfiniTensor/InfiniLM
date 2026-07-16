#pragma once

#include "infinicore/ops/masked_select.hpp"
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_masked_select(py::module &m) {

    m.def("masked_select",
          &op::masked_select,
          py::arg("input"),
          py::arg("mask"),
          R"doc(opertor: torch.masked_select, out-of-place mode)doc");
}

} // namespace infinicore::ops
