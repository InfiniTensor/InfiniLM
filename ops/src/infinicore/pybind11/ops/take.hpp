#pragma once

#include "infinicore/ops/take.hpp"
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_take(py::module &m) {
    m.def("take",
          &op::take,
          py::arg("input"),
          py::arg("indices"),
          R"doc(Extracts elements from the input tensor along the given indices. 
The input tensor is treated as a flattened 1D array.)doc");
    m.def("take_",
          &op::take_,
          py::arg("output"),
          py::arg("input"),
          py::arg("indices"),
          R"doc(Explicit output take operation. Writes the result into the output tensor.)doc");
}

} // namespace infinicore::ops
