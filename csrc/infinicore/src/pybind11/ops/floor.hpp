#pragma once

#include "infinicore/ops/floor.hpp"
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_floor(py::module &m) {
    // 绑定 out-of-place 接口: output = floor(input)
    m.def("floor",
          &op::floor,
          py::arg("input"),
          R"doc(Computes the floor of each element of input.)doc");

    // 绑定 in-place 接口: floor_(output, input)
    m.def("floor_",
          &op::floor_,
          py::arg("output"),
          py::arg("input"),
          R"doc(In-place floor operation. Writes result into output tensor.)doc");
}

} // namespace infinicore::ops
