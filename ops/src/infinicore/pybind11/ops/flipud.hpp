#pragma once

#include "infinicore/ops/flipud.hpp"
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_flipud(py::module &m) {
    // 1. 绑定 out-of-place 接口: output = flipud(input)
    m.def("flipud",
          &op::flipud,
          py::arg("input"),
          R"doc(Flip array in the up/down direction.

    Flips the entries in axis 0 (preserving the shape).

    Args:
        input (Tensor): The input tensor.
    )doc");

    // 2. 绑定 explicit output 接口: flipud_(output, input)
    m.def("flipud_",
          &op::flipud_,
          py::arg("output"),
          py::arg("input"),
          R"doc(Explicit output FlipUD operation. Writes the result into the output tensor.)doc");
}

} // namespace infinicore::ops
