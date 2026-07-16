#pragma once

#include "infinicore/ops/hypot.hpp"
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_hypot(py::module &m) {
    // 绑定 out-of-place 接口: output = hypot(input, other)
    m.def("hypot",
          &op::hypot,
          py::arg("input"),
          py::arg("other"),
          R"doc(Computes the hypotenuse of input and other arguments, i.e. sqrt(input^2 + other^2).)doc");

    // 绑定 in-place / 指定输出接口: hypot_(output, input, other)
    m.def("hypot_",
          &op::hypot_,
          py::arg("output"),
          py::arg("input"),
          py::arg("other"),
          R"doc(In-place hypot operation. Writes result into output tensor.)doc");
}

} // namespace infinicore::ops
