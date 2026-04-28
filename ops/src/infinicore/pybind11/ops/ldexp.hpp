#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/ldexp.hpp"

namespace py = pybind11;

#pragma once

#include "infinicore/ops/ldexp.hpp"
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_ldexp(py::module &m) {
    // 1. 绑定 functional 接口: output = ldexp(input, other)
    m.def("ldexp",
          &op::ldexp,
          py::arg("input"),
          py::arg("other"),
          R"doc(Multiplies input by 2 raised to the power of other.

    Args:
        input (Tensor): The input tensor (mantissa).
        other (Tensor): The exponent tensor.
    )doc");

    // 2. 绑定 explicit output 接口: ldexp_(output, input, other)
    m.def("ldexp_",
          &op::ldexp_,
          py::arg("output"),
          py::arg("input"),
          py::arg("other"),
          R"doc(Explicit output Ldexp operation. Writes result into output tensor.)doc");
}

} // namespace infinicore::ops
