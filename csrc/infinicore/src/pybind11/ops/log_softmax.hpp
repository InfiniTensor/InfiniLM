#pragma once

#include "infinicore/ops/log_softmax.hpp"
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_log_softmax(py::module &m) {
    // 1. 绑定 functional 接口: output = log_softmax(input, dim)
    m.def("log_softmax",
          &op::log_softmax,
          py::arg("input"),
          py::arg("dim"),
          R"doc(Applies a softmax followed by a logarithm.

    Args:
        input (Tensor): The input tensor.
        dim (int): A dimension along which log_softmax will be computed.
    )doc");

    // 2. 绑定 explicit output 接口: log_softmax_(output, input, dim)
    m.def("log_softmax_",
          &op::log_softmax_,
          py::arg("output"),
          py::arg("input"),
          py::arg("dim"),
          R"doc(Explicit output LogSoftmax operation. Writes results into output tensor.)doc");
}

} // namespace infinicore::ops
