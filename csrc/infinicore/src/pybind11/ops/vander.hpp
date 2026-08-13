#pragma once

#include "infinicore/ops/vander.hpp"
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_vander(py::module &m) {
    m.def(
        "vander",
        [](const Tensor &input, int64_t N, bool increasing) {
            return op::vander(input, N, increasing);
        },
        py::arg("input"),
        py::arg("N") = 0,
        py::arg("increasing") = false,
        R"doc(Generates a Vandermonde matrix.

    Args:
        input (Tensor): 1-D input tensor.
        N (int, optional): Number of columns in the output. If 0, defaults to input size (square matrix). Default: 0.
        increasing (bool, optional): Order of the powers. If True, powers increase (x^0, x^1...). Default: False.
    )doc");

    // -------------------------------------------------------------------------
    // 2. 绑定 in-place 接口 (vander_)
    // -------------------------------------------------------------------------
    m.def(
        "vander_",
        [](Tensor &output, const Tensor &input, int64_t N, bool increasing) {
            op::vander_(output, input, N, increasing);
        },
        py::arg("output"),
        py::arg("input"),
        py::arg("N") = 0,
        py::arg("increasing") = false,
        R"doc(Explicit output Vander operation. Writes the result into the output tensor.)doc");
}

} // namespace infinicore::ops
