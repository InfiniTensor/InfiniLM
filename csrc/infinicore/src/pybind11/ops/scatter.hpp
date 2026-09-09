#pragma once

#include "infinicore/ops/scatter.hpp"
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_scatter(py::module &m) {
    // =========================================================================
    // 1. 绑定 out-of-place 接口: scatter
    // =========================================================================
    // 为了匹配测试脚本的行为（将所有 Tensor 作为位置参数传入，属性作为 kwargs 传入），
    // 我们将参数顺序调整为: input, index, src, dim, reduction
    // =========================================================================
    m.def(
        "scatter",
        [](const Tensor &input, const Tensor &index, const Tensor &src, int64_t dim, int64_t reduction) {
            // 调用底层 C++ 实现时，必须恢复正确的参数顺序: (input, dim, index, src, reduction)
            return op::scatter(input, dim, index, src, reduction);
        },
        py::arg("input"),
        py::arg("index"),
        py::arg("src"),
        py::arg("dim"), // 关键修改：将 dim 移到 Tensor 参数之后
        py::arg("reduction") = 0,
        R"doc(
    Scatter operator.
    Note: Parameter order in this binding is adapted for the test runner: (input, index, src, dim, reduction).
    )doc");

    // =========================================================================
    // 2. 绑定 in-place 接口: scatter_
    // =========================================================================
    // 参数顺序调整为: output, input, index, src, dim, reduction
    // =========================================================================
    m.def(
        "scatter_",
        [](Tensor &output, const Tensor &input, const Tensor &index, const Tensor &src, int64_t dim, int64_t reduction) {
            // 调用底层 C++ 实现
            op::scatter_(output, input, dim, index, src, reduction);
        },
        py::arg("output"),
        py::arg("input"),
        py::arg("index"),
        py::arg("src"),
        py::arg("dim"), // 关键修改：将 dim 移到 Tensor 参数之后
        py::arg("reduction") = 0,
        R"doc(
    In-place Scatter operator.
    Writes result into output.
    )doc");
}

} // namespace infinicore::ops
