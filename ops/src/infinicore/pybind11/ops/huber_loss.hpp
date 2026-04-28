#pragma once

#include "infinicore/ops/huber_loss.hpp"
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_huber_loss(py::module &m) {
    m.def(
        "huber_loss",
        [](const Tensor &input, const Tensor &target, float delta, int reduction) {
            return op::huber_loss(input, target, delta, reduction);
        },
        py::arg("input"),
        py::arg("target"),
        py::arg("delta") = 1.0f,
        py::arg("reduction") = 1,
        R"doc(Computes the Huber Loss between input and target.
    
    Args:
        input (Tensor): Input tensor of arbitrary shape.
        target (Tensor): Ground truth labels, same shape as input.
        delta (float, optional): The threshold at which to change between delta-scaled L1 and L2 loss. Default: 1.0.
        reduction (int, optional): Specifies the reduction to apply to the output: 0=None, 1=Mean, 2=Sum. Default: 1.
    )doc");

    // -------------------------------------------------------------------------
    // 2. 绑定 in-place 接口 (huber_loss_)
    // -------------------------------------------------------------------------
    m.def(
        "huber_loss_",
        [](Tensor &output, const Tensor &input, const Tensor &target, float delta, int reduction) {
            // 调用底层
            op::huber_loss_(output, input, target, delta, reduction);
        },
        py::arg("output"),
        py::arg("input"),
        py::arg("target"),
        py::arg("delta") = 1.0f,
        py::arg("reduction") = 1,
        R"doc(Explicit output Huber Loss operation. Writes the result into the output tensor.)doc");
}

} // namespace infinicore::ops
