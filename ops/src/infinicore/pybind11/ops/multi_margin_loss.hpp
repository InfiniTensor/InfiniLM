#pragma once

#include "infinicore/ops/multi_margin_loss.hpp"
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_multi_margin_loss(py::module &m) {
    m.def(
        "multi_margin_loss",
        [](const Tensor &input, const Tensor &target, py::object weight, int p, float margin, int reduction) {
            Tensor weight_tensor;
            if (!weight.is_none()) {
                weight_tensor = weight.cast<Tensor>();
            }
            return op::multi_margin_loss(input, target, weight_tensor, p, margin, reduction);
        },
        py::arg("input"),
        py::arg("target"),
        py::arg("weight") = py::none(), // Python 端看到默认值是 None
        py::arg("p") = 1,
        py::arg("margin") = 1.0f,
        py::arg("reduction") = 1,
        R"doc(Computes the Multi Margin Loss between input and target.
    
    Args:
        input (Tensor): Input tensor of shape (N, C).
        target (Tensor): Ground truth labels of shape (N,).
        weight (Tensor, optional): Manual rescaling weight given to each class. If given, has to be a Tensor of size C.
        p (int, optional): The norm degree for pairwise distance. p=1 or p=2. Default: 1.
        margin (float, optional): Margin value. Default: 1.0.
        reduction (int, optional): Specifies the reduction to apply to the output: 0=None, 1=Mean, 2=Sum. Default: 1.
    )doc");

    m.def(
        "multi_margin_loss_",
        [](Tensor &output, const Tensor &input, const Tensor &target, py::object weight, int p, float margin, int reduction) {
            Tensor weight_tensor;
            if (!weight.is_none()) {
                weight_tensor = weight.cast<Tensor>();
            }
            // 调用底层
            op::multi_margin_loss_(output, input, target, weight_tensor, p, margin, reduction);
        },
        py::arg("output"),
        py::arg("input"),
        py::arg("target"),
        py::arg("weight") = py::none(),
        py::arg("p") = 1,
        py::arg("margin") = 1.0f,
        py::arg("reduction") = 1,
        R"doc(Explicit output Multi Margin Loss operation. Writes the result into the output tensor.)doc");
}

} // namespace infinicore::ops
