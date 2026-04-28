#pragma once

#include "infinicore/ops/smooth_l1_loss.hpp"
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_smooth_l1_loss(py::module &m) {
    // 1. 绑定 out-of-place 接口: output = smooth_l1_loss(input, target, beta, reduction)
    m.def("smooth_l1_loss",
          &op::smooth_l1_loss,
          py::arg("input"),
          py::arg("target"),
          py::arg("beta") = 1.0f,
          py::arg("reduction") = 1,
          R"doc(Computes the Smooth L1 Loss between input and target.
    
    Args:
        input (Tensor): Predicted values.
        target (Tensor): Ground truth values.
        beta (float, optional): The threshold at which to change between L1 and L2 loss. Default: 1.0.
        reduction (int, optional): Specifies the reduction to apply to the output: 0=None, 1=Mean, 2=Sum. Default: 1.
    )doc");
    m.def("smooth_l1_loss_",
          &op::smooth_l1_loss_,
          py::arg("output"),
          py::arg("input"),
          py::arg("target"),
          py::arg("beta") = 1.0f,
          py::arg("reduction") = 1,
          R"doc(Explicit output Smooth L1 Loss operation. Writes the result into the output tensor.)doc");
}

} // namespace infinicore::ops
