#pragma once

#include "infinicore/ops/triplet_margin_loss.hpp"
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_triplet_margin_loss(py::module &m) {
    // 1. 绑定 functional 接口: output = triplet_margin_loss(anchor, positive, negative, ...)
    m.def("triplet_margin_loss",
          &op::triplet_margin_loss,
          py::arg("anchor"),
          py::arg("positive"),
          py::arg("negative"),
          py::arg("margin") = 1.0f,
          py::arg("p") = 2,
          py::arg("eps") = 1e-6f,
          py::arg("swap") = false,
          py::arg("reduction") = 1,
          R"doc(Computes the triplet margin loss.

    Args:
        anchor (Tensor): The anchor tensor.
        positive (Tensor): The positive tensor.
        negative (Tensor): The negative tensor.
        margin (float): Default: 1.0.
        p (int): The norm degree for pairwise distance. Default: 2.
        eps (float): Small constant for numerical stability. Default: 1e-6.
        swap (bool): The distance swap is described in the paper Learning shallow convolutional feature descriptors with triplet losses. Default: False.
        reduction (int): Specifies the reduction to apply to the output: 0 (none), 1 (mean), 2 (sum). Default: 1.
    )doc");

    // 2. 绑定 explicit output 接口: triplet_margin_loss_(output, anchor, positive, negative, ...)
    m.def("triplet_margin_loss_",
          &op::triplet_margin_loss_,
          py::arg("output"),
          py::arg("anchor"),
          py::arg("positive"),
          py::arg("negative"),
          py::arg("margin") = 1.0f,
          py::arg("p") = 2,
          py::arg("eps") = 1e-6f,
          py::arg("swap") = false,
          py::arg("reduction") = 1,
          R"doc(Explicit output TripletMarginLoss operation. Writes the result into the output tensor.)doc");
}

} // namespace infinicore::ops
