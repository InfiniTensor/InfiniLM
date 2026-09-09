#pragma once

#include "infinicore/ops/triplet_margin_with_distance_loss.hpp"
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_triplet_margin_with_distance_loss(py::module &m) {
    m.def("triplet_margin_with_distance_loss",
          &op::triplet_margin_with_distance_loss,
          py::arg("anchor"),
          py::arg("positive"),
          py::arg("negative"),
          py::arg("margin") = 1.0,
          py::arg("swap") = false,
          py::arg("reduction") = 1,
          R"doc(Computes the triplet margin loss with distance.

    Args:
        anchor (Tensor): The anchor input tensor.
        positive (Tensor): The positive input tensor.
        negative (Tensor): The negative input tensor.
        margin (float, optional): Default: 1.0.
        swap (bool, optional): The distance swap is described in the paper Learning shallow convolutional feature descriptors with triplet losses. Default: False.
        reduction (int, optional): Specifies the reduction to apply to the output: 0 (None), 1 (Mean), 2 (Sum). Default: 1.
    )doc");
    m.def("triplet_margin_with_distance_loss_",
          &op::triplet_margin_with_distance_loss_,
          py::arg("output"),
          py::arg("anchor"),
          py::arg("positive"),
          py::arg("negative"),
          py::arg("margin"),
          py::arg("swap"),
          py::arg("reduction"),
          R"doc(Explicit output TripletMarginWithDistanceLoss operation. Writes results into output tensor.)doc");
}

} // namespace infinicore::ops
