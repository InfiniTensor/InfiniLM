#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/moe_topk_softmax.hpp"

namespace py = pybind11;

namespace infinicore::ops {

std::tuple<Tensor, Tensor> py_moe_topk_softmax(Tensor gating_output,
                                               size_t topk,
                                               bool renormalize,
                                               float moe_softcapping,
                                               py::object correction_bias) {
    Tensor bias;
    if (!correction_bias.is_none()) {
        bias = correction_bias.cast<Tensor>();
    }
    return op::moe_topk_softmax(gating_output, topk, renormalize, moe_softcapping, bias);
}

void py_moe_topk_softmax_(Tensor topk_weights,
                          Tensor topk_indices,
                          Tensor gating_output,
                          py::object correction_bias,
                          bool renormalize,
                          float moe_softcapping) {
    Tensor bias;
    if (!correction_bias.is_none()) {
        bias = correction_bias.cast<Tensor>();
    }
    op::moe_topk_softmax_(topk_weights, topk_indices, gating_output, bias, renormalize, moe_softcapping);
}

inline void bind_moe_topk_softmax(py::module &m) {
    m.def("moe_topk_softmax",
          &py_moe_topk_softmax,
          py::arg("gating_output"),
          py::arg("topk"),
          py::arg("renormalize") = false,
          py::arg("moe_softcapping") = 0.0f,
          py::arg("correction_bias") = py::none(),
          R"doc(MoE top-k softmax.)doc");

    m.def("moe_topk_softmax_",
          &py_moe_topk_softmax_,
          py::arg("topk_weights"),
          py::arg("topk_indices"),
          py::arg("gating_output"),
          py::arg("correction_bias") = py::none(),
          py::arg("renormalize") = false,
          py::arg("moe_softcapping") = 0.0f,
          R"doc(In-place MoE top-k softmax.)doc");
}

} // namespace infinicore::ops
