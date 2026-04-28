#pragma once

#include "infinicore/ops/binary_cross_entropy_with_logits.hpp"
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_binary_cross_entropy_with_logits(py::module &m) {
    // 1. 绑定 out-of-place 接口: out = binary_cross_entropy_with_logits(...)
    m.def(
        "binary_cross_entropy_with_logits",
        [](Tensor logits,
           Tensor target,
           py::object weight,
           py::object pos_weight,
           std::string reduction) {
            Tensor w = weight.is_none() ? Tensor() : weight.cast<Tensor>();
            Tensor pw = pos_weight.is_none() ? Tensor() : pos_weight.cast<Tensor>();

            return op::binary_cross_entropy_with_logits(
                logits, target, w, pw, reduction);
        },
        py::arg("input"),
        py::arg("target"),
        py::arg("weight") = py::none(),
        py::arg("pos_weight") = py::none(),
        py::arg("reduction") = "mean",
        R"doc(Measures Binary Cross Entropy between target and output logits.

    
Args:
    input: Tensor of arbitrary shape as unnormalized scores (logits).
    target: Tensor of the same shape as input with values between 0 and 1.
    weight: Optional rescaling weight for each loss component.
    pos_weight: Optional weight for positive examples (must be broadcastable).
    reduction: Specfies the reduction to apply: 'none' | 'mean' | 'sum'.

Returns:
    A tensor representing the loss.
)doc");

    // 2. 绑定指定输出接口: binary_cross_entropy_with_logits_(out, ...)
    m.def(
        "binary_cross_entropy_with_logits_",
        [](Tensor output,
           Tensor logits,
           Tensor target,
           py::object weight,
           py::object pos_weight,
           std::string reduction) {
            Tensor w = weight.is_none() ? Tensor() : weight.cast<Tensor>();
            Tensor pw = pos_weight.is_none() ? Tensor() : pos_weight.cast<Tensor>();

            return op::binary_cross_entropy_with_logits_(
                output, logits, target, w, pw, reduction);
        },
        py::arg("out"),
        py::arg("input"),
        py::arg("target"),
        py::arg("weight") = py::none(),
        py::arg("pos_weight") = py::none(),
        py::arg("reduction") = "mean",
        R"doc(Specified output version of binary_cross_entropy_with_logits.

Args:
    out: The destination tensor to store the loss.
    input: Logits tensor.
    target: Target tensor.
    weight: Optional sample weight.
    pos_weight: Optional positive class weight.
    reduction: Specfies the reduction to apply.
)doc");
}

} // namespace infinicore::ops
