#pragma once

#include "infinicore/ops/unfold.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_unfold(py::module &m) {
    // -------------------------------------------------------------------------
    // 1. 绑定函数式接口 (unfold)
    // -------------------------------------------------------------------------
    m.def(
        "unfold",
        [](const Tensor &input,
           std::vector<int64_t> kernel_sizes,
           std::vector<int64_t> dilations,
           std::vector<int64_t> paddings,
           std::vector<int64_t> strides) {
            return op::unfold(input, kernel_sizes, dilations, paddings, strides);
        },
        py::arg("input"),
        py::arg("kernel_sizes"),
        py::arg("dilations"),
        py::arg("paddings"),
        py::arg("strides"),
        R"doc(Extracts sliding local blocks from a batched input tensor.

    Args:
        input (Tensor): The input tensor.
        kernel_sizes (List[int]): The size of the sliding blocks.
        dilations (List[int]): The parameter that controls the stride of elements within the neighborhood.
        paddings (List[int]): Implicit zero padding to be added on both sides of input.
        strides (List[int]): The stride of the sliding blocks.
    )doc");

    // -------------------------------------------------------------------------
    // 2. 绑定 in-place 接口 (unfold_)
    // -------------------------------------------------------------------------
    m.def(
        "unfold_",
        [](Tensor &output,
           const Tensor &input,
           std::vector<int64_t> kernel_sizes,
           std::vector<int64_t> dilations,
           std::vector<int64_t> paddings,
           std::vector<int64_t> strides) {
            op::unfold_(output, input, kernel_sizes, dilations, paddings, strides);
        },
        py::arg("output"),
        py::arg("input"),
        py::arg("kernel_sizes"),
        py::arg("dilations"),
        py::arg("paddings"),
        py::arg("strides"),
        R"doc(Explicit output Unfold operation. Writes the result into the output tensor.)doc");
}

} // namespace infinicore::ops
