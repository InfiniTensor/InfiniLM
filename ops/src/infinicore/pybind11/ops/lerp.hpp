#pragma once

#include "infinicore/ops/lerp.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_lerp(py::module &m) {
    // 定义函数指针别名，用于区分重载
    using LerpTensorFunc = Tensor (*)(Tensor, Tensor, Tensor);
    using LerpScalarFunc = Tensor (*)(Tensor, Tensor, float);
    using LerpTensorInplaceFunc = void (*)(Tensor, Tensor, Tensor, Tensor);
    using LerpScalarInplaceFunc = void (*)(Tensor, Tensor, Tensor, float);

    // ========================================================================
    // 1. 绑定 functional 接口
    // ========================================================================

    // 重载 1: weight 为 Tensor
    m.def("lerp",
          static_cast<LerpTensorFunc>(&op::lerp),
          py::arg("start"),
          py::arg("end"),
          py::arg("weight"),
          R"doc(Does a linear interpolation of two tensors start and end based on a tensor weight.
          
          output = start + weight * (end - start)
          )doc");

    // 重载 2: weight 为 float
    m.def("lerp",
          static_cast<LerpScalarFunc>(&op::lerp),
          py::arg("start"),
          py::arg("end"),
          py::arg("weight"),
          R"doc(Does a linear interpolation of two tensors start and end based on a scalar weight.)doc");

    // ========================================================================
    // 2. 绑定 explicit output 接口 (In-place)
    // ========================================================================

    // 重载 1: weight 为 Tensor
    m.def("lerp_",
          static_cast<LerpTensorInplaceFunc>(&op::lerp_),
          py::arg("output"),
          py::arg("start"),
          py::arg("end"),
          py::arg("weight"),
          R"doc(Explicit output Lerp operation with tensor weight. Writes the result into the output tensor.)doc");

    // 重载 2: weight 为 float
    m.def("lerp_",
          static_cast<LerpScalarInplaceFunc>(&op::lerp_),
          py::arg("output"),
          py::arg("start"),
          py::arg("end"),
          py::arg("weight"),
          R"doc(Explicit output Lerp operation with scalar weight. Writes the result into the output tensor.)doc");
}

} // namespace infinicore::ops
