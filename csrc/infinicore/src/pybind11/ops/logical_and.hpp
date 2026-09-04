#pragma once
#include <pybind11/pybind11.h>

#include "infinicore/ops/logical_and.hpp"

namespace py = pybind11;
namespace infinicore::ops {

inline void bind_logical_and(py::module &m) {
    // 绑定常规函数: logical_and(input, other) -> Tensor
    m.def("logical_and",
          &op::logical_and,
          py::arg("input"),
          py::arg("other"),
          R"doc(Computes the element-wise logical AND of the given input tensors.)doc");

    // 绑定底层输出指定函数: logical_and_(output, input, other)
    // 对应 Python 调用: _infinicore.logical_and_(out, input, other)
    m.def("logical_and_",
          &op::logical_and_,
          py::arg("output"),
          py::arg("input"),
          py::arg("other"),
          R"doc(Explicit output logical AND computation.)doc");
}

} // namespace infinicore::ops
