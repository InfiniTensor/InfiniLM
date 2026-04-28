#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/addcmul.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_addcmul(py::module &m) {
    // 绑定 out-of-place 接口: out = addcmul(input, t1, t2, value)
    m.def("addcmul",
          &op::addcmul,
          py::arg("input"),
          py::arg("tensor1"),
          py::arg("tensor2"),
          py::arg("value") = 1.0f,
          R"doc(Performs the element-wise multiplication of tensor1 by tensor2, 
multiplies the result by value and adds it to input.

Args:
    input: Tensor to be added
    tensor1: First tensor for multiplication
    tensor2: Second tensor for multiplication
    value: Scalar multiplier for tensor1 * tensor2 (default: 1.0)

Returns:
    The output tensor
)doc");

    // 绑定 in-place / specified output 接口: addcmul_(out, input, t1, t2, value)
    m.def("addcmul_",
          &op::addcmul_,
          py::arg("out"),
          py::arg("input"),
          py::arg("tensor1"),
          py::arg("tensor2"),
          py::arg("value") = 1.0f,
          R"doc(In-place version of addcmul.

Args:
    out: The destination tensor to store the result
    input: Tensor to be added
    tensor1: First tensor for multiplication
    tensor2: Second tensor for multiplication
    value: Scalar multiplier for tensor1 * tensor2 (default: 1.0)
)doc");
}

} // namespace infinicore::ops
