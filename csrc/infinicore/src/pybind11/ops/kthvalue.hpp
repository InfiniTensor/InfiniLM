#pragma once

#include "infinicore/ops/kthvalue.hpp"
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_kthvalue(py::module &m) {
    // 1. 绑定 functional 接口: (values, indices) = kthvalue(input, k, dim, keepdim)
    m.def("kthvalue",
          &op::kthvalue,
          py::arg("input"),
          py::arg("k"),
          py::arg("dim") = -1,
          py::arg("keepdim") = false,
          R"doc(Returns the k-th smallest element of each row of the input tensor in the given dimension.

    Args:
        input (Tensor): The input tensor.
        k (int): The k value.
        dim (int): The dimension to find the k-th value along.
        keepdim (bool): Whether to keep the output dimension.
    )doc");

    // 2. 绑定 explicit output 接口: kthvalue_(values, indices, input, k, dim, keepdim)
    m.def("kthvalue_",
          &op::kthvalue_,
          py::arg("values"),
          py::arg("indices"),
          py::arg("input"),
          py::arg("k"),
          py::arg("dim") = -1,
          py::arg("keepdim") = false,
          R"doc(Explicit output Kthvalue operation. Writes results into values and indices tensors.)doc");
}

} // namespace infinicore::ops
