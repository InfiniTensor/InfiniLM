#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/baddbmm.hpp"

namespace py = pybind11;

namespace infinicore::ops {

Tensor py_baddbmm(Tensor input, Tensor batch1, Tensor batch2, float beta = 1.0f, float alpha = 1.0f) {
    return op::baddbmm(input, batch1, batch2, beta, alpha);
}

void py_baddbmm_(Tensor out, Tensor input, Tensor batch1, Tensor batch2, float beta = 1.0f, float alpha = 1.0f) {
    op::baddbmm_(out, input, batch1, batch2, beta, alpha);
}

inline void bind_baddbmm(py::module &m) {
    m.def("baddbmm",
          &py_baddbmm,
          py::arg("input"),
          py::arg("batch1"),
          py::arg("batch2"),
          py::arg("beta") = 1.0f,
          py::arg("alpha") = 1.0f,
          R"doc(Batched matrix-matrix product with addition.
Args:
    input: Input tensor
    batch1: First batch of matrices
    batch2: Second batch of matrices
    beta: Scaling factor for input tensor
    alpha: Scaling factor for the product of batch1 and batch2
Returns:
    Output tensor after baddbmm operation
)doc");
    m.def("baddbmm_",
          &py_baddbmm_,
          py::arg("out"),
          py::arg("input"),
          py::arg("batch1"),
          py::arg("batch2"),
          py::arg("beta") = 1.0f,
          py::arg("alpha") = 1.0f,
          R"doc(In-place batched matrix-matrix product with addition.
Args:
    out: Output tensor
    input: Input tensor
    batch1: First batch of matrices
    batch2: Second batch of matrices
    beta: Scaling factor for input tensor
    alpha: Scaling factor for the product of batch1 and batch2
)doc");
}

} // namespace infinicore::ops
