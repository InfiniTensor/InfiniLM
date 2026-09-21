#pragma once

#include "infinicore/ops/addbmm.hpp"
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_addbmm(py::module &m) {
    // -----------------------------------------------------------
    // -----------------------------------------------------------
    m.def("addbmm",
          &op::addbmm,
          py::arg("input"),
          py::arg("batch1"),
          py::arg("batch2"),
          py::arg("beta") = 1.0f,
          py::arg("alpha") = 1.0f,
          R"doc(Performs a batch matrix-matrix product of matrices stored in batch1 and batch2,
with a reduced add step (summing over all matrices in the batch).

.. math::
    \text{out} = \beta \times \text{input} + \alpha \times \sum_{i=0}^{b-1} (\text{batch1}_i \mathbin{@} \text{batch2}_i)

Args:
    input (Tensor): Matrix to be added. Shape (n, p).
    batch1 (Tensor): The first batch of matrices to be multiplied. Shape (b, n, m).
    batch2 (Tensor): The second batch of matrices to be multiplied. Shape (b, m, p).
    beta (float, optional): Multiplier for input. Default: 1.0.
    alpha (float, optional): Multiplier for batch1 @ batch2. Default: 1.0.

Returns:
    Tensor: Output tensor of shape (n, p).
)doc");

    // -----------------------------------------------------------
    // -----------------------------------------------------------
    m.def("addbmm_",
          &op::addbmm_,
          py::arg("out"),
          py::arg("input"),
          py::arg("batch1"),
          py::arg("batch2"),
          py::arg("beta") = 1.0f,
          py::arg("alpha") = 1.0f,
          "In-place version of addbmm");
}

} // namespace infinicore::ops
