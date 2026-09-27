#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/topksoftmax.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_topksoftmax(py::module &m) {
    m.def("topksoftmax",
          &op::topksoftmax,
          py::arg("values"),
          py::arg("indices"),
          py::arg("x"),
          py::arg("topk"),
          py::arg("norm") = 0,
          R"doc(In-place Top-k Softmax.

Writes results to pre-allocated values and indices tensors.

Args:
    values: Output tensor for softmax weights [N, topk]
    indices: Output tensor for selected indices [N, topk], int32
    x: Input tensor [N, width], router logits
    topk: Number of top values to select
    norm: Whether to re-normalize top-k probabilities (1=yes, 0=no), default 0
)doc");
}

} // namespace infinicore::ops
