#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/cdist.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_cdist(py::module &m) {
    m.def("cdist",
          &op::cdist,
          py::arg("x1"),
          py::arg("x2"),
          py::arg("p") = 2.0,
          R"doc(Computes batched pairwise distance between vectors in x1 and x2 using p-norm.

Args:
    x1: First set of vectors, shape (M, D)
    x2: Second set of vectors, shape (N, D)
    p: The p-norm to apply (default: 2.0)

Returns:
    A matrix containing pairwise distances, shape (M, N)
)doc");

    m.def("cdist_",
          &op::cdist_,
          py::arg("out"),
          py::arg("x1"),
          py::arg("x2"),
          py::arg("p") = 2.0,
          R"doc(In-place version of cdist. Stores the results in the 'out' tensor.

Args:
    out: The destination tensor, shape (M, N)
    x1: First set of vectors, shape (M, D)
    x2: Second set of vectors, shape (N, D)
    p: The p-norm to apply (default: 2.0)
)doc");
}

} // namespace infinicore::ops
