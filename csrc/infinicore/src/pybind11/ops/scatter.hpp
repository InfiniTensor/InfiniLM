#pragma once

#include "infinicore/ops/scatter.hpp"
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_scatter(py::module &m) {
    // =========================================================================
    // =========================================================================
    // =========================================================================
    m.def(
        "scatter",
        [](const Tensor &input, const Tensor &index, const Tensor &src, int64_t dim, int64_t reduction) {
            return op::scatter(input, dim, index, src, reduction);
        },
        py::arg("input"),
        py::arg("index"),
        py::arg("src"),
        py::arg("dim"),
        py::arg("reduction") = 0,
        R"doc(
    Scatter operator.
    Note: Parameter order in this binding is adapted for the test runner: (input, index, src, dim, reduction).
    )doc");

    // =========================================================================
    // =========================================================================
    // =========================================================================
    m.def(
        "scatter_",
        [](Tensor &output, const Tensor &input, const Tensor &index, const Tensor &src, int64_t dim, int64_t reduction) {
            op::scatter_(output, input, dim, index, src, reduction);
        },
        py::arg("output"),
        py::arg("input"),
        py::arg("index"),
        py::arg("src"),
        py::arg("dim"),
        py::arg("reduction") = 0,
        R"doc(
    In-place Scatter operator.
    Writes result into output.
    )doc");
}

} // namespace infinicore::ops
