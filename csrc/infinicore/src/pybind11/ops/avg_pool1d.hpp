#pragma once

#include <optional>
#include <pybind11/pybind11.h>

#include "infinicore/ops/avg_pool1d.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_avg_pool1d(py::module &m) {
    m.def(
        "avg_pool1d",
        [](::infinicore::Tensor input, size_t kernel_size, std::optional<size_t> stride, size_t padding) {
            return op::avg_pool1d(input, kernel_size, stride.value_or(0), padding);
        },
        py::arg("input"),
        py::arg("kernel_size"),
        py::arg("stride") = py::none(),
        py::arg("padding") = 0,
        R"doc(AvgPool1d out-of-place.)doc");

    m.def(
        "avg_pool1d_",
        [](::infinicore::Tensor output, ::infinicore::Tensor input, size_t kernel_size, std::optional<size_t> stride, size_t padding) {
            op::avg_pool1d_(output, input, kernel_size, stride.value_or(0), padding);
        },
        py::arg("output"),
        py::arg("input"),
        py::arg("kernel_size"),
        py::arg("stride") = py::none(),
        py::arg("padding") = 0,
        R"doc(AvgPool1d in-place variant writing to provided output tensor.)doc");
}

} // namespace infinicore::ops
