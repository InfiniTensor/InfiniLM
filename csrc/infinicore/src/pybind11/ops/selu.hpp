#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/selu.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_selu(py::module &m) {
    m.def(
        "selu",
        &op::selu,
        py::arg("input"),
        R"doc(SELU activation function.)doc");

    m.def(
        "selu_",
        &op::selu_,
        py::arg("output"),
        py::arg("input"),
        R"doc(In-place SELU activation function.)doc");
}

} // namespace infinicore::ops
