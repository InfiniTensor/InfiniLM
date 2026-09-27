#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/hinge_embedding_loss.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_hinge_embedding_loss(py::module &m) {
    m.def(
        "hinge_embedding_loss",
        &op::hinge_embedding_loss,
        py::arg("input"),
        py::arg("target"),
        py::arg("margin") = 1.0,
        py::arg("reduction") = 1,
        R"doc(Hinge embedding loss.)doc");
}

} // namespace infinicore::ops
