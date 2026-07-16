#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/kron.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_kron(py::module &m) {
    m.def(
        "kron",
        &op::kron,
        py::arg("a"),
        py::arg("b"),
        R"doc(Kronecker product.)doc");
}

} // namespace infinicore::ops
