#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "infinicore/ops/block_diag.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_block_diag(py::module &m) {
    m.def(
        "block_diag",
        &op::block_diag,
        py::arg("tensors"),
        R"doc(Construct a block diagonal matrix from a list of 2D tensors.)doc");
}

} // namespace infinicore::ops
