#pragma once

#include <pybind11/pybind11.h>

#include "ops/add.hpp"
#include "ops/cat.hpp"
#include "ops/matmul.hpp"

namespace py = pybind11;

namespace infinilm::runtime::ops {

inline void bind(py::module &m) {
    bind_add(m);
    bind_cat(m);
    bind_matmul(m);
}

} // namespace infinilm::runtime::ops
