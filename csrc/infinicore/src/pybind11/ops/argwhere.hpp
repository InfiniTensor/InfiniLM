#pragma once

#include "infinicore/ops/argwhere.hpp"
#include <pybind11/pybind11.h>
namespace py = pybind11;
namespace infinicore::ops {
inline void bind_argwhere(py::module &m) {
    m.def("argwhere",
          &op::argwhere,
          py::arg("x"),
          R"doc(Argwhere.)doc");
}
} // namespace infinicore::ops
