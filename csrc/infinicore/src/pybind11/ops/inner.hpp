#pragma once

#include "infinicore/ops/inner.hpp"
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_inner(py::module &m) {

    m.def("inner",
          &op::inner,
          py::arg("input"),
          py::arg("other"),
          R"doc(opertor: torch.inner, out-of-place mode)doc");

    m.def("inner_",
          &op::inner_,
          py::arg("out"),
          py::arg("input"),
          py::arg("other"),
          R"doc(opertor: torch.inner, in-place mode)doc");
}

} // namespace infinicore::ops
