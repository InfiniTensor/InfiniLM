#pragma once

#include "infinicore/ops/cat.hpp"
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_cat(py::module &m) {

    m.def("cat",
          &op::cat,
          py::arg("tensors"),
          py::arg("dim") = 0,
          R"doc(opertor: torch.cat, out-of-place mode)doc");

    m.def("cat_",
          &op::cat_,
          py::arg("out"),
          py::arg("tensors"),
          py::arg("dim") = 0,
          R"doc(opertor: torch.cat, in-place mode)doc");
}

} // namespace infinicore::ops
