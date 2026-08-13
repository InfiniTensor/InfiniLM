#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/relu6.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_relu6(py::module &m) {
    m.def("relu6",
          &op::relu6,
          py::arg("input"),
          R"doc(ReLU6 activation.)doc");

    m.def("relu6_",
          &op::relu6_,
          py::arg("out"),
          py::arg("input"),
          R"doc(In-place ReLU6 activation (writes to out).)doc");
}

} // namespace infinicore::ops
