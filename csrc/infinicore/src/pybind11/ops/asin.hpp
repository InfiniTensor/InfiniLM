#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/asin.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_asin(py::module &m) {
    m.def("asin",
          &op::asin,
          py::arg("input"),
          R"doc(Arcsin activation function.)doc");

    m.def("asin_",
          &op::asin_,
          py::arg("output"),
          py::arg("input"),
          R"doc(In-place arcsin activation function.)doc");
}

} // namespace infinicore::ops
