#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/digamma.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_digamma(py::module &m) {
    m.def("digamma",
          &op::digamma,
          py::arg("x"),
          R"doc(Digamma function.)doc");

    m.def("digamma_",
          &op::digamma_,
          py::arg("y"),
          py::arg("x"),
          R"doc(Out variant of digamma.)doc");
}

} // namespace infinicore::ops
