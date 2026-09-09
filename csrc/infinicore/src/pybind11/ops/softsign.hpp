#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/softsign.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_softsign(py::module &m) {
    m.def("softsign",
          &op::softsign,
          py::arg("x"),
          R"doc(Softsign activation function.)doc");

    m.def("softsign_",
          &op::softsign_,
          py::arg("y"),
          py::arg("x"),
          R"doc(In-place softsign activation.)doc");
}

} // namespace infinicore::ops
