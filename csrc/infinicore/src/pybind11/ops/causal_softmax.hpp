#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/causal_softmax.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_causal_softmax(py::module &m) {
    m.def("causal_softmax",
          &op::causal_softmax,
          py::arg("input"),
          R"doc(Causal softmax activation function.)doc");

    m.def("causal_softmax_",
          &op::causal_softmax_,
          py::arg("output"),
          py::arg("input"),
          R"doc(In-place causal softmax activation function.)doc");
}

} // namespace infinicore::ops
