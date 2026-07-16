#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/per_channel_quant_i8.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_per_channel_quant_i8(py::module &m) {
    m.def("per_channel_quant_i8_",
          &op::per_channel_quant_i8_,
          py::arg("x"),
          py::arg("x_packed"),
          py::arg("x_scale"),
          R"doc(Per-channel quantization of a tensor.)doc");
}

} // namespace infinicore::ops
