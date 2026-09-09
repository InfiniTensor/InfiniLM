#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/flash_attention.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_flash_attention(py::module &m) {
    m.def("flash_attention",
          &op::flash_attention,
          py::arg("q"),
          py::arg("k"),
          py::arg("v"),
          py::arg("total_kv_len"),
          py::arg("scale"),
          py::arg("is_causal"));
}

} // namespace infinicore::ops
