#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/mamba_selective_scan.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_mamba_selective_scan(py::module &m) {
    m.def("mamba_selective_scan",
          &op::mamba_selective_scan,
          py::arg("x"),
          py::arg("dt"),
          py::arg("b"),
          py::arg("c"),
          py::arg("a_log"),
          py::arg("d"),
          py::arg("gate"),
          py::arg("dt_bias"),
          py::arg("state"),
          R"doc(Mamba selective scan. Returns out and updates state in-place.

Shapes:
  x, dt, gate, out: [batch, seq_len, intermediate]
  b, c: [batch, seq_len, state_size]
  a_log: [intermediate, state_size]
  d, dt_bias: [intermediate]
  state: [batch, intermediate, state_size], float32
)doc");
}

} // namespace infinicore::ops
