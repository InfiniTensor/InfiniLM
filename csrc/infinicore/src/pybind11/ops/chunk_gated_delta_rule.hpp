#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "infinicore/ops/chunk_gated_delta_rule.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_chunk_gated_delta_rule(py::module &m) {
    m.def("chunk_gated_delta_rule",
          &op::chunk_gated_delta_rule,
          py::arg("q"),
          py::arg("k"),
          py::arg("v"),
          py::arg("g"),
          py::arg("beta"),
          py::arg("initial_state"),
          py::arg("cu_seqlens") = std::nullopt,
          py::arg("initial_state_indices") = std::nullopt,
          py::arg("final_state_indices") = std::nullopt,
          py::arg("use_qk_l2norm") = false,
          py::arg("chunk_size") = 64,
          R"doc(Chunk gated delta rule. Returns out only.

Padded mode:
  q/k: [B, T, Hk, Dk], v/out: [B, T, Hv, Dv], g/beta: [B, T, Hv],
  initial_state: [B, Hv, Dv, Dk].

Continuous-batch mode:
  pass cu_seqlens [B + 1]; q/k: [1, total_tokens, Hk, Dk],
  v/out: [1, total_tokens, Hv, Dv], g/beta: [1, total_tokens, Hv].

Indexed pool mode:
  initial_state is [pool_size, Hv, Dv, Dk]. Provide both initial_state_indices
  and final_state_indices [B]; final state is written in-place to initial_state.
)doc");
}

} // namespace infinicore::ops
