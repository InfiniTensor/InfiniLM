#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "infinicore/ops/causal_conv1d.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_causal_conv1d(py::module &m) {
    m.def("causal_conv1d",
          &op::causal_conv1d,
          py::arg("qkv"),
          py::arg("conv_state"),
          py::arg("weight"),
          py::arg("bias") = std::nullopt,
          py::arg("cu_seqlens") = std::nullopt,
          py::arg("initial_state_indices") = std::nullopt,
          py::arg("final_state_indices") = std::nullopt,
          R"doc(Causal depthwise Conv1d. Returns out only.

Padded mode:
  qkv/out: [B, T, C], conv_state: [B, C, state_len].

Continuous-batch mode:
  pass cu_seqlens [num_requests + 1]; qkv/out: [1, total_tokens, C].

Indexed pool mode:
  conv_state is [pool_size, C, state_len]. Provide initial_state_indices [num_requests]
  to read states. Provide final_state_indices [num_requests] to write final states
  in-place to conv_state. The current backend supports K == 4, where
  weight is [C, 1, K] and conv_state is [*, C, K - 1].
)doc");
}

} // namespace infinicore::ops
