#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/fused_gated_delta_net_gating.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_fused_gated_delta_net_gating(py::module &m) {
    m.def(
        "fused_gated_delta_net_gating",
        [](const Tensor &A_log,
           const Tensor &a,
           const Tensor &b,
           const Tensor &dt_bias,
           float beta,
           float threshold) {
            auto result = op::fused_gated_delta_net_gating(A_log, a, b, dt_bias, beta, threshold);
            return py::make_tuple(result.first, result.second);
        },
        py::arg("A_log"),
        py::arg("a"),
        py::arg("b"),
        py::arg("dt_bias"),
        py::arg("beta") = 1.0f,
        py::arg("threshold") = 20.0f,
        R"doc(Fused GatedDeltaNet gating out-of-place.)doc");

    m.def("fused_gated_delta_net_gating_",
          &op::fused_gated_delta_net_gating_,
          py::arg("g"),
          py::arg("beta_output"),
          py::arg("A_log"),
          py::arg("a"),
          py::arg("b"),
          py::arg("dt_bias"),
          py::arg("beta") = 1.0f,
          py::arg("threshold") = 20.0f,
          R"doc(Fused GatedDeltaNet gating writing to provided outputs.)doc");
}

} // namespace infinicore::ops
