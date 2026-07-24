#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/recurrent_gated_delta_rule.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_recurrent_gated_delta_rule(py::module &m) {
    m.def("recurrent_gated_delta_rule",
          &op::recurrent_gated_delta_rule,
          py::arg("q"),
          py::arg("k"),
          py::arg("v"),
          py::arg("g"),
          py::arg("beta"),
          py::arg("initial_state"),
          py::arg("use_qk_l2norm") = false,
          R"doc(Recurrent gated delta rule. Returns out only.)doc");

    m.def("recurrent_gated_delta_rule_indexed",
          &op::recurrent_gated_delta_rule_indexed,
          py::arg("q"),
          py::arg("k"),
          py::arg("v"),
          py::arg("g"),
          py::arg("beta"),
          py::arg("initial_state"),
          py::arg("initial_state_indices"),
          py::arg("final_state_indices"),
          py::arg("use_qk_l2norm") = false,
          R"doc(Recurrent gated delta rule with indexed in-place state pool. Returns out only.)doc");
}

} // namespace infinicore::ops
