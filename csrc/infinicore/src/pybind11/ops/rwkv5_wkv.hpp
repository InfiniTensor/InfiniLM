#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/rwkv5_wkv.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_rwkv5_wkv(py::module &m) {
    m.def("rwkv5_wkv",
          &op::rwkv5_wkv,
          py::arg("receptance"),
          py::arg("key"),
          py::arg("value"),
          py::arg("time_decay"),
          py::arg("time_faaaa"),
          py::arg("state"),
          R"doc(RWKV5 weighted key-value recurrence. Updates state in-place and returns output.)doc");

    m.def("rwkv5_wkv_",
          &op::rwkv5_wkv_,
          py::arg("out"),
          py::arg("receptance"),
          py::arg("key"),
          py::arg("value"),
          py::arg("time_decay"),
          py::arg("time_faaaa"),
          py::arg("state"),
          R"doc(Explicit-output RWKV5 weighted key-value recurrence. Updates out and state in-place.)doc");
}

} // namespace infinicore::ops
