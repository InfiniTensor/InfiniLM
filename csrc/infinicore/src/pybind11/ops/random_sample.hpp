#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/random_sample.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_random_sample(py::module &m) {
    m.def("random_sample",
          &op::random_sample,
          py::arg("logits"),
          py::arg("random_val"),
          py::arg("topp"),
          py::arg("topk"),
          py::arg("temperature"),
          R"doc(Random sampling: returns an int32 scalar index.)doc");

    m.def("random_sample_",
          &op::random_sample_,
          py::arg("indices"),
          py::arg("logits"),
          py::arg("random_val"),
          py::arg("topp"),
          py::arg("topk"),
          py::arg("temperature"),
          R"doc(In-place random sampling into provided int32 scalar tensor.)doc");
}

} // namespace infinicore::ops
