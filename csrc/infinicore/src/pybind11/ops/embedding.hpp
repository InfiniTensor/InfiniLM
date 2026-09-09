#pragma once

#include "infinicore/ops/embedding.hpp"
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_embedding(py::module &m) {

    m.def("embedding",
          &op::embedding,
          py::arg("input"),
          py::arg("weight"),
          R"doc(Generate a simple lookup table that looks up embeddings in a fixed dictionary and size..)doc");

    m.def("embedding_",
          &op::embedding_,
          py::arg("out"),
          py::arg("input"),
          py::arg("weight"),
          R"doc(In-place, Generate a simple lookup table that looks up embeddings in a fixed dictionary and size..)doc");
}

} // namespace infinicore::ops
