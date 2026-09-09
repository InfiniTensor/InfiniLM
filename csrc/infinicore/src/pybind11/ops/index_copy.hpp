#pragma once

#include "infinicore/ops/index_copy.hpp"
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_index_copy(py::module &m) {
    // 1. Out-of-place version (returns new tensor)
    m.def("index_copy",
          &op::index_copy,
          py::arg("input"),
          py::arg("dim"),
          py::arg("index"),
          py::arg("source"),
          R"doc(Copies elements of source into input at the indices given in index.
                Formula: output[index[i]] = source[i])doc");
    m.def("index_copy_",
          &op::index_copy_,
          py::arg("output"),
          py::arg("input"),
          py::arg("dim"),
          py::arg("index"),
          py::arg("source"),
          R"doc(In-place version of index_copy. Writes result into output tensor.)doc");
}

} // namespace infinicore::ops
