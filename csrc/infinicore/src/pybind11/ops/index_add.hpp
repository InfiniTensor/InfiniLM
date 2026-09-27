#pragma once

#include "infinicore/ops/index_add.hpp"
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_index_add(py::module &m) {
    m.def("index_add",
          &op::index_add,
          py::arg("input"),
          py::arg("dim"),
          py::arg("index"),
          py::arg("source"),
          py::arg("alpha") = 1.0f,
          R"doc(Accumulate elements of source into input by adding to the indices in the order given in index.
                Formula: output[index[i]] = input[index[i]] + alpha * source[i])doc");
    m.def("index_add_",
          &op::index_add_,
          py::arg("output"),
          py::arg("input"),
          py::arg("dim"),
          py::arg("index"),
          py::arg("source"),
          py::arg("alpha") = 1.0f,
          R"doc(In-place version of index_add. Writes result into output tensor.)doc");
}

} // namespace infinicore::ops
