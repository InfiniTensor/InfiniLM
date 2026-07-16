#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/paged_caching.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_paged_caching(py::module &m) {
    m.def("paged_caching_",
          &op::paged_caching_,
          py::arg("k_cache"),
          py::arg("v_cache"),
          py::arg("k"),
          py::arg("v"),
          py::arg("slot_mapping"),
          R"doc(Paged caching of key and value tensors.)doc");
}

} // namespace infinicore::ops
