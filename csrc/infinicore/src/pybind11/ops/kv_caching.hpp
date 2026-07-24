#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/kv_caching.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_kv_caching(py::module &m) {
    m.def("kv_caching_",
          &op::kv_caching_,
          py::arg("k_cache"),
          py::arg("v_cache"),
          py::arg("k"),
          py::arg("v"),
          py::arg("past_kv_lengths"),
          R"doc(In-place Key-Value Caching.

Updates the KV cache in-place with new key and value tensors.

Args:
    k_cache: Key cache tensor to update in-place
    v_cache: Value cache tensor to update in-place
    k: New key tensor to append
    v: New value tensor to append
    past_kv_lengths: Tensor containing current sequence lengths for each batch
)doc");
}

} // namespace infinicore::ops
