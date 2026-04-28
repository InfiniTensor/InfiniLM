#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/attention.hpp"

namespace py = pybind11;

namespace infinicore::ops {

inline void bind_attention(py::module &m) {
    m.def("attention",
          &op::attention,
          py::arg("q"),
          py::arg("k"),
          py::arg("v"),
          py::arg("k_cache"),
          py::arg("v_cache"),
          py::arg("pos"),
          R"doc(Attention mechanism with KV caching.

Args:
    q: Query tensor
    k: Key tensor  
    v: Value tensor
    k_cache: Key cache tensor
    v_cache: Value cache tensor
    pos: Current position in the sequence

Returns:
    Output tensor from attention computation
)doc");

    m.def("attention_",
          &op::Attention::execute,
          py::arg("out"),
          py::arg("q"),
          py::arg("k"),
          py::arg("v"),
          py::arg("k_cache"),
          py::arg("v_cache"),
          py::arg("pos"),
          R"doc(In-place attention mechanism with KV caching.

Args:
    out: Output tensor
    q: Query tensor
    k: Key tensor
    v: Value tensor  
    k_cache: Key cache tensor
    v_cache: Value cache tensor
    pos: Current position in the sequence
)doc");
}

} // namespace infinicore::ops
