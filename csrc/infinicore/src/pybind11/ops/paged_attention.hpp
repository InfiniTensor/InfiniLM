#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/paged_attention.hpp"

namespace py = pybind11;

namespace infinicore::ops {

Tensor py_paged_attention(Tensor q, Tensor k_cache, Tensor v_cache, Tensor block_tables, Tensor cache_lens, pybind11::object alibi_slopes, float scale) {
    std::optional<Tensor> alibi_slopes_tensor = std::nullopt;
    if (!alibi_slopes.is_none()) {
        alibi_slopes_tensor = alibi_slopes.cast<Tensor>();
    }
    return op::paged_attention(q, k_cache, v_cache, block_tables, cache_lens, alibi_slopes_tensor, scale);
}

void py_paged_attention_(Tensor out, Tensor q, Tensor k_cache, Tensor v_cache, Tensor block_tables, Tensor cache_lens, pybind11::object alibi_slopes, float scale) {
    std::optional<Tensor> alibi_slopes_tensor = std::nullopt;
    if (!alibi_slopes.is_none()) {
        alibi_slopes_tensor = alibi_slopes.cast<Tensor>();
    }

    op::paged_attention_(out, q, k_cache, v_cache, block_tables, cache_lens, alibi_slopes_tensor, scale);
}

inline void bind_paged_attention(py::module &m) {
    m.def("paged_attention",
          &ops::py_paged_attention,
          py::arg("q"),
          py::arg("k_cache"),
          py::arg("v_cache"),
          py::arg("block_tables"),
          py::arg("cache_lens"),
          py::arg("alibi_slopes"),
          py::arg("scale"),
          R"doc(Paged attention of query and key cache tensors.)doc");

    m.def("paged_attention_",
          &ops::py_paged_attention_,
          py::arg("out"),
          py::arg("q"),
          py::arg("k_cache"),
          py::arg("v_cache"),
          py::arg("block_tables"),
          py::arg("cache_lens"),
          py::arg("alibi_slopes"),
          py::arg("scale"),
          R"doc(In-place paged attention of query and key cache tensors.)doc");
}

} // namespace infinicore::ops
