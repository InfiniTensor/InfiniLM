#pragma once

#include "infinicore/ops/paged_attention_prefill.hpp"
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace infinicore::ops {

Tensor py_paged_attention_prefill(Tensor q,
                                  Tensor k_cache,
                                  Tensor v_cache,
                                  Tensor block_tables,
                                  Tensor history_lens,
                                  Tensor cu_seqlens_q,
                                  py::object alibi_slopes,
                                  float scale) {
    std::optional<Tensor> alibi_slopes_tensor = std::nullopt;
    if (!alibi_slopes.is_none()) {
        alibi_slopes_tensor = alibi_slopes.cast<Tensor>();
    }
    return op::paged_attention_prefill(
        q, k_cache, v_cache, block_tables, history_lens, cu_seqlens_q, alibi_slopes_tensor, scale);
}

void py_paged_attention_prefill_(Tensor out,
                                 Tensor q,
                                 Tensor k_cache,
                                 Tensor v_cache,
                                 Tensor block_tables,
                                 Tensor history_lens,
                                 Tensor cu_seqlens_q,
                                 py::object alibi_slopes,
                                 float scale) {
    std::optional<Tensor> alibi_slopes_tensor = std::nullopt;
    if (!alibi_slopes.is_none()) {
        alibi_slopes_tensor = alibi_slopes.cast<Tensor>();
    }
    op::paged_attention_prefill_(out, q, k_cache, v_cache, block_tables, history_lens, cu_seqlens_q, alibi_slopes_tensor, scale);
}

inline void bind_paged_attention_prefill(py::module &m) {
    m.def("paged_attention_prefill",
          &ops::py_paged_attention_prefill,
          py::arg("q"),
          py::arg("k_cache"),
          py::arg("v_cache"),
          py::arg("block_tables"),
          py::arg("history_lens"),
          py::arg("cu_seqlens_q"),
          py::arg("alibi_slopes") = py::none(),
          py::arg("scale") = 1.0,
          R"doc(Paged attention prefill for packed variable-length queries.)doc");

    m.def("paged_attention_prefill_",
          &ops::py_paged_attention_prefill_,
          py::arg("out"),
          py::arg("q"),
          py::arg("k_cache"),
          py::arg("v_cache"),
          py::arg("block_tables"),
          py::arg("history_lens"),
          py::arg("cu_seqlens_q"),
          py::arg("alibi_slopes") = py::none(),
          py::arg("scale") = 1.0,
          R"doc(In-place paged attention prefill for packed variable-length queries.)doc");
}

} // namespace infinicore::ops
