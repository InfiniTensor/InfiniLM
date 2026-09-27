#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/mha_kvcache.hpp"

namespace py = pybind11;

namespace infinicore::ops {

Tensor py_mha_kvcache(Tensor q,
                      Tensor k_cache,
                      Tensor v_cache,
                      Tensor seqlens_k,
                      Tensor block_table,
                      pybind11::object alibi_slopes,
                      float scale) {
    std::optional<Tensor> alibi_slopes_tensor = std::nullopt;
    if (!alibi_slopes.is_none()) {
        alibi_slopes_tensor = alibi_slopes.cast<Tensor>();
    }

    return op::mha_kvcache(
        q,
        k_cache,
        v_cache,
        seqlens_k,
        block_table,
        alibi_slopes_tensor,
        scale);
}

void py_mha_kvcache_(Tensor out,
                     Tensor q,
                     Tensor k_cache,
                     Tensor v_cache,
                     Tensor seqlens_k,
                     Tensor block_table,
                     pybind11::object alibi_slopes,
                     float scale) {
    std::optional<Tensor> alibi_slopes_tensor = std::nullopt;
    if (!alibi_slopes.is_none()) {
        alibi_slopes_tensor = alibi_slopes.cast<Tensor>();
    }

    op::mha_kvcache_(
        out,
        q,
        k_cache,
        v_cache,
        seqlens_k,
        block_table,
        alibi_slopes_tensor,
        scale);
}

inline void bind_mha_kvcache(py::module &m) {
    m.def(
        "mha_kvcache",
        &ops::py_mha_kvcache,
        py::arg("q"),
        py::arg("k_cache"),
        py::arg("v_cache"),
        py::arg("seqlens_k"),
        py::arg("block_table"),
        py::arg("alibi_slopes"),
        py::arg("scale"),
        R"doc(Flash attention KV-cache decode for single-step attention over a paged KV cache.

Parameters
----------
q : Tensor
    Query tensor of shape [batch_size, seqlen_q, num_heads, head_size]
k_cache : Tensor
    Key cache tensor of shape [num_blocks, block_size, num_heads_k, head_size] (paged layout)
v_cache : Tensor
    Value cache tensor of shape [num_blocks, block_size, num_heads_k, head_size] (paged layout)
seqlens_k : Tensor
    Total KV length per request of shape [batch_size] (int32)
block_table : Tensor
    Block mapping table of shape [batch_size, max_num_blocks_per_seq] (int32)
alibi_slopes : Optional[Tensor]
    ALiBi slopes tensor, if None then ALiBi is disabled
scale : float
    Scaling factor for attention scores (typically 1.0/sqrt(head_size))

Returns
-------
Tensor
    Output tensor of shape [batch_size, seqlen_q, num_heads, head_size]
)doc");

    m.def(
        "mha_kvcache_",
        &ops::py_mha_kvcache_,
        py::arg("out"),
        py::arg("q"),
        py::arg("k_cache"),
        py::arg("v_cache"),
        py::arg("seqlens_k"),
        py::arg("block_table"),
        py::arg("alibi_slopes"),
        py::arg("scale"),
        R"doc(In-place flash attention KV-cache decode.

Parameters
----------
out : Tensor
    Output tensor of shape [batch_size, seqlen_q, num_heads, head_size]
q : Tensor
    Query tensor of shape [batch_size, seqlen_q, num_heads, head_size]
k_cache : Tensor
    Key cache tensor of shape [num_blocks, block_size, num_heads_k, head_size] (paged layout)
v_cache : Tensor
    Value cache tensor of shape [num_blocks, block_size, num_heads_k, head_size] (paged layout)
seqlens_k : Tensor
    Total KV length per request of shape [batch_size] (int32)
block_table : Tensor
    Block mapping table of shape [batch_size, max_num_blocks_per_seq] (int32)
alibi_slopes : Optional[Tensor]
    ALiBi slopes tensor, if None then ALiBi is disabled
scale : float
    Scaling factor for attention scores (typically 1.0/sqrt(head_size))
)doc");
}

} // namespace infinicore::ops
