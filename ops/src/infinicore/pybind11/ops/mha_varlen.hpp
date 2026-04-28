#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/mha_varlen.hpp"

namespace py = pybind11;

namespace infinicore::ops {

Tensor py_mha_varlen(Tensor q,
                     Tensor k,
                     Tensor v,
                     Tensor cum_seqlens_q,
                     Tensor cum_seqlens_k,
                     Tensor block_table,
                     int max_seqlen_q,
                     int max_seqlen_k,
                     pybind11::object alibi_slopes,
                     float scale) {
    std::optional<Tensor> alibi_slopes_tensor = std::nullopt;
    if (!alibi_slopes.is_none()) {
        alibi_slopes_tensor = alibi_slopes.cast<Tensor>();
    }

    return op::mha_varlen(
        q,
        k,
        v,
        cum_seqlens_q,
        cum_seqlens_k,
        block_table,
        max_seqlen_q,
        max_seqlen_k,
        alibi_slopes_tensor,
        scale);
}

void py_mha_varlen_(Tensor out,
                    Tensor q,
                    Tensor k,
                    Tensor v,
                    Tensor cum_seqlens_q,
                    Tensor cum_seqlens_k,
                    Tensor block_table,
                    int max_seqlen_q,
                    int max_seqlen_k,
                    pybind11::object alibi_slopes,
                    float scale) {
    std::optional<Tensor> alibi_slopes_tensor = std::nullopt;
    if (!alibi_slopes.is_none()) {
        alibi_slopes_tensor = alibi_slopes.cast<Tensor>();
    }

    op::mha_varlen_(
        out,
        q,
        k,
        v,
        cum_seqlens_q,
        cum_seqlens_k,
        block_table,
        max_seqlen_q,
        max_seqlen_k,
        alibi_slopes_tensor,
        scale);
}

inline void bind_mha_varlen(py::module &m) {
    m.def(
        "mha_varlen",
        &ops::py_mha_varlen,
        py::arg("q"),
        py::arg("k"),
        py::arg("v"),
        py::arg("cum_seqlens_q"),
        py::arg("cum_seqlens_k"),
        py::arg("block_table"),
        py::arg("max_seqlen_q"),
        py::arg("max_seqlen_k"),
        py::arg("alibi_slopes"),
        py::arg("scale"),
        R"doc(Variable-length multi-head attention.)doc");

    m.def(
        "mha_varlen_",
        &ops::py_mha_varlen_,
        py::arg("out"),
        py::arg("q"),
        py::arg("k"),
        py::arg("v"),
        py::arg("cum_seqlens_q"),
        py::arg("cum_seqlens_k"),
        py::arg("block_table"),
        py::arg("max_seqlen_q"),
        py::arg("max_seqlen_k"),
        py::arg("alibi_slopes"),
        py::arg("scale"),
        R"doc(In-place variable-length multi-head attention.)doc");
}

} // namespace infinicore::ops
