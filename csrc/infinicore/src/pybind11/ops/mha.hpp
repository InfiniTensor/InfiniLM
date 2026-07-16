#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/mha.hpp"

namespace py = pybind11;

namespace infinicore::ops {

Tensor py_mha(Tensor q,
              Tensor k,
              Tensor v,
              pybind11::object alibi_slopes,
              float scale,
              bool is_causal) {
    std::optional<Tensor> alibi_slopes_tensor = std::nullopt;
    if (!alibi_slopes.is_none()) {
        alibi_slopes_tensor = alibi_slopes.cast<Tensor>();
    }

    return op::mha(
        q,
        k,
        v,
        alibi_slopes_tensor,
        scale,
        is_causal);
}

void py_mha_(Tensor out,
             Tensor q,
             Tensor k,
             Tensor v,
             pybind11::object alibi_slopes,
             float scale,
             bool is_causal) {
    std::optional<Tensor> alibi_slopes_tensor = std::nullopt;
    if (!alibi_slopes.is_none()) {
        alibi_slopes_tensor = alibi_slopes.cast<Tensor>();
    }

    op::mha_(
        out,
        q,
        k,
        v,
        alibi_slopes_tensor,
        scale,
        is_causal);
}

inline void bind_mha(py::module &m) {
    m.def(
        "mha",
        &ops::py_mha,
        py::arg("q"),
        py::arg("k"),
        py::arg("v"),
        py::arg("alibi_slopes"),
        py::arg("scale"),
        py::arg("is_causal"),
        R"doc(Variable-length multi-head attention.)doc");

    m.def(
        "mha_",
        &ops::py_mha_,
        py::arg("out"),
        py::arg("q"),
        py::arg("k"),
        py::arg("v"),
        py::arg("alibi_slopes"),
        py::arg("scale"),
        py::arg("is_causal"),
        R"doc(In-place variable-length multi-head attention.)doc");
}

} // namespace infinicore::ops
