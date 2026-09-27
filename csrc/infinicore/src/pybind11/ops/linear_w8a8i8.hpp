#pragma once

#include <pybind11/pybind11.h>

#include "infinicore/ops/linear_w8a8i8.hpp"

namespace py = pybind11;

namespace infinicore::ops {

Tensor py_linear_w8a8i8(Tensor input,
                        Tensor weight_packed,
                        Tensor weight_scale,
                        pybind11::object bias) {
    std::optional<Tensor> bias_tensor = std::nullopt;
    if (!bias.is_none()) {
        bias_tensor = bias.cast<Tensor>();
    }
    return op::linear_w8a8i8(input, weight_packed, weight_scale, bias_tensor);
}

void py_linear_w8a8i8_(Tensor out,
                       Tensor input,
                       Tensor weight_packed,
                       Tensor weight_scale,
                       pybind11::object bias) {

    std::optional<Tensor> bias_tensor = std::nullopt;
    if (!bias.is_none()) {
        bias_tensor = bias.cast<Tensor>();
    }

    op::linear_w8a8i8_(out, input, weight_packed, weight_scale, bias_tensor);
}

inline void bind_linear_w8a8i8(py::module &m) {
    m.def("linear_w8a8i8",
          &ops::py_linear_w8a8i8,
          py::arg("input"),
          py::arg("weight_packed"),
          py::arg("weight_scale"),
          py::arg("bias") = py::none(),
          R"doc(linear_w8a8i8.)doc");
    m.def("linear_w8a8i8_",
          &ops::py_linear_w8a8i8_,
          py::arg("out"),
          py::arg("input"),
          py::arg("weight_packed"),
          py::arg("weight_scale"),
          py::arg("bias") = py::none(),
          R"doc(linear_w8a8i8_.)doc");
}

} // namespace infinicore::ops
