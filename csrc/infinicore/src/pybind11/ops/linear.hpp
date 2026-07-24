#pragma once

#include "infinicore/ops/linear.hpp"

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace infinicore::ops {

Tensor py_linear(Tensor input,
                 Tensor weight,
                 pybind11::object bias,
                 float alpha = 1.0f) {
    std::optional<Tensor> bias_tensor = std::nullopt;
    if (!bias.is_none()) {
        bias_tensor = bias.cast<Tensor>();
    }
    return op::linear(input, weight, bias_tensor, alpha);
}

void py_linear_(Tensor out,
                Tensor input,
                Tensor weight,
                pybind11::object bias,
                float alpha = 1.0f) {

    std::optional<Tensor> bias_tensor = std::nullopt;
    if (!bias.is_none()) {
        bias_tensor = bias.cast<Tensor>();
    }

    op::linear_(out, input, weight, bias_tensor, alpha);
}

inline void bind_linear(py::module &m) {

    m.def("linear",
          &ops::py_linear,
          py::arg("input"),
          py::arg("weight"),
          py::arg("bias") = py::none(),
          py::arg("alpha") = 1.0f,
          R"doc(Applies a linear transformation to the incoming data: y=alpha*xA^T+b.)doc");

    m.def("linear_",
          &ops::py_linear_,
          py::arg("out"),
          py::arg("input"),
          py::arg("weight"),
          py::arg("bias") = py::none(),
          py::arg("alpha") = 1.0f,
          R"doc(In-place, applies a linear transformation to the incoming data: y=alpha*xA^T+b.)doc");
}

} // namespace infinicore::ops
